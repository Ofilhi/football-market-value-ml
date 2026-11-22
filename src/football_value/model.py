"""Train and evaluate the market value models.

Run it after the features step::

    python -m football_value.model

Five models are compared on the same held-out test set:

* ``baseline``: always predicts the typical market value, ignoring the player;
* ``linear``: plain linear regression on the five original features (goals,
  assists, minutes played, age, position): the reference model;
* ``ridge``: the **main model**. A regularised linear regression on the five
  features plus the league, the strength of the player's team (league table,
  Champions League) and his activity, all from football-data.org. Age may
  follow a curve instead of a straight line;
* ``ridge_adjusted``: the ``ridge`` model with goals and assists multiplied by a
  factor for the difficulty of the league (see ``config.LEAGUE_DIFFICULTY``),
  to test whether a goal is worth more in a harder league;
* ``ridge_tm``: the same plus optional factors from Transfermarkt (career
  history, club in Europe, contract, international caps);
* ``boosting``: gradient boosting on the ``ridge`` factors, kept as a
  comparison to see how much a non-linear model would add.

Every model predicts the *logarithm* of the market value, because the value
itself is heavily skewed. Predictions are converted back to euros before the
errors are measured.

Outputs:

* ``reports/metrics.csv``: test and cross-validation metrics of every model;
* ``reports/errors_by_position.csv``: cross-validated error by position;
* ``reports/coefficients.csv``: coefficients of the main model;
* ``data/processed/predictions.csv``: a prediction for every player, always
  made by a model that did not see that player during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from football_value import config
from football_value.features import (
    ACTIVITY_COLUMNS,
    CATEGORICAL_FEATURES,
    FEATURES_FILE,
    LEAGUE_ADJUSTED_COLUMNS,
    NUMERIC_FEATURES,
    TARGET,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = config.PROJECT_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "metrics.csv"
POSITION_ERRORS_FILE = REPORTS_DIR / "errors_by_position.csv"
COEFFICIENTS_FILE = REPORTS_DIR / "coefficients.csv"
PREDICTIONS_FILE = config.PROCESSED_DATA_DIR / "predictions.csv"

CV_FOLDS = 5
CV_REPEATS = 3
CURVE_DEGREE = 3  # an age curve: age, age squared, age cubed
RIDGE_STRENGTHS = np.logspace(-1, 3, 20)  # tried by cross-validation inside each fit

Estimator = Literal["mean", "ols", "ridge", "boosting"]


@dataclass(frozen=True)
class ModelSpec:
    """Which columns a model uses, how they are prepared, and the estimator.

    Attributes:
        name: short identifier used in the reports.
        numeric: numeric columns, standardised (missing values are replaced
            by the training median first).
        logged: skewed counts, transformed with log(1 + x) then standardised.
        curved: columns that enter as x, x² and x³, to let their effect bend
            instead of being a straight line.
        categorical: text columns, one-hot encoded.
        estimator: ``mean`` (always the same value), ``ols`` (plain linear
            regression), ``ridge`` (linear regression with a penalty on large
            coefficients, which keeps correlated factors stable) or
            ``boosting`` (gradient boosting).
    """

    name: str
    numeric: list[str] = field(default_factory=list)
    logged: list[str] = field(default_factory=list)
    curved: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    estimator: Estimator = "ridge"

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.logged, *self.curved, *self.categorical]


# Factors from football-data.org, on top of the five original features.
TEAM_STRENGTH = [
    "points_per_game",
    "goal_difference_per_game",
    "table_position",
    "win_rate",
    "cl_games",
    "cl_stage",
]
API_NUMERIC = ["goals", "assists", "minutes_played", *ACTIVITY_COLUMNS, *TEAM_STRENGTH]
API_CATEGORICAL = [*CATEGORICAL_FEATURES, "competition"]

# Optional factors from Transfermarkt.
TM_NUMERIC = [
    "club_cl_games",
    "club_el_games",
    "club_prev_cl_games",
    "club_prev_euro_games",
    "career_seasons",
    "prev_goals",
    "prev_assists",
    "matches_all",
    "min_per_match",
    "contract_months",
    "contract_missing",
]
TM_LOGGED = ["prev_minutes", "career_minutes", "minutes_all", "caps", "intl_goals"]

BASELINE = ModelSpec(name="baseline", estimator="mean")
LINEAR = ModelSpec(
    name="linear",
    numeric=NUMERIC_FEATURES,
    categorical=CATEGORICAL_FEATURES,
    estimator="ols",
)
RIDGE = ModelSpec(name="ridge", numeric=API_NUMERIC, curved=["age"], categorical=API_CATEGORICAL)
RIDGE_TM = ModelSpec(
    name="ridge_tm",
    numeric=[*API_NUMERIC, *TM_NUMERIC],
    logged=TM_LOGGED,
    curved=["age"],
    categorical=API_CATEGORICAL,
)
# Goals, assists and goals per match replaced by their league-adjusted versions.
_RAW_STATS = ("goals", "assists", "goals_per_match")
RIDGE_ADJUSTED = ModelSpec(
    name="ridge_adjusted",
    numeric=[
        *[column for column in API_NUMERIC if column not in _RAW_STATS],
        *LEAGUE_ADJUSTED_COLUMNS,
    ],
    curved=["age"],
    categorical=API_CATEGORICAL,
)
BOOSTING = ModelSpec(
    name="boosting",
    numeric=API_NUMERIC,
    curved=["age"],
    categorical=API_CATEGORICAL,
    estimator="boosting",
)
MODEL_SPECS = (BASELINE, LINEAR, RIDGE, RIDGE_ADJUSTED, RIDGE_TM, BOOSTING)
MAIN_SPEC = RIDGE


def build_model(spec: ModelSpec) -> TransformedTargetRegressor:
    """Build an untrained model that predicts euros through a log transform.

    All preprocessing lives inside the scikit-learn pipeline, so it is fitted
    on the training data only: nothing leaks from the test set.
    """
    if spec.estimator == "mean":
        # Predicts the mean of the log values: the "typical" player.
        regressor: Pipeline | DummyRegressor = DummyRegressor(strategy="mean")
    else:
        regressor = Pipeline(
            [
                ("preprocessing", _preprocessing(spec)),
                ("regression", _estimator(spec.estimator)),
            ]
        )
    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


def _preprocessing(spec: ModelSpec) -> ColumnTransformer:
    parts = []
    if spec.numeric:
        parts.append(
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                spec.numeric,
            )
        )
    if spec.logged:
        log = FunctionTransformer(np.log1p, feature_names_out="one-to-one")
        parts.append(("logged", make_pipeline(log, StandardScaler()), spec.logged))
    if spec.curved:
        curve = PolynomialFeatures(degree=CURVE_DEGREE, include_bias=False)
        parts.append(("curved", make_pipeline(curve, StandardScaler()), spec.curved))
    if spec.categorical:
        # With plain regression, one category per column is the reference the
        # others are compared to. A ridge penalty makes the choice of reference
        # arbitrary, so every category is kept.
        encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first" if spec.estimator == "ols" else None,
        )
        parts.append(("categorical", encoder, spec.categorical))
    return ColumnTransformer(parts)


def _estimator(kind: Estimator):
    if kind == "ols":
        return LinearRegression()
    if kind == "ridge":
        return RidgeCV(alphas=RIDGE_STRENGTHS)
    if kind == "boosting":
        return HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=300, random_state=config.RANDOM_SEED
        )
    raise ValueError(f"Unknown estimator: {kind}")


def split_players(players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the players into a training set and a test set, reproducibly."""
    return train_test_split(players, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED)


def evaluate(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Compute the error metrics of a set of predictions.

    * ``mae_eur``: mean absolute error, the typical miss in euros;
    * ``rmse_eur``: root mean squared error, which punishes big misses more;
    * ``r2``: share of the variance of the market value that is explained;
    * ``r2_log``: the same on the log scale, the scale the model is trained on;
    * ``median_pct_error``: the typical miss as a share of the player's real
      value. Unlike the three metrics in euros, it is not dominated by the
      handful of players worth more than €100M.
    """
    return {
        "median_pct_error": float(np.median(np.abs(predicted - actual) / actual) * 100),
        "mae_eur": mean_absolute_error(actual, predicted),
        "rmse_eur": root_mean_squared_error(actual, predicted),
        "r2": r2_score(actual, predicted),
        "r2_log": r2_score(np.log(actual), np.log(predicted)),
    }


def train_and_evaluate(
    spec: ModelSpec, train: pd.DataFrame, test: pd.DataFrame
) -> tuple[TransformedTargetRegressor, dict[str, float]]:
    """Fit a model on the training set and measure it on the test set."""
    model = build_model(spec).fit(train[spec.columns], train[TARGET])
    metrics = evaluate(test[TARGET], model.predict(test[spec.columns]))
    return model, {"model": spec.name, **metrics}


def cross_validate(
    spec: ModelSpec, players: pd.DataFrame, repeats: int = CV_REPEATS
) -> tuple[dict[str, float], pd.Series]:
    """Repeated k-fold cross-validation on all the players.

    The test set is small (274 players), so one split gives a noisy picture.
    Here every player is predicted by a model that did not see him, and the
    whole procedure is repeated with different folds and averaged.

    Returns:
        The averaged metrics, and the typical percentage error by position.
    """
    metrics, by_position = [], []
    for repeat in range(repeats):
        folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED + repeat)
        predicted = cross_val_predict(
            build_model(spec), players[spec.columns], players[TARGET], cv=folds
        )
        metrics.append(evaluate(players[TARGET], predicted))
        errors = np.abs(predicted - players[TARGET]) / players[TARGET] * 100
        by_position.append(errors.groupby(players["position"]).median())
    return pd.DataFrame(metrics).mean().to_dict(), pd.concat(by_position, axis=1).mean(axis=1)


def coefficient_table(model: TransformedTargetRegressor) -> pd.DataFrame:
    """Return the coefficients of a fitted linear model in a readable form.

    Because the target is a logarithm, a coefficient ``c`` means the market
    value is multiplied by ``exp(c)``. ``effect_percent`` expresses that as a
    percentage change, for one standard deviation of a numeric factor (they
    are standardised, which makes them comparable with each other).

    Two cautions when reading the table:

    * the factors are correlated (a club that plays in the Champions League
      also has more points), so a coefficient is the effect of a factor
      *with the others held constant*, not its total effect;
    * the age curve (age, age², age³) only makes sense as a whole: its three
      coefficients cannot be read one by one.

    ``one_std`` is the size of one standard deviation of a plain numeric
    factor in its own unit, measured on the training set.
    """
    pipeline = model.regressor_
    preprocessing = pipeline.named_steps["preprocessing"]
    coefficients = pipeline.named_steps["regression"].coef_

    names = [_readable_name(name) for name in preprocessing.get_feature_names_out()]
    table = pd.DataFrame({"feature": names, "coefficient": coefficients})

    one_std: dict[str, float] = {}
    for group, transformer, columns in preprocessing.transformers_:
        if group == "numeric":
            scale = transformer.named_steps["standardscaler"].scale_
            one_std = dict(zip(columns, scale, strict=True))
    table["one_std"] = table["feature"].map(one_std)

    table["effect_percent"] = (np.exp(table["coefficient"]) - 1) * 100
    return table.sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


def _readable_name(name: str) -> str:
    """Turn ``logged__caps`` into ``log(caps)`` and ``curved__age^2`` into ``age²``."""
    group, _, feature = name.partition("__")
    if group == "logged":
        return f"log({feature})"
    if group == "curved":
        return feature.replace("^2", "²").replace("^3", "³")
    return feature


def out_of_sample_predictions(
    spec: ModelSpec,
    model: TransformedTargetRegressor,
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> pd.DataFrame:
    """Predict every player with a model that was not trained on that player.

    Test players are predicted by the final model. Training players are
    predicted by cross-validation: the training set is cut into folds and
    each fold is predicted by a model fitted on the other folds.
    """
    folds = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    train_predictions = cross_val_predict(
        build_model(spec), train[spec.columns], train[TARGET], cv=folds
    )

    predictions = pd.concat(
        [
            train.assign(split="train", predicted_value_eur=train_predictions),
            test.assign(split="test", predicted_value_eur=model.predict(test[spec.columns])),
        ]
    )
    predictions["predicted_value_eur"] = predictions["predicted_value_eur"].round(-3).astype(int)
    # Above 1: the model thinks the player is worth more than the market says.
    predictions["predicted_to_actual_ratio"] = (
        predictions["predicted_value_eur"] / predictions[TARGET]
    ).round(3)
    return predictions.sort_values("player_id").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    players = pd.read_csv(FEATURES_FILE)
    train, test = split_players(players)
    logger.info("%d players: %d for training, %d for testing", len(players), len(train), len(test))

    fitted, rows, by_position = {}, [], {}
    for spec in MODEL_SPECS:
        model, test_metrics = train_and_evaluate(spec, train, test)
        cv_metrics, position_errors = cross_validate(spec, players)
        fitted[spec.name] = model
        rows.append(
            {
                **test_metrics,
                "cv_median_pct_error": cv_metrics["median_pct_error"],
                "cv_r2_log": cv_metrics["r2_log"],
            }
        )
        by_position[spec.name] = position_errors

    metrics = pd.DataFrame(rows)
    position_errors = pd.DataFrame(by_position).T.rename_axis("model")
    coefficients = coefficient_table(fitted[MAIN_SPEC.name])
    predictions = out_of_sample_predictions(MAIN_SPEC, fitted[MAIN_SPEC.name], train, test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(METRICS_FILE, index=False)
    position_errors.round(2).to_csv(POSITION_ERRORS_FILE)
    coefficients.round(4).to_csv(COEFFICIENTS_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    logger.info("Metrics:\n%s", metrics.round(3).to_string(index=False))
    logger.info("Typical error by position (cross-validated, %%):\n%s", position_errors.round(1))
    logger.info("Largest coefficients of '%s':\n%s", MAIN_SPEC.name, coefficients.head(12).round(3))


if __name__ == "__main__":
    main()
