"""Train and evaluate the market value models.

Run it after the features step::

    python -m football_value.model

Three models are compared on the same held-out test set:

* ``baseline``: always predicts the typical market value, ignoring the player;
* ``linear``: linear regression on the five features of the project (goals,
  assists, minutes played, age, position);
* ``extended``: the same plus the league and age squared, two additions
  suggested by the exploratory analysis.

Every model predicts the *logarithm* of the market value, because the value
itself is heavily skewed. Predictions are converted back to euros before the
errors are measured.

Outputs:

* ``reports/metrics.csv``: test metrics of the three models;
* ``reports/coefficients.csv``: coefficients of the ``linear`` model;
* ``data/processed/predictions.csv``: a prediction for every player, always
  made by a model that did not see that player during training.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler

from football_value import config
from football_value.features import CATEGORICAL_FEATURES, FEATURES_FILE, NUMERIC_FEATURES, TARGET

logger = logging.getLogger(__name__)

REPORTS_DIR = config.PROJECT_ROOT / "reports"
METRICS_FILE = REPORTS_DIR / "metrics.csv"
COEFFICIENTS_FILE = REPORTS_DIR / "coefficients.csv"
PREDICTIONS_FILE = config.PROCESSED_DATA_DIR / "predictions.csv"

CV_FOLDS = 5


@dataclass(frozen=True)
class ModelSpec:
    """Which columns a model uses, and how.

    Attributes:
        name: short identifier used in the reports.
        numeric: numeric columns, standardised before the regression.
        categorical: text columns, one-hot encoded.
        squared: numeric columns that also enter the model squared, to let
            their effect bend instead of being a straight line.
    """

    name: str
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    squared: list[str] = field(default_factory=list)

    @property
    def columns(self) -> list[str]:
        return [*self.numeric, *self.squared, *self.categorical]


BASELINE = ModelSpec(name="baseline")
LINEAR = ModelSpec(name="linear", numeric=NUMERIC_FEATURES, categorical=CATEGORICAL_FEATURES)
EXTENDED = ModelSpec(
    name="extended",
    numeric=["goals", "assists", "minutes_played"],
    categorical=[*CATEGORICAL_FEATURES, "competition"],
    squared=["age"],
)
MODEL_SPECS = (BASELINE, LINEAR, EXTENDED)


def build_model(spec: ModelSpec) -> TransformedTargetRegressor:
    """Build an untrained model that predicts euros through a log transform.

    All preprocessing lives inside the scikit-learn pipeline, so it is fitted
    on the training data only: nothing leaks from the test set.
    """
    if not spec.columns:
        # Predicts the mean of the log values: the "typical" player.
        regressor: Pipeline | DummyRegressor = DummyRegressor(strategy="mean")
    else:
        preprocessing = ColumnTransformer(
            [
                ("numeric", StandardScaler(), spec.numeric),
                (
                    "squared",
                    make_pipeline(
                        PolynomialFeatures(degree=2, include_bias=False), StandardScaler()
                    ),
                    spec.squared,
                ),
                # drop="first": one category is the reference the others are compared to.
                ("categorical", OneHotEncoder(drop="first"), spec.categorical),
            ],
            verbose_feature_names_out=False,
        )
        regressor = Pipeline([("preprocessing", preprocessing), ("regression", LinearRegression())])

    return TransformedTargetRegressor(regressor=regressor, func=np.log, inverse_func=np.exp)


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


def coefficient_table(model: TransformedTargetRegressor) -> pd.DataFrame:
    """Return the coefficients of a fitted linear model in a readable form.

    Because the target is a logarithm, a coefficient ``c`` means the market
    value is multiplied by ``exp(c)``. ``effect_percent`` expresses that as a
    percentage change:

    * for a numeric feature, per standard deviation of the feature (they are
      standardised, which also makes them comparable with each other);
    * for a category, compared with the reference category.

    ``one_std`` gives the size of one standard deviation of a numeric feature
    in its own unit (goals, minutes, years), measured on the training set.
    """
    pipeline = model.regressor_
    preprocessing = pipeline.named_steps["preprocessing"]
    names = preprocessing.get_feature_names_out()
    coefficients = pipeline.named_steps["regression"].coef_

    scaler = preprocessing.named_transformers_["numeric"]
    one_std = dict(zip(scaler.feature_names_in_, scaler.scale_, strict=True))

    table = pd.DataFrame({"feature": names, "coefficient": coefficients})
    table["one_std"] = table["feature"].map(one_std)
    table["effect_percent"] = (np.exp(table["coefficient"]) - 1) * 100
    return table.sort_values("coefficient", key=np.abs, ascending=False).reset_index(drop=True)


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

    results = {spec.name: train_and_evaluate(spec, train, test) for spec in MODEL_SPECS}
    metrics = pd.DataFrame([result[1] for result in results.values()])
    linear_model = results[LINEAR.name][0]
    coefficients = coefficient_table(linear_model)
    predictions = out_of_sample_predictions(LINEAR, linear_model, train, test)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.round(4).to_csv(METRICS_FILE, index=False)
    coefficients.round(4).to_csv(COEFFICIENTS_FILE, index=False)
    predictions.to_csv(PREDICTIONS_FILE, index=False)

    logger.info("Test metrics:\n%s", metrics.round(3).to_string(index=False))
    logger.info(
        "Coefficients of the linear model:\n%s", coefficients.round(3).to_string(index=False)
    )


if __name__ == "__main__":
    main()
