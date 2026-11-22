import numpy as np
import pandas as pd
import pytest

from football_value import model
from football_value.features import OUTPUT_COLUMNS, TARGET


@pytest.fixture
def players():
    """Synthetic players whose value follows a known rule plus a little noise.

    log(value) = 16 + 0.10 * goals - 0.08 * age + 0.3 * points per game
                 (+ 0.5 for midfielders)
    """
    rng = np.random.default_rng(0)
    size = 300
    frame = pd.DataFrame(
        {
            "player_id": range(size),
            "position": rng.choice(["Attack", "Midfield", "Defence"], size),
            "competition": rng.choice(["PL", "PD"], size),
            "goals": rng.integers(1, 25, size),
            "assists": rng.integers(0, 12, size),
            "minutes_played": rng.integers(100, 3400, size),
            "age": rng.uniform(18, 36, size),
            "played_matches": rng.integers(5, 38, size),
            "availability": rng.uniform(0.1, 1, size),
            "goals_per_match": rng.uniform(0.03, 1, size),
            "table_position": rng.integers(1, 21, size),
            "points_per_game": rng.uniform(0.5, 2.5, size),
            "goal_difference_per_game": rng.uniform(-1, 1.2, size),
            "win_rate": rng.uniform(0.1, 0.7, size),
            "cl_games": rng.integers(0, 17, size),
            "cl_stage": rng.integers(0, 7, size),
        }
    )
    multiplier = rng.uniform(0.7, 1.4, size)  # stands for the league difficulty factor
    frame["goals_adjusted"] = frame["goals"] * multiplier
    frame["assists_adjusted"] = frame["assists"] * multiplier
    frame["adjusted_goals_per_match"] = frame["goals_per_match"] * multiplier
    # Optional Transfermarkt factors: noise that the Ridge penalty should learn to ignore.
    for column in [*model.TM_NUMERIC, *model.TM_LOGGED]:
        frame[column] = rng.integers(0, 50, size).astype(float)
    frame.loc[::10, "contract_months"] = np.nan  # some contracts are unknown

    log_value = (
        16
        + 0.10 * frame["goals"]
        - 0.08 * frame["age"]
        + 0.3 * frame["points_per_game"]
        + 0.5 * (frame["position"] == "Midfield")
        + rng.normal(0, 0.05, size)
    )
    frame[TARGET] = np.exp(log_value)
    return frame


def test_every_model_only_uses_columns_of_the_features_table():
    for spec in model.MODEL_SPECS:
        assert set(spec.columns) <= set(OUTPUT_COLUMNS), spec.name


def test_split_is_reproducible_and_has_no_overlap(players):
    train, test = model.split_players(players)
    train_again, _ = model.split_players(players)

    assert len(test) == 60
    assert set(train["player_id"]).isdisjoint(test["player_id"])
    assert list(train["player_id"]) == list(train_again["player_id"])


def test_evaluate_on_perfect_predictions():
    actual = pd.Series([1e6, 5e6, 2e7])

    metrics = model.evaluate(actual, actual.to_numpy())

    assert metrics["mae_eur"] == 0
    assert metrics["rmse_eur"] == 0
    assert metrics["median_pct_error"] == 0
    assert metrics["r2"] == pytest.approx(1.0)
    assert metrics["r2_log"] == pytest.approx(1.0)


def test_median_pct_error_is_relative_to_the_actual_value():
    actual = pd.Series([10e6, 10e6, 10e6])
    predicted = np.array([11e6, 5e6, 14e6])

    assert model.evaluate(actual, predicted)["median_pct_error"] == pytest.approx(40.0)


def test_baseline_predicts_the_same_value_for_everyone(players):
    train, test = model.split_players(players)

    baseline, _ = model.train_and_evaluate(model.BASELINE, train, test)
    predictions = baseline.predict(test[model.BASELINE.columns])

    assert len(set(predictions.round(2))) == 1
    # The mean of the logs, converted back to euros.
    assert predictions[0] == pytest.approx(np.exp(np.log(train[TARGET]).mean()))


def test_linear_model_beats_the_baseline(players):
    train, test = model.split_players(players)

    _, baseline_metrics = model.train_and_evaluate(model.BASELINE, train, test)
    _, linear_metrics = model.train_and_evaluate(model.LINEAR, train, test)

    assert linear_metrics["r2_log"] > 0.8
    assert linear_metrics["mae_eur"] < baseline_metrics["mae_eur"]


def test_team_factors_improve_on_the_five_original_features(players):
    train, test = model.split_players(players)

    _, linear_metrics = model.train_and_evaluate(model.LINEAR, train, test)
    _, ridge_metrics = model.train_and_evaluate(model.RIDGE, train, test)

    # The rule depends on points per game, which only the ridge model can see.
    assert ridge_metrics["r2_log"] > linear_metrics["r2_log"]
    assert ridge_metrics["r2_log"] > 0.95


def test_coefficients_recover_the_rule_behind_the_data(players):
    train, test = model.split_players(players)
    linear, _ = model.train_and_evaluate(model.LINEAR, train, test)

    table = model.coefficient_table(linear).set_index("feature")

    # Numeric coefficients are per standard deviation: divide to get per unit.
    assert table.loc["goals", "coefficient"] / table.loc["goals", "one_std"] == pytest.approx(
        0.10, abs=0.01
    )
    assert table.loc["age", "coefficient"] / table.loc["age", "one_std"] == pytest.approx(
        -0.08, abs=0.01
    )
    # Midfielders are worth exp(0.5) - 1 = 65% more than the reference (Attack). The tolerance is
    # wide because this model does not see points per game, which adds noise on 240 players.
    assert table.loc["position_Midfield", "effect_percent"] == pytest.approx(65, abs=10)
    assert abs(table.loc["assists", "coefficient"]) < 0.02


def test_ridge_coefficient_names_are_readable(players):
    train, test = model.split_players(players)
    ridge_tm, _ = model.train_and_evaluate(model.RIDGE_TM, train, test)

    names = set(model.coefficient_table(ridge_tm)["feature"])

    assert {"age", "age²", "age³", "competition_PL", "log(caps)", "points_per_game"} <= names
    assert not any("__" in name for name in names)


def test_missing_contract_months_are_filled_in_not_rejected(players):
    train, test = model.split_players(players)
    assert train["contract_months"].isna().any()

    fitted, _ = model.train_and_evaluate(model.RIDGE_TM, train, test)
    predictions = fitted.predict(test[model.RIDGE_TM.columns])

    assert np.isfinite(predictions).all()


def test_league_adjusted_model_swaps_raw_statistics_for_adjusted_ones():
    adjusted = set(model.RIDGE_ADJUSTED.columns)

    assert {"goals_adjusted", "assists_adjusted", "adjusted_goals_per_match"} <= adjusted
    assert not adjusted & {"goals", "assists", "goals_per_match"}
    # Everything else is the same as in the main model.
    assert adjusted - {"goals_adjusted", "assists_adjusted", "adjusted_goals_per_match"} == set(
        model.RIDGE.columns
    ) - {"goals", "assists", "goals_per_match"}


def test_league_adjusted_model_trains(players):
    train, test = model.split_players(players)

    _, metrics = model.train_and_evaluate(model.RIDGE_ADJUSTED, train, test)

    # Lower than for the main model here: the data's rule uses raw goals, and the
    # adjusted goals of this fixture are a noisy version of them.
    assert metrics["r2_log"] > 0.8


def test_boosting_model_trains_on_the_same_factors_as_ridge(players):
    train, test = model.split_players(players)

    _, metrics = model.train_and_evaluate(model.BOOSTING, train, test)

    assert model.BOOSTING.columns == model.RIDGE.columns
    assert metrics["r2_log"] > 0.5


def test_cross_validation_reports_the_error_of_every_position(players):
    metrics, by_position = model.cross_validate(model.LINEAR, players, repeats=2)

    assert metrics["r2_log"] > 0.8
    assert set(by_position.index) == {"Attack", "Midfield", "Defence"}
    assert (by_position > 0).all()


def test_every_player_gets_one_out_of_sample_prediction(players):
    train, test = model.split_players(players)
    linear, _ = model.train_and_evaluate(model.LINEAR, train, test)

    predictions = model.out_of_sample_predictions(model.LINEAR, linear, train, test)

    assert len(predictions) == len(players)
    assert predictions["player_id"].is_unique
    assert predictions["split"].value_counts().to_dict() == {"train": 240, "test": 60}
    assert (predictions["predicted_value_eur"] > 0).all()
