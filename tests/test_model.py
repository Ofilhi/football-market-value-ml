import numpy as np
import pandas as pd
import pytest

from football_value import model
from football_value.features import TARGET


@pytest.fixture
def players():
    """Synthetic players whose value follows a known rule plus a little noise.

    log(value) = 16 + 0.10 * goals - 0.08 * age (+ 0.5 for midfielders)
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
        }
    )
    log_value = (
        16
        + 0.10 * frame["goals"]
        - 0.08 * frame["age"]
        + 0.5 * (frame["position"] == "Midfield")
        + rng.normal(0, 0.05, size)
    )
    frame[TARGET] = np.exp(log_value)
    return frame


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

    assert linear_metrics["r2_log"] > 0.95
    assert linear_metrics["mae_eur"] < baseline_metrics["mae_eur"]


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
    # Midfielders are worth exp(0.5) - 1 = 65% more than the reference (Attack).
    assert table.loc["position_Midfield", "effect_percent"] == pytest.approx(65, abs=3)
    assert abs(table.loc["assists", "coefficient"]) < 0.02


def test_extended_model_uses_league_and_age_squared(players):
    train, test = model.split_players(players)
    extended, _ = model.train_and_evaluate(model.EXTENDED, train, test)

    names = set(extended.regressor_.named_steps["preprocessing"].get_feature_names_out())

    assert {"age", "age^2", "competition_PL"} <= names


def test_every_player_gets_one_out_of_sample_prediction(players):
    train, test = model.split_players(players)
    linear, _ = model.train_and_evaluate(model.LINEAR, train, test)

    predictions = model.out_of_sample_predictions(model.LINEAR, linear, train, test)

    assert len(predictions) == len(players)
    assert predictions["player_id"].is_unique
    assert predictions["split"].value_counts().to_dict() == {"train": 240, "test": 60}
    assert (predictions["predicted_value_eur"] > 0).all()
