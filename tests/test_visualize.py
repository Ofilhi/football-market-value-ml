import numpy as np
import pandas as pd
import pytest

from football_value import visualize
from football_value.features import TARGET


@pytest.fixture
def predictions():
    rng = np.random.default_rng(0)
    size = 80
    actual = 10 ** rng.uniform(5.5, 8, size)
    ratio = 10 ** rng.normal(0, 0.3, size)
    return pd.DataFrame(
        {
            "name": [f"Player {index}" for index in range(size)],
            "team": "Some FC",
            "competition": rng.choice(["PL", "PD", "SA", "BL1", "FL1"], size),
            "age": rng.uniform(18, 36, size),
            "minutes_played": rng.integers(100, 3400, size),
            TARGET: actual,
            visualize.PREDICTED: actual * ratio,
            visualize.RATIO: ratio,
        }
    )


@pytest.mark.parametrize(
    ("ratio", "expected"),
    [(5, "5× higher"), (0.1, "10× lower"), (0.5, "2× lower"), (1, "same")],
)
def test_format_ratio(ratio, expected):
    assert visualize.format_ratio(ratio) == expected


def test_value_gaps_returns_the_largest_disagreements_first(predictions):
    model_higher, model_lower = visualize.value_gaps(predictions, min_minutes=900, size=5)

    assert len(model_higher) == len(model_lower) == 5
    assert (model_higher[visualize.RATIO] > 1).all()
    assert (model_lower[visualize.RATIO] < 1).all()
    assert model_higher[visualize.RATIO].is_monotonic_decreasing
    assert model_lower[visualize.RATIO].is_monotonic_increasing


def test_value_gaps_ignores_players_with_few_minutes(predictions):
    predictions.loc[0, ["minutes_played", visualize.RATIO]] = [50, 1000.0]

    model_higher, _ = visualize.value_gaps(predictions, min_minutes=900, size=5)

    assert "Player 0" not in set(model_higher["name"])
    assert (model_higher["minutes_played"] >= 900).all()


def test_run_writes_every_chart(predictions, tmp_path):
    paths = visualize.run(predictions, directory=tmp_path)

    assert len(paths) == 3
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
