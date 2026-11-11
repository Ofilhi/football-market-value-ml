import numpy as np
import pandas as pd
import pytest

from football_value import eda, plotting
from football_value.features import TARGET


@pytest.fixture
def players():
    """A small synthetic dataset where value falls with age and is skewed."""
    rng = np.random.default_rng(0)
    size = 60
    age = rng.uniform(18, 36, size)
    return pd.DataFrame(
        {
            "position": rng.choice(["Attack", "Midfield", "Defence"], size),
            "competition": rng.choice(["PL", "PD", "SA", "BL1", "FL1"], size),
            "goals": rng.integers(1, 20, size),
            "assists": rng.integers(0, 10, size),
            "minutes_played": rng.integers(100, 3400, size),
            "age": age,
            TARGET: (10 ** (8.5 - 0.08 * age + rng.normal(0, 0.3, size))).astype(int),
        }
    )


@pytest.mark.parametrize(
    ("amount", "expected"),
    [(200_000_000, "€200M"), (2_500_000, "€2.5M"), (300_000, "€300K"), (500, "€500")],
)
def test_format_euros(amount, expected):
    assert plotting.format_euros(amount) == expected


def test_log_transform_reduces_skewness(players):
    skew = eda.skewness(players)

    assert abs(skew[eda.LOG_TARGET]) < abs(skew[TARGET])


def test_correlation_matrix_is_square_and_finds_the_age_effect(players):
    matrix = eda.correlation_matrix(players)

    assert list(matrix.index) == list(matrix.columns)
    assert matrix.loc["age", "age"] == pytest.approx(1.0)
    assert matrix.loc["age", eda.LOG_TARGET] < -0.5


def test_run_writes_every_chart(players, tmp_path):
    paths = eda.run(players, directory=tmp_path)

    assert len(paths) == 4
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
