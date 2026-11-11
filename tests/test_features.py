import pandas as pd
import pytest

from football_value import features


def make_merged(**overrides):
    """Two single-league players and one who changed league mid-season."""
    data = {
        "player_id": [1, 2, 3, 3],
        "name": ["Striker", "Midfielder", "Traveller", "Traveller"],
        "date_of_birth": ["2000-06-30", "1996-01-01", "2001-03-15", "2001-03-15"],
        "position": ["Offence", "Central Midfield", "Defence", "Defence"],
        "team": ["Club A", "Club B", "Club C", "Club D"],
        "competition": ["PL", "PD", "PL", "SA"],
        "goals": [20, 3, 1, 2],
        "assists": [5.0, None, None, 4.0],
        "minutes_played": [3000.0, 1500.0, 400.0, 1200.0],
        "valuation_date": ["2026-06-01", "2026-05-20", "2026-06-10", "2026-06-10"],
        "market_value_eur": [80e6, 10e6, 5e6, 5e6],
    }
    return pd.DataFrame({**data, **overrides})


def test_missing_assists_become_zero():
    result = features.fill_missing_assists(make_merged())

    assert list(result["assists"]) == [5, 0, 0, 4]


def test_positions_are_grouped_into_three_categories():
    result = features.group_positions(make_merged())

    assert list(result["position"]) == ["Attack", "Midfield", "Defence", "Defence"]


def test_unknown_position_is_reported():
    merged = make_merged(position=["Offence", "Sweeper", "Defence", "Defence"])

    with pytest.raises(ValueError, match="Sweeper"):
        features.group_positions(merged)


def test_stale_valuations_are_dropped():
    merged = make_merged(valuation_date=["2026-06-01", "2024-06-01", "2026-06-10", "2026-06-10"])

    result = features.drop_stale_valuations(merged, season_start="2025-07-01")

    assert list(result["player_id"]) == [1, 3, 3]


def test_league_rows_are_combined_and_main_team_is_kept():
    merged = features.fill_missing_assists(make_merged())

    result = features.combine_league_rows(merged).set_index("player_id")

    assert len(result) == 3
    traveller = result.loc[3]
    assert traveller["goals"] == 3
    assert traveller["assists"] == 4
    assert traveller["minutes_played"] == 1600
    assert traveller["team"] == "Club D"  # most minutes played there
    assert traveller["competition"] == "SA"
    assert traveller["market_value_eur"] == 5e6  # a value, not a sum


def test_age_is_computed_at_the_reference_date():
    result = features.add_age(make_merged(), as_of="2026-06-30")

    assert result.loc[0, "age"] == pytest.approx(26.0, abs=0.01)
    assert result.loc[1, "age"] == pytest.approx(30.49, abs=0.01)


def test_build_features_returns_a_clean_model_ready_table():
    result = features.build_features(make_merged())

    assert list(result.columns) == features.OUTPUT_COLUMNS
    assert list(result["player_id"]) == [1, 2, 3]
    assert not result.isna().any().any()
    assert set(result["position"]) <= {"Attack", "Midfield", "Defence"}
