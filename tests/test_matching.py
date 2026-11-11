import pandas as pd
import pytest

from football_value.matching import match_players, normalize_name


def make_players(*rows):
    return pd.DataFrame(rows, columns=["name", "date_of_birth", "competition_id"])


def make_candidates(*rows):
    return pd.DataFrame(rows, columns=["tm_player_id", "name", "date_of_birth", "competition_id"])


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Kylian Mbappé", "kylian mbappe"),
        ("Matt O'Riley", "matt o riley"),
        ("  Aron   Dønnum ", "aron donnum"),
        ("Łukasz Skorupski", "lukasz skorupski"),
        ("Mickey van de Ven", "mickey van de ven"),
    ],
)
def test_normalize_name(raw, expected):
    assert normalize_name(raw) == expected


def test_match_on_name_and_birth_date_ignores_accents():
    players = make_players(("Kylian Mbappe", "1998-12-20", "ES1"))
    candidates = make_candidates((342229, "Kylian Mbappé", "1998-12-20", "ES1"))

    result = match_players(players, candidates)

    assert result.loc[0, "tm_player_id"] == 342229
    assert result.loc[0, "match_method"] == "name_and_birth_date"


def test_match_on_birth_date_and_shared_name_part():
    players = make_players(("Marcus Thuram-Ulien", "1997-08-06", "IT1"))
    candidates = make_candidates(
        (318528, "Marcus Thuram", "1997-08-06", "IT1"),
        (111, "Someone Else", "1997-08-06", "IT1"),
    )

    result = match_players(players, candidates)

    assert result.loc[0, "tm_player_id"] == 318528
    assert result.loc[0, "match_method"] == "birth_date_and_name_part"


def test_match_on_name_only_when_birth_dates_disagree():
    players = make_players(("Jarrod Bowen", "1996-01-01", "GB1"))
    candidates = make_candidates((314875, "Jarrod Bowen", "1996-12-20", "GB1"))

    result = match_players(players, candidates)

    assert result.loc[0, "tm_player_id"] == 314875
    assert result.loc[0, "match_method"] == "name_only"


def test_ambiguous_candidates_are_left_unmatched():
    players = make_players(("Pedro Silva", "2000-01-01", "ES1"))
    candidates = make_candidates(
        (1, "Pedro Gomes", "2000-01-01", "ES1"),
        (2, "Pedro Santos", "2000-01-01", "ES1"),
    )

    result = match_players(players, candidates)

    assert pd.isna(result.loc[0, "tm_player_id"])
    assert result.loc[0, "match_method"] is None


def test_candidates_from_another_league_are_ignored():
    players = make_players(("Kylian Mbappé", "1998-12-20", "ES1"))
    candidates = make_candidates((342229, "Kylian Mbappé", "1998-12-20", "FR1"))

    result = match_players(players, candidates)

    assert pd.isna(result.loc[0, "tm_player_id"])
