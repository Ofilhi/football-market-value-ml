import pandas as pd

from football_value.merge import merge_sources


def make_inputs():
    players = pd.DataFrame(
        [
            (10, "Erling Haaland", "2000-07-21", "PL", 27),
            (11, "Unknown Player", "1999-01-01", "PL", 1),
            (12, "No Value Player", "2001-05-05", "PL", 2),
        ],
        columns=["player_id", "name", "date_of_birth", "competition", "goals"],
    )
    tm_players = pd.DataFrame(
        [
            (418560, "Erling Haaland", "2000-07-21"),
            (500, "No Value Player", "2001-05-05"),
        ],
        columns=["player_id", "name", "date_of_birth"],
    )
    tm_stats = pd.DataFrame(
        [
            (418560, "GB1", 35, 2959, 27, 8),
            (500, "GB1", 10, 400, 2, 0),
        ],
        columns=[
            "player_id",
            "competition_id",
            "tm_matches",
            "minutes_played",
            "tm_goals",
            "tm_assists",
        ],
    )
    tm_valuations = pd.DataFrame(
        [(418560, "2026-06-03", 200_000_000)],
        columns=["player_id", "valuation_date", "market_value_eur"],
    )
    return players, tm_players, tm_stats, tm_valuations


def test_matched_player_gets_minutes_and_market_value():
    merged, _ = merge_sources(*make_inputs())

    assert len(merged) == 1
    row = merged.iloc[0]
    assert row["name"] == "Erling Haaland"
    assert row["tm_player_id"] == 418560
    assert row["minutes_played"] == 2959
    assert row["market_value_eur"] == 200_000_000
    assert row["valuation_date"] == "2026-06-03"


def test_players_without_match_or_value_are_reported_not_dropped_silently():
    players, *_ = inputs = make_inputs()

    merged, left_out = merge_sources(*inputs)

    assert sorted(left_out["name"]) == ["No Value Player", "Unknown Player"]
    assert list(left_out.columns) == list(players.columns)
    assert len(merged) + len(left_out) == len(players)
