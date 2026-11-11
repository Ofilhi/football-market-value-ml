import pandas as pd

from football_value import transfermarkt


def make_appearances(*rows):
    columns = ["player_id", "date", "competition_id", "goals", "assists", "minutes_played"]
    return pd.DataFrame(rows, columns=columns)


def test_season_league_stats_totals_each_player_per_league():
    appearances = make_appearances(
        (1, "2025-08-20", "GB1", 1, 0, 90),
        (1, "2025-09-01", "GB1", 2, 1, 75),
        (1, "2026-02-01", "IT1", 0, 1, 60),
        (2, "2025-10-10", "GB1", 0, 0, 15),
    )

    stats = transfermarkt.season_league_stats(
        appearances, ["GB1", "IT1"], season_start="2025-07-01", season_end="2026-06-30"
    )

    assert stats.to_dict("records") == [
        {
            "player_id": 1,
            "competition_id": "GB1",
            "tm_matches": 2,
            "minutes_played": 165,
            "tm_goals": 3,
            "tm_assists": 1,
        },
        {
            "player_id": 1,
            "competition_id": "IT1",
            "tm_matches": 1,
            "minutes_played": 60,
            "tm_goals": 0,
            "tm_assists": 1,
        },
        {
            "player_id": 2,
            "competition_id": "GB1",
            "tm_matches": 1,
            "minutes_played": 15,
            "tm_goals": 0,
            "tm_assists": 0,
        },
    ]


def test_season_league_stats_ignores_other_seasons_and_competitions():
    appearances = make_appearances(
        (1, "2025-05-20", "GB1", 1, 0, 90),  # previous season
        (1, "2026-08-20", "GB1", 1, 0, 90),  # next season
        (1, "2025-09-20", "CL", 1, 0, 90),  # not a domestic league
        (1, "2025-09-27", "GB1", 0, 0, 45),
    )

    stats = transfermarkt.season_league_stats(
        appearances, ["GB1"], season_start="2025-07-01", season_end="2026-06-30"
    )

    assert len(stats) == 1
    assert stats.loc[0, "minutes_played"] == 45


def test_latest_valuations_keeps_the_most_recent_value_before_the_cutoff():
    valuations = pd.DataFrame(
        [
            (1, "2025-12-15", 40_000_000),
            (1, "2026-06-03", 50_000_000),
            (1, "2026-09-01", 60_000_000),  # after the cutoff
            (2, "2024-06-01", 1_000_000),
        ],
        columns=["player_id", "date", "market_value_in_eur"],
    )

    latest = transfermarkt.latest_valuations(valuations, as_of="2026-06-30")

    assert latest.to_dict("records") == [
        {"player_id": 1, "valuation_date": "2026-06-03", "market_value_eur": 50_000_000},
        {"player_id": 2, "valuation_date": "2024-06-01", "market_value_eur": 1_000_000},
    ]


def test_download_table_skips_files_already_on_disk(tmp_path):
    existing = tmp_path / "players.csv.gz"
    existing.write_bytes(b"already here")

    # An unreachable URL proves that no request is attempted.
    path = transfermarkt.download_table("players", raw_dir=tmp_path, base_url="http://invalid")

    assert path == existing
    assert existing.read_bytes() == b"already here"
