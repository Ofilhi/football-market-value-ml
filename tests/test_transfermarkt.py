import pandas as pd
import pytest

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


def make_extras_inputs():
    columns = [
        "game_id",
        "player_id",
        "player_club_id",
        "date",
        "competition_id",
        "goals",
        "assists",
        "minutes_played",
    ]
    appearances = pd.DataFrame(
        [
            # Player 1, club 10: two league matches and one Champions League match this season.
            (1, 1, 10, "2025-08-20", "GB1", 1, 0, 90),
            (2, 1, 10, "2025-09-01", "GB1", 0, 1, 60),
            (3, 1, 10, "2025-10-01", "CL", 0, 0, 45),
            # Player 2, club 10: Champions League only this season, so no club of the season.
            (4, 2, 10, "2025-11-01", "CL", 0, 0, 90),
            # Previous season (2024-25): player 1 plays, and club 10 plays in Europe.
            (5, 1, 10, "2025-02-01", "GB1", 1, 2, 80),
            (6, 3, 10, "2025-03-01", "CL", 0, 0, 90),
            (7, 3, 10, "2025-04-01", "EL", 0, 0, 90),
            # Before that (2023-24).
            (8, 1, 10, "2023-12-01", "GB1", 0, 0, 70),
        ],
        columns=columns,
    )
    players = pd.DataFrame(
        {
            "player_id": [1, 2, 4],
            "international_caps": [12, None, None],
            "international_goals": [3, None, None],
            "contract_expiration_date": ["2028-06-30 00:00:00", None, None],
        }
    )
    return appearances, players


def test_player_extras_summarise_season_history_and_club_in_europe():
    appearances, players = make_extras_inputs()

    extras = transfermarkt.player_extras(
        appearances,
        players,
        league_ids=["GB1"],
        season_start="2025-07-01",
        season_end="2026-06-30",
        previous_season_start="2024-07-01",
    )

    player = extras.loc[1]
    assert player["minutes_all"] == 195  # 90 + 60 + 45, Champions League included
    assert player["matches_all"] == 3
    assert player["min_per_match"] == pytest.approx(65)
    assert player["prev_minutes"] == 80
    assert player["prev_goals"] == 1
    assert player["prev_assists"] == 2
    assert player["career_minutes"] == 150  # 80 + 70, before this season
    assert player["career_seasons"] == 2
    # Club 10 played games 3 and 4 in the Champions League this season...
    assert player["club_cl_games"] == 2
    assert player["club_el_games"] == 0
    # ...and games 6 (Champions League) and 7 (Europa League) last season.
    assert player["club_prev_cl_games"] == 1
    assert player["club_prev_euro_games"] == 2
    assert player["caps"] == 12
    assert player["intl_goals"] == 3
    assert player["contract_months"] == pytest.approx(24.0, abs=0.1)
    assert player["contract_missing"] == 0


def test_player_extras_for_a_player_without_league_club_or_history():
    appearances, players = make_extras_inputs()

    extras = transfermarkt.player_extras(
        appearances, players, ["GB1"], "2025-07-01", "2026-06-30", "2024-07-01"
    )

    no_league = extras.loc[2]  # only played in Europe: no club of the season
    assert no_league["club_cl_games"] == 0
    assert no_league["minutes_all"] == 90
    unknown = extras.loc[4]  # not in any match, no profile data
    assert unknown["minutes_all"] == 0
    assert unknown["caps"] == 0
    assert pd.isna(unknown["contract_months"])  # missing stays missing...
    assert unknown["contract_missing"] == 1  # ...and is flagged


def test_player_extras_have_one_row_per_player_of_the_players_table():
    appearances, players = make_extras_inputs()

    extras = transfermarkt.player_extras(
        appearances, players, ["GB1"], "2025-07-01", "2026-06-30", "2024-07-01"
    )

    assert list(extras.index) == [1, 2, 4]
