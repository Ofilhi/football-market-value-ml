import pytest

from football_value import teams


def standings_payload(code="PL"):
    def row(position, name, played, won, points, goals_for, goals_against):
        return {
            "position": position,
            "team": {"name": name},
            "playedGames": played,
            "won": won,
            "points": points,
            "goalsFor": goals_for,
            "goalsAgainst": goals_against,
        }

    return {
        "competition": {"code": code},
        "standings": [
            {
                "type": "TOTAL",
                "table": [
                    row(1, "Arsenal", 38, 26, 85, 71, 27),
                    row(2, "Spurs", 38, 20, 64, 60, 50),
                ],
            },
            {"type": "HOME", "table": [row(1, "Spurs", 19, 15, 50, 40, 10)]},
        ],
    }


def match(home, away, home_goals, away_goals, status="FINISHED", stage="REGULAR_SEASON"):
    return {
        "status": status,
        "stage": stage,
        "homeTeam": {"name": home},
        "awayTeam": {"name": away},
        "score": {"fullTime": {"home": home_goals, "away": away_goals}},
    }


def test_standings_use_the_total_table_not_home_or_away():
    frame = teams.standings_to_frame(standings_payload())

    assert list(frame["team"]) == ["Arsenal", "Spurs"]
    assert frame.loc[0, "points"] == 85
    assert frame.loc[0, "competition"] == "PL"
    assert frame.loc[1, "goals_against"] == 50


def test_clean_sheet_rate_counts_finished_matches_only():
    payload = {
        "matches": [
            match("Arsenal", "Spurs", 2, 0),  # Arsenal keeps a clean sheet
            match("Spurs", "Arsenal", 1, 1),  # nobody does
            match("Arsenal", "Spurs", None, None, status="SCHEDULED"),  # ignored
        ]
    }

    rates = teams.clean_sheet_rates(payload)

    assert rates["Arsenal"] == pytest.approx(0.5)
    assert rates["Spurs"] == 0


def test_champions_league_stats_give_games_and_furthest_round():
    payload = {
        "matches": [
            match("Arsenal", "PSG", 1, 0, stage="LEAGUE_STAGE"),
            match("Arsenal", "Bayern", 2, 2, stage="QUARTER_FINALS"),
            match("Bayern", "PSG", 0, 0, stage="SEMI_FINALS"),
            match("Bayern", None, 0, 0, stage="FINAL"),  # opponent not known yet
        ]
    }
    payload["matches"][3]["awayTeam"] = {"name": None}

    stats = teams.champions_league_stats(payload)

    assert stats.loc["Arsenal"].to_dict() == {"cl_games": 2, "cl_stage": 4}
    assert stats.loc["Bayern"].to_dict() == {"cl_games": 3, "cl_stage": 6}
    assert stats.loc["PSG"].to_dict() == {"cl_games": 2, "cl_stage": 5}


class FakeClient:
    def __init__(self):
        self.paths = []

    def get(self, path, params=None):
        self.paths.append(path)
        if path.endswith("/standings"):
            return standings_payload(code=path.split("/")[1])
        if path == "competitions/CL/matches":
            return {"matches": [match("Arsenal", "PSG", 1, 0, stage="LEAGUE_STAGE")]}
        return {"matches": [match("Arsenal", "Spurs", 2, 0), match("Spurs", "Arsenal", 1, 1)]}


def test_team_table_combines_standings_results_and_champions_league():
    table = teams.collect_team_table(FakeClient(), {"PL": "Premier League"}, season=2025)
    table = table.set_index("team")

    assert table.loc["Arsenal", "points_per_game"] == pytest.approx(85 / 38)
    assert table.loc["Arsenal", "goal_difference_per_game"] == pytest.approx(44 / 38)
    assert table.loc["Arsenal", "win_rate"] == pytest.approx(26 / 38)
    assert table.loc["Arsenal", "clean_sheet_rate"] == pytest.approx(0.5)
    assert table.loc["Arsenal", "cl_games"] == 1
    assert table.loc["Arsenal", "cl_stage"] == 1
    # A team that never played in the Champions League gets zeros, not missing values.
    assert table.loc["Spurs", "cl_games"] == 0
    assert table.loc["Spurs", "cl_stage"] == 0


def test_every_stage_has_a_label():
    assert set(teams.CL_STAGES.values()) | {0} == set(teams.CL_STAGE_LABELS)
