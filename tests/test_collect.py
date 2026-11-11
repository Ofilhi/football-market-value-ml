import pandas as pd

from football_value.collect import COLUMNS, collect_players, scorers_to_frame


def make_payload(code, scorers):
    return {
        "filters": {"season": "2025", "limit": 1000},
        "competition": {"code": code},
        "scorers": scorers,
    }


def make_scorer(player_id, name, **stats):
    return {
        "player": {
            "id": player_id,
            "name": name,
            "dateOfBirth": "2000-07-21",
            "nationality": "Norway",
            "section": "Offence",
            "position": None,
        },
        "team": {"id": 65, "name": "Manchester City FC"},
        "playedMatches": 36,
        "goals": 27,
        "assists": 8,
        "penalties": 3,
        **stats,
    }


class FakeClient:
    """Serves canned payloads, keyed by request path."""

    def __init__(self, payloads):
        self._payloads = payloads
        self.requests = []

    def get(self, path, params=None):
        self.requests.append((path, params))
        return self._payloads[path]


def test_scorers_to_frame_flattens_one_row_per_player():
    frame = scorers_to_frame(make_payload("PL", [make_scorer(1, "Erling Haaland")]))

    assert list(frame.columns) == COLUMNS
    assert frame.iloc[0].to_dict() == {
        "player_id": 1,
        "name": "Erling Haaland",
        "date_of_birth": "2000-07-21",
        "nationality": "Norway",
        "position": "Offence",
        "team": "Manchester City FC",
        "competition": "PL",
        "season": 2025,
        "played_matches": 36,
        "goals": 27,
        "assists": 8,
        "penalties": 3,
    }


def test_missing_statistics_are_kept_as_missing():
    payload = make_payload("PL", [make_scorer(1, "A Player", assists=None, penalties=None)])

    frame = scorers_to_frame(payload)

    assert pd.isna(frame.loc[0, "assists"])
    assert pd.isna(frame.loc[0, "penalties"])


def test_empty_response_gives_an_empty_frame_with_all_columns():
    frame = scorers_to_frame(make_payload("PL", []))

    assert frame.empty
    assert list(frame.columns) == COLUMNS


def test_collect_players_combines_every_league():
    client = FakeClient(
        {
            "competitions/PL/scorers": make_payload("PL", [make_scorer(1, "Player One")]),
            "competitions/SA/scorers": make_payload(
                "SA", [make_scorer(2, "Player Two"), make_scorer(3, "Player Three")]
            ),
        }
    )

    players = collect_players(client, {"PL": "Premier League", "SA": "Serie A"}, season=2025)

    assert list(players["competition"]) == ["PL", "SA", "SA"]
    assert list(players["player_id"]) == [1, 2, 3]
    assert client.requests[0] == ("competitions/PL/scorers", {"season": 2025, "limit": 1000})
