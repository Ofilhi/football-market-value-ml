"""Collect team-level factors from football-data.org.

A player's market value depends a lot on the club he plays for, and the free
plan of the API gives three ways to measure a club's strength:

* the **league table** (points, goals scored and conceded);
* the **match results** of the league (clean sheets);
* the **Champions League** matches (how many, and how far the club went).

Run it after the players collection::

    python -m football_value.teams

It writes ``data/raw/football_data_teams.csv``, one row per team.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from football_value import config
from football_value.collect import RAW_CACHE_DIR
from football_value.football_data_client import FootballDataClient

logger = logging.getLogger(__name__)

OUTPUT_FILE = config.RAW_DATA_DIR / "football_data_teams.csv"
CHAMPIONS_LEAGUE = "CL"

# Rounds of the Champions League, as an ordered scale: the further a club
# goes, the higher the number. A club that did not play in it gets 0.
CL_STAGES: dict[str, int] = {
    "LEAGUE_STAGE": 1,
    "PLAYOFFS": 2,
    "LAST_16": 3,
    "QUARTER_FINALS": 4,
    "SEMI_FINALS": 5,
    "FINAL": 6,
}
CL_STAGE_LABELS: dict[int, str] = {
    0: "did not play in the Champions League",
    1: "league stage",
    2: "playoffs",
    3: "round of 16",
    4: "quarter-finals",
    5: "semi-finals",
    6: "final",
}


def standings_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten the full-season league table into one row per team."""
    competition = payload["competition"]["code"]
    total = next(table for table in payload["standings"] if table["type"] == "TOTAL")

    rows = [
        {
            "team": row["team"]["name"],
            "competition": competition,
            "table_position": row["position"],
            "played": row["playedGames"],
            "won": row["won"],
            "points": row["points"],
            "goals_for": row["goalsFor"],
            "goals_against": row["goalsAgainst"],
        }
        for row in total["table"]
    ]
    return pd.DataFrame(rows)


def clean_sheet_rates(payload: Mapping[str, Any]) -> pd.Series:
    """Share of the finished matches in which each team conceded no goal."""
    conceded = []
    for match in payload["matches"]:
        if match["status"] != "FINISHED":
            continue
        score = match["score"]["fullTime"]
        conceded.append((match["homeTeam"]["name"], score["away"]))
        conceded.append((match["awayTeam"]["name"], score["home"]))

    frame = pd.DataFrame(conceded, columns=["team", "conceded"])
    return (frame["conceded"] == 0).groupby(frame["team"]).mean().rename("clean_sheet_rate")


def champions_league_stats(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Number of Champions League matches and furthest round reached, per team."""
    rows = []
    for match in payload["matches"]:
        for side in ("homeTeam", "awayTeam"):
            name = match[side].get("name")
            if name:  # a round not drawn yet has no team name
                rows.append({"team": name, "stage": CL_STAGES[match["stage"]]})

    frame = pd.DataFrame(rows, columns=["team", "stage"])
    return frame.groupby("team").agg(cl_games=("stage", "size"), cl_stage=("stage", "max"))


def collect_team_table(
    client: FootballDataClient,
    competitions: Mapping[str, str] = config.COMPETITIONS,
    season: int = config.SEASON,
) -> pd.DataFrame:
    """Build the team table for every league: one row per team."""
    params = {"season": season}
    tables, rates = [], []
    for code in competitions:
        tables.append(standings_to_frame(client.get(f"competitions/{code}/standings", params)))
        rates.append(clean_sheet_rates(client.get(f"competitions/{code}/matches", params)))
    teams = pd.concat(tables, ignore_index=True)

    teams = teams.merge(pd.concat(rates), left_on="team", right_index=True, how="left")
    champions = champions_league_stats(
        client.get(f"competitions/{CHAMPIONS_LEAGUE}/matches", params)
    )
    teams = teams.merge(champions, left_on="team", right_index=True, how="left")
    teams[["cl_games", "cl_stage"]] = teams[["cl_games", "cl_stage"]].fillna(0).astype(int)

    return add_rates(teams)


def add_rates(teams: pd.DataFrame) -> pd.DataFrame:
    """Add per-match rates, which compare teams that played different numbers of matches."""
    return teams.assign(
        points_per_game=teams["points"] / teams["played"],
        goal_difference_per_game=(teams["goals_for"] - teams["goals_against"]) / teams["played"],
        goals_against_per_game=teams["goals_against"] / teams["played"],
        win_rate=teams["won"] / teams["played"],
    )


def main(output_file: Path = OUTPUT_FILE) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = config.load_settings()
    client = FootballDataClient(
        api_key=settings.require_football_data_api_key(), cache_dir=RAW_CACHE_DIR
    )
    teams = collect_team_table(client)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    teams.to_csv(output_file, index=False)
    logger.info("Saved %d teams to %s", len(teams), output_file)


if __name__ == "__main__":
    main()
