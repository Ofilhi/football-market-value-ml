"""Collect player statistics from football-data.org.

For each league, the *scorers* endpoint returns every player who scored at
least one goal in the season, with their goals, assists, matches played, date
of birth and position. Players who did not score are not listed by the API,
so they are outside the scope of this dataset.

Run it with::

    python -m football_value.collect
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from football_value import config
from football_value.football_data_client import FootballDataClient

logger = logging.getLogger(__name__)

# Higher than the number of scorers in any league, so one request gets them all.
SCORERS_LIMIT = 1000

RAW_CACHE_DIR = config.RAW_DATA_DIR / "football_data"
OUTPUT_FILE = config.RAW_DATA_DIR / "football_data_players.csv"

COLUMNS = [
    "player_id",
    "name",
    "date_of_birth",
    "nationality",
    "position",
    "team",
    "competition",
    "season",
    "played_matches",
    "goals",
    "assists",
    "penalties",
]


def fetch_scorers(client: FootballDataClient, competition: str, season: int) -> dict[str, Any]:
    """Fetch the raw scorers response for one league and one season."""
    return client.get(
        f"competitions/{competition}/scorers",
        params={"season": season, "limit": SCORERS_LIMIT},
    )


def scorers_to_frame(payload: Mapping[str, Any]) -> pd.DataFrame:
    """Flatten a raw scorers response into one row per player.

    Values are kept exactly as the API returns them (missing assists stay
    missing); cleaning happens later, in the feature engineering step.
    """
    competition = payload["competition"]["code"]
    season = int(payload["filters"]["season"])

    rows = [
        {
            "player_id": entry["player"]["id"],
            "name": entry["player"]["name"],
            "date_of_birth": entry["player"].get("dateOfBirth"),
            "nationality": entry["player"].get("nationality"),
            # The API puts the broad position (Offence, Midfield, ...) in "section".
            "position": entry["player"].get("section") or entry["player"].get("position"),
            "team": entry["team"]["name"],
            "competition": competition,
            "season": season,
            "played_matches": entry.get("playedMatches"),
            "goals": entry.get("goals"),
            "assists": entry.get("assists"),
            "penalties": entry.get("penalties"),
        }
        for entry in payload["scorers"]
    ]
    return pd.DataFrame(rows, columns=COLUMNS)


def collect_players(
    client: FootballDataClient,
    competitions: Mapping[str, str] = config.COMPETITIONS,
    season: int = config.SEASON,
) -> pd.DataFrame:
    """Collect the players of every league into a single table."""
    frames = []
    for code, league_name in competitions.items():
        frame = scorers_to_frame(fetch_scorers(client, code, season))
        logger.info("%s: %d players", league_name, len(frame))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def main(output_file: Path = OUTPUT_FILE) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    settings = config.load_settings()
    client = FootballDataClient(
        api_key=settings.require_football_data_api_key(),
        cache_dir=RAW_CACHE_DIR,
    )

    players = collect_players(client)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    players.to_csv(output_file, index=False)
    logger.info("Saved %d players to %s", len(players), output_file)


if __name__ == "__main__":
    main()
