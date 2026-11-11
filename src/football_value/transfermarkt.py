"""Load market values and minutes played from the public Transfermarkt dataset.

The dataset is published as compressed CSV files by the open-source project
https://github.com/dcaribou/transfermarkt-datasets. Three tables are used:

* ``players``: one row per player (name, date of birth);
* ``appearances``: one row per player and match (minutes, goals, assists);
* ``player_valuations``: the history of each player's market value.

Files are downloaded once into ``data/raw/transfermarkt/`` and reused afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests

from football_value import config

logger = logging.getLogger(__name__)

RAW_DIR = config.RAW_DATA_DIR / "transfermarkt"
TABLES = ("players", "appearances", "player_valuations")

DOWNLOAD_TIMEOUT_SECONDS = 120
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def download_table(
    table: str,
    raw_dir: Path = RAW_DIR,
    base_url: str = config.TRANSFERMARKT_BASE_URL,
) -> Path:
    """Download one table unless it is already on disk, and return its path."""
    destination = raw_dir / f"{table}.csv.gz"
    if destination.exists():
        logger.debug("Already downloaded: %s", destination.name)
        return destination

    raw_dir.mkdir(parents=True, exist_ok=True)
    url = f"{base_url}/{table}.csv.gz"
    logger.info("Downloading %s", url)

    # Write to a temporary file first so an interrupted download is never
    # mistaken for a complete one.
    partial = destination.with_suffix(".part")
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        with partial.open("wb") as file:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                file.write(chunk)
    partial.rename(destination)
    return destination


def load_players(path: Path) -> pd.DataFrame:
    """Load the players table with a clean ``YYYY-MM-DD`` date of birth."""
    players = pd.read_csv(path, usecols=["player_id", "name", "date_of_birth"])
    players["date_of_birth"] = players["date_of_birth"].str[:10]
    return players


def load_appearances(path: Path) -> pd.DataFrame:
    """Load the columns of the appearances table that the project needs."""
    return pd.read_csv(
        path,
        usecols=["player_id", "date", "competition_id", "goals", "assists", "minutes_played"],
    )


def load_valuations(path: Path) -> pd.DataFrame:
    """Load the market value history."""
    return pd.read_csv(path, usecols=["player_id", "date", "market_value_in_eur"])


def season_league_stats(
    appearances: pd.DataFrame,
    competition_ids: Iterable[str],
    season_start: str = config.SEASON_START,
    season_end: str = config.SEASON_END,
) -> pd.DataFrame:
    """Total each player's league statistics over one season.

    Returns one row per player and league, so a player who changed league
    during the season gets one row for each of them.
    """
    in_season = appearances["date"].between(season_start, season_end)
    in_leagues = appearances["competition_id"].isin(list(competition_ids))

    return (
        appearances[in_season & in_leagues]
        .groupby(["player_id", "competition_id"], as_index=False)
        .agg(
            tm_matches=("minutes_played", "size"),
            minutes_played=("minutes_played", "sum"),
            tm_goals=("goals", "sum"),
            tm_assists=("assists", "sum"),
        )
    )


def latest_valuations(valuations: pd.DataFrame, as_of: str = config.SEASON_END) -> pd.DataFrame:
    """Return each player's most recent market value on or before ``as_of``.

    Using a fixed date keeps the target consistent with the season the
    statistics come from, whenever the dataset is downloaded.
    """
    known = valuations[valuations["date"] <= as_of].dropna(subset=["market_value_in_eur"])
    latest = known.sort_values("date").groupby("player_id", as_index=False).last()
    return latest.rename(
        columns={"date": "valuation_date", "market_value_in_eur": "market_value_eur"}
    )
