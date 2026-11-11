"""Clean the merged data and build the model-ready dataset.

Run it after the merge step::

    python -m football_value.features

The output, ``data/processed/players_features.csv``, has one row per player
with the five features (goals, assists, minutes played, age, position) and
the target (market value).

The position is kept as a readable category here. It is turned into numbers
inside the model pipeline, so that the encoding is learned from the training
set only.
"""

from __future__ import annotations

import logging

import pandas as pd

from football_value import config, merge

logger = logging.getLogger(__name__)

FEATURES_FILE = config.PROCESSED_DATA_DIR / "players_features.csv"

NUMERIC_FEATURES = ["goals", "assists", "minutes_played", "age"]
CATEGORICAL_FEATURES = ["position"]
TARGET = "market_value_eur"

OUTPUT_COLUMNS = [
    "player_id",
    "name",
    "team",
    "competition",
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
    TARGET,
]

# football-data.org mostly uses three broad positions, but a few players carry
# a more detailed one. Everything is mapped to the three broad groups.
# (No goalkeeper appears in the data: it only contains players who scored.)
POSITION_GROUPS: dict[str, str] = {
    "Offence": "Attack",
    "Centre-Forward": "Attack",
    "Left Winger": "Attack",
    "Right Winger": "Attack",
    "Midfield": "Midfield",
    "Attacking Midfield": "Midfield",
    "Central Midfield": "Midfield",
    "Defensive Midfield": "Midfield",
    "Left Midfield": "Midfield",
    "Right Midfield": "Midfield",
    "Defence": "Defence",
    "Centre-Back": "Defence",
    "Left-Back": "Defence",
    "Right-Back": "Defence",
}

DAYS_PER_YEAR = 365.25


def fill_missing_assists(players: pd.DataFrame) -> pd.DataFrame:
    """Replace missing assists with zero.

    The API leaves the field empty instead of returning 0: the lowest value
    it ever reports is 1.
    """
    return players.assign(assists=players["assists"].fillna(0).astype(int))


def group_positions(players: pd.DataFrame) -> pd.DataFrame:
    """Map every position to Attack, Midfield or Defence.

    Raises:
        ValueError: if a position is not in :data:`POSITION_GROUPS`, so a new
            label from the API is noticed instead of silently lost.
    """
    unknown = sorted(set(players["position"].dropna()) - set(POSITION_GROUPS))
    if unknown:
        raise ValueError(f"Unknown positions, add them to POSITION_GROUPS: {unknown}")
    return players.assign(position=players["position"].map(POSITION_GROUPS))


def drop_stale_valuations(
    players: pd.DataFrame, season_start: str = config.SEASON_START
) -> pd.DataFrame:
    """Remove players whose market value was last updated before the season.

    Such a value cannot reflect the season's performance.
    """
    return players[players["valuation_date"] >= season_start]


def combine_league_rows(players: pd.DataFrame) -> pd.DataFrame:
    """Merge the rows of a player who changed league during the season.

    Goals, assists and minutes are added up. The team and league kept are
    those where the player spent the most minutes.
    """
    ordered = players.sort_values("minutes_played", ascending=False)
    totals = ordered.groupby("player_id", sort=False)[["goals", "assists", "minutes_played"]].sum()

    main_rows = ordered.drop_duplicates("player_id").set_index("player_id")
    main_rows[totals.columns] = totals
    return main_rows.reset_index()


def add_age(players: pd.DataFrame, as_of: str = config.SEASON_END) -> pd.DataFrame:
    """Add the player's age in years on a reference date (end of the season)."""
    days = pd.Timestamp(as_of) - pd.to_datetime(players["date_of_birth"])
    return players.assign(age=(days.dt.days / DAYS_PER_YEAR).round(2))


def build_features(merged: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step and return the model-ready table."""
    players = (
        merged.pipe(fill_missing_assists)
        .pipe(group_positions)
        .pipe(drop_stale_valuations)
        .pipe(combine_league_rows)
        .pipe(add_age)
    )
    players = players.astype({"minutes_played": int, TARGET: int})
    return players[OUTPUT_COLUMNS].sort_values("player_id").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    merged = pd.read_csv(merge.MERGED_FILE)
    features = build_features(merged)

    missing = int(features.isna().sum().sum())
    if missing:
        raise ValueError(f"{missing} missing values remain in the features table.")

    features.to_csv(FEATURES_FILE, index=False)
    logger.info("%d rows in, %d players out -> %s", len(merged), len(features), FEATURES_FILE)


if __name__ == "__main__":
    main()
