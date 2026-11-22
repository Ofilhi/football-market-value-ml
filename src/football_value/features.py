"""Clean the merged data and build the model-ready dataset.

Run it after the merge and teams steps::

    python -m football_value.features

The output, ``data/processed/players_features.csv``, has one row per player
with all the candidate factors and the target (market value):

* the five original features: goals, assists, minutes played, age, position;
* **player activity**, from football-data.org: matches played, share of the
  team's matches, goals per match;
* **team factors**, from football-data.org: league table and Champions League;
* **extra factors**, from Transfermarkt (optional): career history, club in
  Europe, contract, international caps.

Which of these a model uses is decided in :mod:`football_value.model`. Text
columns (position, league) are kept readable here and encoded inside the model
pipeline, so that the encoding is learned from the training set only.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping

import pandas as pd

from football_value import config, merge, teams

logger = logging.getLogger(__name__)

FEATURES_FILE = config.PROCESSED_DATA_DIR / "players_features.csv"

# The five original features.
NUMERIC_FEATURES = ["goals", "assists", "minutes_played", "age"]
CATEGORICAL_FEATURES = ["position"]
TARGET = "market_value_eur"

# Derived from the player's own statistics and his team's number of matches.
ACTIVITY_COLUMNS = ["played_matches", "availability", "goals_per_match"]

# Goals and assists multiplied by a factor for the difficulty of the league.
LEAGUE_ADJUSTED_COLUMNS = ["goals_adjusted", "assists_adjusted", "adjusted_goals_per_match"]

# Copied from the team table, by the player's team.
TEAM_COLUMNS = [
    "table_position",
    "points",
    "points_per_game",
    "goal_difference_per_game",
    "goals_against_per_game",
    "clean_sheet_rate",
    "win_rate",
    "cl_games",
    "cl_stage",
]

# Optional factors, computed by the merge step from Transfermarkt.
EXTRA_COLUMNS = [
    "minutes_all",
    "matches_all",
    "min_per_match",
    "prev_minutes",
    "prev_goals",
    "prev_assists",
    "career_minutes",
    "career_seasons",
    "club_cl_games",
    "club_el_games",
    "club_prev_cl_games",
    "club_prev_euro_games",
    "caps",
    "intl_goals",
    "contract_months",
    "contract_missing",
]

# Columns allowed to stay empty: the model fills them in (see model.py).
OPTIONAL_COLUMNS = ["contract_months"]

OUTPUT_COLUMNS = [
    "player_id",
    "tm_player_id",
    "name",
    "team",
    "competition",
    *CATEGORICAL_FEATURES,
    *NUMERIC_FEATURES,
    *ACTIVITY_COLUMNS,
    *LEAGUE_ADJUSTED_COLUMNS,
    *TEAM_COLUMNS,
    *EXTRA_COLUMNS,
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

    Goals, assists, minutes and matches played are added up. The team and
    league kept are those where the player spent the most minutes.
    """
    summed = [
        "goals",
        "assists",
        "goals_adjusted",
        "assists_adjusted",
        "minutes_played",
        "played_matches",
    ]
    ordered = players.sort_values("minutes_played", ascending=False)
    totals = ordered.groupby("player_id", sort=False)[summed].sum()

    main_rows = ordered.drop_duplicates("player_id").set_index("player_id")
    main_rows[totals.columns] = totals
    return main_rows.reset_index()


def league_factors(
    difficulty: Mapping[str, float] = config.LEAGUE_DIFFICULTY,
) -> dict[str, float]:
    """Turn the difficulty scores of the leagues into multipliers.

    A multiplier is the score divided by the average score, so the average
    league has a factor of 1, an easier league below 1 and a harder one above
    1. Ten goals in a league twice as hard as another count for twice as much.

    >>> league_factors({"A": 9.0, "B": 3.0})
    {'A': 1.5, 'B': 0.5}
    """
    average = sum(difficulty.values()) / len(difficulty)
    return {league: score / average for league, score in difficulty.items()}


def add_league_adjusted_stats(
    players: pd.DataFrame, factors: Mapping[str, float] | None = None
) -> pd.DataFrame:
    """Add goals and assists multiplied by the difficulty factor of the league.

    This must run on the rows of a player *per league*, before they are added
    up, so that a player who changed league is adjusted in each of them.

    Raises:
        ValueError: if a league has no factor.
    """
    factors = league_factors() if factors is None else factors
    multiplier = players["competition"].map(factors)
    if multiplier.isna().any():
        unknown = sorted(set(players.loc[multiplier.isna(), "competition"]))
        raise ValueError(f"No difficulty factor for the leagues: {unknown}")
    return players.assign(
        goals_adjusted=players["goals"] * multiplier,
        assists_adjusted=players["assists"] * multiplier,
    )


def add_age(players: pd.DataFrame, as_of: str = config.SEASON_END) -> pd.DataFrame:
    """Add the player's age in years on a reference date (end of the season)."""
    days = pd.Timestamp(as_of) - pd.to_datetime(players["date_of_birth"])
    return players.assign(age=(days.dt.days / DAYS_PER_YEAR).round(2))


def add_team_factors(players: pd.DataFrame, team_table: pd.DataFrame) -> pd.DataFrame:
    """Attach the factors of each player's team, and the player's activity.

    Added columns: everything in :data:`TEAM_COLUMNS`, plus

    * ``availability``: share of his team's league matches the player appeared in
      (capped at 1, since a player who changed league can have played more);
    * ``goals_per_match`` and ``adjusted_goals_per_match``: goals (raw and
      adjusted for the league) divided by matches played.

    Raises:
        ValueError: if a team is missing from the team table.
    """
    wanted = [*TEAM_COLUMNS, "played"]
    table = team_table.set_index("team")[wanted]

    missing = sorted(set(players["team"]) - set(table.index))
    if missing:
        raise ValueError(f"Teams missing from the team table: {missing}")

    joined = players.join(table, on="team")
    matches = joined["played_matches"].clip(lower=1)
    return joined.assign(
        availability=(joined["played_matches"] / joined["played"]).clip(upper=1),
        goals_per_match=joined["goals"] / matches,
        adjusted_goals_per_match=joined["goals_adjusted"] / matches,
    ).drop(columns="played")


def build_features(merged: pd.DataFrame, team_table: pd.DataFrame) -> pd.DataFrame:
    """Run every cleaning step and return the model-ready table."""
    players = (
        merged.pipe(fill_missing_assists)
        .pipe(group_positions)
        .pipe(drop_stale_valuations)
        .pipe(add_league_adjusted_stats)
        .pipe(combine_league_rows)
        .pipe(add_age)
        .pipe(add_team_factors, team_table)
    )
    players = players.astype({"minutes_played": int, TARGET: int, "tm_player_id": int})
    return players[OUTPUT_COLUMNS].sort_values("player_id").reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    merged = pd.read_csv(merge.MERGED_FILE)
    team_table = pd.read_csv(teams.OUTPUT_FILE)
    features = build_features(merged, team_table)

    required = features.drop(columns=OPTIONAL_COLUMNS)
    missing = int(required.isna().sum().sum())
    if missing:
        raise ValueError(f"{missing} missing values remain in the features table.")

    features.to_csv(FEATURES_FILE, index=False)
    logger.info("%d rows in, %d players out -> %s", len(merged), len(features), FEATURES_FILE)


if __name__ == "__main__":
    main()
