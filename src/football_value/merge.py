"""Join the football-data.org statistics with Transfermarkt values and minutes.

Run it after the collection step::

    python -m football_value.merge

It writes two files in ``data/processed/``:

* ``players_merged.csv``: the matched players, with their minutes played and
  market value;
* ``unmatched_players.csv``: the players that could not be matched, kept as a
  report so the loss is visible rather than silent.
"""

from __future__ import annotations

import logging

import pandas as pd

from football_value import collect, config, transfermarkt
from football_value.matching import match_players

logger = logging.getLogger(__name__)

MERGED_FILE = config.PROCESSED_DATA_DIR / "players_merged.csv"
UNMATCHED_FILE = config.PROCESSED_DATA_DIR / "unmatched_players.csv"


def merge_sources(
    players: pd.DataFrame,
    tm_players: pd.DataFrame,
    tm_stats: pd.DataFrame,
    tm_valuations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach minutes played and market value to the football-data.org players.

    Args:
        players: output of the collection step (one row per player and league).
        tm_players: Transfermarkt players (``player_id``, ``name``, ``date_of_birth``).
        tm_stats: output of :func:`transfermarkt.season_league_stats`.
        tm_valuations: output of :func:`transfermarkt.latest_valuations`.

    Returns:
        The merged table and the table of players left out, in that order.
        A player is left out when no Transfermarkt record matches, or when
        the matched record has no market value.
    """
    players = players.assign(
        competition_id=players["competition"].map(config.TRANSFERMARKT_COMPETITION_IDS)
    )

    # Candidates: Transfermarkt players who actually appeared in each league.
    candidates = tm_stats.merge(tm_players, on="player_id").rename(
        columns={"player_id": "tm_player_id"}
    )
    matched = match_players(players, candidates)

    merged = matched.merge(
        tm_stats.rename(columns={"player_id": "tm_player_id"}),
        on=["tm_player_id", "competition_id"],
        how="left",
    ).merge(
        tm_valuations.rename(columns={"player_id": "tm_player_id"}),
        on="tm_player_id",
        how="left",
    )

    is_complete = merged["tm_player_id"].notna() & merged["market_value_eur"].notna()
    complete = merged[is_complete].drop(columns="competition_id").reset_index(drop=True)
    left_out = merged.loc[~is_complete, list(players.columns.drop("competition_id"))]
    return complete, left_out.reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    players = pd.read_csv(collect.OUTPUT_FILE)
    paths = {table: transfermarkt.download_table(table) for table in transfermarkt.TABLES}

    tm_players = transfermarkt.load_players(paths["players"])
    tm_stats = transfermarkt.season_league_stats(
        transfermarkt.load_appearances(paths["appearances"]),
        competition_ids=config.TRANSFERMARKT_COMPETITION_IDS.values(),
    )
    tm_valuations = transfermarkt.latest_valuations(
        transfermarkt.load_valuations(paths["player_valuations"])
    )

    merged, left_out = merge_sources(players, tm_players, tm_stats, tm_valuations)

    config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(MERGED_FILE, index=False)
    left_out.to_csv(UNMATCHED_FILE, index=False)

    logger.info(
        "Matched %d of %d players (%.1f%%)",
        len(merged),
        len(players),
        100 * len(merged) / len(players),
    )
    for method, count in merged["match_method"].value_counts().items():
        logger.info("  %s: %d", method, count)
    logger.info("Left out: %d (see %s)", len(left_out), UNMATCHED_FILE.name)


if __name__ == "__main__":
    main()
