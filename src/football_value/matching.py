"""Match football-data.org players to their Transfermarkt record.

The two sources share no identifier, and they do not always agree on how a
name is written ("Sávio" / "Savinho") or even on a date of birth. Matching is
therefore done in three passes, from the strictest rule to the loosest, and
always among the Transfermarkt players who appeared in the same league during
the season:

1. ``name_and_birth_date``: same normalised name and same date of birth;
2. ``birth_date_and_name_part``: same date of birth and at least one name
   part in common, when exactly one candidate fits;
3. ``name_only``: same normalised name, when exactly one candidate fits
   (covers a wrong date of birth in one of the sources).

A player matched by none of the rules is left unmatched rather than guessed.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

MATCH_METHODS = ("name_and_birth_date", "birth_date_and_name_part", "name_only")

# Letters that Unicode does not split into "base letter + accent".
_SPECIAL_LETTERS = str.maketrans(
    {"ø": "o", "đ": "d", "ð": "d", "ł": "l", "ı": "i", "ß": "ss", "æ": "ae", "œ": "oe", "þ": "th"}
)


def normalize_name(name: str) -> str:
    """Lower-case a name and strip accents, punctuation and extra spaces.

    >>> normalize_name("Kylian Mbappé")
    'kylian mbappe'
    >>> normalize_name("Matt O'Riley")
    'matt o riley'
    >>> normalize_name("Aron Dønnum")
    'aron donnum'
    """
    decomposed = unicodedata.normalize("NFKD", str(name).lower().translate(_SPECIAL_LETTERS))
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return " ".join(re.sub(r"[^a-z]+", " ", without_accents).split())


def match_players(players: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    """Find the Transfermarkt id of each football-data.org player.

    Args:
        players: one row per player and league, with the columns ``name``,
            ``date_of_birth`` and ``competition_id``.
        candidates: Transfermarkt players who appeared in each league, with
            the columns ``tm_player_id``, ``name``, ``date_of_birth`` and
            ``competition_id``.

    Returns:
        ``players`` with two extra columns: ``tm_player_id`` (missing when no
        match was found) and ``match_method``.
    """
    candidates = candidates.assign(name_key=candidates["name"].map(normalize_name))
    by_league = {league: group for league, group in candidates.groupby("competition_id")}

    matches = [
        _match_one(row.name, row.date_of_birth, by_league.get(row.competition_id))
        for row in players.itertuples(index=False)
    ]

    result = players.copy()
    result["tm_player_id"] = pd.array([match[0] for match in matches], dtype="Int64")
    result["match_method"] = [match[1] for match in matches]
    return result


def _match_one(
    name: str, date_of_birth: str, pool: pd.DataFrame | None
) -> tuple[int | None, str | None]:
    if pool is None:
        return None, None

    name_key = normalize_name(name)
    name_parts = set(name_key.split())
    same_birth_date = pool[pool["date_of_birth"] == date_of_birth]

    same_name_and_birth_date = same_birth_date[same_birth_date["name_key"] == name_key]
    shares_a_name_part = same_birth_date[
        same_birth_date["name_key"].map(lambda key: bool(name_parts & set(key.split())))
    ]
    same_name = pool[pool["name_key"] == name_key]

    attempts = zip(
        MATCH_METHODS, (same_name_and_birth_date, shares_a_name_part, same_name), strict=True
    )
    for method, found in attempts:
        if len(found) == 1:
            return int(found["tm_player_id"].iloc[0]), method
    return None, None
