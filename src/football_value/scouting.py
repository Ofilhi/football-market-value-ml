"""Turn the model's numbers into readable text with an LLM.

Two commands, both run after the model step::

    python -m football_value.scouting player "Vedat Muriqi"
    python -m football_value.scouting model

* ``player`` writes a short scouting report on one player: his statistics,
  what the model thinks he is worth, and how that compares with the market.
* ``model`` writes a plain-language summary of the regression itself.

The LLM never predicts anything. It only receives numbers that the pipeline
already computed, and is asked to comment on them without adding facts of its
own. The known weaknesses of the model are only passed along when they apply to
the player being discussed, and nothing is explained when the model and the
market agree.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

import pandas as pd
from openai import OpenAI, OpenAIError

from football_value import config
from football_value.features import TARGET
from football_value.matching import normalize_name
from football_value.model import (
    COEFFICIENTS_FILE,
    MAIN_SPEC,
    METRICS_FILE,
    PREDICTIONS_FILE,
)
from football_value.teams import CL_STAGE_LABELS

logger = logging.getLogger(__name__)

TEMPERATURE = 0.5
REQUEST_TIMEOUT_SECONDS = 60
# No automatic retries: each one counts against the daily quota of free plans.
MAX_RETRIES = 0

MODEL_DESCRIPTION = (
    "A regularised linear regression predicts the logarithm of a football player's market "
    "value. It uses his goals, assists, minutes and matches played, age (with a curved "
    "effect), position, league, and the strength of his club this season (points per game, "
    "goal difference, league rank, Champions League games and round reached). It is trained "
    "on players of the top 5 European leagues (2025-26 season) who scored at least one goal. "
    "Market values come from Transfermarkt."
)

# A prediction more than this many times above or below the market value is
# "a gap worth explaining"; anything closer is within the model's normal error.
GAP_THRESHOLD = 2.0
CHEAP_PLAYER_EUR = 3_000_000
EXPENSIVE_PLAYER_EUR = 30_000_000
FEW_MINUTES = 300

# Weaknesses of the model, each told to the LLM only when it applies to the player.
LIMITATION_DEFENDING = (
    "The model only sees goals, assists and minutes. It cannot see defending, passing or "
    "ball-winning, which is most of what a defender or a holding midfielder is valued for."
)
LIMITATION_CHEAP_PLAYER = (
    "Predictions are pulled toward the average, so players with a low market value tend to "
    "be predicted too high."
)
LIMITATION_EXPENSIVE_PLAYER = (
    "Predictions are pulled toward the average, so players with a very high market value "
    "tend to be predicted too low."
)
LIMITATION_FEW_MINUTES = (
    "With very few minutes played, goals and assists say little about a player's level."
)

# Weaknesses of the model as a whole, told when the model itself is explained.
MODEL_LIMITATIONS = [
    "The data only contains players who scored at least one goal: there is no goalkeeper, "
    "and few defenders or midfielders who never scored.",
    "No statistic measures defending or passing, so the model cannot tell a good defender "
    "from an average one at the same club.",
    "A large part of the model is the strength of the club (league table, Champions League), "
    "so it says as much about where a player plays as about the player himself.",
    "Predictions are pulled toward the average: the cheapest players tend to be predicted too "
    "high and the most expensive ones too low.",
]

SYSTEM_PROMPT = (
    "You are a football analyst writing for a data science portfolio project. "
    "You are given a JSON document with numbers produced by a statistical model. "
    "Use only the facts in that document: do not add transfers, injuries, trophies "
    "or any other information from memory. Do not describe a player's role more precisely "
    "than the data does (a position is only Attack, Midfield or Defence). Never mention the "
    "document, its fields or its lists: write as if you simply know the facts. Be concrete, "
    "quote the numbers, and be honest about what the model cannot see. Write in plain "
    "English, without headings or bullet points."
)

PLAYER_INSTRUCTIONS = (
    "Write a scouting note of 80 to 120 words about this player. Start with whatever stands "
    "out most about his season or his club, not with a standard introduction. Say what the "
    "model thinks he is worth compared with the market. If the assessment says the model and "
    "the market broadly agree, say so in one sentence and stop there: do not look for a "
    "reason and do not mention any limitation. Otherwise explain the gap using only the club "
    "context and the limitations that apply to this player; if none applies, say that the "
    "available data does not explain the gap. Do not finish with a verdict or a summary "
    "sentence."
)

MODEL_INSTRUCTIONS = (
    "Explain this model to a football fan with no statistics background, in 150 to 200 "
    "words. Say what it does, how accurate it is compared with the baseline and with the "
    "simple five-feature model, which factors matter most and in which direction, and what "
    "it cannot see."
)


class PlayerNotFoundError(LookupError):
    """Raised when a name matches no player, or more than one."""


class LLMError(RuntimeError):
    """Raised when the LLM provider cannot produce an answer."""


# --- Facts given to the LLM --------------------------------------------------


def find_player(predictions: pd.DataFrame, name: str) -> pd.Series:
    """Find one player by name, ignoring accents and letter case.

    An exact match wins; otherwise the name may be any part of the full name
    ("muriqi" finds "Vedat Muriqi").

    Raises:
        PlayerNotFoundError: if no player or several players match.
    """
    wanted = normalize_name(name)
    names = predictions["name"].map(normalize_name)

    matches = predictions[names == wanted]
    if matches.empty:
        matches = predictions[names.str.contains(wanted, regex=False)]

    if matches.empty:
        raise PlayerNotFoundError(f"No player matches '{name}'.")
    if len(matches) > 1:
        candidates = ", ".join(sorted(matches["name"])[:10])
        raise PlayerNotFoundError(f"Several players match '{name}': {candidates}.")
    return matches.iloc[0]


def format_euros(value: float) -> str:
    """Write an amount in euros compactly, for the LLM to quote.

    >>> format_euros(84_511_000)
    '€84.5M'
    >>> format_euros(4_500_000)
    '€4.5M'
    >>> format_euros(300_000)
    '€300K'
    """
    if value >= 1_000_000:
        return f"€{value / 1_000_000:.1f}".removesuffix(".0") + "M"
    return f"€{value / 1_000:.0f}K"


def describe_gap(ratio: float, typical_error_percent: float) -> str:
    """Say in words whether the model and the market disagree meaningfully."""
    if ratio >= GAP_THRESHOLD:
        return f"The model values him {ratio:.1f} times higher than the market."
    if ratio <= 1 / GAP_THRESHOLD:
        return f"The model values him {1 / ratio:.1f} times lower than the market."
    return (
        "The model and the market broadly agree: the difference is within the model's normal "
        f"margin of error (it is typically off by about {typical_error_percent:.0f}%)."
    )


def relevant_limitations(player: pd.Series) -> list[str]:
    """Pick the weaknesses of the model that can explain this player's gap.

    Returns an empty list when the model and the market broadly agree, or when
    no known weakness applies.
    """
    ratio = float(player["predicted_to_actual_ratio"])
    model_higher = ratio >= GAP_THRESHOLD
    model_lower = ratio <= 1 / GAP_THRESHOLD
    if not (model_higher or model_lower):
        return []

    limitations = []
    if player["position"] in ("Defence", "Midfield"):
        limitations.append(LIMITATION_DEFENDING)
    if model_higher and player[TARGET] < CHEAP_PLAYER_EUR:
        limitations.append(LIMITATION_CHEAP_PLAYER)
    if model_lower and player[TARGET] >= EXPENSIVE_PLAYER_EUR:
        limitations.append(LIMITATION_EXPENSIVE_PLAYER)
    if model_higher and player["minutes_played"] < FEW_MINUTES:
        limitations.append(LIMITATION_FEW_MINUTES)
    return limitations


def player_facts(
    player: pd.Series, predictions: pd.DataFrame, typical_error_percent: float
) -> dict[str, Any]:
    """Gather everything the LLM may say about one player."""
    same_position = predictions[predictions["position"] == player["position"]]

    def percentile(column: str) -> int:
        """Share of same-position players that this player beats or equals."""
        return int(round((same_position[column] <= player[column]).mean() * 100))

    ratio = float(player["predicted_to_actual_ratio"])
    return {
        "player": {
            "name": player["name"],
            "team": player["team"],
            "league": config.COMPETITIONS[player["competition"]],
            "position": player["position"],
            "age": round(float(player["age"]), 1),
        },
        "season_statistics": {
            "goals": int(player["goals"]),
            "assists": int(player["assists"]),
            "minutes_played": int(player["minutes_played"]),
            "league_matches_played": int(player["played_matches"]),
        },
        "percentile_among_same_position": {
            "goals": percentile("goals"),
            "assists": percentile("assists"),
            "minutes_played": percentile("minutes_played"),
        },
        "club_context": {
            "league_position": int(player["table_position"]),
            "points": int(player["points"]),
            "champions_league": CL_STAGE_LABELS[int(player["cl_stage"])],
        },
        "valuation": {
            "market_value": format_euros(player[TARGET]),
            "model_prediction": format_euros(player["predicted_value_eur"]),
            "prediction_divided_by_market_value": round(ratio, 2),
            "assessment": describe_gap(ratio, typical_error_percent),
        },
        "model": MODEL_DESCRIPTION,
        "known_limitations_that_apply": relevant_limitations(player),
    }


def model_facts(metrics: pd.DataFrame, coefficients: pd.DataFrame) -> dict[str, Any]:
    """Gather everything the LLM may say about the model itself."""
    return {
        "model": MODEL_DESCRIPTION,
        "metrics_by_model": {
            "explanation": {
                "median_pct_error": "typical error, as a percentage of the real value (test set)",
                "cv_median_pct_error": "the same, by cross-validation on all players",
                "r2_log": "share of the variance of the log value explained (0 to 1)",
                "baseline": "gives every player the same typical value",
                "linear": "plain linear regression on goals, assists, minutes, age and position",
                "ridge": "the main model described above",
                "ridge_tm": "the main model plus career, contract and international factors",
                "boosting": "gradient boosting on the main model's factors, for comparison",
            },
            "values": metrics.round(3).to_dict(orient="records"),
        },
        "largest_coefficients_of_the_main_model": {
            "explanation": (
                "effect_percent is the change in market value for one standard deviation of "
                "a factor, the others being held constant. The factors are correlated, so "
                "read each one with care. Age, age squared and age cubed form one curve and "
                "cannot be read separately."
            ),
            "values": json.loads(coefficients.head(12).round(3).to_json(orient="records")),
        },
        "known_limitations": MODEL_LIMITATIONS,
    }


def build_messages(facts: dict[str, Any], instructions: str) -> list[dict[str, str]]:
    """Build the chat messages: fixed rules, then the facts and the task."""
    document = json.dumps(facts, indent=2, ensure_ascii=False)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instructions}\n\nDATA:\n{document}"},
    ]


# --- LLM call ----------------------------------------------------------------


def create_client(settings: config.Settings) -> OpenAI:
    """Create a client for the OpenAI-compatible gateway set in ``.env``."""
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.require_llm_api_key(),
        timeout=REQUEST_TIMEOUT_SECONDS,
        max_retries=MAX_RETRIES,
    )


def generate_text(client: OpenAI, model: str, messages: list[dict[str, str]]) -> str:
    """Send the messages to the LLM and return its answer.

    Raises:
        LLMError: if the provider fails or returns an empty answer.
    """
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=TEMPERATURE
        )
    except OpenAIError as error:
        raise LLMError(f"The LLM provider returned an error: {error}") from error

    text = response.choices[0].message.content if response.choices else None
    if not text or not text.strip():
        raise LLMError("The LLM returned an empty answer.")
    return text.strip()


def main_model_error(metrics: pd.DataFrame) -> float:
    """Typical percentage error of the main model, by cross-validation."""
    row = metrics[metrics["model"] == MAIN_SPEC.name]
    return float(row["cv_median_pct_error"].iloc[0])


def write_player_report(
    name: str, predictions: pd.DataFrame, metrics: pd.DataFrame, client: OpenAI, model: str
) -> str:
    """Write the scouting report of one player."""
    player = find_player(predictions, name)
    facts = player_facts(player, predictions, main_model_error(metrics))
    return generate_text(client, model, build_messages(facts, PLAYER_INSTRUCTIONS))


def write_model_summary(
    metrics: pd.DataFrame, coefficients: pd.DataFrame, client: OpenAI, model: str
) -> str:
    """Write the plain-language summary of the regression."""
    facts = model_facts(metrics, coefficients)
    return generate_text(client, model, build_messages(facts, MODEL_INSTRUCTIONS))


# --- Command line ------------------------------------------------------------


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text about the model with an LLM.")
    commands = parser.add_subparsers(dest="command", required=True)

    player = commands.add_parser("player", help="scouting report on one player")
    player.add_argument("name", help='player name or part of it, e.g. "Muriqi"')

    commands.add_parser("model", help="plain-language summary of the model")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    options = parse_arguments(arguments)

    try:
        settings = config.load_settings()
        client = create_client(settings)
        if options.command == "player":
            text = write_player_report(
                options.name,
                pd.read_csv(PREDICTIONS_FILE),
                pd.read_csv(METRICS_FILE),
                client,
                settings.llm_model,
            )
        else:
            text = write_model_summary(
                pd.read_csv(METRICS_FILE),
                pd.read_csv(COEFFICIENTS_FILE),
                client,
                settings.llm_model,
            )
    except (PlayerNotFoundError, LLMError, config.MissingSettingError) as error:
        logger.error("%s", error)
        return 1

    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
