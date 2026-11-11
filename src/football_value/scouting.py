"""Turn the model's numbers into readable text with an LLM.

Two commands, both run after the model step::

    python -m football_value.scouting player "Vedat Muriqi"
    python -m football_value.scouting model

* ``player`` writes a short scouting report on one player: his statistics,
  what the model thinks he is worth, and how that compares with the market.
* ``model`` writes a plain-language summary of the regression itself.

The LLM never predicts anything. It only receives numbers that the pipeline
already computed, together with the model's known weaknesses, and is asked to
comment on them without adding facts of its own.
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
from football_value.model import COEFFICIENTS_FILE, METRICS_FILE, PREDICTIONS_FILE

logger = logging.getLogger(__name__)

TEMPERATURE = 0.3

MODEL_DESCRIPTION = (
    "A linear regression predicts the logarithm of a football player's market value "
    "from five features: goals, assists, minutes played, age and position. It is trained "
    "on players of the top 5 European leagues (2025-26 season) who scored at least one "
    "goal. Market values come from Transfermarkt."
)

# Told to the LLM so that it explains a gap honestly instead of over-selling it.
KNOWN_LIMITATIONS = [
    "The model only knows goals, assists, minutes, age and position. It knows nothing "
    "about defending, passing, injuries, contracts, reputation or the club's wealth.",
    "The league is not a feature: Premier League players are predicted about twice too low.",
    "Age has a straight-line effect, so players over 30 tend to be predicted too high.",
    "Predictions stay close to the average: stars are predicted too low and cheap "
    "players too high.",
]

SYSTEM_PROMPT = (
    "You are a football analyst writing for a data science portfolio project. "
    "You are given a JSON document with numbers produced by a statistical model. "
    "Use only the facts in that document: do not add transfers, injuries, trophies "
    "or any other information from memory. Be concrete, quote the numbers, and be "
    "honest about what the model cannot see. Write in plain English, without headings "
    "or bullet points."
)

PLAYER_INSTRUCTIONS = (
    "Write a scouting note of 120 to 160 words about this player. Cover, in order: "
    "his season in numbers; what the model thinks he is worth compared with the market; "
    "the most likely reason for the gap, using the known limitations when they apply. "
    "Finish with one sentence saying whether the gap is more likely a real market "
    "inefficiency or a blind spot of the model."
)

MODEL_INSTRUCTIONS = (
    "Explain this model to a football fan with no statistics background, in 150 to 200 "
    "words. Say what it does, how accurate it is compared with the baseline, which "
    "features matter most and in which direction, and what it cannot see."
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


def player_facts(player: pd.Series, predictions: pd.DataFrame) -> dict[str, Any]:
    """Gather everything the LLM may say about one player."""
    same_position = predictions[predictions["position"] == player["position"]]

    def percentile(column: str) -> int:
        """Share of same-position players that this player beats or equals."""
        return int(round((same_position[column] <= player[column]).mean() * 100))

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
        },
        "percentile_among_same_position": {
            "goals": percentile("goals"),
            "assists": percentile("assists"),
            "minutes_played": percentile("minutes_played"),
        },
        "valuation_eur": {
            "market_value": int(player[TARGET]),
            "model_prediction": int(player["predicted_value_eur"]),
            "prediction_divided_by_market_value": round(
                float(player["predicted_to_actual_ratio"]), 2
            ),
        },
        "model": MODEL_DESCRIPTION,
        "known_limitations": KNOWN_LIMITATIONS,
    }


def model_facts(metrics: pd.DataFrame, coefficients: pd.DataFrame) -> dict[str, Any]:
    """Gather everything the LLM may say about the model itself."""
    return {
        "model": MODEL_DESCRIPTION,
        "test_set_metrics": {
            "explanation": {
                "median_pct_error": "typical error as a percentage of the real value",
                "mae_eur": "mean absolute error in euros",
                "r2_log": "share of the variance of the log value explained (0 to 1)",
                "baseline": "a model that gives every player the same typical value",
                "extended": "the linear model plus the league and age squared",
            },
            "values": metrics.round(3).to_dict(orient="records"),
        },
        "linear_model_coefficients": {
            "explanation": (
                "effect_percent is the change in market value for one standard deviation "
                "(one_std, in the feature's own unit) of a numeric feature, or compared "
                "with an attacker for a position."
            ),
            "values": json.loads(coefficients.round(3).to_json(orient="records")),
        },
        "known_limitations": KNOWN_LIMITATIONS,
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
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.require_llm_api_key())


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


def write_player_report(name: str, predictions: pd.DataFrame, client: OpenAI, model: str) -> str:
    """Write the scouting report of one player."""
    facts = player_facts(find_player(predictions, name), predictions)
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
                options.name, pd.read_csv(PREDICTIONS_FILE), client, settings.llm_model
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
