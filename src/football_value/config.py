"""Central configuration: project paths, constants and secrets.

Secrets are read from environment variables, which are loaded from a local
``.env`` file when one exists. Nothing secret is ever hard-coded here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

# --- Data collection ---------------------------------------------------------

FOOTBALL_DATA_BASE_URL = "https://api.football-data.org/v4"

# Free plan of football-data.org: 10 requests per minute.
FOOTBALL_DATA_REQUESTS_PER_MINUTE = 10

# The five major European leagues, by football-data.org competition code.
COMPETITIONS: dict[str, str] = {
    "PL": "Premier League",
    "PD": "La Liga",
    "SA": "Serie A",
    "BL1": "Bundesliga",
    "FL1": "Ligue 1",
}

# A season is identified by the year it starts in: 2025 means 2025-26.
SEASON = 2025
SEASON_START = f"{SEASON}-07-01"
SEASON_END = f"{SEASON + 1}-06-30"

# Public Transfermarkt dataset (https://github.com/dcaribou/transfermarkt-datasets).
TRANSFERMARKT_BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"

# football-data.org competition code -> Transfermarkt competition id.
TRANSFERMARKT_COMPETITION_IDS: dict[str, str] = {
    "PL": "GB1",
    "PD": "ES1",
    "SA": "IT1",
    "BL1": "L1",
    "FL1": "FR1",
}

# --- Modelling ---------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.2

# --- LLM defaults (overridable from the environment) -------------------------

DEFAULT_LLM_BASE_URL = "https://gateway.respan.ai/v1/"
DEFAULT_LLM_MODEL = "gemini-4-carbon:free"


class MissingSettingError(RuntimeError):
    """Raised when a required environment variable is not set."""


@dataclass(frozen=True)
class Settings:
    """Values that come from the environment (``.env`` file or shell)."""

    football_data_api_key: str | None
    llm_api_key: str | None
    llm_base_url: str
    llm_model: str

    def require_football_data_api_key(self) -> str:
        """Return the football-data.org key, or fail with a helpful message."""
        return _require(self.football_data_api_key, "FOOTBALL_DATA_API_KEY")

    def require_llm_api_key(self) -> str:
        """Return the LLM provider key, or fail with a helpful message."""
        return _require(self.llm_api_key, "RESPAN_API_KEY")


def _require(value: str | None, name: str) -> str:
    if not value:
        raise MissingSettingError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value


def load_settings(env_file: Path | None = PROJECT_ROOT / ".env") -> Settings:
    """Build the settings from the environment.

    Variables already set in the shell take precedence over the ``.env`` file.
    Pass ``env_file=None`` to skip the file entirely (useful in tests).
    """
    if env_file is not None:
        load_dotenv(env_file)

    return Settings(
        football_data_api_key=os.getenv("FOOTBALL_DATA_API_KEY") or None,
        llm_api_key=os.getenv("RESPAN_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or DEFAULT_LLM_BASE_URL,
        llm_model=os.getenv("LLM_MODEL") or DEFAULT_LLM_MODEL,
    )
