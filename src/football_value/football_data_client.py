"""Minimal client for the football-data.org v4 API.

Two things make it safe to use on the free plan:

* every response is cached on disk, so a request is only ever sent once;
* requests are spaced out to stay under the plan's rate limit.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import requests

from football_value import config

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT_SECONDS = 30
RATE_LIMIT_STATUS = 429
MAX_RATE_LIMIT_RETRIES = 3


class FootballDataError(RuntimeError):
    """Raised when the API answers with an error we cannot recover from."""


class FootballDataClient:
    """Read-only client with an on-disk cache and client-side rate limiting.

    Args:
        api_key: football-data.org token, sent in the ``X-Auth-Token`` header.
        cache_dir: folder where raw JSON responses are stored.
        requests_per_minute: maximum request rate allowed by the plan.
        session: HTTP session to use (injectable for tests).
        sleep: function used to wait (injectable for tests).
        clock: monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        api_key: str,
        cache_dir: Path,
        *,
        base_url: str = config.FOOTBALL_DATA_BASE_URL,
        requests_per_minute: int = config.FOOTBALL_DATA_REQUESTS_PER_MINUTE,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._api_key = api_key
        self._cache_dir = cache_dir
        self._base_url = base_url.rstrip("/")
        self._min_interval = 60.0 / requests_per_minute
        self._session = session or requests.Session()
        self._sleep = sleep
        self._clock = clock
        self._last_request_at: float | None = None

    def get(self, path: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Return the JSON body for ``path``, from the cache when available."""
        params = dict(params or {})
        cache_file = self._cache_dir / cache_filename(path, params)

        if cache_file.exists():
            logger.debug("Cache hit: %s", cache_file.name)
            return json.loads(cache_file.read_text(encoding="utf-8"))

        payload = self._request(path, params)

        self._cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        return payload

    def _request(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = f"{self._base_url}/{path.lstrip('/')}"

        for _attempt in range(MAX_RATE_LIMIT_RETRIES + 1):
            self._wait_for_rate_limit()
            logger.info("GET %s %s", url, params or "")
            response = self._session.get(
                url,
                params=params,
                headers={"X-Auth-Token": self._api_key},
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            self._last_request_at = self._clock()

            if response.status_code == RATE_LIMIT_STATUS:
                logger.warning("Rate limit reached, waiting 60 seconds before retrying.")
                self._sleep(60.0)
                continue
            if not response.ok:
                raise FootballDataError(
                    f"football-data.org answered {response.status_code} for {path}: "
                    f"{_error_message(response)}"
                )
            return response.json()

        raise FootballDataError(f"Rate limit still exceeded after retries for {path}.")

    def _wait_for_rate_limit(self) -> None:
        """Sleep just long enough to respect the minimum gap between requests."""
        if self._last_request_at is None:
            return
        elapsed = self._clock() - self._last_request_at
        if elapsed < self._min_interval:
            self._sleep(self._min_interval - elapsed)


def cache_filename(path: str, params: Mapping[str, Any]) -> str:
    """Build a readable, deterministic file name for a request.

    >>> cache_filename("competitions/PL/scorers", {"season": 2025, "limit": 1000})
    'competitions_PL_scorers__limit-1000_season-2025.json'
    """
    name = path.strip("/").replace("/", "_")
    if params:
        query = "_".join(f"{key}-{params[key]}" for key in sorted(params))
        name = f"{name}__{query}"
    return f"{name}.json"


def _error_message(response: requests.Response) -> str:
    try:
        return str(response.json().get("message", response.text))
    except ValueError:
        return response.text
