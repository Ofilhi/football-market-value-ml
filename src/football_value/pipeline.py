"""Run the whole pipeline with a single command::

    python -m football_value.pipeline

The steps run in order, each one reading the files written by the previous one:

1. ``collect``: player statistics from football-data.org (cached);
2. ``merge``: market values and minutes from Transfermarkt;
3. ``features``: cleaning and feature engineering;
4. ``eda``: exploratory charts;
5. ``model``: training, evaluation, predictions;
6. ``visualize``: charts of the results.

The LLM scouting reports (:mod:`football_value.scouting`) are left out on
purpose: they call an external service with its own usage limits.
"""

from __future__ import annotations

import logging
import time
from types import ModuleType

from football_value import collect, eda, features, merge, model, visualize

logger = logging.getLogger(__name__)

STEPS: tuple[ModuleType, ...] = (collect, merge, features, eda, model, visualize)


def run() -> None:
    """Run every step in order. The first failing step stops the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    started = time.monotonic()

    for number, step in enumerate(STEPS, start=1):
        name = step.__name__.rsplit(".", 1)[-1]
        logger.info("[%d/%d] %s", number, len(STEPS), name)
        step.main()

    logger.info("Pipeline finished in %.0f seconds.", time.monotonic() - started)


if __name__ == "__main__":
    run()
