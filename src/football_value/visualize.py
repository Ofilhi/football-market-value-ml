"""Charts of the model's results.

Run it after the model step::

    python -m football_value.visualize

It reads ``data/processed/predictions.csv`` (one out-of-sample prediction per
player) and saves three charts in ``reports/figures/``:

* predicted against actual market value;
* the residuals, overall and by league;
* the players the model disagrees with the market on the most.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FixedLocator, FuncFormatter

from football_value import config, plotting
from football_value.features import TARGET
from football_value.model import PREDICTIONS_FILE

logger = logging.getLogger(__name__)

PREDICTED = "predicted_value_eur"
RATIO = "predicted_to_actual_ratio"

# The ranking only considers regular players: about ten full matches.
MIN_MINUTES_FOR_RANKING = 900
RANKING_SIZE = 10

RATIO_TICKS = [0.1, 0.2, 0.5, 1, 2, 5, 10]


def format_ratio(ratio: float, _position: float | None = None) -> str:
    """Describe a predicted / actual ratio in words.

    >>> format_ratio(5)
    '5× higher'
    >>> format_ratio(0.1)
    '10× lower'
    >>> format_ratio(1)
    'same'
    """
    if np.isclose(ratio, 1):
        return "same"
    if ratio > 1:
        return f"{ratio:g}× higher"
    return f"{1 / ratio:g}× lower"


def value_gaps(
    predictions: pd.DataFrame,
    min_minutes: int = MIN_MINUTES_FOR_RANKING,
    size: int = RANKING_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find the regular players where model and market disagree the most.

    Returns:
        Two tables of ``size`` players, largest gap first:

        * players the model values *higher* than the market does;
        * players the model values *lower* than the market does.
    """
    regulars = predictions[predictions["minutes_played"] >= min_minutes].sort_values(RATIO)
    model_higher = regulars[regulars[RATIO] > 1].tail(size).iloc[::-1]
    model_lower = regulars[regulars[RATIO] < 1].head(size)
    return model_higher.reset_index(drop=True), model_lower.reset_index(drop=True)


# --- Charts ------------------------------------------------------------------


def plot_predicted_vs_actual(predictions: pd.DataFrame, directory: Path) -> Path:
    """Scatter plot of predicted against actual value, with the ideal line."""
    fig, ax = plt.subplots(figsize=(8.2, 7.6))
    low = min(predictions[TARGET].min(), predictions[PREDICTED].min()) * 0.7
    high = max(predictions[TARGET].max(), predictions[PREDICTED].max()) * 1.5

    ax.plot([low, high], [low, high], color=plotting.INK_MUTED, linewidth=1.5, linestyle="--")
    ax.text(
        low * 2.2,
        low * 2.9,
        "perfect prediction",
        color=plotting.INK_SECONDARY,
        rotation=45,
        rotation_mode="anchor",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    ax.scatter(
        predictions[TARGET],
        predictions[PREDICTED],
        s=16,
        color=plotting.BLUE,
        alpha=0.35,
        edgecolors="none",
    )

    # Name the two most valuable players only. Labels are staggered
    # vertically (highest prediction on top) so they do not overlap.
    stars = predictions.nlargest(2, TARGET).sort_values(PREDICTED, ascending=False)
    for vertical_offset, player in zip((10, -6), stars.itertuples(), strict=True):
        ax.annotate(
            player.name,
            (getattr(player, TARGET), getattr(player, PREDICTED)),
            xytext=(-8, vertical_offset),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color=plotting.INK,
        )

    plotting.log_euro_axis(ax, "x")
    plotting.log_euro_axis(ax, "y")
    ax.set_xlim(low, high)
    ax.set_ylim(low, high)
    ax.set_aspect("equal")
    ax.set_xlabel("Actual market value (log scale)")
    ax.set_ylabel("Predicted market value (log scale)")

    plotting.set_figure_title(
        fig,
        "The model gets the order of magnitude, and plays it safe at both ends",
        "Each dot is a player, predicted by a model that never saw him. Cheap players are\n"
        "predicted too high and stars too low: predictions stay closer to the average.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return plotting.save_figure(fig, "predicted_vs_actual", directory)


def plot_residuals(predictions: pd.DataFrame, directory: Path) -> Path:
    """Residuals against the predicted value, and their spread by league."""
    league_order = predictions.groupby("competition")[RATIO].median().sort_values().index.to_list()
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.5, 5), sharey=True, gridspec_kw={"width_ratios": [1.15, 1]}
    )

    left.scatter(
        predictions[PREDICTED],
        predictions[RATIO],
        s=14,
        color=plotting.BLUE,
        alpha=0.35,
        edgecolors="none",
    )
    plotting.log_euro_axis(left, "x")
    left.set_xlabel("Predicted market value (log scale)")
    left.set_ylabel("Prediction compared with the real value")
    left.set_title("No pattern left along the predictions")

    right.boxplot(
        [predictions.loc[predictions["competition"] == code, RATIO] for code in league_order],
        tick_labels=[config.COMPETITIONS[code] for code in league_order],
        widths=0.5,
        patch_artist=True,
        boxprops={"facecolor": plotting.BLUE, "edgecolor": plotting.BLUE, "alpha": 0.35},
        medianprops={"color": plotting.INK, "linewidth": 2},
        whiskerprops={"color": plotting.INK_MUTED},
        capprops={"color": plotting.INK_MUTED},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": plotting.INK_MUTED,
            "markeredgecolor": "none",
            "alpha": 0.5,
        },
    )
    right.grid(axis="x", visible=False)
    right.set_title("Premier League players are predicted too low")

    for ax in (left, right):
        ax.set_yscale("log")
        ax.axhline(1, color=plotting.INK, linewidth=1.2)
        ax.yaxis.set_major_locator(FixedLocator(RATIO_TICKS))
        ax.yaxis.set_major_formatter(FuncFormatter(format_ratio))
        ax.yaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))

    plotting.set_figure_title(
        fig,
        "The errors are balanced overall, but not across leagues",
        "The league is not a feature of the model, so its effect ends up in the errors.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return plotting.save_figure(fig, "residuals", directory)


def plot_value_gaps(predictions: pd.DataFrame, directory: Path) -> Path:
    """Bar chart of the players where model and market disagree the most."""
    model_higher, model_lower = value_gaps(predictions)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(10.5, 9.4), sharex=True)

    _gap_bars(top, model_higher, plotting.BLUE)
    top.set_title("Model higher than the market: mostly veterans, whose value drops faster")
    _gap_bars(bottom, model_lower, plotting.RED)
    bottom.set_title("Model lower than the market: top-club stars valued for more than goals")
    bottom.set_xlabel("Model's value compared with the market value")

    plotting.set_figure_title(
        fig,
        "Where the model and the market disagree the most",
        f"Players with at least {MIN_MINUTES_FOR_RANKING} minutes. "
        "Labels show: market value → model's value.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return plotting.save_figure(fig, "value_gaps", directory)


def _gap_bars(ax: plt.Axes, players: pd.DataFrame, color: str) -> None:
    labels = [f"{player.name} ({player.age:.0f}, {player.team})" for player in players.itertuples()]
    positions = np.arange(len(players))[::-1]

    # Bars start at 1 ("same value") and extend to the ratio.
    ax.barh(positions, players[RATIO] - 1, left=1, height=0.62, color=color)
    ax.set_yticks(positions, labels)
    ax.set_xscale("log")
    ax.set_xlim(0.04, 25)
    ax.axvline(1, color=plotting.INK, linewidth=1.2)
    ax.xaxis.set_major_locator(FixedLocator(RATIO_TICKS))
    ax.xaxis.set_major_formatter(FuncFormatter(format_ratio))
    ax.xaxis.set_minor_formatter(FuncFormatter(lambda *_: ""))
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)

    for position, player in zip(positions, players.itertuples(), strict=True):
        ratio = getattr(player, RATIO)
        text = (
            f"{plotting.format_euros(getattr(player, TARGET))} → "
            f"{plotting.format_euros(round(getattr(player, PREDICTED), -5))}"
        )
        # Always written on the right: after the bar when it points right,
        # after the "same value" line when it points left.
        ax.annotate(
            text,
            (max(ratio, 1), position),
            xytext=(6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color=plotting.INK_SECONDARY,
        )


def run(predictions: pd.DataFrame, directory: Path = config.FIGURES_DIR) -> list[Path]:
    """Create every chart and return the paths of the files written."""
    plotting.apply_style()
    return [
        plot_predicted_vs_actual(predictions, directory),
        plot_residuals(predictions, directory),
        plot_value_gaps(predictions, directory),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    predictions = pd.read_csv(PREDICTIONS_FILE)

    columns = ["name", "team", "age", "goals", "assists", TARGET, PREDICTED, RATIO]
    model_higher, model_lower = value_gaps(predictions)
    logger.info("Model higher than the market:\n%s", model_higher[columns].to_string(index=False))
    logger.info("Model lower than the market:\n%s", model_lower[columns].to_string(index=False))

    for path in run(predictions):
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
