"""Exploratory data analysis: understand the data before modelling it.

Run it after the features step::

    python -m football_value.eda

It prints a few key statistics and saves four charts in ``reports/figures/``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from football_value import config, plotting
from football_value.features import FEATURES_FILE, NUMERIC_FEATURES, TARGET

logger = logging.getLogger(__name__)

LOG_TARGET = "log_market_value"

FEATURE_LABELS = {
    "goals": "Goals",
    "assists": "Assists",
    "minutes_played": "Minutes played",
    "age": "Age",
    TARGET: "Market value",
    LOG_TARGET: "Market value (log)",
}

POSITION_ORDER = ["Attack", "Midfield", "Defence"]


def add_log_target(players: pd.DataFrame) -> pd.DataFrame:
    """Add the base-10 logarithm of the market value."""
    return players.assign(**{LOG_TARGET: np.log10(players[TARGET])})


def skewness(players: pd.DataFrame) -> pd.Series:
    """Skewness of the market value, before and after the log transform.

    A value near 0 means a symmetric distribution; a large positive value
    means a long tail of very high values.
    """
    return add_log_target(players)[[TARGET, LOG_TARGET]].skew()


def correlation_matrix(players: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlations between the numeric features and the target."""
    columns = [*NUMERIC_FEATURES, TARGET, LOG_TARGET]
    return add_log_target(players)[columns].corr()


# --- Charts ------------------------------------------------------------------


def plot_target_distribution(players: pd.DataFrame, directory: Path) -> Path:
    """Histogram of the market value, on a linear and on a log scale."""
    skew = skewness(players)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.6))
    bar_style = {"color": plotting.BLUE, "edgecolor": plotting.SURFACE, "linewidth": 1}

    left.hist(players[TARGET] / 1e6, bins=40, **bar_style)
    left.set_title(f"Linear scale (skewness {skew[TARGET]:.1f})")
    left.set_xlabel("Market value (€ million)")
    left.set_ylabel("Number of players")

    log_bins = np.logspace(np.log10(players[TARGET].min()), np.log10(players[TARGET].max()), num=30)
    right.hist(players[TARGET], bins=log_bins, **bar_style)
    plotting.log_euro_axis(right, axis="x")
    right.set_title(f"Log scale (skewness {skew[LOG_TARGET]:.1f})")
    right.set_xlabel("Market value (log scale)")

    for ax in (left, right):
        ax.grid(axis="x", visible=False)

    plotting.set_figure_title(
        fig,
        "Market value is heavily skewed; its logarithm is not",
        "A few stars are worth far more than everyone else, so the model will predict the log.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return plotting.save_figure(fig, "market_value_distribution", directory)


def plot_features_vs_target(players: pd.DataFrame, directory: Path) -> Path:
    """One scatter plot per numeric feature against the market value."""
    correlations = correlation_matrix(players)[LOG_TARGET]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2), sharey=True)

    for ax, feature in zip(axes.flat, NUMERIC_FEATURES, strict=True):
        ax.scatter(
            players[feature],
            players[TARGET],
            s=14,
            color=plotting.BLUE,
            alpha=0.35,
            edgecolors="none",
        )
        plotting.log_euro_axis(ax)
        ax.set_title(f"{FEATURE_LABELS[feature]}  (r = {correlations[feature]:+.2f})")
        ax.set_xlabel(FEATURE_LABELS[feature])

    for ax in axes[:, 0]:
        ax.set_ylabel("Market value (log scale)")

    plotting.set_figure_title(
        fig,
        "Age is the strongest single signal; goals, assists and minutes all help a little",
        "Each dot is a player. r is the correlation with the log of the market value.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return plotting.save_figure(fig, "features_vs_market_value", directory)


def plot_value_by_group(players: pd.DataFrame, directory: Path) -> Path:
    """Box plots of the market value by position and by league."""
    league_order = (
        players.groupby("competition")[TARGET].median().sort_values(ascending=False).index
    )
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)

    _boxplot(left, players, "position", POSITION_ORDER)
    left.set_title("By position: almost no difference")
    left.set_ylabel("Market value (log scale)")

    _boxplot(right, players, "competition", list(league_order))
    right.set_xticklabels([config.COMPETITIONS[code] for code in league_order])
    right.set_title("By league: the Premier League stands apart")

    plotting.set_figure_title(
        fig,
        "Position barely moves market value, the league does",
        "The line in each box is the median; the box holds the middle half of the players.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return plotting.save_figure(fig, "market_value_by_group", directory)


def _boxplot(ax: plt.Axes, players: pd.DataFrame, column: str, order: list[str]) -> None:
    ax.boxplot(
        [players.loc[players[column] == group, TARGET] for group in order],
        tick_labels=order,
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
    plotting.log_euro_axis(ax)
    ax.grid(axis="x", visible=False)


def plot_correlation_matrix(players: pd.DataFrame, directory: Path) -> Path:
    """Heatmap of the correlations between the features and the log target."""
    columns = [*NUMERIC_FEATURES, LOG_TARGET]
    matrix = correlation_matrix(players).loc[columns, columns]
    labels = [FEATURE_LABELS[column] for column in columns]

    # Two opposite hues around a neutral midpoint: red = negative, blue = positive.
    colormap = LinearSegmentedColormap.from_list(
        "red_neutral_blue", [plotting.RED, plotting.NEUTRAL, plotting.BLUE]
    )

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap=colormap, vmin=-1, vmax=1)

    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.grid(visible=False)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)

    for row in range(len(columns)):
        for column in range(len(columns)):
            value = matrix.iloc[row, column]
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="#ffffff" if abs(value) > 0.6 else plotting.INK,
            )

    # White gaps between the cells.
    ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(columns)), minor=True)
    ax.grid(which="minor", color=plotting.SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Correlation")

    plotting.set_figure_title(
        fig,
        "The features are only mildly correlated with each other",
        "So each one brings its own information to the model.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return plotting.save_figure(fig, "correlation_matrix", directory)


def run(players: pd.DataFrame, directory: Path = config.FIGURES_DIR) -> list[Path]:
    """Create every chart and return the paths of the files written."""
    plotting.apply_style()
    return [
        plot_target_distribution(players, directory),
        plot_features_vs_target(players, directory),
        plot_value_by_group(players, directory),
        plot_correlation_matrix(players, directory),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    players = pd.read_csv(FEATURES_FILE)

    logger.info("%d players", len(players))
    logger.info(
        "Market value: median %s, mean %s, max %s",
        plotting.format_euros(players[TARGET].median()),
        plotting.format_euros(round(players[TARGET].mean(), -5)),
        plotting.format_euros(players[TARGET].max()),
    )
    logger.info("Skewness:\n%s", skewness(players).round(2).to_string())
    logger.info(
        "Correlation with the log market value:\n%s",
        correlation_matrix(players)[LOG_TARGET][NUMERIC_FEATURES].round(2).to_string(),
    )

    for path in run(players):
        logger.info("Saved %s", path)


if __name__ == "__main__":
    main()
