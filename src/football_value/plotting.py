"""Shared look for every chart of the project.

Keeping colours and styling in one place makes the figures consistent and
lets a chart function focus on what it shows.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # render to files, no display needed

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from football_value import config  # noqa: E402

# --- Colours -----------------------------------------------------------------

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
RED = "#e34948"
NEUTRAL = "#f0efec"

# One fixed colour per position, so a position looks the same in every chart.
# This trio stays distinguishable for colour-blind readers.
POSITION_COLORS: dict[str, str] = {"Attack": BLUE, "Midfield": ORANGE, "Defence": AQUA}

DPI = 160


def apply_style() -> None:
    """Set the matplotlib defaults used by all figures."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK_SECONDARY,
            "axes.edgecolor": AXIS,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "legend.frameon": False,
        }
    )


def format_euros(value: float, _position: float | None = None) -> str:
    """Format an amount in euros compactly.

    >>> format_euros(200_000_000)
    '€200M'
    >>> format_euros(2_500_000)
    '€2.5M'
    >>> format_euros(300_000)
    '€300K'
    """
    if abs(value) >= 1_000_000:
        return f"€{value / 1_000_000:g}M"
    if abs(value) >= 1_000:
        return f"€{value / 1_000:g}K"
    return f"€{value:g}"


EURO_FORMATTER = FuncFormatter(format_euros)


def set_figure_title(fig: Figure, title: str, subtitle: str | None = None) -> None:
    """Add a left-aligned title, with an optional explanatory subtitle."""
    fig.suptitle(title, x=0.01, ha="left", fontsize=13, fontweight="bold", color=INK)
    if subtitle:
        fig.text(0.01, 0.925, subtitle, ha="left", va="top", fontsize=10, color=INK_SECONDARY)


def log_euro_axis(ax: Axes, axis: str = "y") -> None:
    """Put an axis on a log scale with readable euro labels."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    (ax.set_yscale if axis == "y" else ax.set_xscale)("log")
    target.set_major_formatter(EURO_FORMATTER)
    target.set_minor_formatter(FuncFormatter(lambda *_: ""))


def save_figure(fig: Figure, name: str, directory: Path = config.FIGURES_DIR) -> Path:
    """Save a figure as PNG and close it."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
