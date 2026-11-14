"""Shared figure style: one look for every exhibit. Colors are the validated reference palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"      # categorical slots 1-3, in fixed order
MUTED = "#c9c8c2"                                          # background / "everything else"
GROUP_COLOR = {"Migros group": BLUE, "Coop group": ORANGE, "Aldi + Lidl": AQUA}

ASSETS = Path(__file__).resolve().parents[2] / "assets"


def setup():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.edgecolor": INK_2, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2,
        "legend.frameon": False, "figure.dpi": 150,
    })


def save(fig, pillar: int, step: str, name: str):
    """assets/p{pillar}/m{module}/{step}_{name}.png, e.g. assets/p1/m1/1_1_overlap_map.png (step "1_1" -> m1)"""
    out = ASSETS / f"p{pillar}" / f"m{step.split('_')[1]}" / f"{step}_{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def source(ax, text):
    ax.figure.text(0.01, -0.02, text, fontsize=7.5, color=INK_2, ha="left", va="top")


def map_axes(ax, cantons):
    cantons.boundary.plot(ax=ax, color=GRID, linewidth=0.5, zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")
