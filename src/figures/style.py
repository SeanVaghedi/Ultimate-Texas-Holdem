"""Shared styling for report figures."""

from pathlib import Path

import matplotlib as mpl


FIGURES_DIR = Path(__file__).resolve().parents[2] / "results" / "figures"

EV_OPTIMAL_HOUSE_EDGE = 5.6275
EV_OPTIMAL_MEAN_CHIPS = -0.0563
RANDOM_MEAN_CHIPS = -0.5417
MC_TABULAR_HOUSE_EDGE = 17.6385


def apply_style() -> None:
    """Apply consistent styling to all figures."""
    mpl.rcParams["figure.figsize"] = (8, 5)
    mpl.rcParams["figure.dpi"] = 150
    mpl.rcParams["savefig.dpi"] = 150
    mpl.rcParams["savefig.bbox"] = "tight"
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.grid"] = True
    mpl.rcParams["grid.alpha"] = 0.3
    mpl.rcParams["font.size"] = 11
    mpl.rcParams["axes.facecolor"] = "white"
    mpl.rcParams["figure.facecolor"] = "white"


COLORS = {
    "mc": "#1f77b4",
    "sarsa": "#ff7f0e",
    "q_learning": "#d62728",
    "sarsa_lambda": "#9467bd",
    "linear_fa": "#2ca02c",
    "optimal": "#000000",
    "naive": "#7f7f7f",
    "random": "#bcbd22",
    "always_fold": "#e377c2",
}


def ensure_figures_dir() -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR
