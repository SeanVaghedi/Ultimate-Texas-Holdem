"""Fig 1: Monte Carlo convergence at 50k / 500k / 5M training scales."""

import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import (  # noqa: E402
    EV_OPTIMAL_MEAN_CHIPS,
    RANDOM_MEAN_CHIPS,
    apply_style,
    ensure_figures_dir,
)


SCALES = [
    ("results/mc_training_log_50k.csv", "MC, 50k episodes", "#9ecae1"),
    ("results/mc_training_log_500k.csv", "MC, 500k episodes", "#4292c6"),
    ("results/mc_training_log_5M.csv", "MC, 5M episodes", "#08519c"),
]


def _read_log(path: str) -> Tuple[List[int], List[float]]:
    episodes: List[int] = []
    rolling: List[float] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = float(row["rolling_mean_chips"])
            if math.isnan(r):
                continue
            episodes.append(int(row["episode"]))
            rolling.append(r)
    return episodes, rolling


def main() -> None:
    apply_style()
    fig, ax = plt.subplots()

    for path, label, color in SCALES:
        ep, rm = _read_log(path)
        ax.plot(ep, rm, label=label, color=color, linewidth=1.5)

    ax.axhline(
        EV_OPTIMAL_MEAN_CHIPS,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=f"EV-Optimal Ceiling ({EV_OPTIMAL_MEAN_CHIPS:+.4f})",
    )
    ax.axhline(
        RANDOM_MEAN_CHIPS,
        linestyle="--",
        color="#bcbd22",
        linewidth=1.0,
        label=f"Random Play ({RANDOM_MEAN_CHIPS:+.4f})",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Training episodes (log scale)")
    ax.set_ylabel("Rolling mean net chips per hand")
    ax.set_title(
        "Monte Carlo Training Convergence at Three Episode Scales"
    )
    ax.legend(loc="lower right", framealpha=0.95)

    out = ensure_figures_dir() / "fig1_mc_convergence.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
