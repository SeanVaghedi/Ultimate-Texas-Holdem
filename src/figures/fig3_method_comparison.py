"""Fig 3: training-curve comparison of MC, SARSA, and Q-learning at 5M episodes.

Each method's raw rolling-mean is plotted as a faint background line,
overlaid with a thick moving-average smoothing (window of 50 log
entries = ~250k episodes since LOG_INTERVAL is 5000).
"""

import csv
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import (  # noqa: E402
    COLORS,
    EV_OPTIMAL_MEAN_CHIPS,
    RANDOM_MEAN_CHIPS,
    apply_style,
    ensure_figures_dir,
)


SERIES = [
    ("results/mc_training_log_5M.csv", "Monte Carlo", COLORS["mc"]),
    ("results/sarsa_training_log_5M.csv", "SARSA", COLORS["sarsa"]),
    ("results/q_training_log_5M.csv", "Q-Learning", COLORS["q_learning"]),
]
SMOOTH_WINDOW = 50  # log entries; LOG_INTERVAL=5000 → ~250k-episode window


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


def smooth(values: List[float], window: int = SMOOTH_WINDOW) -> np.ndarray:
    if len(values) < window:
        return np.array(values)
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def main() -> None:
    apply_style()
    fig, ax = plt.subplots()

    for path, label, color in SERIES:
        ep, rm = _read_log(path)

        # Raw rolling-mean as faint background context.
        ax.plot(ep, rm, color=color, linewidth=0.9, alpha=0.25)

        # Smoothed overlay; align x-positions to the right edge of each window.
        smoothed = smooth(rm, SMOOTH_WINDOW)
        if len(smoothed) == len(rm):
            ep_smoothed = ep
        else:
            ep_smoothed = ep[SMOOTH_WINDOW - 1:]
        ax.plot(
            ep_smoothed,
            smoothed,
            label=label,
            color=color,
            linewidth=2.5,
            alpha=1.0,
        )

    ax.axhline(
        EV_OPTIMAL_MEAN_CHIPS,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=f"EV-Optimal ({EV_OPTIMAL_MEAN_CHIPS:+.4f})",
    )
    ax.axhline(
        RANDOM_MEAN_CHIPS,
        linestyle="--",
        color="#bcbd22",
        linewidth=1.0,
        label=f"Random Play ({RANDOM_MEAN_CHIPS:+.4f})",
    )

    ax.set_xlabel("Training episode")
    ax.set_ylabel("Rolling mean net chips per hand")
    ax.set_title(
        "Training Convergence: Monte Carlo vs. SARSA vs. Q-Learning"
    )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))
    ax.legend(loc="lower right", framealpha=0.95)

    out = ensure_figures_dir() / "fig3_method_comparison.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
