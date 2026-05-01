"""Fig 6: pre-flop policy heatmap — trained MC vs. EV-Optimal."""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import apply_style, ensure_figures_dir  # noqa: E402
from qtable import ACTIONS_BY_STAGE, NUM_STATES_BY_STAGE, QTable  # noqa: E402


QTABLE_PATH = ROOT / "results" / "mc_qtable_5M.npz"
OPTIMAL_DIST_PATH = ROOT / "results" / "optimal_action_dist.json"

BUCKET_LABELS = [
    "Premium pairs (AA/KK/QQ)",
    "Mid pairs (JJ/TT/99/88)",
    "Low pairs (77-22)",
    "Suited ace-broadway (AKs-ATs)",
    "Suited ace-low (A9s-A2s)",
    "Offsuit ace-broadway (AKo-ATo)",
    "Offsuit ace-low (A9o-A2o)",
    "Suited Kx high (KQs/KJs/KTs)",
    "Suited K low / offsuit K high",
    "Offsuit K low (K9o-K2o)",
    "Suited QJ/QT/JT",
    "Offsuit QJ/QT/JT",
    "Suited connectors mid (T9s-53s)",
    "Other suited",
    "Other offsuit",
]


def main() -> None:
    apply_style()

    actions = ACTIONS_BY_STAGE["preflop"]
    n_buckets = NUM_STATES_BY_STAGE["preflop"]

    qt = QTable()
    qt.load(str(QTABLE_PATH))
    q_grid = np.zeros((n_buckets, len(actions)), dtype=np.float64)
    for s in range(n_buckets):
        for a in range(len(actions)):
            q_grid[s, a] = qt.get_q("preflop", s, a)

    with open(OPTIMAL_DIST_PATH) as f:
        opt_raw = json.load(f)
    opt_grid = np.zeros((n_buckets, len(actions)), dtype=np.float64)
    for s in range(n_buckets):
        probs = opt_raw["preflop"][str(s)]
        for a_idx, a_name in enumerate(actions):
            opt_grid[s, a_idx] = probs.get(a_name, 0.0) * 100.0

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))

    q_abs_max = float(np.max(np.abs(q_grid))) or 1.0
    im_left = axes[0].imshow(
        q_grid,
        cmap="coolwarm",
        vmin=-q_abs_max,
        vmax=q_abs_max,
        aspect="auto",
    )
    axes[0].set_title("Trained MC Agent — Q-values")
    axes[0].set_xticks(range(len(actions)))
    axes[0].set_xticklabels(actions)
    axes[0].set_yticks(range(n_buckets))
    axes[0].set_yticklabels(BUCKET_LABELS)
    axes[0].grid(False)
    for s in range(n_buckets):
        for a in range(len(actions)):
            v = q_grid[s, a]
            text_color = "white" if abs(v) > 0.6 * q_abs_max else "black"
            axes[0].text(
                a, s, f"{v:+.2f}",
                ha="center", va="center", fontsize=8, color=text_color,
            )
    cbar1 = fig.colorbar(im_left, ax=axes[0], shrink=0.85)
    cbar1.set_label("Q-value (net chips)")

    im_right = axes[1].imshow(
        opt_grid, cmap="viridis", vmin=0, vmax=100, aspect="auto"
    )
    axes[1].set_title("EV-Optimal — Action Probability (%)")
    axes[1].set_xticks(range(len(actions)))
    axes[1].set_xticklabels(actions)
    axes[1].set_yticks(range(n_buckets))
    axes[1].set_yticklabels([])  # share row labels with left
    axes[1].grid(False)
    for s in range(n_buckets):
        for a in range(len(actions)):
            v = opt_grid[s, a]
            text_color = "white" if v < 50 else "black"
            axes[1].text(
                a, s, f"{v:.0f}%",
                ha="center", va="center", fontsize=8, color=text_color,
            )
    cbar2 = fig.colorbar(im_right, ax=axes[1], shrink=0.85)
    cbar2.set_label("Probability (%)")

    fig.suptitle(
        "Pre-flop Policy: Trained MC Agent vs. EV-Optimal",
        fontsize=14, y=1.02,
    )
    fig.tight_layout()

    out = ensure_figures_dir() / "fig6_policy_heatmap.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
