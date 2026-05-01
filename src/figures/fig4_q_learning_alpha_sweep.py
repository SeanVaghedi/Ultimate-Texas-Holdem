"""Fig 4: Q-learning house-edge sensitivity to alpha (1M episodes per alpha)."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import (  # noqa: E402
    COLORS,
    MC_TABULAR_HOUSE_EDGE,
    apply_style,
    ensure_figures_dir,
)


# Source: src/evaluation/eval_one_qtable.py outputs from the prior alpha
# sweep on Q-learning at 1M episodes per alpha (results/q_qtable_1M_a*.npz).
ALPHA_SWEEP = [
    (0.01, 17.30),
    (0.05, 42.24),
    (0.10, 46.82),
    (0.30, 59.39),
]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots()

    labels = [f"α={a:g}" for a, _ in ALPHA_SWEEP]
    edges = [edge for _, edge in ALPHA_SWEEP]
    x = list(range(len(ALPHA_SWEEP)))

    bars = ax.bar(
        x, edges, color=COLORS["q_learning"], edgecolor="black", linewidth=0.5
    )

    ax.axhline(
        MC_TABULAR_HOUSE_EDGE,
        linestyle="--",
        color=COLORS["mc"],
        linewidth=1.2,
        label=f"MC reference (+{MC_TABULAR_HOUSE_EDGE:.2f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Learning rate α")
    ax.set_ylabel("House edge (% per ante)")
    ax.set_title(
        "Q-Learning Sensitivity to Learning Rate (1M episodes per α)"
    )

    ymax = max(edges) * 1.15
    ax.set_ylim(0, ymax)
    for bar, edge in zip(bars, edges):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            edge + ymax * 0.01,
            f"+{edge:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax.legend(loc="upper left", framealpha=0.95)

    out = ensure_figures_dir() / "fig4_q_learning_alpha_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
