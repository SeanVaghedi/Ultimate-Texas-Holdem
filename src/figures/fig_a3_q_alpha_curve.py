"""Fig A3: Q-learning house edge vs. learning rate (1M episodes per alpha)."""

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


# (alpha, house edge %, CI lower %, CI upper %)
# Source: src/evaluation/eval_one_qtable.py outputs from the alpha sweep
# at 1M episodes per alpha.
SWEEP = [
    (0.01, 17.30, 14.08, 20.51),
    (0.05, 42.24, 39.38, 45.09),
    (0.10, 46.82, 44.21, 49.43),
    (0.30, 59.39, 56.92, 61.86),
]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots()

    alphas = [a for a, _, _, _ in SWEEP]
    edges = [e for _, e, _, _ in SWEEP]
    yerr_lower = [e - lo for _, e, lo, _ in SWEEP]
    yerr_upper = [hi - e for _, e, _, hi in SWEEP]

    ax.errorbar(
        alphas,
        edges,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        color=COLORS["q_learning"],
        ecolor=COLORS["q_learning"],
        elinewidth=1.5,
        capsize=4,
        markersize=8,
        linewidth=2.0,
        label="Q-learning",
    )

    ax.axhline(
        MC_TABULAR_HOUSE_EDGE,
        linestyle="--",
        color=COLORS["mc"],
        linewidth=1.2,
        label=f"MC reference (+{MC_TABULAR_HOUSE_EDGE:.2f}%)",
    )

    for a, e, _, hi in SWEEP:
        ax.text(
            a, hi + 1.5,
            f"+{e:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color=COLORS["q_learning"],
        )

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate α (log scale)")
    ax.set_ylabel("House edge (% per ante)")
    ax.set_title(
        "Q-Learning House Edge vs. Learning Rate (1M episodes per α)"
    )
    ax.legend(loc="lower right", framealpha=0.95)

    ax.set_ylim(0, max(hi for _, _, _, hi in SWEEP) * 1.18)
    ax.set_xticks(alphas)
    ax.set_xticklabels([f"{a:g}" for a in alphas])

    out = ensure_figures_dir() / "fig_a3_q_alpha_curve.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
