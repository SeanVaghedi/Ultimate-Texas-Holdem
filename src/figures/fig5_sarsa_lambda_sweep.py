"""Fig 5: SARSA(lambda) house edge across the TD-MC bias-variance spectrum."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import (  # noqa: E402
    COLORS,
    EV_OPTIMAL_HOUSE_EDGE,
    apply_style,
    ensure_figures_dir,
)


# (lambda value, display label, house edge %)
# lam=0 is one-step SARSA; lam=1 is mathematically equivalent to MC for
# UTH (3-step episodes, gamma=1, terminal-only reward).
SWEEP = [
    (0.0, "SARSA\n(one-step)", 51.92, COLORS["sarsa"]),
    (0.5, "SARSA(λ=0.5)", 49.37, COLORS["sarsa_lambda"]),
    (0.9, "SARSA(λ=0.9)", 43.62, COLORS["sarsa_lambda"]),
    (0.99, "SARSA(λ=0.99)", 45.91, COLORS["sarsa_lambda"]),
    (1.0, "MC tabular\n(λ=1 effective)", 17.64, COLORS["mc"]),
]


def main() -> None:
    apply_style()
    fig, ax = plt.subplots()

    labels = [s[1] for s in SWEEP]
    edges = [s[2] for s in SWEEP]
    colors = [s[3] for s in SWEEP]
    x = list(range(len(SWEEP)))

    bars = ax.bar(x, edges, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(
        EV_OPTIMAL_HOUSE_EDGE,
        linestyle="--",
        color="black",
        linewidth=1.0,
        label=f"EV-Optimal (+{EV_OPTIMAL_HOUSE_EDGE:.2f}%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlabel("λ (eligibility trace decay)")
    ax.set_ylabel("House edge (% per ante)")
    ax.set_title(
        "Bias-Variance Spectrum: SARSA(λ) Across the TD-MC Range"
    )

    ymax = max(edges) * 1.18
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

    ax.legend(loc="upper right", framealpha=0.95)

    fig.text(
        0.5,
        -0.02,
        "λ=0 is pure one-step bootstrapping; λ=1 is equivalent to Monte Carlo",
        ha="center",
        fontsize=9,
        style="italic",
        color="#444444",
    )

    out = ensure_figures_dir() / "fig5_sarsa_lambda_sweep.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
