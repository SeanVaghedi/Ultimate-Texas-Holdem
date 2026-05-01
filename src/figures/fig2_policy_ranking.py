"""Fig 2: final house edge by policy, sorted best-to-worst."""

import json
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


JSON_PATH = ROOT / "results" / "baseline_evaluation.json"

# (json key, display label, color key)
POLICY_DISPLAY = [
    ("optimal", "EV-Optimal", "optimal"),
    ("naive", "Naive Heuristic", "naive"),
    ("random", "Random", "random"),
    ("always_fold", "Always-Fold", "always_fold"),
    ("mc_trained_5M", "MC tabular (5M)", "mc"),
    ("mc_linear_fa_5M", "MC + Linear FA (5M)", "linear_fa"),
    ("sarsa_trained_5M", "SARSA (5M)", "sarsa"),
    ("sarsa_lambda_5M_lam09", "SARSA(λ=0.9) (5M)", "sarsa_lambda"),
    ("q_trained_5M", "Q-learning (5M)", "q_learning"),
]


def main() -> None:
    apply_style()
    with open(JSON_PATH) as f:
        data = json.load(f)

    rows = []
    for key, label, color_key in POLICY_DISPLAY:
        if key not in data:
            continue
        rows.append(
            (label, data[key]["house_edge_percent"], COLORS[color_key])
        )
    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    edges = [r[1] for r in rows]
    colors = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = list(range(len(rows)))[::-1]  # best at top
    bars = ax.barh(y_pos, edges, color=colors, edgecolor="black", linewidth=0.5)

    ax.axvline(
        EV_OPTIMAL_HOUSE_EDGE,
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.6,
        label=f"EV-Optimal ({EV_OPTIMAL_HOUSE_EDGE:+.2f}%)",
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("House edge (% per ante)")
    ax.set_title("Final House Edge by Policy (100,000 evaluation hands)")
    ax.legend(loc="center right", framealpha=0.95)

    xmax = max(edges) * 1.18
    ax.set_xlim(0, xmax)
    for bar, edge in zip(bars, edges):
        ax.text(
            edge + xmax * 0.005,
            bar.get_y() + bar.get_height() / 2,
            f"+{edge:.2f}%",
            va="center",
            ha="left",
            fontsize=9,
        )

    out = ensure_figures_dir() / "fig2_policy_ranking.png"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
