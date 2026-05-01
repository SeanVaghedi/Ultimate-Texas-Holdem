"""Fig A2: action choice by (stage, bucket) for MC, SARSA, Q-learning, EV-Optimal.

30-row grid (15 preflop + 8 flop + 7 river) x 4 method columns. Cell color
encodes the chosen action; cells where a trained method disagrees with
EV-Optimal are outlined in red.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from figures.style import apply_style, ensure_figures_dir  # noqa: E402
from qtable import ACTIONS_BY_STAGE, NUM_STATES_BY_STAGE, QTable  # noqa: E402


METHOD_COLUMNS = [
    ("MC (5M)", "results/mc_qtable_5M.npz"),
    ("SARSA (5M)", "results/sarsa_qtable_5M.npz"),
    ("Q-learning (5M)", "results/q_qtable_5M.npz"),
    ("EV-Optimal", None),  # derived from optimal_action_dist.json
]
OPTIMAL_DIST_PATH = ROOT / "results" / "optimal_action_dist.json"

BUCKET_DESCRIPTIONS: Dict[str, Dict[int, str]] = {
    "preflop": {
        0: "Premium pairs (AA/KK/QQ)",
        1: "Mid pairs (JJ/TT/99/88)",
        2: "Low pairs (77-22)",
        3: "Suited ace-broadway (AKs-ATs)",
        4: "Suited ace-low (A9s-A2s)",
        5: "Offsuit ace-broadway (AKo-ATo)",
        6: "Offsuit ace-low (A9o-A2o)",
        7: "Suited Kx high (KQs/KJs/KTs)",
        8: "Suited K low / offsuit K high",
        9: "Offsuit K low (K9o-K2o)",
        10: "Suited QJ/QT/JT",
        11: "Offsuit QJ/QT/JT",
        12: "Suited connectors mid (T9s-53s)",
        13: "Other suited",
        14: "Other offsuit",
    },
    "flop": {
        0: "Straight or better",
        1: "Three of a kind",
        2: "Two pair",
        3: "Top pair (uses hole, >= board)",
        4: "Other one pair",
        5: "4-flush or OESD (no pair)",
        6: "Gutshot or high 3-flush",
        7: "High-card air",
    },
    "river": {
        0: "Royal / straight flush",
        1: "Quads or full house",
        2: "Flush or straight",
        3: "Trips or two pair",
        4: "Top pair (uses hole, >= board)",
        5: "Other one pair",
        6: "High card",
    },
}

# (color, initial) per (stage, action_name)
ACTION_STYLE: Dict[Tuple[str, str], Tuple[str, str]] = {
    ("preflop", "check"):  ("#ffffff", "C"),
    ("preflop", "bet_3x"): ("#9ecae1", "3"),
    ("preflop", "bet_4x"): ("#08519c", "4"),
    ("flop",    "check"):  ("#ffffff", "C"),
    ("flop",    "bet_2x"): ("#08519c", "2"),
    ("river",   "fold"):   ("#fcdbe6", "F"),
    ("river",   "bet_1x"): ("#08519c", "1"),
}

STAGE_ORDER = ("preflop", "flop", "river")
STAGE_GAP = 1  # blank rows between stages
DISAGREE_COLOR = "#d62728"


def _load_qtable_actions(path: str) -> Dict[str, List[str]]:
    """Return {stage: [chosen_action_name per bucket]} via argmax of Q-values."""
    qt = QTable()
    qt.load(path)
    out: Dict[str, List[str]] = {}
    for stage in STAGE_ORDER:
        actions = ACTIONS_BY_STAGE[stage]
        chosen = []
        for s in range(NUM_STATES_BY_STAGE[stage]):
            qs = [qt.get_q(stage, s, a) for a in range(len(actions))]
            chosen.append(actions[int(np.argmax(qs))])
        out[stage] = chosen
    return out


def _load_optimal_actions() -> Dict[str, List[str]]:
    """Return {stage: [chosen_action_name per bucket]} via argmax of probabilities."""
    with open(OPTIMAL_DIST_PATH) as f:
        raw = json.load(f)
    out: Dict[str, List[str]] = {}
    for stage in STAGE_ORDER:
        actions = ACTIONS_BY_STAGE[stage]
        chosen = []
        for s in range(NUM_STATES_BY_STAGE[stage]):
            probs = raw[stage][str(s)]
            best = max(actions, key=lambda a: probs.get(a, 0.0))
            chosen.append(best)
        out[stage] = chosen
    return out


def _row_layout() -> List[Tuple[str, int, int]]:
    """Return [(stage, bucket_idx, y_position)] from top to bottom with stage gaps."""
    rows: List[Tuple[str, int, int]] = []
    y = 0
    for s_i, stage in enumerate(STAGE_ORDER):
        n = NUM_STATES_BY_STAGE[stage]
        for b in range(n):
            rows.append((stage, b, y))
            y += 1
        if s_i < len(STAGE_ORDER) - 1:
            y += STAGE_GAP
    return rows


def main() -> None:
    apply_style()

    method_actions: Dict[str, Dict[str, List[str]]] = {}
    for label, path in METHOD_COLUMNS:
        if path is None:
            method_actions[label] = _load_optimal_actions()
        else:
            method_actions[label] = _load_qtable_actions(path)

    rows = _row_layout()
    n_methods = len(METHOD_COLUMNS)
    method_labels = [m[0] for m in METHOD_COLUMNS]
    optimal_label = method_labels[-1]

    total_y = max(y for _, _, y in rows) + 1

    fig, ax = plt.subplots(figsize=(11, 13))

    for stage, b, y in rows:
        # y_top is the visual y of the top of the cell. We want row 0 at the
        # top, so flip: visual_y = (total_y - 1 - y).
        visual_y = total_y - 1 - y

        opt_action = method_actions[optimal_label][stage][b]

        for col, label in enumerate(method_labels):
            action = method_actions[label][stage][b]
            color, initial = ACTION_STYLE[(stage, action)]

            rect = Rectangle(
                (col, visual_y),
                1,
                1,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
            )
            ax.add_patch(rect)

            # Disagreement border (skip for the optimal column itself).
            if label != optimal_label and action != opt_action:
                border = Rectangle(
                    (col + 0.05, visual_y + 0.05),
                    0.90,
                    0.90,
                    facecolor="none",
                    edgecolor=DISAGREE_COLOR,
                    linewidth=2.0,
                )
                ax.add_patch(border)

            # Label initial. Pick contrasting text color against fill.
            text_color = "white" if color == "#08519c" else "black"
            ax.text(
                col + 0.5,
                visual_y + 0.5,
                initial,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
                color=text_color,
            )

    # Y tick labels for non-gap rows.
    yticks = []
    yticklabels = []
    for stage, b, y in rows:
        visual_y = total_y - 1 - y
        yticks.append(visual_y + 0.5)
        yticklabels.append(BUCKET_DESCRIPTIONS[stage][b])

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)

    ax.set_xticks([c + 0.5 for c in range(n_methods)])
    ax.set_xticklabels(method_labels, fontsize=11, fontweight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")
    ax.tick_params(axis="x", which="both", length=0, pad=8)
    ax.tick_params(axis="y", which="both", length=0)

    ax.set_xlim(0, n_methods)
    ax.set_ylim(0, total_y)
    ax.set_aspect("equal")
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Stage band labels in the right margin so they don't collide with the
    # bucket descriptions on the left.
    for stage in STAGE_ORDER:
        ys = [total_y - 1 - y for s, _, y in rows if s == stage]
        ycenter = (min(ys) + max(ys)) / 2 + 0.5
        ax.text(
            n_methods + 0.25,
            ycenter,
            stage.upper(),
            ha="left",
            va="center",
            fontsize=11,
            fontweight="bold",
            rotation=90,
        )

    # Build a compact legend describing colors + disagreement marker.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_handles = [
        Patch(facecolor="#ffffff", edgecolor="black", label="check"),
        Patch(facecolor="#9ecae1", edgecolor="black", label="bet_3x"),
        Patch(facecolor="#08519c", edgecolor="black", label="bet_4x / bet_2x / bet_1x"),
        Patch(facecolor="#fcdbe6", edgecolor="black", label="fold"),
        Line2D(
            [0], [0],
            marker="s", linestyle="",
            markerfacecolor="white",
            markeredgecolor=DISAGREE_COLOR,
            markeredgewidth=2.0, markersize=10,
            label="disagrees with EV-Optimal",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=3,
        frameon=False,
        fontsize=9,
    )

    ax.set_title(
        "Action Choice by Method: 30 (Stage, Bucket) States × 4 Methods "
        "(5M Training Episodes)",
        pad=42,
        fontsize=12,
    )

    out = ensure_figures_dir() / "fig_a2_method_disagreements.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
