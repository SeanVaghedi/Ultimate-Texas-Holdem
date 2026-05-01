"""Table A2: full 9-policy comparison sourced from baseline_evaluation.json."""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


JSON_PATH = ROOT / "results" / "baseline_evaluation.json"
TABLES_DIR = ROOT / "results" / "tables"
OUT_PATH = TABLES_DIR / "table_a2_policy_comparison.md"

# (json_key, display label) — order rendered in the table
POLICY_ORDER: List[Tuple[str, str]] = [
    ("random", "Random"),
    ("always_fold", "Always-Fold"),
    ("naive", "Naive Heuristic"),
    ("optimal", "EV-Optimal"),
    ("mc_trained_5M", "MC tabular (5M)"),
    ("q_trained_5M", "Q-learning (5M)"),
    ("sarsa_trained_5M", "SARSA (5M)"),
    ("sarsa_lambda_5M_lam09", "SARSA(λ=0.9) (5M)"),
    ("mc_linear_fa_5M", "MC + Linear FA (5M)"),
]


def _pct_dist(counts: Dict[str, int], action_order: List[str]) -> str:
    """Format an action-distribution dict as 'name=xx.x% / ...' in a fixed order."""
    total = sum(counts.values())
    if total == 0:
        return "—"
    parts = []
    for a in action_order:
        v = counts.get(a, 0)
        parts.append(f"{a}={v / total * 100:.1f}%")
    return " / ".join(parts)


def _row(label: str, entry: Dict) -> str:
    mean = entry["mean_net_chips"]
    se = entry["std_error"]
    he = entry["house_edge_percent"]
    he_lo = entry["house_edge_ci_lower_percent"]
    he_hi = entry["house_edge_ci_upper_percent"]
    fr = entry["fold_rate"] * 100
    ac = entry["action_counts"]

    preflop = _pct_dist(ac["preflop"], ["check", "bet_3x", "bet_4x"])
    flop = _pct_dist(ac["flop"], ["check", "bet_2x"])
    river = _pct_dist(ac["river"], ["fold", "bet_1x"])

    cells = [
        label,
        f"{mean:+.4f}",
        f"{se:.4f}",
        f"{he:+.2f}%",
        f"[{he_lo:+.2f}%, {he_hi:+.2f}%]",
        f"{fr:.1f}%",
        preflop,
        flop,
        river,
    ]
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    with open(JSON_PATH) as f:
        data = json.load(f)

    headers = [
        "Policy",
        "Mean / hand",
        "SE",
        "House edge",
        "95% CI on house edge",
        "Fold rate",
        "Pre-flop dist",
        "Flop dist",
        "River dist",
    ]
    align = ["---"] * 2 + ["---:", "---:", "---", "---:", "---", "---", "---"]

    out: List[str] = []
    out.append("# Table A2 — Policy comparison (100,000 evaluation hands per policy)")
    out.append("")
    out.append(
        f"Source: `{JSON_PATH.relative_to(ROOT).as_posix()}`. "
        "All trained agents were evaluated greedy (epsilon=0)."
    )
    out.append("")
    out.append("| " + " | ".join(headers) + " |")
    out.append("| " + " | ".join(align) + " |")

    for key, label in POLICY_ORDER:
        if key not in data:
            continue
        out.append(_row(label, data[key]))

    out.append("")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
