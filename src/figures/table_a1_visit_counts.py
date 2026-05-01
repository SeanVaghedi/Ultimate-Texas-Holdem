"""Table A1: per-bucket visit counts for the trained MC Q-table (5M episodes)."""

import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from qtable import ACTIONS_BY_STAGE, NUM_STATES_BY_STAGE, QTable  # noqa: E402


QTABLE_PATH = ROOT / "results" / "mc_qtable_5M.npz"
TABLES_DIR = ROOT / "results" / "tables"
OUT_PATH = TABLES_DIR / "table_a1_visit_counts.md"
LOW_VISIT_THRESHOLD = 100

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

STAGE_TITLE = {
    "preflop": "Pre-flop",
    "flop": "Flop",
    "river": "River",
}


def _section(stage: str, qt: QTable) -> List[str]:
    actions = ACTIONS_BY_STAGE[stage]
    n = NUM_STATES_BY_STAGE[stage]

    header_cells = ["Bucket", "Description"] + list(actions) + ["Total"]
    align_row = ["---", "---"] + ["---:"] * len(actions) + ["---:"]

    lines: List[str] = []
    lines.append(f"## {STAGE_TITLE[stage]} visit counts")
    lines.append("")
    lines.append("| " + " | ".join(header_cells) + " |")
    lines.append("| " + " | ".join(align_row) + " |")

    cell_totals: List[int] = []
    for b in range(n):
        per_action = [qt.get_visits(stage, b, a) for a in range(len(actions))]
        total = sum(per_action)
        cell_totals.extend(per_action)
        per_action_strs = [f"{v:,}" for v in per_action]
        row = (
            [str(b), BUCKET_DESCRIPTIONS[stage][b]]
            + per_action_strs
            + [f"{total:,}"]
        )
        lines.append("| " + " | ".join(row) + " |")

    low = sum(1 for v in cell_totals if v < LOW_VISIT_THRESHOLD)
    lines.append("")
    lines.append(
        f"_Min visits: {min(cell_totals):,}. "
        f"Max visits: {max(cell_totals):,}. "
        f"Cells with <{LOW_VISIT_THRESHOLD} visits: {low}._"
    )
    lines.append("")
    return lines


def main() -> None:
    qt = QTable()
    qt.load(str(QTABLE_PATH))

    out: List[str] = []
    out.append("# Table A1 — Per-bucket visit counts")
    out.append("")
    out.append(
        f"Source: `{QTABLE_PATH.relative_to(ROOT).as_posix()}` "
        "(Monte Carlo, 5M training episodes)."
    )
    out.append("")

    for stage in ("preflop", "flop", "river"):
        out.extend(_section(stage, qt))

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
