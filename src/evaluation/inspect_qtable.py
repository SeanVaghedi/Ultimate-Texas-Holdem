"""Inspect the trained MC Q-table and compare against the EV-optimal baseline.

Run with:

    python -m src.evaluation.inspect_qtable [--qtable PATH]

Loads the given Q-table (default: results/mc_qtable_500k.npz), samples
optimal_policy's per-bucket action distribution (cached to
results/optimal_action_dist.json), and prints three formatted sections
plus a disagreement summary.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
from treys import Card  # noqa: E402

from abstraction import flop_bucket, preflop_bucket, river_bucket  # noqa: E402
from qtable import (  # noqa: E402
    ACTIONS_BY_STAGE,
    NUM_STATES_BY_STAGE,
    QTable,
    STAGES,
)
from strategies.optimal import optimal_policy  # noqa: E402


DEFAULT_QTABLE_PATH = "results/mc_qtable_500k.npz"
CACHE_PATH = "results/optimal_action_dist.json"
SAMPLING_SEED = 12345
SAMPLES_PER_BUCKET = 1000
MAX_HANDS_PER_STAGE = 100_000
DOMINANT_THRESHOLD = 0.80


_BUCKET_DESCRIPTIONS: Dict[str, Dict[int, str]] = {
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


_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_DECK_TEMPLATE = [Card.new(r + s) for r in _RANKS for s in _SUITS]


def _bucket_for_stage(stage: str, hole: List[int], community: List[int]) -> int:
    if stage == "preflop":
        return preflop_bucket(hole)
    if stage == "flop":
        return flop_bucket(hole, community[:3])
    return river_bucket(hole, community)


def _state_for_stage(
    stage: str, hole: List[int], community: List[int]
) -> Dict:
    if stage == "preflop":
        revealed: List[int] = []
    elif stage == "flop":
        revealed = community[:3]
    else:
        revealed = community
    return {
        "hole_cards": hole,
        "community_cards_revealed": revealed,
        "actions_taken_so_far": [],
    }


def sample_optimal_action_dist() -> Dict:
    """Build optimal_policy's action distribution per (stage, bucket).

    Forced-check semantics: each stage is queried independently with a
    freshly dealt hand, so optimal sees every bucket regardless of what
    it would have done at earlier stages.
    """
    rng = random.Random(SAMPLING_SEED)
    counts = {
        stage: {
            b: {a: 0 for a in ACTIONS_BY_STAGE[stage]}
            for b in range(NUM_STATES_BY_STAGE[stage])
        }
        for stage in STAGES
    }

    for stage in STAGES:
        n_buckets = NUM_STATES_BY_STAGE[stage]
        bucket_totals = [0] * n_buckets
        hands_dealt = 0
        while hands_dealt < MAX_HANDS_PER_STAGE:
            if all(t >= SAMPLES_PER_BUCKET for t in bucket_totals):
                break
            deck = list(_DECK_TEMPLATE)
            rng.shuffle(deck)
            hole = [deck.pop(), deck.pop()]
            community = [deck.pop() for _ in range(5)]
            hands_dealt += 1

            b = _bucket_for_stage(stage, hole, community)
            if bucket_totals[b] >= SAMPLES_PER_BUCKET:
                continue
            state = _state_for_stage(stage, hole, community)
            action = optimal_policy(stage, state)
            counts[stage][b][action] += 1
            bucket_totals[b] += 1

    dist: Dict = {}
    for stage in STAGES:
        dist[stage] = {}
        for b in range(NUM_STATES_BY_STAGE[stage]):
            total = sum(counts[stage][b].values())
            if total > 0:
                dist[stage][b] = {
                    a: c / total for a, c in counts[stage][b].items()
                }
            else:
                dist[stage][b] = {a: 0.0 for a in ACTIONS_BY_STAGE[stage]}
    return dist


def load_or_compute_optimal_dist() -> Dict:
    """Load the cached optimal action distribution, or sample it once."""
    cache = Path(CACHE_PATH)
    if cache.exists():
        with open(cache) as f:
            raw = json.load(f)
        return {
            stage: {int(b): probs for b, probs in buckets.items()}
            for stage, buckets in raw.items()
        }

    print("Sampling optimal policy action distribution (one-time, cached)...")
    dist = sample_optimal_action_dist()
    cache.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        stage: {str(b): probs for b, probs in buckets.items()}
        for stage, buckets in dist.items()
    }
    with open(cache, "w") as f:
        json.dump(serializable, f, indent=2)
    return dist


def _dominant_action(probs: Dict[str, float]) -> str:
    for a, p in probs.items():
        if p >= DOMINANT_THRESHOLD:
            return a
    return "MIXED"


def _stage_title(stage: str) -> str:
    return {"preflop": "Pre-flop", "flop": "Flop", "river": "River"}[stage]


def print_qtable_section(stage: str, q_table: QTable) -> None:
    """Print the trained agent's Q-values and best action per bucket."""
    actions = ACTIONS_BY_STAGE[stage]
    descs = _BUCKET_DESCRIPTIONS[stage]
    print(f"=== {_stage_title(stage)} Q-values (trained MC agent) ===")

    action_header = " | ".join(f"{a:>7}" for a in actions)
    print(
        f"Bucket | {'Description':<32} | {action_header} | "
        f"{'Best':<8} | Visits"
    )
    sep_actions = "-+-".join(["-" * 7] * len(actions))
    print(
        f"{'-' * 6}-+-{'-' * 32}-+-{sep_actions}-+-{'-' * 8}-+-{'-' * 8}"
    )

    for b in range(NUM_STATES_BY_STAGE[stage]):
        qs = [q_table.get_q(stage, b, a) for a in range(len(actions))]
        visits = sum(
            q_table.get_visits(stage, b, a) for a in range(len(actions))
        )
        best_idx = int(np.argmax(qs))
        best_name = actions[best_idx]
        q_str = " | ".join(f"{q:+7.2f}" for q in qs)
        print(
            f"  {b:>4} | {descs[b]:<32} | {q_str} | "
            f"{best_name:<8} | {visits:>8,}"
        )
    print()


def print_optimal_section(stage: str, optimal_dist: Dict) -> None:
    """Print optimal_policy's action distribution per bucket."""
    actions = ACTIONS_BY_STAGE[stage]
    descs = _BUCKET_DESCRIPTIONS[stage]
    print(f"=== {_stage_title(stage)} Optimal Action Distribution ===")

    action_header = " | ".join(f"{a:>7}" for a in actions)
    print(
        f"Bucket | {'Description':<32} | {action_header} | "
        f"Optimal pick (>= 80%)"
    )
    sep_actions = "-+-".join(["-" * 7] * len(actions))
    print(
        f"{'-' * 6}-+-{'-' * 32}-+-{sep_actions}-+-{'-' * 22}"
    )

    for b in range(NUM_STATES_BY_STAGE[stage]):
        probs = optimal_dist[stage][b]
        pct_str = " | ".join(
            f"{probs[a] * 100:>6.1f}%" for a in actions
        )
        pick = _dominant_action(probs)
        print(f"  {b:>4} | {descs[b]:<32} | {pct_str} | {pick}")
    print()


def find_disagreements(q_table: QTable, optimal_dist: Dict) -> List[Dict]:
    """Buckets where agent's argmax differs from optimal's dominant action."""
    rows: List[Dict] = []
    for stage in STAGES:
        actions = ACTIONS_BY_STAGE[stage]
        for b in range(NUM_STATES_BY_STAGE[stage]):
            qs = [q_table.get_q(stage, b, a) for a in range(len(actions))]
            visits = sum(
                q_table.get_visits(stage, b, a)
                for a in range(len(actions))
            )
            agent_idx = int(np.argmax(qs))
            agent_name = actions[agent_idx]
            optimal_pick = _dominant_action(optimal_dist[stage][b])
            if optimal_pick == "MIXED" or agent_name == optimal_pick:
                continue
            optimal_idx = actions.index(optimal_pick)
            margin = qs[agent_idx] - qs[optimal_idx]
            rows.append(
                {
                    "stage": stage,
                    "bucket": b,
                    "description": _BUCKET_DESCRIPTIONS[stage][b],
                    "agent": agent_name,
                    "optimal": optimal_pick,
                    "visits": visits,
                    "margin": margin,
                }
            )
    return rows


def print_disagreements(rows: List[Dict]) -> None:
    """Print the disagreement table and a per-stage tally."""
    print("=== Disagreements with Optimal ===")
    if not rows:
        print("(none — agent matches optimal on every dominant bucket)")
        print()
        return

    print(
        f"{'Stage':<8} | {'Bucket':>6} | {'Description':<32} | "
        f"{'Agent':<8} | {'Optimal':<8} | {'Visits':>8} | {'Q-margin':>9}"
    )
    print(
        f"{'-' * 8}-+-{'-' * 6}-+-{'-' * 32}-+-{'-' * 8}-+-{'-' * 8}-+-"
        f"{'-' * 8}-+-{'-' * 9}"
    )
    for r in rows:
        print(
            f"{r['stage']:<8} | {r['bucket']:>6} | "
            f"{r['description']:<32} | "
            f"{r['agent']:<8} | {r['optimal']:<8} | "
            f"{r['visits']:>8,} | {r['margin']:+9.2f}"
        )
    print()
    by_stage = {s: 0 for s in STAGES}
    for r in rows:
        by_stage[r["stage"]] += 1
    summary = ", ".join(f"{s}: {n}" for s, n in by_stage.items())
    print(f"Total disagreements: {len(rows)}  ({summary})")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect a trained MC Q-table.")
    parser.add_argument(
        "--qtable",
        type=str,
        default=DEFAULT_QTABLE_PATH,
        help="Path to the .npz Q-table to inspect (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    print(f"Inspecting Q-table: {args.qtable}\n")
    q_table = QTable()
    q_table.load(args.qtable)
    optimal_dist = load_or_compute_optimal_dist()

    for stage in STAGES:
        print_qtable_section(stage, q_table)
    for stage in STAGES:
        print_optimal_section(stage, optimal_dist)

    rows = find_disagreements(q_table, optimal_dist)
    print_disagreements(rows)


if __name__ == "__main__":
    main()
