"""Quick 50k-hand evaluation of a single Q-table file.

Run with:

    python -m src.evaluation.eval_one_qtable --qtable PATH

Loads the .npz at PATH, wraps it in a frozen Q-learning agent, and
prints a one-line house-edge summary with 95% CI. Used by the
learning-rate sweep to compare runs without running the full
run_baselines pipeline each time.
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.q_learning_agent import QLearningAgent  # noqa: E402
from evaluation.evaluate import evaluate_policy  # noqa: E402
from qtable import QTable  # noqa: E402


NUM_HANDS = 50_000
SEED = 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained Q-table for house edge."
    )
    parser.add_argument(
        "--qtable",
        type=str,
        required=True,
        help="Path to .npz Q-table to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    q_table = QTable()
    q_table.load(args.qtable)
    agent = QLearningAgent(q_table, random.Random(SEED))

    results = evaluate_policy(
        agent.frozen_policy_fn, NUM_HANDS, seed=SEED, show_progress=False
    )

    he = results["house_edge_percent"]
    he_lo = results["house_edge_ci_lower_percent"]
    he_hi = results["house_edge_ci_upper_percent"]
    mean = results["mean_net_chips"]
    print(
        f"{args.qtable}: mean={mean:+.4f}  "
        f"house_edge={he:+.2f}%  "
        f"CI95=[{he_lo:+.2f}%, {he_hi:+.2f}%]"
    )


if __name__ == "__main__":
    main()
