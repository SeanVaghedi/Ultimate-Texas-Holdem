"""Monte Carlo training driver for tabular UTH RL.

Run from the repo root:

    python -m src.training.train_mc [--episodes N] [--output-suffix S]

Or directly:

    python src/training/train_mc.py [--episodes N] [--output-suffix S]
"""

import argparse
import collections
import csv
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_agent import MonteCarloAgent  # noqa: E402
from game import UTHGame  # noqa: E402
from qtable import (  # noqa: E402
    ACTIONS_BY_STAGE,
    NUM_STATES_BY_STAGE,
    QTable,
    STAGES,
)

from tqdm import tqdm  # noqa: E402


DEFAULT_NUM_EPISODES = 500_000
EPSILON_START = 1.0
EPSILON_MIN = 0.05
EPSILON_DECAY_FRACTION = 0.8
LOG_INTERVAL = 5_000
ROLLING_WINDOW = 10_000
SEED = 42


def _epsilon_at(episode: int, num_episodes: int) -> float:
    """Linear decay from EPSILON_START to EPSILON_MIN over the first decay window."""
    decay_end = max(1, int(EPSILON_DECAY_FRACTION * num_episodes))
    if episode >= decay_end:
        return EPSILON_MIN
    frac = episode / decay_end
    return EPSILON_START + (EPSILON_MIN - EPSILON_START) * frac


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monte Carlo UTH trainer.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_NUM_EPISODES,
        help="Number of training episodes (default: %(default)s).",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix appended to output filenames (default: empty).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    num_episodes = args.episodes
    suffix = args.output_suffix
    output_qtable = f"results/mc_qtable{suffix}.npz"
    output_log = f"results/mc_training_log{suffix}.csv"

    print("=== MC Training Run ===")
    print(f"Episodes:   {num_episodes:,}")
    print(f"Suffix:     {suffix!r}")
    print(f"Seed:       {SEED}")
    print(f"Q-table:    {output_qtable}")
    print(f"Log CSV:    {output_log}")

    rng = random.Random(SEED)
    game = UTHGame(seed=SEED)
    q_table = QTable()
    agent = MonteCarloAgent(q_table, rng)

    Path(output_qtable).parent.mkdir(parents=True, exist_ok=True)
    Path(output_log).parent.mkdir(parents=True, exist_ok=True)

    rolling: collections.deque = collections.deque(maxlen=ROLLING_WINDOW)

    with open(output_log, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["episode", "rolling_mean_chips", "epsilon"])

        for ep in tqdm(range(num_episodes)):
            eps = _epsilon_at(ep, num_episodes)
            agent.set_epsilon(eps)
            agent.start_hand()
            result = game.play_hand(agent.policy_fn)
            net = result["net_chips"]
            rolling.append(net)
            agent.learn_from_hand(net)

            if (ep + 1) % LOG_INTERVAL == 0:
                if len(rolling) == ROLLING_WINDOW:
                    rmean = sum(rolling) / len(rolling)
                else:
                    rmean = math.nan
                writer.writerow([ep + 1, rmean, eps])

    q_table.save(output_qtable)

    final_rmean = (sum(rolling) / len(rolling)) if rolling else math.nan

    visited_pairs = 0
    unvisited_pairs = 0
    visit_counts: list = []
    for stage in STAGES:
        n_states = NUM_STATES_BY_STAGE[stage]
        n_actions = len(ACTIONS_BY_STAGE[stage])
        for s in range(n_states):
            for a in range(n_actions):
                v = q_table.get_visits(stage, s, a)
                if v > 0:
                    visited_pairs += 1
                    visit_counts.append(v)
                else:
                    unvisited_pairs += 1

    print(f"Final rolling mean net chips: {final_rmean:.4f}")
    print(f"Visited (state, action) pairs: {visited_pairs}")
    print(f"Unvisited (state, action) pairs: {unvisited_pairs}")
    if visit_counts:
        print(f"Min visits among visited:      {min(visit_counts)}")
        print(f"Max visits among visited:      {max(visit_counts)}")


if __name__ == "__main__":
    main()
