"""MC + linear function-approximation training driver for UTH RL.

Run from the repo root:

    python -m src.training.train_mc_linear_fa [--episodes N] [--output-suffix S] [--alpha A]

Or directly:

    python src/training/train_mc_linear_fa.py [--episodes N] [--output-suffix S] [--alpha A]
"""

import argparse
import collections
import csv
import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_linear_fa import MonteCarloLinearFAAgent  # noqa: E402
from features import (  # noqa: E402
    FLOP_FEATURE_SIZE,
    PREFLOP_FEATURE_SIZE,
    RIVER_FEATURE_SIZE,
)
from game import UTHGame  # noqa: E402
from linear_q import LinearQ, STAGES  # noqa: E402

from tqdm import tqdm  # noqa: E402


DEFAULT_NUM_EPISODES = 5_000_000
DEFAULT_ALPHA = 0.01
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
    parser = argparse.ArgumentParser(
        description="MC + linear FA UTH trainer."
    )
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Semi-gradient learning rate (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    num_episodes = args.episodes
    suffix = args.output_suffix
    alpha = args.alpha
    output_weights = f"results/mc_linear_fa{suffix}.npz"
    output_log = f"results/mc_linear_fa_log{suffix}.csv"

    print("=== MC + Linear FA Training Run ===")
    print(f"Episodes:   {num_episodes:,}")
    print(f"Suffix:     {suffix!r}")
    print(f"Seed:       {SEED}")
    print(f"Alpha:      {alpha}")
    print(f"Feature sizes:")
    print(f"  preflop:  {PREFLOP_FEATURE_SIZE}")
    print(f"  flop:     {FLOP_FEATURE_SIZE}")
    print(f"  river:    {RIVER_FEATURE_SIZE}")
    print(f"Weights:    {output_weights}")
    print(f"Log CSV:    {output_log}")

    rng = random.Random(SEED)
    game = UTHGame(seed=SEED)
    linear_q = LinearQ()
    agent = MonteCarloLinearFAAgent(linear_q, rng, alpha=alpha)

    Path(output_weights).parent.mkdir(parents=True, exist_ok=True)
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

    linear_q.save(output_weights)

    final_rmean = (sum(rolling) / len(rolling)) if rolling else math.nan
    print(f"Final rolling mean net chips: {final_rmean:.4f}")
    print("Per-stage updates and weight magnitudes:")
    for stage in STAGES:
        w = linear_q.get_weights(stage)
        n_updates = linear_q.get_update_count(stage)
        max_abs = float(np.max(np.abs(w)))
        mean_abs = float(np.mean(np.abs(w)))
        print(
            f"  {stage:<8} updates={n_updates:>12,}  "
            f"shape={w.shape}  "
            f"max|w|={max_abs:8.4f}  mean|w|={mean_abs:8.4f}"
        )


if __name__ == "__main__":
    main()
