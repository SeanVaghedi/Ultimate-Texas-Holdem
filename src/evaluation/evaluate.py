"""Policy evaluation harness for Ultimate Texas Hold'em.

Plays a fixed number of hands against a deterministic UTHGame and
reports aggregate net-chip statistics, action histograms, and the
fold rate.
"""

import math
from typing import Callable, Dict

from tqdm import tqdm

from game import UTHGame


_STAGES = ("preflop", "flop", "river")
_ACTIONS_BY_STAGE: Dict[str, tuple] = {
    "preflop": ("check", "bet_3x", "bet_4x"),
    "flop": ("check", "bet_2x"),
    "river": ("fold", "bet_1x"),
}


def evaluate_policy(
    policy_fn: Callable[[str, Dict], str],
    num_hands: int,
    seed: int = 0,
    show_progress: bool = True,
) -> Dict:
    """Play num_hands hands of UTH under policy_fn and return aggregate stats."""
    game = UTHGame(seed=seed)

    action_counts: Dict[str, Dict[str, int]] = {
        stage: {a: 0 for a in _ACTIONS_BY_STAGE[stage]} for stage in _STAGES
    }
    total = 0.0
    sum_sq = 0.0
    folds = 0

    iterator = range(num_hands)
    if show_progress:
        iterator = tqdm(iterator)

    for _ in iterator:
        result = game.play_hand(policy_fn)
        net = result["net_chips"]
        total += net
        sum_sq += net * net
        for stage, action in result["player_action_history"]:
            action_counts[stage][action] += 1
        if result["final_player_category"] is None:
            folds += 1

    mean = total / num_hands
    variance = sum_sq / num_hands - mean * mean
    std_error = math.sqrt(max(variance, 0.0) / num_hands)
    half_ci = 1.96 * std_error

    return {
        "num_hands": int(num_hands),
        "total_net_chips": float(total),
        "mean_net_chips": float(mean),
        "std_error": float(std_error),
        "ci_95_lower": float(mean - half_ci),
        "ci_95_upper": float(mean + half_ci),
        "house_edge_percent": float(-100.0 * mean),
        "house_edge_ci_lower_percent": float(-100.0 * (mean + half_ci)),
        "house_edge_ci_upper_percent": float(-100.0 * (mean - half_ci)),
        "action_counts": action_counts,
        "fold_rate": float(folds / num_hands),
    }


def print_evaluation_summary(label: str, results: Dict) -> None:
    """Pretty-print an evaluate_policy result dict to stdout."""
    print(f"=== {label} ===")
    print(f"Hands:        {results['num_hands']:,}")
    print(
        f"Mean/hand:    {results['mean_net_chips']:+.4f} "
        f"(SE: {results['std_error']:.4f})"
    )
    he = results["house_edge_percent"]
    he_lo = results["house_edge_ci_lower_percent"]
    he_hi = results["house_edge_ci_upper_percent"]
    print(
        f"House edge:   {he:+.2f}% [{he_lo:+.2f}%, {he_hi:+.2f}%] per ante"
    )
    print(f"Fold rate:    {results['fold_rate'] * 100:.1f}%")
    print("Action counts:")
    for stage in _STAGES:
        counts = results["action_counts"][stage]
        total = sum(counts.values())
        if total == 0:
            parts = ", ".join(
                f"{a}=0 (0.0%)" for a in _ACTIONS_BY_STAGE[stage]
            )
        else:
            parts = ", ".join(
                f"{a}={counts[a]} ({counts[a] / total * 100:.1f}%)"
                for a in _ACTIONS_BY_STAGE[stage]
            )
        label_field = (stage + ":").ljust(8)
        print(f"  {label_field} {parts}")
