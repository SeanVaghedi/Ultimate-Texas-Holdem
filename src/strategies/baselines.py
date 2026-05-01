"""Baseline UTH policies for evaluation comparisons.

All functions match the UTHGame.policy_fn signature
``(stage: str, state: Dict) -> str``. random_policy is a factory that
binds an rng and returns the closure.
"""

import random
from typing import Callable, Dict

from treys import Card, Evaluator


_ACTIONS_BY_STAGE = {
    "preflop": ("check", "bet_3x", "bet_4x"),
    "flop": ("check", "bet_2x"),
    "river": ("fold", "bet_1x"),
}

_EVALUATOR = Evaluator()


def random_policy(rng: random.Random) -> Callable[[str, Dict], str]:
    """Return a closure that picks uniformly among legal actions per stage."""

    def _policy(stage: str, state: Dict) -> str:
        return rng.choice(_ACTIONS_BY_STAGE[stage])

    return _policy


def always_fold_policy(stage: str, state: Dict) -> str:
    """Check, check, fold — concedes ante + blind on every hand."""
    if stage == "preflop":
        return "check"
    if stage == "flop":
        return "check"
    if stage == "river":
        return "fold"
    raise ValueError(f"Unknown stage: {stage!r}")


def naive_heuristic_policy(stage: str, state: Dict) -> str:
    """Crude rule-based policy: bet on any pair / ace pre, pair-or-better post."""
    hole = state["hole_cards"]
    community = state["community_cards_revealed"]

    if stage == "preflop":
        r1 = Card.get_rank_int(hole[0]) + 2  # 2..14, Ace=14
        r2 = Card.get_rank_int(hole[1]) + 2
        if r1 == r2 or r1 == 14 or r2 == 14:
            return "bet_4x"
        return "check"

    if stage == "flop":
        cls = _EVALUATOR.get_rank_class(_EVALUATOR.evaluate(community, hole))
        return "bet_2x" if cls <= 8 else "check"

    if stage == "river":
        cls = _EVALUATOR.get_rank_class(_EVALUATOR.evaluate(community, hole))
        return "bet_1x" if cls <= 8 else "fold"

    raise ValueError(f"Unknown stage: {stage!r}")
