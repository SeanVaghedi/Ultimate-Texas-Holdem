"""Monte Carlo control with linear function approximation for UTH RL."""

import random
from typing import Dict, List, Tuple

import numpy as np

from features import (
    flop_features,
    preflop_features,
    river_features,
)
from linear_q import LinearQ
from qtable import action_index_to_name


class MonteCarloLinearFAAgent:
    """Epsilon-greedy MC control with linear function approximation.

    During a hand the agent records each (stage, features, action_idx)
    decision in its trajectory. After the hand resolves, it applies the
    same terminal return (net chips) to every visited (stage, features,
    action) via a semi-gradient MC update on the corresponding weight
    column.

    alpha defaults to 0.01 — much smaller than the tabular default
    (0.1) because gradient updates here touch ~600 weights and the
    larger step size becomes unstable.
    """

    def __init__(
        self,
        linear_q: LinearQ,
        rng: random.Random,
        alpha: float = 0.01,
    ) -> None:
        self._q = linear_q
        self._rng = rng
        self._alpha = alpha
        self._current_trajectory: List[Tuple[str, np.ndarray, int]] = []
        self._epsilon: float = 0.0

    def start_hand(self) -> None:
        """Reset the trajectory buffer; call before each new hand."""
        self._current_trajectory = []

    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration rate used by subsequent policy_fn calls."""
        self._epsilon = epsilon

    def _features_for(self, stage: str, state: Dict) -> np.ndarray:
        hole_cards = state["hole_cards"]
        if stage == "preflop":
            return preflop_features(hole_cards)
        if stage == "flop":
            return flop_features(hole_cards, state["community_cards_revealed"])
        if stage == "river":
            return river_features(
                hole_cards, state["community_cards_revealed"]
            )
        raise ValueError(f"Unknown stage: {stage!r}")

    def policy_fn(self, stage: str, state: Dict) -> str:
        """UTHGame.policy_fn-compatible decision callback.

        Computes stage-appropriate features, samples an epsilon-greedy
        action against the current linear Q, records (stage, features,
        action_idx) in the trajectory, and returns the action's name.
        """
        phi = self._features_for(stage, state)
        action_idx = self._q.epsilon_greedy_action(
            stage, phi, self._epsilon, self._rng
        )
        self._current_trajectory.append((stage, phi, action_idx))
        return action_index_to_name(stage, action_idx)

    def frozen_policy_fn(self, stage: str, state: Dict) -> str:
        """Greedy (epsilon=0) policy for evaluation. Does not record trajectory."""
        phi = self._features_for(stage, state)
        action_idx = self._q.best_action(stage, phi)
        return action_index_to_name(stage, action_idx)

    def learn_from_hand(self, net_chips: float) -> None:
        """Apply MC return = net_chips to every visited (state, action) weight column."""
        for stage, phi, action_idx in self._current_trajectory:
            self._q.update_mc(stage, phi, action_idx, net_chips, self._alpha)

    def get_trajectory(self) -> List[Tuple[str, np.ndarray, int]]:
        """Return a copy of the trajectory recorded so far this hand."""
        return list(self._current_trajectory)
