"""Monte Carlo agent for tabular UTH RL."""

import random
from typing import Dict, List, Tuple

from abstraction import flop_bucket, preflop_bucket, river_bucket
from qtable import (
    ACTIONS_BY_STAGE,
    QTable,
    action_index_to_name,
    action_name_to_index,
)


class MonteCarloAgent:
    """Epsilon-greedy Monte Carlo control agent.

    Records every (stage, state_bucket, action_index) decision made
    during a hand. After the hand resolves, applies the same terminal
    return (net chips) to each visited (state, action) pair via the
    Q-table's incremental Monte Carlo update.
    """

    def __init__(self, q_table: QTable, rng: random.Random) -> None:
        self._q_table = q_table
        self._rng = rng
        self._current_trajectory: List[Tuple[str, int, int]] = []
        self._epsilon: float = 0.0

    def start_hand(self) -> None:
        """Reset the trajectory buffer; call before each new hand."""
        self._current_trajectory = []

    def set_epsilon(self, epsilon: float) -> None:
        """Set the exploration rate used by subsequent policy_fn calls."""
        self._epsilon = epsilon

    def policy_fn(self, stage: str, state: Dict) -> str:
        """UTHGame.policy_fn-compatible decision callback.

        Buckets the current state, samples an epsilon-greedy action,
        records the (stage, state_idx, action_idx) triple in the
        trajectory, and returns the action's name string.
        """
        hole_cards = state["hole_cards"]
        if stage == "preflop":
            state_idx = preflop_bucket(hole_cards)
        elif stage == "flop":
            state_idx = flop_bucket(
                hole_cards, state["community_cards_revealed"]
            )
        elif stage == "river":
            state_idx = river_bucket(
                hole_cards, state["community_cards_revealed"]
            )
        else:
            raise ValueError(f"Unknown stage: {stage!r}")

        action_idx = self._q_table.epsilon_greedy_action(
            stage, state_idx, self._epsilon, self._rng
        )
        self._current_trajectory.append((stage, state_idx, action_idx))
        return action_index_to_name(stage, action_idx)

    def frozen_policy_fn(self, stage: str, state: Dict) -> str:
        """Greedy (epsilon=0) policy for evaluation. Does not record trajectory."""
        hole_cards = state["hole_cards"]
        if stage == "preflop":
            state_idx = preflop_bucket(hole_cards)
        elif stage == "flop":
            state_idx = flop_bucket(
                hole_cards, state["community_cards_revealed"]
            )
        elif stage == "river":
            state_idx = river_bucket(
                hole_cards, state["community_cards_revealed"]
            )
        else:
            raise ValueError(f"Unknown stage: {stage!r}")
        action_idx = self._q_table.best_action(stage, state_idx)
        return action_index_to_name(stage, action_idx)

    def learn_from_hand(self, net_chips: float) -> None:
        """Apply MC return = net_chips to every visited (state, action)."""
        for stage, state_idx, action_idx in self._current_trajectory:
            self._q_table.update_q_mc(
                stage, state_idx, action_idx, net_chips
            )

    def get_trajectory(self) -> List[Tuple[str, int, int]]:
        """Return a copy of the trajectory recorded so far this hand."""
        return list(self._current_trajectory)
