"""Tabular Q-learning agent for UTH RL."""

import random
from typing import Dict, List, Tuple

from abstraction import flop_bucket, preflop_bucket, river_bucket
from qtable import (
    ACTIONS_BY_STAGE,
    QTable,
    action_index_to_name,
    action_name_to_index,
)


class QLearningAgent:
    """Epsilon-greedy tabular Q-learning agent.

    Records every (stage, state_bucket, action_index) decision during a
    hand. After the hand resolves, applies a Q-learning TD update to
    each visited (state, action) using the standard rule

        Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

    where ``target = net_chips`` for the trajectory's last (terminal)
    entry and ``target = gamma * max_a Q(s', a)`` otherwise.
    """

    def __init__(
        self,
        q_table: QTable,
        rng: random.Random,
        alpha: float = 0.1,
        gamma: float = 1.0,
    ) -> None:
        self._q_table = q_table
        self._rng = rng
        self._alpha = alpha
        self._gamma = gamma
        self._current_trajectory: List[Tuple[str, int, int]] = []
        self._epsilon: float = 0.0

    def start_hand(self) -> None:
        """Reset trajectory; call before each hand."""
        self._current_trajectory = []

    def set_epsilon(self, epsilon: float) -> None:
        """Set exploration rate used by subsequent policy_fn calls."""
        self._epsilon = epsilon

    def policy_fn(self, stage: str, state: Dict) -> str:
        """UTHGame.policy_fn-compatible decision callback (records trajectory)."""
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
        """Apply chronological Q-learning updates to the trajectory.

        Each step's bootstrap target is computed against the current
        Q-table values for the next state, using the values that exist
        at the moment of update — so a non-terminal step looks up the
        next state's Q before that next step's own update has been
        applied.
        """
        n = len(self._current_trajectory)
        for i, (stage_i, state_i, action_i) in enumerate(
            self._current_trajectory
        ):
            if i == n - 1:
                target = net_chips
            else:
                stage_next, state_next, _ = self._current_trajectory[i + 1]
                n_next_actions = len(ACTIONS_BY_STAGE[stage_next])
                max_next_q = max(
                    self._q_table.get_q(stage_next, state_next, a)
                    for a in range(n_next_actions)
                )
                target = self._gamma * max_next_q
            self._q_table.update_q(
                stage_i, state_i, action_i, target, self._alpha
            )

    def get_trajectory(self) -> List[Tuple[str, int, int]]:
        """Return a copy of the trajectory recorded so far this hand."""
        return list(self._current_trajectory)
