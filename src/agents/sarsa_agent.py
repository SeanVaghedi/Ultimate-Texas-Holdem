"""Tabular SARSA agent for UTH RL."""

import random
from typing import Dict, List, Tuple

from abstraction import flop_bucket, preflop_bucket, river_bucket
from qtable import (
    ACTIONS_BY_STAGE,
    QTable,
    action_index_to_name,
    action_name_to_index,
)


class SarsaAgent:
    """Epsilon-greedy tabular SARSA agent.

    Records every (stage, state_bucket, action_index) decision during a
    hand. After the hand resolves, applies the SARSA on-policy TD update
    to each visited (state, action) using

        Q(s, a) <- Q(s, a) + alpha * (target - Q(s, a))

    where ``target = net_chips`` for the trajectory's last (terminal)
    entry and ``target = gamma * Q(s', a')`` otherwise — a' is the
    action the agent actually took next under its current epsilon-greedy
    policy, NOT the max over actions.
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
        """Apply chronological SARSA updates to the trajectory.

        Each non-terminal step's bootstrap target is gamma * Q(s', a')
        where a' is the next action actually taken — read from the
        next entry in the trajectory, not the max over actions. The
        last entry uses net_chips as the terminal target.
        """
        n = len(self._current_trajectory)
        for i, (stage_i, state_i, action_i) in enumerate(
            self._current_trajectory
        ):
            if i == n - 1:
                target = net_chips
            else:
                stage_next, state_next, action_next = (
                    self._current_trajectory[i + 1]
                )
                next_q = self._q_table.get_q(
                    stage_next, state_next, action_next
                )
                target = self._gamma * next_q
            self._q_table.update_q(
                stage_i, state_i, action_i, target, self._alpha
            )

    def get_trajectory(self) -> List[Tuple[str, int, int]]:
        """Return a copy of the trajectory recorded so far this hand."""
        return list(self._current_trajectory)
