"""Tabular SARSA(lambda) agent with accumulating eligibility traces.

At lambda=0 reduces to one-step SARSA. At lambda=1 (with gamma=1, as
in UTH) every visited (state, action) in the episode receives the same
TD error from the terminal step, recovering Monte Carlo. Intermediate
lambda blends the two.

Per-episode update (Sutton & Barto Eq. 12.7-12.11, accumulating
traces):

    For each step i in the trajectory (chronological order):
        delta_i = (r_i + gamma * Q(s_{i+1}, a_{i+1})) - Q(s_i, a_i)
                  (terminal step uses delta = net_chips - Q(s_i, a_i))
        e(s_i, a_i) += 1
        For each (state, action) with non-zero trace:
            Q(state, action) += alpha * delta_i * e(state, action)
        For each trace:
            e(state, action) *= gamma * lam

UTH episodes are <=3 steps and intermediate rewards are 0.
"""

import random
from typing import Dict, List, Tuple

from abstraction import flop_bucket, preflop_bucket, river_bucket
from qtable import (
    QTable,
    action_index_to_name,
)


class SarsaLambdaAgent:
    """Epsilon-greedy tabular SARSA(lambda) with accumulating traces."""

    def __init__(
        self,
        q_table: QTable,
        rng: random.Random,
        alpha: float = 0.1,
        gamma: float = 1.0,
        lam: float = 0.9,
    ) -> None:
        self._q_table = q_table
        self._rng = rng
        self._alpha = alpha
        self._gamma = gamma
        self._lam = lam
        self._trajectory: List[Tuple[str, int, int]] = []
        self._epsilon: float = 0.0

    def start_hand(self) -> None:
        """Reset trajectory; call before each hand."""
        self._trajectory = []

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
        self._trajectory.append((stage, state_idx, action_idx))
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
        """Apply SARSA(lambda) accumulating-trace updates over the trajectory."""
        traj = self._trajectory
        if not traj:
            return

        traces: Dict[Tuple[str, int, int], float] = {}

        for i, (stage, state, action) in enumerate(traj):
            current_q = self._q_table.get_q(stage, state, action)
            if i + 1 < len(traj):
                next_stage, next_state, next_action = traj[i + 1]
                next_q = self._q_table.get_q(
                    next_stage, next_state, next_action
                )
                td_target = self._gamma * next_q
            else:
                td_target = net_chips
            td_error = td_target - current_q

            key = (stage, state, action)
            traces[key] = traces.get(key, 0.0) + 1.0

            for (t_stage, t_state, t_action), trace in traces.items():
                cur = self._q_table.get_q(t_stage, t_state, t_action)
                self._q_table.set_q(
                    t_stage,
                    t_state,
                    t_action,
                    cur + self._alpha * td_error * trace,
                )

            decay = self._gamma * self._lam
            for trace_key in traces:
                traces[trace_key] *= decay

        # Visit-count bookkeeping. set_q does not bump visit counts, so
        # we run a no-op TD update (alpha=0, target=current) on every
        # visited (stage, state, action) to advance the counter.
        for stage, state, action in traj:
            current_q = self._q_table.get_q(stage, state, action)
            self._q_table.update_q(
                stage, state, action, target=current_q, alpha=0.0
            )

    def get_trajectory(self) -> List[Tuple[str, int, int]]:
        """Return a copy of the trajectory recorded so far this hand."""
        return list(self._trajectory)
