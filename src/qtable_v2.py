"""Tabular Q-table for Ultimate Texas Hold'em RL — v2 (richer abstraction).

Same public API as ``qtable.QTable`` but sized for the v2 state space
defined in ``abstraction_v2``: pre-flop is unchanged at 15 buckets,
flop expands to 120 (15 pre x 8 post), river expands to 105 (15 x 7).
Action conventions are identical to v1.
"""

import random
from typing import Dict, Tuple

import numpy as np

from qtable import (
    ACTIONS_BY_STAGE,
    action_index_to_name,
    action_name_to_index,
)


STAGES: Tuple[str, ...] = ("preflop", "flop", "river")

NUM_STATES_BY_STAGE_V2: Dict[str, int] = {
    "preflop": 15,
    "flop": 120,
    "river": 105,
}

_EXPECTED_SHAPES: Dict[str, Tuple[int, int]] = {
    stage: (NUM_STATES_BY_STAGE_V2[stage], len(ACTIONS_BY_STAGE[stage]))
    for stage in STAGES
}


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(
            f"Invalid stage {stage!r}; expected one of {STAGES}"
        )


def _validate_state(stage: str, state: int) -> None:
    n = NUM_STATES_BY_STAGE_V2[stage]
    if not isinstance(state, (int, np.integer)) or state < 0 or state >= n:
        raise ValueError(
            f"Invalid state index {state!r} for stage {stage!r}; "
            f"expected int in [0, {n - 1}]"
        )


def _validate_action(stage: str, action: int) -> None:
    n = len(ACTIONS_BY_STAGE[stage])
    if not isinstance(action, (int, np.integer)) or action < 0 or action >= n:
        raise ValueError(
            f"Invalid action index {action!r} for stage {stage!r}; "
            f"expected int in [0, {n - 1}]"
        )


class QTableV2:
    """Q-values and visit counts for tabular UTH RL under the v2 abstraction.

    Internal layout: one (n_states, n_actions) float64 array per stage
    for Q-values and a parallel int64 array per stage for visit counts.
    Shapes are (15, 3), (120, 2), (105, 2).
    """

    def __init__(self) -> None:
        self._q: Dict[str, np.ndarray] = {
            stage: np.zeros(_EXPECTED_SHAPES[stage], dtype=np.float64)
            for stage in STAGES
        }
        self._visits: Dict[str, np.ndarray] = {
            stage: np.zeros(_EXPECTED_SHAPES[stage], dtype=np.int64)
            for stage in STAGES
        }

    def get_q(self, stage: str, state: int, action: int) -> float:
        _validate_stage(stage)
        _validate_state(stage, state)
        _validate_action(stage, action)
        return float(self._q[stage][state, action])

    def set_q(
        self, stage: str, state: int, action: int, value: float
    ) -> None:
        """Overwrite a Q-value (testing/debugging only; not a training op)."""
        _validate_stage(stage)
        _validate_state(stage, state)
        _validate_action(stage, action)
        self._q[stage][state, action] = float(value)

    def update_q(
        self,
        stage: str,
        state: int,
        action: int,
        target: float,
        alpha: float,
    ) -> None:
        """TD-style update: Q <- Q + alpha * (target - Q). Bumps visit count."""
        _validate_stage(stage)
        _validate_state(stage, state)
        _validate_action(stage, action)
        current = self._q[stage][state, action]
        self._q[stage][state, action] = current + alpha * (target - current)
        self._visits[stage][state, action] += 1

    def update_q_mc(
        self, stage: str, state: int, action: int, return_value: float
    ) -> None:
        """Incremental Monte Carlo average of returns for a (state, action).

        Increments the visit count first, then sets
        Q <- Q + (return - Q) / N where N is the new count. Equivalent
        to maintaining a running mean of all returns observed.
        """
        _validate_stage(stage)
        _validate_state(stage, state)
        _validate_action(stage, action)
        self._visits[stage][state, action] += 1
        n = self._visits[stage][state, action]
        current = self._q[stage][state, action]
        self._q[stage][state, action] = current + (return_value - current) / n

    def get_visits(self, stage: str, state: int, action: int) -> int:
        _validate_stage(stage)
        _validate_state(stage, state)
        _validate_action(stage, action)
        return int(self._visits[stage][state, action])

    def best_action(self, stage: str, state: int) -> int:
        """Argmax action; ties broken by lowest action index."""
        _validate_stage(stage)
        _validate_state(stage, state)
        return int(np.argmax(self._q[stage][state]))

    def epsilon_greedy_action(
        self,
        stage: str,
        state: int,
        epsilon: float,
        rng: random.Random,
    ) -> int:
        """Pick a random action with probability epsilon, else best_action."""
        _validate_stage(stage)
        _validate_state(stage, state)
        if rng.random() < epsilon:
            return rng.randrange(len(ACTIONS_BY_STAGE[stage]))
        return self.best_action(stage, state)

    def save(self, filepath: str) -> None:
        """Persist all six arrays to a .npz file."""
        np.savez(
            filepath,
            q_preflop=self._q["preflop"],
            q_flop=self._q["flop"],
            q_river=self._q["river"],
            visits_preflop=self._visits["preflop"],
            visits_flop=self._visits["flop"],
            visits_river=self._visits["river"],
        )

    def load(self, filepath: str) -> None:
        """Replace internal arrays from a .npz file. Validates shapes."""
        with np.load(filepath) as data:
            for stage in STAGES:
                q_key = f"q_{stage}"
                v_key = f"visits_{stage}"
                if q_key not in data.files or v_key not in data.files:
                    raise ValueError(
                        f"Missing array {q_key!r} or {v_key!r} in {filepath!r}"
                    )
                expected = _EXPECTED_SHAPES[stage]
                if data[q_key].shape != expected:
                    raise ValueError(
                        f"Loaded {q_key} has shape {data[q_key].shape}; "
                        f"expected {expected}"
                    )
                if data[v_key].shape != expected:
                    raise ValueError(
                        f"Loaded {v_key} has shape {data[v_key].shape}; "
                        f"expected {expected}"
                    )
            new_q = {
                stage: data[f"q_{stage}"].astype(np.float64).copy()
                for stage in STAGES
            }
            new_visits = {
                stage: data[f"visits_{stage}"].astype(np.int64).copy()
                for stage in STAGES
            }
        self._q = new_q
        self._visits = new_visits

    def __repr__(self) -> str:
        total_visits = sum(int(v.sum()) for v in self._visits.values())
        return (
            "QTableV2(preflop=15x3, flop=120x2, river=105x2, "
            f"total_visits={total_visits})"
        )
