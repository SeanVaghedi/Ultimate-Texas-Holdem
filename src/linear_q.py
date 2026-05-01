"""Linear Q-function for tabular UTH RL with feature approximation.

Stores one weight matrix per stage of shape ``(feature_size, n_actions)``.
Q(s, a) = phi(s) @ w[stage][:, a]. Updates are semi-gradient on the
weights for the chosen action only.

Action conventions match ``qtable.QTable``:
    preflop: 0=check, 1=bet_3x, 2=bet_4x
    flop:    0=check, 1=bet_2x
    river:   0=fold,  1=bet_1x
"""

import random
from typing import Dict, Tuple

import numpy as np

from features import (
    FLOP_FEATURE_SIZE,
    PREFLOP_FEATURE_SIZE,
    RIVER_FEATURE_SIZE,
)
from qtable import ACTIONS_BY_STAGE


STAGES: Tuple[str, ...] = ("preflop", "flop", "river")

FEATURE_SIZE_BY_STAGE: Dict[str, int] = {
    "preflop": PREFLOP_FEATURE_SIZE,
    "flop": FLOP_FEATURE_SIZE,
    "river": RIVER_FEATURE_SIZE,
}

_EXPECTED_SHAPES: Dict[str, Tuple[int, int]] = {
    stage: (FEATURE_SIZE_BY_STAGE[stage], len(ACTIONS_BY_STAGE[stage]))
    for stage in STAGES
}


def _validate_stage(stage: str) -> None:
    if stage not in STAGES:
        raise ValueError(
            f"Invalid stage {stage!r}; expected one of {STAGES}"
        )


def _validate_action(stage: str, action: int) -> None:
    n = len(ACTIONS_BY_STAGE[stage])
    if not isinstance(action, (int, np.integer)) or action < 0 or action >= n:
        raise ValueError(
            f"Invalid action index {action!r} for stage {stage!r}; "
            f"expected int in [0, {n - 1}]"
        )


def _validate_features(stage: str, features: np.ndarray) -> None:
    expected = FEATURE_SIZE_BY_STAGE[stage]
    if features.shape != (expected,):
        raise ValueError(
            f"Feature vector for stage {stage!r} has shape "
            f"{features.shape}; expected ({expected},)"
        )


class LinearQ:
    """Per-stage linear Q-function with semi-gradient MC updates.

    Tracks per-stage update counts for diagnostics — these are gradient
    steps, not visit counts in the tabular sense, so they live separate
    from any per-cell counter.
    """

    def __init__(self) -> None:
        self._w: Dict[str, np.ndarray] = {
            stage: np.zeros(_EXPECTED_SHAPES[stage], dtype=np.float64)
            for stage in STAGES
        }
        self._update_count: Dict[str, int] = {
            stage: 0 for stage in STAGES
        }

    def get_weights(self, stage: str) -> np.ndarray:
        """Return the weight matrix for a stage (shape (feat, n_actions))."""
        _validate_stage(stage)
        return self._w[stage]

    def get_update_count(self, stage: str) -> int:
        _validate_stage(stage)
        return self._update_count[stage]

    def q_value(
        self, stage: str, features: np.ndarray, action: int
    ) -> float:
        """Return Q(s, a) = features dot w[stage][:, action]."""
        _validate_stage(stage)
        _validate_action(stage, action)
        _validate_features(stage, features)
        return float(features @ self._w[stage][:, action])

    def q_values_all_actions(
        self, stage: str, features: np.ndarray
    ) -> np.ndarray:
        """Return shape (n_actions,) of Q-values for every action at this stage."""
        _validate_stage(stage)
        _validate_features(stage, features)
        return features @ self._w[stage]

    def best_action(self, stage: str, features: np.ndarray) -> int:
        """Argmax action; ties broken by lowest action index."""
        return int(np.argmax(self.q_values_all_actions(stage, features)))

    def epsilon_greedy_action(
        self,
        stage: str,
        features: np.ndarray,
        epsilon: float,
        rng: random.Random,
    ) -> int:
        """Pick a random action with probability epsilon, else best_action."""
        _validate_stage(stage)
        if rng.random() < epsilon:
            return rng.randrange(len(ACTIONS_BY_STAGE[stage]))
        return self.best_action(stage, features)

    def update_mc(
        self,
        stage: str,
        features: np.ndarray,
        action: int,
        return_value: float,
        alpha: float,
    ) -> None:
        """Semi-gradient MC update: w[:, a] += alpha * (G - phi @ w[:, a]) * phi."""
        _validate_stage(stage)
        _validate_action(stage, action)
        _validate_features(stage, features)
        current = float(features @ self._w[stage][:, action])
        error = return_value - current
        self._w[stage][:, action] += alpha * error * features
        self._update_count[stage] += 1

    def save(self, filepath: str) -> None:
        """Persist all three weight matrices and update counts to a .npz file."""
        np.savez(
            filepath,
            w_preflop=self._w["preflop"],
            w_flop=self._w["flop"],
            w_river=self._w["river"],
            updates_preflop=np.int64(self._update_count["preflop"]),
            updates_flop=np.int64(self._update_count["flop"]),
            updates_river=np.int64(self._update_count["river"]),
        )

    def load(self, filepath: str) -> None:
        """Replace internal weights from a .npz file. Validates shapes."""
        with np.load(filepath) as data:
            for stage in STAGES:
                w_key = f"w_{stage}"
                if w_key not in data.files:
                    raise ValueError(
                        f"Missing array {w_key!r} in {filepath!r}"
                    )
                expected = _EXPECTED_SHAPES[stage]
                if data[w_key].shape != expected:
                    raise ValueError(
                        f"Loaded {w_key} has shape {data[w_key].shape}; "
                        f"expected {expected}"
                    )
            new_w = {
                stage: data[f"w_{stage}"].astype(np.float64).copy()
                for stage in STAGES
            }
            new_counts = {}
            for stage in STAGES:
                key = f"updates_{stage}"
                new_counts[stage] = (
                    int(data[key]) if key in data.files else 0
                )
        self._w = new_w
        self._update_count = new_counts

    def __repr__(self) -> str:
        shapes = ", ".join(
            f"{stage}={self._w[stage].shape[0]}x{self._w[stage].shape[1]}"
            for stage in STAGES
        )
        total_updates = sum(self._update_count.values())
        return f"LinearQ({shapes}, total_updates={total_updates})"
