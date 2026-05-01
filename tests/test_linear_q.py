"""Unit tests for src/linear_q.py."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features import (  # noqa: E402
    FLOP_FEATURE_SIZE,
    PREFLOP_FEATURE_SIZE,
    RIVER_FEATURE_SIZE,
)
from linear_q import LinearQ, STAGES  # noqa: E402
from qtable import ACTIONS_BY_STAGE  # noqa: E402


def _onehot(size: int, idx: int, bias: bool = True) -> np.ndarray:
    phi = np.zeros(size, dtype=np.float64)
    phi[idx] = 1.0
    if bias:
        phi[-1] = 1.0
    return phi


def test_initialization_zeros():
    q = LinearQ()
    for stage in STAGES:
        w = q.get_weights(stage)
        assert w.shape[1] == len(ACTIONS_BY_STAGE[stage])
        assert np.all(w == 0.0)
        assert q.get_update_count(stage) == 0


def test_q_value_zero_when_weights_zero():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 3)
    for a in range(len(ACTIONS_BY_STAGE["preflop"])):
        assert q.q_value("preflop", phi, a) == 0.0


def test_q_values_all_actions_shape():
    q = LinearQ()
    phi = np.zeros(FLOP_FEATURE_SIZE, dtype=np.float64)
    qs = q.q_values_all_actions("flop", phi)
    assert qs.shape == (len(ACTIONS_BY_STAGE["flop"]),)
    assert np.all(qs == 0.0)


def test_mc_update_alpha_one_with_onehot():
    """With w=0 and a one-hot phi[k]=1 + bias, alpha=1, target=10:
    error = 10 - 0 = 10; gradient step adds 10*phi to w[:, action];
    so w[k] = 10 and w[bias] = 10."""
    q = LinearQ()
    k = 4
    phi = _onehot(PREFLOP_FEATURE_SIZE, k)
    q.update_mc("preflop", phi, action=1, return_value=10.0, alpha=1.0)
    w = q.get_weights("preflop")
    assert w[k, 1] == pytest.approx(10.0)
    assert w[-1, 1] == pytest.approx(10.0)
    assert w[k, 0] == 0.0  # other actions untouched
    assert w[k, 2] == 0.0
    assert q.get_update_count("preflop") == 1


def test_mc_update_alpha_smaller_step():
    """alpha=0.5, target=10, w=0, phi=onehot: w[k] becomes 5."""
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 0)
    q.update_mc("preflop", phi, action=0, return_value=10.0, alpha=0.5)
    w = q.get_weights("preflop")
    assert w[0, 0] == pytest.approx(5.0)
    assert w[-1, 0] == pytest.approx(5.0)


def test_best_action_returns_argmax():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 2)
    q.update_mc("preflop", phi, action=1, return_value=10.0, alpha=1.0)
    assert q.best_action("preflop", phi) == 1


def test_best_action_ties_lowest_index():
    q = LinearQ()
    phi = np.zeros(PREFLOP_FEATURE_SIZE, dtype=np.float64)
    assert q.best_action("preflop", phi) == 0


def test_epsilon_greedy_zero_returns_argmax():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 0)
    q.update_mc("preflop", phi, action=2, return_value=10.0, alpha=1.0)
    rng = random.Random(0)
    for _ in range(100):
        assert q.epsilon_greedy_action("preflop", phi, 0.0, rng) == 2


def test_epsilon_greedy_one_uniform_random():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 0)
    rng = random.Random(0)
    counts = [0] * len(ACTIONS_BY_STAGE["preflop"])
    for _ in range(1000):
        a = q.epsilon_greedy_action("preflop", phi, 1.0, rng)
        counts[a] += 1
    for c in counts:
        assert c > 0


def test_save_and_load_roundtrip(tmp_path):
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 3)
    q.update_mc("preflop", phi, action=1, return_value=10.0, alpha=1.0)
    phi2 = _onehot(FLOP_FEATURE_SIZE, 100)
    q.update_mc("flop", phi2, action=0, return_value=-2.0, alpha=0.5)
    fp = str(tmp_path / "linear_q.npz")
    q.save(fp)

    q2 = LinearQ()
    q2.load(fp)
    for stage in STAGES:
        np.testing.assert_array_equal(
            q2.get_weights(stage), q.get_weights(stage)
        )
        assert q2.get_update_count(stage) == q.get_update_count(stage)


def test_invalid_stage_raises():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 0)
    with pytest.raises(ValueError):
        q.q_value("turn", phi, 0)


def test_invalid_action_raises():
    q = LinearQ()
    phi = _onehot(PREFLOP_FEATURE_SIZE, 0)
    with pytest.raises(ValueError):
        q.q_value("preflop", phi, 99)


def test_invalid_feature_shape_raises():
    q = LinearQ()
    phi = np.zeros(7, dtype=np.float64)  # wrong size
    with pytest.raises(ValueError):
        q.q_value("preflop", phi, 0)


def test_load_wrong_shape_raises(tmp_path):
    fp = tmp_path / "bad.npz"
    np.savez(
        fp,
        w_preflop=np.zeros((10, 3), dtype=np.float64),  # wrong: should be (16, 3)
        w_flop=np.zeros((FLOP_FEATURE_SIZE, 2), dtype=np.float64),
        w_river=np.zeros((RIVER_FEATURE_SIZE, 2), dtype=np.float64),
    )
    q = LinearQ()
    with pytest.raises(ValueError):
        q.load(str(fp))
