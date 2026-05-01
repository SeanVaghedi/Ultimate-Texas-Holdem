"""Unit tests for src/qtable_v2.py."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qtable import (  # noqa: E402
    ACTIONS_BY_STAGE,
    action_index_to_name,
    action_name_to_index,
)
from qtable_v2 import (  # noqa: E402
    NUM_STATES_BY_STAGE_V2,
    QTableV2,
    STAGES,
)


def test_initialization_zeros():
    q = QTableV2()
    for stage in STAGES:
        n_states = NUM_STATES_BY_STAGE_V2[stage]
        n_actions = len(ACTIONS_BY_STAGE[stage])
        for s in range(n_states):
            for a in range(n_actions):
                assert q.get_q(stage, s, a) == 0.0
                assert q.get_visits(stage, s, a) == 0


def test_state_space_shapes():
    """Spot-check the v2 shape contract."""
    assert NUM_STATES_BY_STAGE_V2["preflop"] == 15
    assert NUM_STATES_BY_STAGE_V2["flop"] == 120
    assert NUM_STATES_BY_STAGE_V2["river"] == 105


def test_set_get_q_roundtrip():
    q = QTableV2()
    q.set_q("preflop", 5, 1, 3.14)
    assert q.get_q("preflop", 5, 1) == pytest.approx(3.14)
    q.set_q("flop", 119, 1, -2.5)
    assert q.get_q("flop", 119, 1) == pytest.approx(-2.5)
    q.set_q("river", 104, 0, 7.0)
    assert q.get_q("river", 104, 0) == pytest.approx(7.0)


def test_update_q_math():
    q = QTableV2()
    q.update_q("preflop", 0, 0, target=10.0, alpha=0.5)
    assert q.get_q("preflop", 0, 0) == pytest.approx(5.0)
    assert q.get_visits("preflop", 0, 0) == 1
    q.update_q("preflop", 0, 0, target=10.0, alpha=0.5)
    assert q.get_q("preflop", 0, 0) == pytest.approx(7.5)
    assert q.get_visits("preflop", 0, 0) == 2


def test_update_q_mc_averaging():
    q = QTableV2()
    q.update_q_mc("flop", 50, 0, return_value=10.0)
    assert q.get_q("flop", 50, 0) == pytest.approx(10.0)
    assert q.get_visits("flop", 50, 0) == 1
    q.update_q_mc("flop", 50, 0, return_value=20.0)
    assert q.get_q("flop", 50, 0) == pytest.approx(15.0)
    assert q.get_visits("flop", 50, 0) == 2
    q.update_q_mc("flop", 50, 0, return_value=0.0)
    assert q.get_q("flop", 50, 0) == pytest.approx(10.0)
    assert q.get_visits("flop", 50, 0) == 3


def test_best_action_argmax():
    q = QTableV2()
    q.set_q("preflop", 0, 0, 1.0)
    q.set_q("preflop", 0, 1, 3.0)
    q.set_q("preflop", 0, 2, 2.0)
    assert q.best_action("preflop", 0) == 1


def test_best_action_ties_lowest_index():
    q = QTableV2()
    assert q.best_action("preflop", 0) == 0
    assert q.best_action("flop", 0) == 0
    assert q.best_action("river", 0) == 0


def test_epsilon_greedy_zero_returns_argmax():
    q = QTableV2()
    q.set_q("preflop", 0, 0, 1.0)
    q.set_q("preflop", 0, 1, 3.0)
    q.set_q("preflop", 0, 2, 2.0)
    rng = random.Random(0)
    for _ in range(100):
        assert q.epsilon_greedy_action("preflop", 0, 0.0, rng) == 1


def test_epsilon_greedy_one_uniform_random():
    q = QTableV2()
    q.set_q("preflop", 0, 0, 1.0)
    rng = random.Random(0)
    counts = [0, 0, 0]
    for _ in range(1000):
        a = q.epsilon_greedy_action("preflop", 0, 1.0, rng)
        counts[a] += 1
    for c in counts:
        assert c > 0


def test_action_index_name_roundtrip():
    """Action name <-> index helpers are reused from v1; verify consistency."""
    for stage in STAGES:
        for idx, name in enumerate(ACTIONS_BY_STAGE[stage]):
            assert action_name_to_index(stage, name) == idx
            assert action_index_to_name(stage, idx) == name


def test_save_and_load(tmp_path):
    q = QTableV2()
    q.set_q("preflop", 0, 1, 1.5)
    q.set_q("flop", 100, 0, -0.7)
    q.set_q("river", 90, 1, 2.25)
    q.update_q("preflop", 0, 1, target=10.0, alpha=0.1)
    fp = str(tmp_path / "qtable_v2.npz")
    q.save(fp)

    q2 = QTableV2()
    q2.load(fp)
    for stage in STAGES:
        n_states = NUM_STATES_BY_STAGE_V2[stage]
        n_actions = len(ACTIONS_BY_STAGE[stage])
        for s in range(n_states):
            for a in range(n_actions):
                assert q2.get_q(stage, s, a) == pytest.approx(
                    q.get_q(stage, s, a)
                )
                assert q2.get_visits(stage, s, a) == q.get_visits(stage, s, a)


def test_invalid_stage_raises():
    q = QTableV2()
    with pytest.raises(ValueError):
        q.get_q("turn", 0, 0)


def test_invalid_state_raises():
    q = QTableV2()
    with pytest.raises(ValueError):
        q.get_q("preflop", 99, 0)
    with pytest.raises(ValueError):
        q.get_q("flop", 120, 0)
    with pytest.raises(ValueError):
        q.get_q("river", 105, 0)


def test_invalid_action_raises():
    q = QTableV2()
    with pytest.raises(ValueError):
        q.get_q("preflop", 0, 99)


def test_load_wrong_shape_raises(tmp_path):
    """Loading a v1-shaped (15,3)/(8,2)/(7,2) file into QTableV2 must fail."""
    fp = tmp_path / "bad.npz"
    np.savez(
        fp,
        q_preflop=np.zeros((15, 3), dtype=np.float64),
        q_flop=np.zeros((8, 2), dtype=np.float64),  # wrong: should be (120, 2)
        q_river=np.zeros((7, 2), dtype=np.float64),
        visits_preflop=np.zeros((15, 3), dtype=np.int64),
        visits_flop=np.zeros((8, 2), dtype=np.int64),
        visits_river=np.zeros((7, 2), dtype=np.int64),
    )
    q = QTableV2()
    with pytest.raises(ValueError):
        q.load(str(fp))
