"""Tests for src/agents/mc_linear_fa.py."""

import random
import sys
from pathlib import Path

import numpy as np
import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_linear_fa import MonteCarloLinearFAAgent  # noqa: E402
from features import (  # noqa: E402
    FLOP_FEATURE_SIZE,
    PREFLOP_FEATURE_SIZE,
    RIVER_FEATURE_SIZE,
)
from game import UTHGame  # noqa: E402
from linear_q import LinearQ  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


def _make_agent(alpha: float = 0.01, epsilon: float = 0.0):
    q = LinearQ()
    rng = random.Random(0)
    agent = MonteCarloLinearFAAgent(q, rng, alpha=alpha)
    agent.set_epsilon(epsilon)
    return q, agent


def test_trajectory_recording_three_stages():
    q, agent = _make_agent()
    agent.start_hand()
    hole = _h("As", "Ah")
    flop = _h("Qs", "Js", "Ts")
    river_comm = flop + _h("2h", "3h")

    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )
    agent.policy_fn(
        "river", {"hole_cards": hole, "community_cards_revealed": river_comm}
    )

    traj = agent.get_trajectory()
    assert len(traj) == 3

    stages = [t[0] for t in traj]
    assert stages == ["preflop", "flop", "river"]

    feature_shapes = [t[1].shape for t in traj]
    assert feature_shapes == [
        (PREFLOP_FEATURE_SIZE,),
        (FLOP_FEATURE_SIZE,),
        (RIVER_FEATURE_SIZE,),
    ]

    actions = [t[2] for t in traj]
    assert actions == [0, 0, 0]  # zero weights → ties go to action 0


def test_start_hand_clears_trajectory():
    q, agent = _make_agent()
    agent.start_hand()
    hole = _h("As", "Ah")
    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    assert len(agent.get_trajectory()) == 1
    agent.start_hand()
    assert agent.get_trajectory() == []


def test_learn_from_hand_propagates_to_weights():
    """After one hand with a known trajectory, weights for the chosen
    actions should move toward the return."""
    q, agent = _make_agent(alpha=1.0, epsilon=0.0)
    agent.start_hand()
    hole = _h("As", "Ah")
    flop = _h("Qs", "Js", "Ts")
    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )
    agent.learn_from_hand(10.0)

    # alpha=1, w=0, error=10 → w[k] = 10 * phi[k] for active features
    assert q.get_update_count("preflop") == 1
    assert q.get_update_count("flop") == 1

    # Preflop: action 0 column for AA (preflop bucket 0) and bias.
    w_pre = q.get_weights("preflop")
    assert w_pre[0, 0] == pytest.approx(10.0)  # bucket-0 one-hot
    assert w_pre[-1, 0] == pytest.approx(10.0)  # bias
    assert np.all(w_pre[:, 1] == 0.0)
    assert np.all(w_pre[:, 2] == 0.0)


def test_frozen_policy_does_not_record_or_explore():
    q, agent = _make_agent(epsilon=1.0)  # set high; frozen should ignore
    agent.start_hand()
    hole = _h("As", "Ah")
    name = agent.frozen_policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    assert isinstance(name, str)
    assert agent.get_trajectory() == []


def test_smoke_1000_hands(tmp_path):
    rng = random.Random(0)
    game = UTHGame(seed=0)
    q = LinearQ()
    agent = MonteCarloLinearFAAgent(q, rng, alpha=0.01)
    agent.set_epsilon(1.0)

    for _ in range(1000):
        agent.start_hand()
        result = game.play_hand(agent.policy_fn)
        agent.learn_from_hand(result["net_chips"])

    # All three stages should have non-zero update counts.
    assert q.get_update_count("preflop") > 0
    assert q.get_update_count("flop") > 0
    assert q.get_update_count("river") > 0

    # Weights should have moved off zero somewhere.
    for stage in ("preflop", "flop", "river"):
        assert np.any(q.get_weights(stage) != 0.0)

    fp = str(tmp_path / "smoke_linear.npz")
    q.save(fp)
    q2 = LinearQ()
    q2.load(fp)
