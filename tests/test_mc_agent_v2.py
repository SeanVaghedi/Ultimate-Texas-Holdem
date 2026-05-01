"""Tests for src/agents/mc_agent_v2.py."""

import random
import sys
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstraction_v2 import (  # noqa: E402
    NUM_FLOP_BUCKETS_V1,
    NUM_RIVER_BUCKETS_V1,
)
from agents.mc_agent_v2 import MonteCarloAgentV2  # noqa: E402
from game import UTHGame  # noqa: E402
from qtable_v2 import QTableV2  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


def _make_agent(epsilon: float = 0.0):
    qt = QTableV2()
    rng = random.Random(0)
    agent = MonteCarloAgentV2(qt, rng)
    agent.set_epsilon(epsilon)
    return qt, agent


def test_trajectory_recording_premium_pair():
    """AA → preflop=0, flop top-pair v1=3 → joint=3, river top-pair v1=4 → joint=4."""
    qt, agent = _make_agent(epsilon=0.0)
    agent.start_hand()

    hole = _h("As", "Ah")  # preflop bucket 0
    flop = _h("Qs", "Js", "Ts")  # flop v1 bucket 3 (top pair, A>Q)
    river_comm = flop + _h("2h", "3h")  # river v1 bucket 4 (top pair, A>Q)

    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )
    agent.policy_fn(
        "river", {"hole_cards": hole, "community_cards_revealed": river_comm}
    )

    expected_flop_idx = 0 * NUM_FLOP_BUCKETS_V1 + 3
    expected_river_idx = 0 * NUM_RIVER_BUCKETS_V1 + 4
    assert agent.get_trajectory() == [
        ("preflop", 0, 0),
        ("flop", expected_flop_idx, 0),
        ("river", expected_river_idx, 0),
    ]


def test_trajectory_uses_joint_encoding_for_nonzero_preflop():
    """Non-zero preflop bucket shifts the flop/river joint indices.

    Hole 7s2h is preflop bucket 14 (other offsuit). With a flop of
    2d 2c Ks the agent has trips, v1 flop bucket 1, so the joint
    flop index must be 14*8 + 1 = 113.
    """
    qt, agent = _make_agent(epsilon=0.0)
    agent.start_hand()

    hole = _h("7s", "2h")  # preflop bucket 14
    flop = _h("2d", "2c", "Ks")  # flop v1 bucket 1 (trips)

    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )

    expected_flop_idx = 14 * NUM_FLOP_BUCKETS_V1 + 1
    assert expected_flop_idx == 113
    assert agent.get_trajectory() == [
        ("preflop", 14, 0),
        ("flop", 113, 0),
    ]


def test_start_hand_clears_trajectory():
    qt, agent = _make_agent(epsilon=0.0)
    agent.start_hand()
    hole = _h("As", "Ah")
    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    assert len(agent.get_trajectory()) == 1
    agent.start_hand()
    assert agent.get_trajectory() == []


def test_learn_from_hand_applies_updates():
    qt, agent = _make_agent(epsilon=0.0)
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

    agent.learn_from_hand(10.0)

    expected_flop_idx = 0 * NUM_FLOP_BUCKETS_V1 + 3
    expected_river_idx = 0 * NUM_RIVER_BUCKETS_V1 + 4
    for stage, s, a in [
        ("preflop", 0, 0),
        ("flop", expected_flop_idx, 0),
        ("river", expected_river_idx, 0),
    ]:
        assert qt.get_q(stage, s, a) == pytest.approx(10.0)
        assert qt.get_visits(stage, s, a) == 1


def test_smoke_1000_hands(tmp_path):
    rng = random.Random(0)
    game = UTHGame(seed=0)
    qt = QTableV2()
    agent = MonteCarloAgentV2(qt, rng)
    agent.set_epsilon(1.0)  # uniform random for max coverage

    for _ in range(1000):
        agent.start_hand()
        result = game.play_hand(agent.policy_fn)
        agent.learn_from_hand(result["net_chips"])

    visited_preflop = sum(
        1
        for s in range(15)
        for a in range(3)
        if qt.get_visits("preflop", s, a) > 0
    )
    assert visited_preflop >= 23, (
        f"Only {visited_preflop}/45 preflop pairs visited"
    )

    fp = str(tmp_path / "smoke_v2.npz")
    qt.save(fp)
    qt2 = QTableV2()
    qt2.load(fp)
