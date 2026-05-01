"""Tests for src/agents/mc_agent.py."""

import random
import sys
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.mc_agent import MonteCarloAgent  # noqa: E402
from game import UTHGame  # noqa: E402
from qtable import QTable  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


def _make_agent(epsilon: float = 0.0):
    qt = QTable()
    rng = random.Random(0)
    agent = MonteCarloAgent(qt, rng)
    agent.set_epsilon(epsilon)
    return qt, agent


def test_trajectory_recording():
    qt, agent = _make_agent(epsilon=0.0)
    agent.start_hand()

    hole = _h("As", "Ah")  # preflop bucket 0
    flop = _h("Qs", "Js", "Ts")  # flop bucket 3 (top pair, A>Q)
    river_comm = flop + _h("2h", "3h")  # river bucket 4 (top pair, A>Q)

    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )
    agent.policy_fn(
        "river", {"hole_cards": hole, "community_cards_revealed": river_comm}
    )

    # With Q=0 everywhere and epsilon=0, argmax ties to action index 0.
    assert agent.get_trajectory() == [
        ("preflop", 0, 0),
        ("flop", 3, 0),
        ("river", 4, 0),
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

    for stage, s, a in [
        ("preflop", 0, 0),
        ("flop", 3, 0),
        ("river", 4, 0),
    ]:
        assert qt.get_q(stage, s, a) == pytest.approx(10.0)
        assert qt.get_visits(stage, s, a) == 1


def test_smoke_1000_hands(tmp_path):
    rng = random.Random(0)
    game = UTHGame(seed=0)
    qt = QTable()
    agent = MonteCarloAgent(qt, rng)
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

    fp = str(tmp_path / "smoke.npz")
    qt.save(fp)
    qt2 = QTable()
    qt2.load(fp)
