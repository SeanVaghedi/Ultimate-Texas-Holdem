"""Tests for src/agents/q_learning_agent.py."""

import random
import sys
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.q_learning_agent import QLearningAgent  # noqa: E402
from game import UTHGame  # noqa: E402
from qtable import QTable  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


def _make_agent(alpha: float = 0.1, gamma: float = 1.0, epsilon: float = 0.0):
    qt = QTable()
    rng = random.Random(0)
    agent = QLearningAgent(qt, rng, alpha=alpha, gamma=gamma)
    agent.set_epsilon(epsilon)
    return qt, agent


def test_trajectory_recording():
    qt, agent = _make_agent()
    agent.start_hand()
    hole = _h("As", "Ah")  # preflop bucket 0
    flop = _h("Qs", "Js", "Ts")  # flop bucket 3
    river_comm = flop + _h("2h", "3h")  # river bucket 4

    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    agent.policy_fn(
        "flop", {"hole_cards": hole, "community_cards_revealed": flop}
    )
    agent.policy_fn(
        "river", {"hole_cards": hole, "community_cards_revealed": river_comm}
    )

    assert agent.get_trajectory() == [
        ("preflop", 0, 0),
        ("flop", 3, 0),
        ("river", 4, 0),
    ]


def test_start_hand_clears_trajectory():
    qt, agent = _make_agent()
    agent.start_hand()
    hole = _h("As", "Ah")
    agent.policy_fn(
        "preflop", {"hole_cards": hole, "community_cards_revealed": []}
    )
    assert len(agent.get_trajectory()) == 1
    agent.start_hand()
    assert agent.get_trajectory() == []


def test_terminal_only_update_alpha_one():
    """Single-entry trajectory: target = net_chips, no bootstrap."""
    qt, agent = _make_agent(alpha=1.0, gamma=1.0)
    agent.start_hand()
    agent._current_trajectory.append(("preflop", 0, 2))
    agent.learn_from_hand(5.0)
    assert qt.get_q("preflop", 0, 2) == pytest.approx(5.0)
    assert qt.get_visits("preflop", 0, 2) == 1


def test_bootstrap_update_chronological_order():
    """Two-step trajectory: chronological updates use pre-update next-state Q.

    With alpha=1.0, gamma=1.0 and a baseline Q(flop, 3, 1) = 10:

    1. Preflop step is non-terminal. Target = gamma * max_a Q(flop, 3, a)
       = 1.0 * 10 = 10. Q(preflop, 0, 0) becomes 10.0.
    2. Flop step is terminal. Target = net_chips = 7. Q(flop, 3, 1)
       becomes 7.0 (overwrites the previous 10).

    Implementation choice: process trajectory chronologically, with
    each lookup using whatever is in the table at the time of update.
    The non-terminal preflop step therefore reads the original 10.0,
    not the post-update 7.0.
    """
    qt = QTable()
    rng = random.Random(0)
    agent = QLearningAgent(qt, rng, alpha=1.0, gamma=1.0)
    qt.set_q("flop", 3, 1, 10.0)

    agent.start_hand()
    agent._current_trajectory.append(("preflop", 0, 0))
    agent._current_trajectory.append(("flop", 3, 1))
    agent.learn_from_hand(7.0)

    assert qt.get_q("preflop", 0, 0) == pytest.approx(10.0)
    assert qt.get_q("flop", 3, 1) == pytest.approx(7.0)
    assert qt.get_visits("preflop", 0, 0) == 1
    assert qt.get_visits("flop", 3, 1) == 1


def test_smoke_1000_hands(tmp_path):
    rng = random.Random(0)
    game = UTHGame(seed=0)
    qt = QTable()
    agent = QLearningAgent(qt, rng, alpha=0.1, gamma=1.0)
    agent.set_epsilon(1.0)

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
