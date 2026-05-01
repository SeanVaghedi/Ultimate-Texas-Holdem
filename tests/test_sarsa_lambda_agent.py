"""Tests for src/agents/sarsa_lambda_agent.py."""

import random
import sys
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agents.sarsa_lambda_agent import SarsaLambdaAgent  # noqa: E402
from game import UTHGame  # noqa: E402
from qtable import QTable  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


def _make_agent(
    alpha: float = 0.1,
    gamma: float = 1.0,
    lam: float = 0.9,
    epsilon: float = 0.0,
):
    qt = QTable()
    rng = random.Random(0)
    agent = SarsaLambdaAgent(qt, rng, alpha=alpha, gamma=gamma, lam=lam)
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


def test_lambda_zero_reduces_to_one_step_sarsa():
    """With lam=0 the trace decays to 0 immediately after each visit, so
    each step's update only touches the current (state, action) — exactly
    one-step SARSA.

    Setup: alpha=1, gamma=1, lam=0; pre-set Q(flop, 3, 1) = 10.
    Trajectory: [(preflop, 0, 0), (flop, 3, 1)]; net_chips = 7.

    Step 0 (non-terminal): delta = 0 + 1*Q(flop,3,1) - 0 = 10.
        trace[(preflop,0,0)] = 1; Q(preflop,0,0) += 1*10*1 = 10.
        Then trace decays *= gamma*lam = 0, so trace becomes 0.
    Step 1 (terminal):    delta = 7 - 10 = -3.
        trace[(flop,3,1)] = 1; Q(flop,3,1) += 1*-3*1 = -3, so Q = 7.

    Expected: Q(preflop,0,0) = 10, Q(flop,3,1) = 7.
    """
    qt = QTable()
    rng = random.Random(0)
    agent = SarsaLambdaAgent(qt, rng, alpha=1.0, gamma=1.0, lam=0.0)
    qt.set_q("flop", 3, 1, 10.0)

    agent.start_hand()
    agent._trajectory.append(("preflop", 0, 0))
    agent._trajectory.append(("flop", 3, 1))
    agent.learn_from_hand(7.0)

    assert qt.get_q("preflop", 0, 0) == pytest.approx(10.0)
    assert qt.get_q("flop", 3, 1) == pytest.approx(7.0)
    assert qt.get_visits("preflop", 0, 0) == 1
    assert qt.get_visits("flop", 3, 1) == 1


def test_lambda_one_propagates_terminal_reward_to_all():
    """With lam=1 and gamma=1 the trace never decays, so every visited
    (state, action) receives the terminal TD error — equivalent to MC.

    Setup: alpha=1, gamma=1, lam=1; Q starts at zero everywhere.
    Trajectory: [(preflop, 0, 0), (flop, 3, 1)]; net_chips = 7.

    Step 0 (non-terminal): delta = 0 + 1*0 - 0 = 0; no Q changes; trace
        for (preflop,0,0) = 1 and survives the gamma*lam = 1 decay.
    Step 1 (terminal):    delta = 7 - 0 = 7.
        trace[(flop,3,1)] = 1, trace[(preflop,0,0)] = 1.
        Both Q-values gain 1*7*1 = 7.

    Expected: Q(preflop,0,0) = 7, Q(flop,3,1) = 7 — matches MC.
    """
    qt = QTable()
    rng = random.Random(0)
    agent = SarsaLambdaAgent(qt, rng, alpha=1.0, gamma=1.0, lam=1.0)

    agent.start_hand()
    agent._trajectory.append(("preflop", 0, 0))
    agent._trajectory.append(("flop", 3, 1))
    agent.learn_from_hand(7.0)

    assert qt.get_q("preflop", 0, 0) == pytest.approx(7.0)
    assert qt.get_q("flop", 3, 1) == pytest.approx(7.0)
    assert qt.get_visits("preflop", 0, 0) == 1
    assert qt.get_visits("flop", 3, 1) == 1


def test_smoke_1000_hands(tmp_path):
    rng = random.Random(0)
    game = UTHGame(seed=0)
    qt = QTable()
    agent = SarsaLambdaAgent(qt, rng, alpha=0.1, gamma=1.0, lam=0.9)
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
