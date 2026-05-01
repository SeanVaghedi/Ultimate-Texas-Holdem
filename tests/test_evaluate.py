"""Tests for the evaluation pipeline."""

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.evaluate import evaluate_policy  # noqa: E402
from strategies.baselines import (  # noqa: E402
    always_fold_policy,
    naive_heuristic_policy,
    random_policy,
)


_EXPECTED_KEYS = {
    "num_hands",
    "total_net_chips",
    "mean_net_chips",
    "std_error",
    "ci_95_lower",
    "ci_95_upper",
    "house_edge_percent",
    "house_edge_ci_lower_percent",
    "house_edge_ci_upper_percent",
    "action_counts",
    "fold_rate",
}


def test_evaluate_policy_returns_expected_keys():
    results = evaluate_policy(
        always_fold_policy, 1000, seed=0, show_progress=False
    )
    assert set(results.keys()) == _EXPECTED_KEYS
    assert isinstance(results["num_hands"], int)
    assert isinstance(results["total_net_chips"], float)
    assert isinstance(results["mean_net_chips"], float)
    assert isinstance(results["std_error"], float)
    assert isinstance(results["fold_rate"], float)
    assert isinstance(results["action_counts"], dict)
    for stage in ("preflop", "flop", "river"):
        assert stage in results["action_counts"]
        assert isinstance(results["action_counts"][stage], dict)


def test_always_fold_mean_is_minus_two():
    results = evaluate_policy(
        always_fold_policy, 1000, seed=0, show_progress=False
    )
    assert results["mean_net_chips"] == pytest.approx(-2.0)
    assert results["total_net_chips"] == pytest.approx(-2000.0)


def test_always_fold_fold_rate_is_one():
    results = evaluate_policy(
        always_fold_policy, 1000, seed=0, show_progress=False
    )
    assert results["fold_rate"] == pytest.approx(1.0)


def test_always_fold_action_counts():
    n = 1000
    results = evaluate_policy(
        always_fold_policy, n, seed=0, show_progress=False
    )
    counts = results["action_counts"]
    assert counts["preflop"] == {"check": n, "bet_3x": 0, "bet_4x": 0}
    assert counts["flop"] == {"check": n, "bet_2x": 0}
    assert counts["river"] == {"fold": n, "bet_1x": 0}


def test_random_policy_exercises_all_actions():
    rand_policy = random_policy(random.Random(0))
    results = evaluate_policy(
        rand_policy, 5000, seed=0, show_progress=False
    )
    for stage in ("preflop", "flop", "river"):
        for action, count in results["action_counts"][stage].items():
            assert count > 0, (
                f"action {action!r} at stage {stage!r} never taken"
            )


def test_naive_heuristic_smoke():
    results = evaluate_policy(
        naive_heuristic_policy, 1000, seed=0, show_progress=False
    )
    assert results["num_hands"] == 1000
