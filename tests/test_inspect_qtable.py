"""Smoke test for src/evaluation/inspect_qtable.py."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluation.inspect_qtable import (  # noqa: E402
    find_disagreements,
    print_disagreements,
    print_optimal_section,
    print_qtable_section,
)
from qtable import (  # noqa: E402
    ACTIONS_BY_STAGE,
    NUM_STATES_BY_STAGE,
    QTable,
    STAGES,
)


def _fake_uniform_optimal_dist():
    dist = {}
    for stage in STAGES:
        actions = ACTIONS_BY_STAGE[stage]
        dist[stage] = {
            b: {a: 1.0 / len(actions) for a in actions}
            for b in range(NUM_STATES_BY_STAGE[stage])
        }
    return dist


def test_inspect_smoke(capsys):
    qtable_path = ROOT / "results" / "mc_qtable.npz"
    qt = QTable()
    if qtable_path.exists():
        qt.load(str(qtable_path))
    else:
        qt.set_q("preflop", 0, 2, 1.0)
        qt.set_q("flop", 3, 1, 0.5)
        qt.set_q("river", 4, 1, 0.5)

    optimal_dist = _fake_uniform_optimal_dist()

    for stage in STAGES:
        print_qtable_section(stage, qt)
    for stage in STAGES:
        print_optimal_section(stage, optimal_dist)

    rows = find_disagreements(qt, optimal_dist)
    print_disagreements(rows)

    captured = capsys.readouterr().out
    assert "Q-values (trained MC agent)" in captured
    assert "Optimal Action Distribution" in captured
    assert "Disagreements with Optimal" in captured
    assert isinstance(rows, list)
    assert len(rows) >= 0
