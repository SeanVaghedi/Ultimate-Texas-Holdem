"""Tests for src/features.py."""

import sys
from pathlib import Path

import numpy as np
import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstraction import flop_bucket, preflop_bucket  # noqa: E402
from features import (  # noqa: E402
    FLOP_FEATURE_SIZE,
    NUM_FLOP,
    NUM_PREFLOP,
    NUM_RIVER,
    PREFLOP_FEATURE_SIZE,
    RIVER_FEATURE_SIZE,
    flop_features,
    preflop_features,
    river_features,
)


def _h(*labels):
    return [Card.new(s) for s in labels]


def test_feature_size_constants():
    assert PREFLOP_FEATURE_SIZE == NUM_PREFLOP + 1 == 16
    assert FLOP_FEATURE_SIZE == NUM_PREFLOP + NUM_FLOP + NUM_PREFLOP * NUM_FLOP + 1 == 144
    assert (
        RIVER_FEATURE_SIZE
        == NUM_PREFLOP + NUM_RIVER + NUM_PREFLOP * NUM_RIVER + 1
        == 128
    )


def test_preflop_features_shape():
    phi = preflop_features(_h("As", "Ah"))
    assert phi.shape == (16,)


def test_flop_features_shape():
    phi = flop_features(_h("As", "Ah"), _h("Qs", "Js", "Ts"))
    assert phi.shape == (144,)


def test_river_features_shape():
    phi = river_features(
        _h("As", "Ah"), _h("Qs", "Js", "Ts", "2h", "3h")
    )
    assert phi.shape == (128,)


def test_bias_is_one():
    assert preflop_features(_h("As", "Ah"))[-1] == 1.0
    assert flop_features(_h("As", "Ah"), _h("Qs", "Js", "Ts"))[-1] == 1.0
    assert river_features(
        _h("As", "Ah"), _h("Qs", "Js", "Ts", "2h", "3h")
    )[-1] == 1.0


def test_preflop_features_sum_is_two():
    """One preflop one-hot bit + bias = 2."""
    phi = preflop_features(_h("As", "Ah"))
    assert phi.sum() == pytest.approx(2.0)


def test_flop_features_sum_is_four():
    """preflop one-hot + flop one-hot + cross one-hot + bias = 4."""
    phi = flop_features(_h("As", "Ah"), _h("Qs", "Js", "Ts"))
    assert phi.sum() == pytest.approx(4.0)


def test_river_features_sum_is_four():
    """preflop one-hot + river one-hot + cross one-hot + bias = 4."""
    phi = river_features(
        _h("As", "Ah"), _h("Qs", "Js", "Ts", "2h", "3h")
    )
    assert phi.sum() == pytest.approx(4.0)


def test_flop_cross_feature_index_correctness():
    """For As2h + AdKc7h, verify cross-feature lights up at the expected slot."""
    hole = _h("As", "2h")
    flop = _h("Ad", "Kc", "7h")
    pre = preflop_bucket(hole)
    flp = flop_bucket(hole, flop)
    expected_cross_idx = NUM_PREFLOP + NUM_FLOP + pre * NUM_FLOP + flp

    phi = flop_features(hole, flop)
    assert phi[pre] == 1.0
    assert phi[NUM_PREFLOP + flp] == 1.0
    assert phi[expected_cross_idx] == 1.0
    # Marginal (preflop), marginal (flop), cross, bias — the only non-zeros.
    nonzero_indices = set(np.flatnonzero(phi))
    assert nonzero_indices == {
        pre,
        NUM_PREFLOP + flp,
        expected_cross_idx,
        FLOP_FEATURE_SIZE - 1,
    }


def test_different_hands_different_features():
    """Two distinct hands must produce distinct feature vectors."""
    phi_aa = preflop_features(_h("As", "Ah"))
    phi_72o = preflop_features(_h("7s", "2h"))
    assert not np.array_equal(phi_aa, phi_72o)

    phi_flop1 = flop_features(_h("As", "Ah"), _h("Qs", "Js", "Ts"))
    phi_flop2 = flop_features(_h("7s", "2h"), _h("2d", "2c", "Ks"))
    assert not np.array_equal(phi_flop1, phi_flop2)
