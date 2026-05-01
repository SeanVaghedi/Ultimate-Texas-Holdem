"""Tests for src/abstraction_v2.py (joint preflop x postflop bucketing)."""

import random
import sys
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstraction import flop_bucket, preflop_bucket, river_bucket  # noqa: E402
from abstraction_v2 import (  # noqa: E402
    NUM_FLOP_BUCKETS_V1,
    NUM_FLOP_BUCKETS_V2,
    NUM_PREFLOP_BUCKETS,
    NUM_RIVER_BUCKETS_V1,
    NUM_RIVER_BUCKETS_V2,
    decode_flop_v2,
    decode_river_v2,
    flop_bucket_v2,
    preflop_bucket_v2,
    river_bucket_v2,
)


_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_DECK_TEMPLATE = [Card.new(r + s) for r in _RANKS for s in _SUITS]


def _h(*labels):
    return [Card.new(s) for s in labels]


def test_constants():
    assert NUM_FLOP_BUCKETS_V2 == NUM_PREFLOP_BUCKETS * NUM_FLOP_BUCKETS_V1
    assert NUM_RIVER_BUCKETS_V2 == NUM_PREFLOP_BUCKETS * NUM_RIVER_BUCKETS_V1
    assert NUM_FLOP_BUCKETS_V2 == 120
    assert NUM_RIVER_BUCKETS_V2 == 105


def test_preflop_v2_matches_v1():
    """preflop_bucket_v2 is identical to preflop_bucket for any hole pair."""
    rng = random.Random(0)
    for _ in range(100):
        deck = list(_DECK_TEMPLATE)
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        assert preflop_bucket_v2(hole) == preflop_bucket(hole)


def test_flop_v2_index_range():
    """For 100 random hands, flop_bucket_v2 returns indices in [0, 120)."""
    rng = random.Random(1)
    for _ in range(100):
        deck = list(_DECK_TEMPLATE)
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        flop = [deck.pop() for _ in range(3)]
        idx = flop_bucket_v2(hole, flop)
        assert 0 <= idx < NUM_FLOP_BUCKETS_V2


def test_river_v2_index_range():
    """For 100 random hands, river_bucket_v2 returns indices in [0, 105)."""
    rng = random.Random(2)
    for _ in range(100):
        deck = list(_DECK_TEMPLATE)
        rng.shuffle(deck)
        hole = [deck.pop(), deck.pop()]
        community = [deck.pop() for _ in range(5)]
        idx = river_bucket_v2(hole, community)
        assert 0 <= idx < NUM_RIVER_BUCKETS_V2


def test_flop_encode_decode_roundtrip():
    """Every (pre, flp) pair encodes/decodes losslessly."""
    for pre in range(NUM_PREFLOP_BUCKETS):
        for flp in range(NUM_FLOP_BUCKETS_V1):
            joint = pre * NUM_FLOP_BUCKETS_V1 + flp
            assert decode_flop_v2(joint) == (pre, flp)


def test_river_encode_decode_roundtrip():
    """Every (pre, riv) pair encodes/decodes losslessly."""
    for pre in range(NUM_PREFLOP_BUCKETS):
        for riv in range(NUM_RIVER_BUCKETS_V1):
            joint = pre * NUM_RIVER_BUCKETS_V1 + riv
            assert decode_river_v2(joint) == (pre, riv)


def test_premium_pair_flopped_trips():
    """AA preflop + flopping trips → preflop=0, flop_v1=1, joint=1."""
    hole = _h("As", "Ah")
    flop = _h("Ad", "Kc", "2h")
    assert preflop_bucket_v2(hole) == 0
    assert flop_bucket(hole, flop) == 1
    assert flop_bucket_v2(hole, flop) == 0 * NUM_FLOP_BUCKETS_V1 + 1
    assert flop_bucket_v2(hole, flop) == 1


def test_junk_offsuit_flopped_trips():
    """27o preflop + flopping trips on a 22 board → preflop=14, flop_v1=1, joint=113."""
    hole = _h("2s", "7h")
    flop = _h("2d", "2c", "Ks")
    assert preflop_bucket_v2(hole) == 14
    assert flop_bucket(hole, flop) == 1
    assert flop_bucket_v2(hole, flop) == 14 * NUM_FLOP_BUCKETS_V1 + 1
    assert flop_bucket_v2(hole, flop) == 113


def test_v2_distinguishes_what_v1_conflated():
    """Same v1 flop bucket but different preflop bucket → distinct v2 indices.

    AA + flopped trips and 27o + flopped trips both land in v1 flop bucket 1,
    but v2 places them in different cells (1 vs. 113), giving the agent
    room to learn distinct policies for each case.
    """
    hole_aa = _h("As", "Ah")
    flop_aa = _h("Ad", "Kc", "2h")

    hole_junk = _h("2s", "7h")
    flop_junk = _h("2d", "2c", "Ks")

    v1_aa = flop_bucket(hole_aa, flop_aa)
    v1_junk = flop_bucket(hole_junk, flop_junk)
    assert v1_aa == v1_junk == 1, (
        "Setup precondition: both should be v1 flop bucket 1 (trips)"
    )

    v2_aa = flop_bucket_v2(hole_aa, flop_aa)
    v2_junk = flop_bucket_v2(hole_junk, flop_junk)
    assert v2_aa != v2_junk, (
        f"v2 must distinguish AA-trips ({v2_aa}) from junk-trips ({v2_junk})"
    )
    assert decode_flop_v2(v2_aa) == (0, 1)
    assert decode_flop_v2(v2_junk) == (14, 1)


def test_river_v2_distinguishes_preflop_conditioning():
    """Two distinct preflop buckets reaching the same river bucket get different joints."""
    hole_premium = _h("As", "Ah")
    community_premium = _h("Ad", "Kc", "2h", "5d", "9s")

    hole_junk = _h("2s", "7h")
    community_junk = _h("2d", "2c", "Ks", "5d", "9s")

    pre_premium = preflop_bucket(hole_premium)
    pre_junk = preflop_bucket(hole_junk)
    assert pre_premium != pre_junk

    v2_premium = river_bucket_v2(hole_premium, community_premium)
    v2_junk = river_bucket_v2(hole_junk, community_junk)

    p1, r1 = decode_river_v2(v2_premium)
    p2, r2 = decode_river_v2(v2_junk)
    assert p1 == pre_premium
    assert p2 == pre_junk
    assert v2_premium != v2_junk
