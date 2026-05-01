"""Richer state abstraction (v2) for tabular UTH RL.

Extends the v1 abstraction by carrying the pre-flop bucket forward into
the flop and river buckets, so the agent can condition post-flop
decisions on starting-hand strength. The flop and river buckets are
joint encodings of (preflop_bucket, postflop_bucket).

Rationale: under v1, two distinct strategic situations — e.g. flopping
"other one pair" with AA preflop vs. flopping the same one pair with a
weak suited connector — collapse into the same flop bucket and share a
Q-row. Splitting them by pre-flop bucket gives the agent room to
distinguish "I have a strong hand and a weak board" from "I have a
weak hand and a weak board", at the cost of inflating the post-flop
state space by 15x.

Encoding: ``joint = pre * NUM_POSTFLOP_BUCKETS + post``. Decoders
return ``(pre, post)`` via divmod.
"""

from typing import List, Tuple

from abstraction import flop_bucket, preflop_bucket, river_bucket


NUM_PREFLOP_BUCKETS = 15
NUM_FLOP_BUCKETS_V1 = 8
NUM_RIVER_BUCKETS_V1 = 7
NUM_FLOP_BUCKETS_V2 = NUM_PREFLOP_BUCKETS * NUM_FLOP_BUCKETS_V1   # 120
NUM_RIVER_BUCKETS_V2 = NUM_PREFLOP_BUCKETS * NUM_RIVER_BUCKETS_V1  # 105


def preflop_bucket_v2(hole_cards: List[int]) -> int:
    """Pre-flop bucket index. Identical to v1 — pre-flop is already
    conditioned on hole cards, so no joint encoding is needed."""
    return preflop_bucket(hole_cards)


def flop_bucket_v2(hole_cards: List[int], flop: List[int]) -> int:
    """Joint (preflop_bucket, flop_bucket) index in [0, 120).

    Encoded as ``pre * NUM_FLOP_BUCKETS_V1 + flp`` so that all v1 flop
    buckets for a given pre-flop bucket occupy a contiguous block of
    indices. ``decode_flop_v2`` is the inverse.
    """
    pre = preflop_bucket(hole_cards)
    flp = flop_bucket(hole_cards, flop)
    return pre * NUM_FLOP_BUCKETS_V1 + flp


def river_bucket_v2(hole_cards: List[int], community: List[int]) -> int:
    """Joint (preflop_bucket, river_bucket) index in [0, 105).

    Encoded as ``pre * NUM_RIVER_BUCKETS_V1 + riv``. ``decode_river_v2``
    is the inverse.
    """
    pre = preflop_bucket(hole_cards)
    riv = river_bucket(hole_cards, community)
    return pre * NUM_RIVER_BUCKETS_V1 + riv


def decode_flop_v2(joint_idx: int) -> Tuple[int, int]:
    """Inverse of ``flop_bucket_v2``. Returns ``(preflop_idx, flop_idx)``."""
    return divmod(joint_idx, NUM_FLOP_BUCKETS_V1)


def decode_river_v2(joint_idx: int) -> Tuple[int, int]:
    """Inverse of ``river_bucket_v2``. Returns ``(preflop_idx, river_idx)``."""
    return divmod(joint_idx, NUM_RIVER_BUCKETS_V1)
