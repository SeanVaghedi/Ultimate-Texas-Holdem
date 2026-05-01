"""Feature extraction for linear function approximation in UTH RL.

Each stage has its own feature space; weights are stored per (stage,
action) so the feature vector itself is action-blocked (no action
indicator goes inside phi).

Layout per stage (in fixed positional order, matching the offsets used
inside the helpers below):

    preflop (16):
        [0..14]   one-hot over 15 pre-flop buckets
        [15]      bias = 1.0

    flop (144):
        [0..14]   marginal one-hot over 15 pre-flop buckets
        [15..22]  marginal one-hot over 8 flop buckets
        [23..142] cross one-hot over the 120 (preflop, flop) pairs
        [143]     bias = 1.0

    river (128):
        [0..14]   marginal one-hot over 15 pre-flop buckets
        [15..21]  marginal one-hot over 7 river buckets
        [22..126] cross one-hot over the 105 (preflop, river) pairs
        [127]     bias = 1.0

The cross block matches the v2-tabular abstraction; the marginal
blocks let the model share information across cells that are sparsely
visited under the cross block alone.
"""

from typing import List

import numpy as np

from abstraction import flop_bucket, preflop_bucket, river_bucket


NUM_PREFLOP = 15
NUM_FLOP = 8
NUM_RIVER = 7

PREFLOP_FEATURE_SIZE = NUM_PREFLOP + 1                                  # 16
FLOP_FEATURE_SIZE = NUM_PREFLOP + NUM_FLOP + NUM_PREFLOP * NUM_FLOP + 1   # 144
RIVER_FEATURE_SIZE = NUM_PREFLOP + NUM_RIVER + NUM_PREFLOP * NUM_RIVER + 1  # 128


def preflop_features(hole_cards: List[int]) -> np.ndarray:
    """Return shape (16,): preflop one-hot + bias."""
    phi = np.zeros(PREFLOP_FEATURE_SIZE, dtype=np.float64)
    pre = preflop_bucket(hole_cards)
    phi[pre] = 1.0
    phi[-1] = 1.0
    return phi


def flop_features(hole_cards: List[int], flop: List[int]) -> np.ndarray:
    """Return shape (144,): marginal preflop + marginal flop + cross + bias."""
    phi = np.zeros(FLOP_FEATURE_SIZE, dtype=np.float64)
    pre = preflop_bucket(hole_cards)
    flp = flop_bucket(hole_cards, flop)
    phi[pre] = 1.0
    phi[NUM_PREFLOP + flp] = 1.0
    cross_idx = NUM_PREFLOP + NUM_FLOP + pre * NUM_FLOP + flp
    phi[cross_idx] = 1.0
    phi[-1] = 1.0
    return phi


def river_features(hole_cards: List[int], community: List[int]) -> np.ndarray:
    """Return shape (128,): marginal preflop + marginal river + cross + bias."""
    phi = np.zeros(RIVER_FEATURE_SIZE, dtype=np.float64)
    pre = preflop_bucket(hole_cards)
    riv = river_bucket(hole_cards, community)
    phi[pre] = 1.0
    phi[NUM_PREFLOP + riv] = 1.0
    cross_idx = NUM_PREFLOP + NUM_RIVER + pre * NUM_RIVER + riv
    phi[cross_idx] = 1.0
    phi[-1] = 1.0
    return phi
