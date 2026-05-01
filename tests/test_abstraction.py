"""Unit tests for src/abstraction.py bucket functions."""

import sys
from itertools import combinations
from pathlib import Path

import pytest
from treys import Card

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from abstraction import flop_bucket, preflop_bucket, river_bucket  # noqa: E402


def _h(*labels):
    return [Card.new(s) for s in labels]


@pytest.mark.parametrize(
    "labels,expected",
    [
        (("As", "Ah"), 0),
        (("Ks", "Kh"), 0),
        (("Qs", "Qh"), 0),
        (("Js", "Jh"), 1),
        (("8s", "8h"), 1),
        (("7s", "7h"), 2),
        (("2s", "2h"), 2),
        (("As", "Ks"), 3),
        (("As", "Ts"), 3),
        (("As", "2s"), 4),
        (("As", "Kh"), 5),
        (("As", "Th"), 5),
        (("As", "2h"), 6),
        (("Ks", "Qs"), 7),
        (("Ks", "9s"), 8),
        (("Ks", "Qh"), 8),
        (("Ks", "9h"), 9),
        (("Qs", "Js"), 10),
        (("Js", "Ts"), 10),
        (("Qs", "Jh"), 11),
        (("9s", "8s"), 12),
        (("8s", "7s"), 12),
        (("7s", "4s"), 13),
    ],
)
def test_preflop_bucket_examples(labels, expected):
    assert preflop_bucket(_h(*labels)) == expected


def test_preflop_bucket_exhaustive_coverage():
    deck = [Card.new(r + s) for r in "23456789TJQKA" for s in "shdc"]
    for c1, c2 in combinations(deck, 2):
        b = preflop_bucket([c1, c2])
        assert 0 <= b <= 14, f"bucket {b} out of range for cards {c1},{c2}"


def test_flop_set():
    assert flop_bucket(_h("6s", "6h"), _h("6c", "Kd", "2h")) == 1


def test_flop_two_pair():
    assert flop_bucket(_h("As", "Kh"), _h("Ah", "Kd", "2c")) == 2


def test_flop_top_pair():
    assert flop_bucket(_h("As", "7h"), _h("Ad", "5c", "2h")) == 3


def test_flop_middle_pair():
    assert flop_bucket(_h("7s", "2h"), _h("7d", "Kc", "5h")) == 4


def test_flop_oesd_no_pair():
    assert flop_bucket(_h("7s", "8h"), _h("6c", "9d", "2h")) == 5


def test_flop_4flush_no_pair():
    assert flop_bucket(_h("As", "2s"), _h("5s", "9s", "Kh")) == 5


def test_flop_gutshot_no_pair():
    assert flop_bucket(_h("7s", "8h"), _h("6c", "Td", "2c")) == 6


def test_flop_air():
    assert flop_bucket(_h("7s", "2h"), _h("Kc", "9d", "5c")) == 7


def test_flop_royal_flush_made():
    # Hole AsKs, flop QsJsTs — royal already made on the flop
    assert flop_bucket(_h("As", "Ks"), _h("Qs", "Js", "Ts")) == 0


def test_river_royal_flush():
    assert river_bucket(_h("As", "Ks"), _h("Qs", "Js", "Ts", "2h", "3h")) == 0


def test_river_quads():
    assert river_bucket(_h("As", "Ah"), _h("Ad", "Ac", "2h", "3c", "4d")) == 1


def test_river_flush():
    assert river_bucket(_h("As", "2s"), _h("5s", "9s", "Ks", "3h", "4d")) == 2


def test_river_trips():
    assert river_bucket(_h("7s", "7h"), _h("7d", "Kc", "2h", "3d", "4c")) == 3


def test_river_top_pair():
    assert river_bucket(_h("As", "2h"), _h("Ad", "5c", "9d", "3h", "7c")) == 4


def test_river_other_pair_board_only():
    assert river_bucket(_h("2s", "3h"), _h("7d", "7c", "9d", "Th", "Kc")) == 5


def test_river_high_card():
    assert river_bucket(_h("2s", "5h"), _h("7d", "9c", "Jh", "Kd", "3c")) == 6
