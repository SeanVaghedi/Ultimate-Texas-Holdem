"""State abstraction for tabular UTH Q-learning.

Each function maps a raw card state into a small finite bucket index
suitable as a key in a Q-table:

- preflop_bucket: 15 buckets indexed by 2 hole cards.
- flop_bucket: 8 buckets indexed by 2 hole cards + 3 flop cards.
- river_bucket: 7 buckets indexed by 2 hole cards + 5 community cards.

All inputs are treys Card integers. Bucket assignment is rule-based and
deterministic; rules are evaluated in order and the first matching rule
wins.
"""

from collections import Counter
from typing import List

from treys import Card, Evaluator

_EVALUATOR = Evaluator()


def _extract_rank(card: int) -> int:
    """Return rank in 2..14 (2=Two, ..., 14=Ace) for a treys card int."""
    return Card.get_rank_int(card) + 2


def _extract_suit(card: int) -> int:
    """Return the treys suit integer for a card."""
    return Card.get_suit_int(card)


def _is_pair_using_hole_card(
    hole_cards: List[int], board_cards: List[int]
) -> bool:
    """True iff the made one-pair hand includes at least one hole card.

    Identifies the paired rank by counting rank occurrences across all
    cards (hole + board); when the caller has already verified a single
    pair (treys class 8), exactly one rank appears twice. Returns False
    if no rank is paired.
    """
    all_ranks = [_extract_rank(c) for c in list(hole_cards) + list(board_cards)]
    counts = Counter(all_ranks)
    paired = next((r for r, n in counts.items() if n == 2), None)
    if paired is None:
        return False
    hole_ranks = [_extract_rank(c) for c in hole_cards]
    return paired in hole_ranks


def _is_top_pair(
    hole_cards: List[int], board_cards: List[int]
) -> bool:
    """True iff the made pair uses a hole card and outranks every board card."""
    if not _is_pair_using_hole_card(hole_cards, board_cards):
        return False
    all_ranks = [_extract_rank(c) for c in list(hole_cards) + list(board_cards)]
    counts = Counter(all_ranks)
    paired = next(r for r, n in counts.items() if n == 2)
    board_ranks = [_extract_rank(c) for c in board_cards]
    return paired >= max(board_ranks)


def _augmented_rank_set(cards: List[int]) -> set:
    """Set of ranks (2..14); when an Ace is present, also include 1.

    The added 1 lets straight-draw checks recognise the wheel
    (A-2-3-4-5) and broadway-extreme draws.
    """
    ranks = {_extract_rank(c) for c in cards}
    if 14 in ranks:
        ranks = ranks | {1}
    return ranks


def _has_oesd(cards: List[int]) -> bool:
    """True iff the cards contain a non-extreme open-ended straight draw.

    OESD = four ranks that form four consecutive integers [a..a+3]
    where 2 <= a and a+3 <= 13. The boundary cases A-2-3-4 and J-Q-K-A
    are deliberately excluded — they only draw on one end and are
    classified as gutshots.
    """
    ranks = _augmented_rank_set(cards)
    for low in range(2, 11):
        if all(r in ranks for r in (low, low + 1, low + 2, low + 3)):
            return True
    return False


def _has_4_in_5_window(cards: List[int]) -> bool:
    """True iff some 5-rank window covers exactly 4 of the cards' ranks.

    Windows considered run from {1,2,3,4,5} (wheel) up through
    {10,J,Q,K,A}. A wider check than gutshot — used together with NOT
    OESD to detect strict gutshot draws.
    """
    ranks = _augmented_rank_set(cards)
    for low in range(1, 11):
        window = set(range(low, low + 5))
        if len(window & ranks) == 4:
            return True
    return False


def _has_gutshot(cards: List[int]) -> bool:
    """True iff there is a gutshot straight draw and no OESD.

    Includes the boundary draws A-2-3-4 (needs 5) and J-Q-K-A (needs T)
    plus any interior one-gap draw such as {5,6,8,9} (needs 7).
    """
    return _has_4_in_5_window(cards) and not _has_oesd(cards)


def _has_4_card_flush(cards: List[int]) -> bool:
    """True iff some single suit appears in exactly four of the cards."""
    suit_counts = Counter(_extract_suit(c) for c in cards)
    return any(count == 4 for count in suit_counts.values())


def _has_high_3card_flush(
    hole_cards: List[int], board_cards: List[int]
) -> bool:
    """True iff some suit appears 3+ times AND a hole card of rank >= J shares it."""
    all_cards = list(hole_cards) + list(board_cards)
    suit_counts = Counter(_extract_suit(c) for c in all_cards)
    for suit, count in suit_counts.items():
        if count < 3:
            continue
        for hc in hole_cards:
            if _extract_suit(hc) == suit and _extract_rank(hc) >= 11:
                return True
    return False


def preflop_bucket(hole_cards: List[int]) -> int:
    """Return a bucket index 0..14 classifying two hole cards.

    Buckets, evaluated in order (first match wins):

        0  Pairs of A, K, or Q.
        1  Pairs of J, T, 9, 8.
        2  Pairs of 7..2.
        3  Suited, ace-high, kicker in {K, Q, J, T}.
        4  Suited, ace-high, kicker in {9..2}.
        5  Offsuit, ace-high, kicker in {K, Q, J, T}.
        6  Offsuit, ace-high, kicker in {9..2}.
        7  Suited, king-high, kicker in {Q, J, T}.
        8  Suited K with kicker <= 9, OR offsuit K with kicker in {Q, J, T}.
        9  Offsuit K with kicker in {9..2}.
       10  Suited QJ, QT, or JT.
       11  Offsuit QJ, QT, or JT.
       12  Suited connector / one-gapper, high in {T..5}, gap in {1, 2}.
       13  Any other suited hand.
       14  Any other offsuit hand.
    """
    r1 = _extract_rank(hole_cards[0])
    r2 = _extract_rank(hole_cards[1])
    suited = _extract_suit(hole_cards[0]) == _extract_suit(hole_cards[1])
    high = max(r1, r2)
    low = min(r1, r2)

    if high == low:
        if high in (14, 13, 12):
            return 0
        if high in (11, 10, 9, 8):
            return 1
        return 2

    if high == 14:
        if low in (13, 12, 11, 10):
            return 3 if suited else 5
        return 4 if suited else 6

    if high == 13:
        if suited:
            if low in (12, 11, 10):
                return 7
            return 8
        if low in (12, 11, 10):
            return 8
        return 9

    if high in (12, 11) and low in (11, 10) and high > low:
        return 10 if suited else 11

    if suited and high in (10, 9, 8, 7, 6, 5) and (high - low) in (1, 2):
        return 12

    return 13 if suited else 14


def flop_bucket(hole_cards: List[int], flop: List[int]) -> int:
    """Return a bucket index 0..7 given two hole cards and three flop cards.

    Buckets:

        0  Made hand straight or better (treys class 1..5).
        1  Three of a kind (class 6).
        2  Two pair (class 7).
        3  Top pair using a hole card: pair (class 8), uses a hole card,
           and paired rank >= every flop rank.
        4  Any other one pair (board pair or below-top hole pair).
        5  High card with 4-flush or open-ended straight draw.
        6  High card with gutshot or 3-flush including hole card >= J.
        7  High card with no draw.
    """
    cards = list(hole_cards) + list(flop)
    rank = _EVALUATOR.evaluate(list(flop), list(hole_cards))
    cls = _EVALUATOR.get_rank_class(rank)

    if cls <= 5:
        return 0
    if cls == 6:
        return 1
    if cls == 7:
        return 2
    if cls == 8:
        return 3 if _is_top_pair(hole_cards, flop) else 4

    if _has_4_card_flush(cards) or _has_oesd(cards):
        return 5
    if _has_gutshot(cards) or _has_high_3card_flush(hole_cards, flop):
        return 6
    return 7


def river_bucket(hole_cards: List[int], community: List[int]) -> int:
    """Return a bucket index 0..6 given two hole cards and five community cards.

    Buckets:

        0  Straight flush or royal flush (treys class 1).
        1  Four of a kind or full house (classes 2, 3).
        2  Flush or straight (classes 4, 5).
        3  Three of a kind or two pair (classes 6, 7).
        4  One pair (class 8) using a hole card whose rank >= every
           community rank.
        5  Any other one pair.
        6  High card (class 9).
    """
    rank = _EVALUATOR.evaluate(list(community), list(hole_cards))
    cls = _EVALUATOR.get_rank_class(rank)

    if cls in (0, 1):
        return 0
    if cls in (2, 3):
        return 1
    if cls in (4, 5):
        return 2
    if cls in (6, 7):
        return 3
    if cls == 8:
        return 4 if _is_top_pair(hole_cards, community) else 5
    return 6
