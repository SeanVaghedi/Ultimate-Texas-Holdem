"""Wrapper around treys for 7-card poker hand evaluation."""

from typing import List, Tuple

from treys import Evaluator

_EVALUATOR = Evaluator()

_TREYS_CLASS_TO_CATEGORY = {
    "Royal Flush": "ROYAL_FLUSH",
    "Straight Flush": "STRAIGHT_FLUSH",
    "Four of a Kind": "FOUR_OF_A_KIND",
    "Full House": "FULL_HOUSE",
    "Flush": "FLUSH",
    "Straight": "STRAIGHT",
    "Three of a Kind": "THREE_OF_A_KIND",
    "Two Pair": "TWO_PAIR",
    "Pair": "PAIR",
    "High Card": "HIGH_CARD",
}


def evaluate_seven(
    hole_cards: List[int],
    community_cards: List[int],
) -> Tuple[int, str]:
    """Evaluate the best 5-card poker hand from 2 hole + 5 community cards.

    Args:
        hole_cards: Two treys Card integers.
        community_cards: Five treys Card integers.

    Returns:
        (rank, category) where rank is the treys integer rank (lower is
        stronger; 1 is the unique royal flush) and category is one of the
        BLIND_BONUS keys in payouts.py.
    """
    rank = _EVALUATOR.evaluate(community_cards, hole_cards)
    class_str = _EVALUATOR.class_to_string(_EVALUATOR.get_rank_class(rank))
    return rank, _TREYS_CLASS_TO_CATEGORY[class_str]
