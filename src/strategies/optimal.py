"""Hard-coded near-optimal strategy for Ultimate Texas Hold'em.

Used as a simulator sanity baseline: a faithful UTH simulator paired with
this policy should land in the 2-3% house-edge range, close to the
published optimal of ~2.185%. The rules below intentionally simplify the
true optimal strategy (notably: never raise 3x preflop).
"""

from typing import Dict, List

from treys import Card, Evaluator

from hand_eval import evaluate_seven

_EVALUATOR = Evaluator()

# Treys rank ints — 2=0, 3=1, ..., 8=6, 9=7, T=8, J=9, Q=10, K=11, A=12.


def optimal_policy(stage: str, state: Dict) -> str:
    """Near-optimal action for the given stage and state.

    Matches the UTHGame.policy_fn signature: returns a legal action string
    for the supplied stage ("preflop", "flop", or "river").
    """
    hole = state["hole_cards"]
    community = state["community_cards_revealed"]

    if stage == "preflop":
        return "bet_4x" if _preflop_qualifies_4x(hole) else "check"
    if stage == "flop":
        return "bet_2x" if _flop_should_raise(hole, community) else "check"
    if stage == "river":
        return "bet_1x" if _river_should_bet(hole, community) else "fold"
    raise ValueError(f"Unknown stage: {stage!r}")


def _preflop_qualifies_4x(hole_cards: List[int]) -> bool:
    """Simplified preflop 4x raise rule (else check; never raise 3x)."""
    r1 = Card.get_rank_int(hole_cards[0])
    r2 = Card.get_rank_int(hole_cards[1])
    suited = Card.get_suit_int(hole_cards[0]) == Card.get_suit_int(hole_cards[1])
    high, low = (r1, r2) if r1 >= r2 else (r2, r1)

    # Pocket pair of 3s or higher (3 = rank 1; pocket 2s = rank 0 excluded)
    if high == low:
        return high >= 1

    # Any ace
    if high == 12:
        return True

    # King: any kicker if suited; offsuit kicker must be 5+ (5 = rank 3)
    if high == 11:
        return suited or low >= 3

    # Queen: kicker 8+ (8 = rank 6) any suit, plus Q6s and Q7s
    if high == 10:
        return low >= 6 or (suited and low in (4, 5))

    # Jack: JT (T = rank 8) any suit, plus J9s (rank 7) and J8s (rank 6)
    if high == 9:
        return low == 8 or (suited and low in (6, 7))

    return False


def _flop_should_raise(hole_cards: List[int], flop_cards: List[int]) -> bool:
    """Simplified flop 2x raise rule (else check)."""
    rank = _EVALUATOR.evaluate(hole_cards, flop_cards)
    rank_class = _EVALUATOR.get_rank_class(rank)

    # Two pair (class 7) or stronger (classes 1-7)
    if rank_class <= 7:
        return True

    # Pair (class 8) — only if it's top pair or an overpair, with a hole
    # card actually involved (paired-board pair with no hole help doesn't
    # count).
    if rank_class == 8 and _is_top_pair_or_overpair(hole_cards, flop_cards):
        return True

    # 4-card flush draw with at least one J/Q/K/A hole card of that suit
    if _has_high_4card_flush(hole_cards, flop_cards):
        return True

    return False


def _river_should_bet(
    hole_cards: List[int], community_cards: List[int]
) -> bool:
    """Bet 1x at river iff the player has at least a pair (else fold)."""
    _, category = evaluate_seven(hole_cards, community_cards)
    return category != "HIGH_CARD"


def _is_top_pair_or_overpair(
    hole_cards: List[int], flop_cards: List[int]
) -> bool:
    """Pocket overpair to the flop, or top pair using a hole card."""
    hole_ranks = [Card.get_rank_int(c) for c in hole_cards]
    flop_ranks = [Card.get_rank_int(c) for c in flop_cards]
    max_flop = max(flop_ranks)

    if hole_ranks[0] == hole_ranks[1] and hole_ranks[0] > max_flop:
        return True
    return max_flop in hole_ranks


def _has_high_4card_flush(
    hole_cards: List[int], flop_cards: List[int]
) -> bool:
    """Four cards to a flush with at least one J/Q/K/A hole card of that suit."""
    suits = [Card.get_suit_int(c) for c in hole_cards + flop_cards]
    for suit in set(suits):
        if suits.count(suit) < 4:
            continue
        for hc in hole_cards:
            if (
                Card.get_suit_int(hc) == suit
                and Card.get_rank_int(hc) >= 9  # J or higher
            ):
                return True
    return False
