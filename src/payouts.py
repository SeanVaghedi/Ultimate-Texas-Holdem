"""Ultimate Texas Hold'em payout tables and per-hand chip computation."""

from typing import Optional

BLIND_BONUS = {
    "ROYAL_FLUSH": 500,
    "STRAIGHT_FLUSH": 50,
    "FOUR_OF_A_KIND": 10,
    "FULL_HOUSE": 3,
    "FLUSH": 1.5,
    "STRAIGHT": 1,
    "THREE_OF_A_KIND": 0,
    "TWO_PAIR": 0,
    "PAIR": 0,
    "HIGH_CARD": 0,
}


def compute_payouts(
    player_hand_category: str,
    player_wins: Optional[bool],
    dealer_qualifies: bool,
    ante_bet_size: float,
    blind_bet_size: float,
    play_bet_size: float,
) -> float:
    """Net chip change for a single UTH hand that reaches showdown.

    Fold outcomes are not handled here; the caller resolves a fold
    directly (forfeit of ante + blind, no play bet at risk).

    Args:
        player_hand_category: One of the BLIND_BONUS keys (e.g. "FLUSH").
        player_wins: True if the player won showdown, False if lost,
            None on tie.
        dealer_qualifies: True if the dealer made at least a pair.
        ante_bet_size: Ante posted (typically 1).
        blind_bet_size: Blind posted (typically 1).
        play_bet_size: Play bet placed during the hand. Must be > 0;
            reaching showdown requires a committed play bet.

    Returns:
        Net chips won (positive) or lost (negative) on the hand.
    """
    assert play_bet_size > 0, (
        "compute_payouts is for showdowns only; "
        "fold outcomes are handled by the caller"
    )

    if player_wins is None:
        return 0.0

    if not dealer_qualifies:
        ante_payoff = 0.0
    elif player_wins:
        ante_payoff = ante_bet_size
    else:
        ante_payoff = -ante_bet_size

    play_payoff = play_bet_size if player_wins else -play_bet_size

    if player_wins:
        blind_payoff = blind_bet_size * BLIND_BONUS[player_hand_category]
    else:
        blind_payoff = -blind_bet_size

    return ante_payoff + play_payoff + blind_payoff
