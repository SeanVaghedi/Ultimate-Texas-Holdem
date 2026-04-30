"""Single-player Ultimate Texas Hold'em hand simulator."""

import random
from typing import Callable, Dict, List, Optional, Tuple

from treys import Card

from hand_eval import evaluate_seven
from payouts import compute_payouts

_RANKS = "23456789TJQKA"
_SUITS = "shdc"
_DECK_TEMPLATE: List[int] = [Card.new(r + s) for r in _RANKS for s in _SUITS]

_LEGAL_ACTIONS: Dict[str, set] = {
    "preflop": {"check", "bet_3x", "bet_4x"},
    "flop": {"check", "bet_2x"},
    "river": {"fold", "bet_1x"},
}

_PLAY_BET_BY_ACTION: Dict[str, int] = {
    "bet_4x": 4,
    "bet_3x": 3,
    "bet_2x": 2,
    "bet_1x": 1,
}

PolicyFn = Callable[[str, Dict], str]


class UTHGame:
    """Single-player Ultimate Texas Hold'em hand simulator.

    Each call to play_hand draws a fresh deck (seeded by the constructor's
    RNG), walks the policy through up to three decisions (preflop, flop,
    river), and returns the showdown outcome. Ante and blind are fixed at
    one unit each; the play bet scales with the action chosen.
    """

    def __init__(self, seed: int):
        """Construct with a fixed RNG seed for reproducibility."""
        self._rng = random.Random(seed)

    def play_hand(self, policy_fn: PolicyFn) -> Dict:
        """Play one full hand and return its outcome.

        Args:
            policy_fn: Invoked at each live decision point. Receives a
                stage name ("preflop", "flop", "river") and a state dict
                with keys: hole_cards, community_cards_revealed,
                actions_taken_so_far. Must return a legal action for the
                stage; an illegal action raises ValueError.

        Returns:
            Dict with keys: net_chips, player_action_history,
            final_player_category, dealer_qualified, player_won.
        """
        deck = list(_DECK_TEMPLATE)
        self._rng.shuffle(deck)

        player_hole = [deck.pop(), deck.pop()]
        dealer_hole = [deck.pop(), deck.pop()]
        community = [deck.pop() for _ in range(5)]

        action_history: List[Tuple[str, str]] = []
        ante = 1
        blind = 1
        play_bet = 0
        bet_placed = False

        # Preflop
        action = self._ask(policy_fn, "preflop", player_hole, [], action_history)
        action_history.append(("preflop", action))
        if action in _PLAY_BET_BY_ACTION:
            play_bet = _PLAY_BET_BY_ACTION[action]
            bet_placed = True

        # Flop — only if the player checked preflop
        if not bet_placed:
            flop = community[:3]
            action = self._ask(
                policy_fn, "flop", player_hole, flop, action_history
            )
            action_history.append(("flop", action))
            if action in _PLAY_BET_BY_ACTION:
                play_bet = _PLAY_BET_BY_ACTION[action]
                bet_placed = True

            # River — only if the player checked through preflop and flop
            if not bet_placed:
                action = self._ask(
                    policy_fn, "river", player_hole, community, action_history
                )
                action_history.append(("river", action))

                if action == "fold":
                    return {
                        "net_chips": -(ante + blind),
                        "player_action_history": action_history,
                        "final_player_category": None,
                        "dealer_qualified": None,
                        "player_won": False,
                    }

                play_bet = _PLAY_BET_BY_ACTION[action]
                bet_placed = True

        # Showdown
        player_rank, player_category = evaluate_seven(player_hole, community)
        dealer_rank, dealer_category = evaluate_seven(dealer_hole, community)
        dealer_qualifies = dealer_category != "HIGH_CARD"

        if player_rank < dealer_rank:
            player_wins: Optional[bool] = True
        elif player_rank > dealer_rank:
            player_wins = False
        else:
            player_wins = None

        net = compute_payouts(
            player_hand_category=player_category,
            player_wins=player_wins,
            dealer_qualifies=dealer_qualifies,
            ante_bet_size=ante,
            blind_bet_size=blind,
            play_bet_size=play_bet,
        )

        return {
            "net_chips": net,
            "player_action_history": action_history,
            "final_player_category": player_category,
            "dealer_qualified": dealer_qualifies,
            "player_won": player_wins is True,
        }

    @staticmethod
    def _ask(
        policy_fn: PolicyFn,
        stage: str,
        hole_cards: List[int],
        community_revealed: List[int],
        history: List[Tuple[str, str]],
    ) -> str:
        state = {
            "hole_cards": list(hole_cards),
            "community_cards_revealed": list(community_revealed),
            "actions_taken_so_far": list(history),
        }
        action = policy_fn(stage, state)
        if action not in _LEGAL_ACTIONS[stage]:
            raise ValueError(
                f"Illegal action {action!r} at stage {stage!r}; "
                f"legal: {sorted(_LEGAL_ACTIONS[stage])}"
            )
        return action
