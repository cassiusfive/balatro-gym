# score_with_balatro.py

import numpy as np
from balatro_gym.balatro_game import Card, BalatroGame

# -------------------------------------------------------------------
# Helper: convert an integer [0..51] → a Card(rank, suit) instance
# -------------------------------------------------------------------

def int_to_card(idx: int) -> Card:
    """
    Given an integer idx in 0..51, returns a balatro_game.Card object.
    Ranks run 0..12 (2,3,4,5,6,7,8,9,10,J,Q,K,A)
    Suits run 0..3 (SPADES, CLUBS, HEARTS, DIAMONDS)
    This matches Card.encode() = rank + 13*suit.
    """
    rank_val = idx % 13
    suit_val = idx // 13
    rank_enum = Card.Ranks(rank_val)
    suit_enum = Card.Suits(suit_val)
    return Card(rank_enum, suit_enum)

def score_five_balatro(card_indices: np.ndarray) -> float:
    """
    Given a length-5 NumPy array (or list) of integers in [0..51],
    convert each to a Card object and call BalatroGame._evaluate_hand(...)
    to get the raw chip score. Then normalize by 1000.0 so the result lies in [0,1].
    
    Example usage:
        >>> score_five_balatro(np.array([12, 25, 38, 51, 9]))  # 5 card IDs
        0.827  # whatever the normalized chip value is
    """
    # Convert each integer ID → Card object
    cards = [int_to_card(int(idx)) for idx in card_indices]
    raw_chip = BalatroGame._evaluate_hand(cards)
    return raw_chip / 1000.0

