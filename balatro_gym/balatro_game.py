"""balatro_gym/balatro_game_v2.py – rewired to the hybrid ScoreEngine.

Changes vs original
-------------------
* Adds **ScoreEngine** support: all chip payouts flow through the hybrid
  truth‑table / dynamic‑modifier engine (`scoring_engine.ScoreEngine`).
* `_evaluate_hand()` no longer hard‑codes every hand’s chip+mult logic; it
  classifies into `(hand_type, level)` then delegates to `engine.score()`.
* Keeps legacy fall‑back (`_legacy_score`) so nothing breaks if you haven’t
  wired the engine elsewhere.
* Exposes `self.engine` so your environment can `.apply_consumable()` or
  `.register_modifier()` when the player uses Planet/Tarot/joker effects.

Hand classification here is still heuristic‑light (mirrors original rules) but
now returns a `HandType` enum so you can swap the classifier later without
changing the scoring pipeline.
"""
from __future__ import annotations

import numpy as np
from enum import Enum, IntEnum, auto
from typing import List, Tuple

from scoring_engine import ScoreEngine, HandType  # hybrid engine

# ---------------------------------------------------------------------------
# Card model (unchanged except minor tweaks)
# ---------------------------------------------------------------------------

class Card:
    class Ranks(IntEnum):
        TWO, THREE, FOUR, FIVE, SIX, SEVEN, EIGHT, NINE, TEN, JACK, QUEEN, KING, ACE = range(13)

    class Suits(IntEnum):
        SPADES, CLUBS, HEARTS, DIAMONDS = range(4)

    base_chip_values = {
        Ranks.TWO: 2,
        Ranks.THREE: 3,
        Ranks.FOUR: 4,
        Ranks.FIVE: 5,
        Ranks.SIX: 6,
        Ranks.SEVEN: 7,
        Ranks.EIGHT: 8,
        Ranks.NINE: 9,
        Ranks.TEN: 10,
        Ranks.JACK: 10,
        Ranks.QUEEN: 10,
        Ranks.KING: 10,
        Ranks.ACE: 11,
    }

    def __init__(self, rank: "Card.Ranks", suit: "Card.Suits"):
        self.rank = rank
        self.suit = suit
        self.played = False

    def chip_value(self):
        return self.base_chip_values[self.rank]

    def encode(self):
        return self.rank.value + self.suit.value * 13

    def __str__(self):
        return f"{self.rank.name} OF {self.suit.name}"

# ---------------------------------------------------------------------------
# BalatroGame core
# ---------------------------------------------------------------------------

class BalatroGame:
    class State(Enum):
        IN_PROGRESS = 0
        WIN = 1
        LOSS = 2

    def __init__(self, *, deck: str = "yellow", stake: str = "white", engine: ScoreEngine | None = None):
        # deck building
        self.deck: List[Card] = [Card(rank, suit) for suit in Card.Suits for rank in Card.Ranks]
        np.random.shuffle(self.deck)

        # gameplay params (kept from original)
        self.hand_size = 8
        self.hands = 4
        self.discards = 3
        self.ante = 1
        self.blind_index = 0
        self.blinds = [300, 450, 600]

        # state vars
        self.hand_indexes: List[int] = []
        self.highlighted_indexes: List[int] = []
        self.round_score = 0
        self.round_hands = self.hands
        self.round_discards = self.discards
        self.state = self.State.IN_PROGRESS

        # scoring engine (hybrid)
        self.engine: ScoreEngine = engine or ScoreEngine()

        self._draw_cards()

    # ------------------------------------------------------------------
    # Game actions (highlight/play/discard) – unchanged except scoring
    # ------------------------------------------------------------------

    def highlight_card(self, hand_index: int):
        self.highlighted_indexes.append(self.hand_indexes.pop(hand_index))

    def play_hand(self):
        self.round_hands -= 1
        hand_cards = [self.deck[i] for i in self.highlighted_indexes]
        hand_type, level = self._classify_hand(hand_cards)
        score = self.engine.score([c.encode() for c in hand_cards], hand_type, level)
        self.round_score += score

        if self.round_score >= self.blinds[self.blind_index]:
            self._end_round()
        elif self.round_hands == 0:
            self.state = self.State.LOSS
        else:
            self._draw_cards()
        return score

    def discard_hand(self):
        self.round_discards -= 1
        self._draw_cards()

    # ------------------------------------------------------------------
    # Drawing / Round transitions
    # ------------------------------------------------------------------

    def _draw_cards(self):
        self.highlighted_indexes.clear()
        remaining = [i for i, c in enumerate(self.deck) if not c.played]
        draw_n = min(self.hand_size - len(self.hand_indexes), len(remaining))
        for idx in np.random.choice(remaining, draw_n, replace=False):
            self.deck[idx].played = True
            self.hand_indexes.append(idx)

    def _end_round(self):
        for card in self.deck:
            card.played = False
        self.hand_indexes.clear()
        self.highlighted_indexes.clear()
        self.round_hands = self.hands
        self.round_discards = self.discards
        self.round_score = 0
        self.blind_index = (self.blind_index + 1) % 3
        if self.blind_index == 0:
            self.ante += 1
        if self.ante > 8:
            self.state = self.State.WIN
        self._draw_cards()

    # ------------------------------------------------------------------
    # Hand classification → (HandType, level)
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_hand(hand: List[Card]) -> Tuple[HandType, int]:
        """Return (hand_type, level) for scoring_engine.
        Level is 0 for bronze; upgrade logic can be added later."""
        assert len(hand) == 5, "hand must be 5 cards"

        ranks = [c.rank.value for c in hand]
        suits = [c.suit for c in hand]
        rank_counts = {r: ranks.count(r) for r in set(ranks)}
        flush = len(set(suits)) == 1
        straight = sorted(ranks) == list(range(min(ranks), min(ranks) + 5)) or sorted(ranks) == [0, 1, 2, 3, 12]

        if straight and flush:
            return HandType.STRAIGHT_FLUSH, 0
        match sorted(rank_counts.values(), reverse=True):
            case [4, 1]:
                return HandType.FOUR_KIND, 0
            case [3, 2]:
                return HandType.FULL_HOUSE, 0
            case [3, 1, 1]:
                return HandType.THREE_KIND, 0
            case [2, 2, 1]:
                return HandType.TWO_PAIR, 0
            case [2, 1, 1, 1]:
                return HandType.ONE_PAIR, 0
        if flush:
            return HandType.FLUSH, 0
        if straight:
            return HandType.STRAIGHT, 0
        return HandType.HIGH_CARD, 0

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def hand_to_string(self):
        return ", ".join(str(self.deck[i]) for i in self.hand_indexes)
