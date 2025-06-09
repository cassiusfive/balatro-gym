"""scoring_engine.py – Hybrid truth‑table + dynamic modifiers

A lightweight yet extensible scoring core for Balatro RL.

Why a hybrid?
-------------
* **Truth table** (numpy ndarray) encodes the *static* value of each
  `HandType` × `Level` combination after all *global* effects (Planets,
  Spectral cards that permanently change ordering, etc.).
* **Dynamic modifiers** – a list of callables executed at score‑time for
  *contextual* bonuses (jokers that look at suit counts, remaining discards,
  etc.).  These are cheap Python functions because their number is small.

Schema
------
`HandType` – enum 0‑8 (High‑Card .. Straight‑Flush).  `Level` – 0‑2 where
0 = bronze, 1 = silver, 2 = gold (placeholder, adjust to real game).

The table stores a *multiplier* (float32).  Chips are computed as:
    chips = base_hand_value(hand_type, level) * table[hand_type, level]

Consumables mutate the table in‑place:
    engine.apply_consumable(Planet.MARS)  # e.g., ×1.4 to Straight

Dynamic jokers register a function:
    engine.register_modifier(lambda score, hand, ctx: score + 30)

Usage
-----
    engine = ScoreEngine()
    engine.apply_consumable(Planet.MARS)
    engine.register_modifier(RideTheBus())

    chips = engine.score(hand, context)
"""
from __future__ import annotations

import numpy as np
from enum import IntEnum, auto
from typing import Callable, List, Tuple

# ---------------------------------------------------------------------------
# HandType & Level enums (minimal placeholder)
# ---------------------------------------------------------------------------

class HandType(IntEnum):
    HIGH_CARD = 0
    ONE_PAIR = auto()
    TWO_PAIR = auto()
    THREE_KIND = auto()
    STRAIGHT = auto()
    FLUSH = auto()
    FULL_HOUSE = auto()
    FOUR_KIND = auto()
    STRAIGHT_FLUSH = auto()

NUM_HAND_TYPES = len(HandType)
NUM_LEVELS = 3  # bronze, silver, gold  (simplified)

# ---------------------------------------------------------------------------
# Planet effects (subset)
# ---------------------------------------------------------------------------

class Planet(IntEnum):
    MERCURY = 0  # Straight scores ×1.2
    VENUS = 1    # Flush ×1.2
    MARS = 2     # Straight ×1.4
    JUPITER = 3  # Straight Flush ×1.3
    # ... etc.

PLANET_MULT: dict[Planet, Tuple[HandType, float]] = {
    Planet.MERCURY: (HandType.STRAIGHT, 1.2),
    Planet.VENUS: (HandType.FLUSH, 1.2),
    Planet.MARS: (HandType.STRAIGHT, 1.4),
    Planet.JUPITER: (HandType.STRAIGHT_FLUSH, 1.3),
}

# ---------------------------------------------------------------------------
# ScoreEngine
# ---------------------------------------------------------------------------

ModifierFn = Callable[[float, List[int], "ScoreEngine"], float]

class ScoreEngine:
    """Hybrid truth‑table scorer."""

    def __init__(self):
        # base multipliers (float32) – start at 1.0
        self.table = np.ones((NUM_HAND_TYPES, NUM_LEVELS), dtype=np.float32)
        # dynamic modifier functions
        self.modifiers: List[ModifierFn] = []

    # ------------- consumable hooks (truth‑table mutations) -------------

    def apply_consumable(self, consumable):
        """Dispatch based on consumable type."""
        if isinstance(consumable, Planet):
            hand, mult = PLANET_MULT[consumable]
            self.table[hand, :] *= mult
        # TODO: Tarot, Spectral, etc.

    # ------------- dynamic modifiers ------------------------------------

    def register_modifier(self, fn: ModifierFn):
        self.modifiers.append(fn)

    # ------------- scoring ----------------------------------------------

    def base_hand_value(self, hand_type: HandType, level: int) -> float:
        """Return canonical chip value before multipliers (placeholder)."""
        BASE = {
            HandType.HIGH_CARD: 10,
            HandType.ONE_PAIR: 20,
            HandType.TWO_PAIR: 30,
            HandType.THREE_KIND: 40,
            HandType.STRAIGHT: 50,
            HandType.FLUSH: 60,
            HandType.FULL_HOUSE: 80,
            HandType.FOUR_KIND: 120,
            HandType.STRAIGHT_FLUSH: 300,
        }
        return BASE[hand_type] * (1 + level * 0.5)  # bronze/silver/gold scaling

    def score(self, hand_cards: List[int], hand_type: HandType, level: int) -> float:
        """Compute final chip score for a hand."""
        score = self.base_hand_value(hand_type, level)
        score *= self.table[hand_type, level]

        for fn in self.modifiers:
            score = fn(score, hand_cards, self)
        return score

# ---------------------------------------------------------------------------
# Example dynamic modifier implementation
# ---------------------------------------------------------------------------

class RideTheBus:
    """Joker that gains +1 Mult per hand without a face card."""

    def __init__(self):
        self.mult = 1.0

    def __call__(self, base_score: float, hand: List[int], engine: ScoreEngine) -> float:
        if all((c % 13) < 10 for c in hand):  # no face cards
            self.mult += 1.0
        return base_score * self.mult

# ---------------------------------------------------------------------------
# Unit‑test demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    engine = ScoreEngine()
    engine.apply_consumable(Planet.MARS)
    engine.register_modifier(RideTheBus())

    dummy_hand = [0, 1, 2, 3, 4]  # unsuited straight (ranks 2‑6)
    chips = engine.score(dummy_hand, HandType.STRAIGHT, 0)
    print("Straight score with Mars + RideTheBus: ", chips)
