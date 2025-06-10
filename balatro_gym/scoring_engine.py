"""balatro_gym/scoring_engine.py

Hybrid scorer with **dual truth tables**:
  * `chip_table[hand_type, level]` – multiplies the *base chip payout*.
  * `mult_table[hand_type, level]` – multiplies the traditional Mult value.

`apply_consumable()` mutates both tables for Planet cards.
Dynamic jokers / vouchers register with `register_modifier()`; these functions
execute *after* the table lookups, receiving `(score, hand, engine)` and must
return the modified score.

HandType indices must match any other module that references them.
"""
from __future__ import annotations

from enum import IntEnum
from typing import List, Callable, Tuple

import numpy as np

from balatro_gym.planets import Planet, PLANET_MULT

# ---------------------------------------------------------------------------
# HandType enumeration (keep in sync across project)
# ---------------------------------------------------------------------------
class HandType(IntEnum):
    HIGH_CARD       = 0
    ONE_PAIR        = 1
    TWO_PAIR        = 2
    THREE_KIND      = 3
    STRAIGHT        = 4
    FLUSH           = 5
    FULL_HOUSE      = 6
    FOUR_KIND       = 7
    STRAIGHT_FLUSH  = 8

NUM_HAND_TYPES: int = len(HandType)
NUM_LEVELS:     int = 3   # bronze / silver / gold placeholder

# Base chip payout for each HandType (bronze); silver=×1.5, gold=×2
BASE_CHIPS: Tuple[int, ...] = (5, 10, 20, 30, 30, 35, 40, 60, 100)

# Type signature for dynamic joker modifier
ModifierFn = Callable[[float, List[int], "ScoreEngine"], float]

# ---------------------------------------------------------------------------
# ScoreEngine
# ---------------------------------------------------------------------------
class ScoreEngine:
    """Central scoring utility used by BalatroGame and environment."""

    def __init__(self):
        # Two tables initialised to 1.0 (float32 for speed/MEM)
        self.chip_table: np.ndarray = np.ones((NUM_HAND_TYPES, NUM_LEVELS), dtype=np.float32)
        self.mult_table: np.ndarray = np.ones_like(self.chip_table)
        self.modifiers: List[ModifierFn] = []

    # ---------------- handlers ----------------
    def apply_consumable(self, consumable):
        """Mutate tables when a consumable (Planet, Tarot, etc.) is used."""
        if isinstance(consumable, Planet):
            hand_id, factor = PLANET_MULT[consumable]
            self.chip_table[hand_id, :] *= factor
            self.mult_table[hand_id, :] *= factor
        # TODO: add Tarot/Spectral hooks later

    def register_modifier(self, fn: ModifierFn):
        """Register a per-hand modifier (e.g., RideTheBus, GreenJoker)."""
        self.modifiers.append(fn)

    # ---------------- base chip helper ----------------
    @staticmethod
    def _base_chip(hand_type: int, level: int) -> float:
        return BASE_CHIPS[hand_type] * (1.0 + 0.5 * level)  # bronze 1×, silver 1.5×, gold 2×

    # ---------------- public API ----------------
    def score(self, hand_card_ids: List[int], hand_type: int, level: int) -> float:
        """Compute final chip payout for a scored 5‑card hand.

        Args:
            hand_card_ids: list/iterable of 5 ints (0‑51) for dynamic jokers.
            hand_type:     int enum 0‑8 matching HandType.
            level:         0 bronze, 1 silver, 2 gold.
        Returns:
            Final chip float after all multipliers & modifiers.
        """
        # 1. base chip * chip_table multiplier
        base = self._base_chip(hand_type, level) * self.chip_table[hand_type, level]
        # 2. multiply by global hand multiplier
        score = base * self.mult_table[hand_type, level]
        # 3. Apply dynamic jokers / vouchers
        for fn in self.modifiers:
            score = fn(score, hand_card_ids, self)
        return float(score)

    # ---------------- debug helper ----------------
    def table_snapshot(self) -> dict:
        """Return a dict copy of current multiplier tables (for logging)."""
        return {
            "chip_table":  self.chip_table.copy(),
            "mult_table":  self.mult_table.copy(),
        }
