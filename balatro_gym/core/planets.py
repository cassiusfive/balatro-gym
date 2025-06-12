"""balatro_gym/planets.py – expanded planet roster

Nine Celestial consumables mirrored from Balatro’s in‑game Planet cards.
Each planet multiplies both *chip* and *mult* tables for the corresponding
`HandType` when `ScoreEngine.apply_consumable()` is called.

Numbers for the last five planets (Earth, Saturn, Uranus, Neptune, Pluto) are
place‑holders (1.3×) – fine‑tune once exact balance data is confirmed.
"""
from __future__ import annotations

from enum import IntEnum, auto
from typing import Dict, Tuple

# HandType ids (sync with scoring_engine.HandType)
HAND_HIGH_CARD      = 0
HAND_ONE_PAIR       = 1
HAND_TWO_PAIR       = 2
HAND_THREE_KIND     = 3
HAND_STRAIGHT       = 4
HAND_FLUSH          = 5
HAND_FULL_HOUSE     = 6
HAND_FOUR_KIND      = 7
HAND_STRAIGHT_FLUSH = 8

class Planet(IntEnum):
    MERCURY = auto()   # Straight ×1.2
    VENUS   = auto()   # Flush   ×1.2
    EARTH   = auto()   # Straight ×1.3  (placeholder)
    MARS    = auto()   # Straight ×1.4
    JUPITER = auto()   # StrFlsh ×1.3
    SATURN  = auto()   # FullHouse ×1.3 (placeholder)
    URANUS  = auto()   # FourKind  ×1.3 (placeholder)
    NEPTUNE = auto()   # Flush    ×1.3 (placeholder)
    PLUTO   = auto()   # HighCard ×1.3 (placeholder)

# Map planet → (hand_type_id, factor)
PLANET_MULT: Dict[Planet, Tuple[int, float]] = {
    Planet.MERCURY: (HAND_STRAIGHT,       1.2),
    Planet.VENUS:   (HAND_FLUSH,          1.2),
    Planet.EARTH:   (HAND_STRAIGHT,       1.3),  # TODO verify exact effect
    Planet.MARS:    (HAND_STRAIGHT,       1.4),
    Planet.JUPITER: (HAND_STRAIGHT_FLUSH, 1.3),
    Planet.SATURN:  (HAND_FULL_HOUSE,     1.3),  # TODO verify
    Planet.URANUS:  (HAND_FOUR_KIND,      1.3),  # TODO verify
    Planet.NEPTUNE: (HAND_FLUSH,          1.3),  # TODO verify
    Planet.PLUTO:   (HAND_HIGH_CARD,      1.3),  # TODO verify
}

__all__ = ["Planet", "PLANET_MULT"]
