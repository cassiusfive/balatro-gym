"""balatro_gym/planets_corrected.py - Accurate planet card effects

Planet cards in Balatro add flat bonuses to chips and mult, not multipliers.
Each planet upgrades a specific hand type by adding to its base values.
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
HAND_FIVE_KIND      = 9
HAND_FLUSH_HOUSE    = 10
HAND_FLUSH_FIVE     = 11

class Planet(IntEnum):
    PLUTO   = auto()   # High Card
    MERCURY = auto()   # Pair
    VENUS   = auto()   # Two Pair
    EARTH   = auto()   # Three of a Kind
    MARS    = auto()   # Straight
    JUPITER = auto()   # Flush
    SATURN  = auto()   # Full House
    URANUS  = auto()   # Four of a Kind
    NEPTUNE = auto()   # Straight Flush
    # Secret/Spectral planets
    PLANET_X = auto()  # Five of a Kind
    CERES    = auto()  # Flush House
    ERIS     = auto()  # Flush Five

# Map planet → (hand_type_id, chips_bonus, mult_bonus)
# These are the EXACT values from Balatro
PLANET_EFFECTS: Dict[Planet, Tuple[int, int, int]] = {
    Planet.PLUTO:    (HAND_HIGH_CARD,      10, 1),   # High Card: +10 chips, +1 mult
    Planet.MERCURY:  (HAND_ONE_PAIR,       15, 1),   # Pair: +15 chips, +1 mult
    Planet.VENUS:    (HAND_TWO_PAIR,       20, 1),   # Two Pair: +20 chips, +1 mult
    Planet.EARTH:    (HAND_THREE_KIND,     20, 2),   # Three of a Kind: +20 chips, +2 mult
    Planet.MARS:     (HAND_STRAIGHT,       30, 3),   # Straight: +30 chips, +3 mult
    Planet.JUPITER:  (HAND_FLUSH,          15, 2),   # Flush: +15 chips, +2 mult
    Planet.SATURN:   (HAND_FULL_HOUSE,     25, 2),   # Full House: +25 chips, +2 mult
    Planet.URANUS:   (HAND_FOUR_KIND,      30, 3),   # Four of a Kind: +30 chips, +3 mult
    Planet.NEPTUNE:  (HAND_STRAIGHT_FLUSH, 40, 4),   # Straight Flush: +40 chips, +4 mult
    Planet.PLANET_X: (HAND_FIVE_KIND,      35, 3),   # Five of a Kind: +35 chips, +3 mult
    Planet.CERES:    (HAND_FLUSH_HOUSE,    40, 4),   # Flush House: +40 chips, +4 mult
    Planet.ERIS:     (HAND_FLUSH_FIVE,     50, 3),   # Flush Five: +50 chips, +3 mult
}

# Base hand values (level 1) - for reference
BASE_HAND_VALUES = {
    HAND_HIGH_CARD:      {'chips': 5,   'mult': 1},
    HAND_ONE_PAIR:       {'chips': 10,  'mult': 2},
    HAND_TWO_PAIR:       {'chips': 20,  'mult': 2},
    HAND_THREE_KIND:     {'chips': 30,  'mult': 3},
    HAND_STRAIGHT:       {'chips': 30,  'mult': 4},
    HAND_FLUSH:          {'chips': 35,  'mult': 4},
    HAND_FULL_HOUSE:     {'chips': 40,  'mult': 4},
    HAND_FOUR_KIND:      {'chips': 60,  'mult': 7},
    HAND_STRAIGHT_FLUSH: {'chips': 100, 'mult': 8},
    HAND_FIVE_KIND:      {'chips': 120, 'mult': 12},
    HAND_FLUSH_HOUSE:    {'chips': 140, 'mult': 14},
    HAND_FLUSH_FIVE:     {'chips': 160, 'mult': 16},
}

def calculate_hand_value_at_level(hand_type: int, level: int) -> Tuple[int, int]:
    """Calculate the chips and mult for a hand type at a given level"""
    if hand_type not in BASE_HAND_VALUES:
        return 0, 0
    
    base = BASE_HAND_VALUES[hand_type]
    
    # Find which planet upgrades this hand
    planet_bonus = None
    for planet, (h_type, chips_add, mult_add) in PLANET_EFFECTS.items():
        if h_type == hand_type:
            planet_bonus = (chips_add, mult_add)
            break
    
    if not planet_bonus:
        return base['chips'], base['mult']
    
    # Each level adds the planet's bonus
    chips = base['chips'] + (level - 1) * planet_bonus[0]
    mult = base['mult'] + (level - 1) * planet_bonus[1]
    
    return chips, mult

# Spectral cards that create planets
SPECTRAL_PLANET_CREATORS = {
    'Telescope': 'destroy_most_played_planet',  # Destroy most played hand's planet
    'Black Hole': 'upgrade_all_hands',          # Upgrade every hand by 1 level
}

# Tarot cards that can create planets
TAROT_PLANET_CREATORS = {
    'The Star': Planet.PLUTO,      # Creates random planet (simplified to Pluto)
    'The World': Planet.PLANET_X,  # Creates random planet (simplified to Planet X)
}

__all__ = ["Planet", "PLANET_EFFECTS", "BASE_HAND_VALUES", "calculate_hand_value_at_level"]
