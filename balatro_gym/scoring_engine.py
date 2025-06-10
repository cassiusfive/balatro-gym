"""balatro_gym/scoring_engine_accurate.py

Accurate Balatro scoring engine that matches the real game mechanics:
- Additive planet upgrades (not multiplicative)
- Numeric hand levels (1-99, not bronze/silver/gold)
- Proper base values and upgrade amounts
- Dynamic joker modifiers
"""

from __future__ import annotations
from enum import IntEnum
from typing import List, Callable, Dict, Tuple, Optional
import numpy as np

# ---------------------------------------------------------------------------
# HandType enumeration
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
    FIVE_KIND       = 9   # Secret hands
    FLUSH_HOUSE     = 10
    FLUSH_FIVE      = 11
    NONE            = -1  # For invalid hands

# Base values for each hand type at level 1
BASE_HAND_VALUES = {
    HandType.HIGH_CARD:      {'chips': 5,   'mult': 1},
    HandType.ONE_PAIR:       {'chips': 10,  'mult': 2},
    HandType.TWO_PAIR:       {'chips': 20,  'mult': 2},
    HandType.THREE_KIND:     {'chips': 30,  'mult': 3},
    HandType.STRAIGHT:       {'chips': 30,  'mult': 4},
    HandType.FLUSH:          {'chips': 35,  'mult': 4},
    HandType.FULL_HOUSE:     {'chips': 40,  'mult': 4},
    HandType.FOUR_KIND:      {'chips': 60,  'mult': 7},
    HandType.STRAIGHT_FLUSH: {'chips': 100, 'mult': 8},
    HandType.FIVE_KIND:      {'chips': 120, 'mult': 12},
    HandType.FLUSH_HOUSE:    {'chips': 140, 'mult': 14},
    HandType.FLUSH_FIVE:     {'chips': 160, 'mult': 16},
}

# Planet upgrade values (chips_add, mult_add per level)
PLANET_UPGRADES = {
    HandType.HIGH_CARD:      (10, 1),   # Pluto
    HandType.ONE_PAIR:       (15, 1),   # Mercury
    HandType.TWO_PAIR:       (20, 1),   # Venus
    HandType.THREE_KIND:     (20, 2),   # Earth
    HandType.STRAIGHT:       (30, 3),   # Mars
    HandType.FLUSH:          (15, 2),   # Jupiter
    HandType.FULL_HOUSE:     (25, 2),   # Saturn
    HandType.FOUR_KIND:      (30, 3),   # Uranus
    HandType.STRAIGHT_FLUSH: (40, 4),   # Neptune
    HandType.FIVE_KIND:      (35, 3),   # Planet X
    HandType.FLUSH_HOUSE:    (40, 4),   # Ceres
    HandType.FLUSH_FIVE:     (50, 3),   # Eris
}

# Type signature for dynamic modifiers
ModifierFn = Callable[[float, List[int], "ScoreEngine"], float]

# ---------------------------------------------------------------------------
# ScoreEngine
# ---------------------------------------------------------------------------
class ScoreEngine:
    """Accurate Balatro scoring engine with additive upgrades"""

    def __init__(self):
        # Hand levels start at 1
        self.hand_levels: Dict[HandType, int] = {
            hand_type: 1 for hand_type in HandType if hand_type != HandType.NONE
        }
        
        # Dynamic modifiers (jokers, vouchers, etc.)
        self.modifiers: List[ModifierFn] = []
        
        # Track most played hand for jokers like Obelisk
        self.hand_play_counts: Dict[HandType, int] = {
            hand_type: 0 for hand_type in HandType if hand_type != HandType.NONE
        }
        
        # Planet usage tracking for jokers like Constellation
        self.planets_used = 0

    def apply_planet(self, hand_type: HandType):
        """Apply a planet card to upgrade a hand type"""
        if hand_type in self.hand_levels:
            self.hand_levels[hand_type] += 1
            self.planets_used += 1
            return True
        return False

    def apply_consumable(self, consumable):
        """Apply any consumable effect"""
        # Handle planet cards
        if hasattr(consumable, 'value'):  # It's a Planet enum
            # Map planet to hand type
            planet_to_hand = {
                'PLUTO': HandType.HIGH_CARD,
                'MERCURY': HandType.ONE_PAIR,
                'VENUS': HandType.TWO_PAIR,
                'EARTH': HandType.THREE_KIND,
                'MARS': HandType.STRAIGHT,
                'JUPITER': HandType.FLUSH,
                'SATURN': HandType.FULL_HOUSE,
                'URANUS': HandType.FOUR_KIND,
                'NEPTUNE': HandType.STRAIGHT_FLUSH,
                'PLANET_X': HandType.FIVE_KIND,
                'CERES': HandType.FLUSH_HOUSE,
                'ERIS': HandType.FLUSH_FIVE,
            }
            
            planet_name = consumable.name
            if planet_name in planet_to_hand:
                self.apply_planet(planet_to_hand[planet_name])
        
        # TODO: Handle tarot/spectral cards

    def register_modifier(self, fn: ModifierFn):
        """Register a dynamic modifier (joker effect)"""
        self.modifiers.append(fn)

    def unregister_modifier(self, fn: ModifierFn):
        """Remove a modifier (when joker is sold)"""
        if fn in self.modifiers:
            self.modifiers.remove(fn)

    def get_hand_chips_mult(self, hand_type: HandType) -> Tuple[int, int]:
        """Get the current chips and mult for a hand type at its level"""
        if hand_type not in BASE_HAND_VALUES:
            return 0, 0
        
        base = BASE_HAND_VALUES[hand_type]
        level = self.hand_levels.get(hand_type, 1)
        
        if hand_type in PLANET_UPGRADES:
            chips_add, mult_add = PLANET_UPGRADES[hand_type]
            chips = base['chips'] + (level - 1) * chips_add
            mult = base['mult'] + (level - 1) * mult_add
        else:
            chips = base['chips']
            mult = base['mult']
        
        return chips, mult

    def score(self, hand_card_ids: List[int], hand_type: HandType, 
              card_chips: int = 0) -> Tuple[float, float]:
        """
        Calculate score for a hand
        
        Args:
            hand_card_ids: List of card indices (0-51)
            hand_type: The poker hand type
            card_chips: Sum of base chip values from cards
            
        Returns:
            (total_score, mult_applied) tuple
        """
        # Track hand usage
        if hand_type in self.hand_play_counts:
            self.hand_play_counts[hand_type] += 1
        
        # Get base chips and mult for this hand type
        base_chips, base_mult = self.get_hand_chips_mult(hand_type)
        
        # Total chips = hand chips + card chips
        chips = base_chips + card_chips
        mult = base_mult
        
        # Calculate base score
        score = float(chips * mult)
        
        # Apply dynamic modifiers (jokers)
        for modifier in self.modifiers:
            score = modifier(score, hand_card_ids, self)
        
        return score, mult

    def get_most_played_hand(self) -> HandType:
        """Get the most frequently played hand type"""
        if not any(self.hand_play_counts.values()):
            return HandType.HIGH_CARD
        
        return max(self.hand_play_counts.items(), key=lambda x: x[1])[0]

    def reset_hand_counts(self):
        """Reset hand play counts (for new run)"""
        for hand_type in self.hand_play_counts:
            self.hand_play_counts[hand_type] = 0

    def get_hand_level(self, hand_type: HandType) -> int:
        """Get the current level of a hand type"""
        return self.hand_levels.get(hand_type, 1)

    def set_hand_level(self, hand_type: HandType, level: int):
        """Set hand level (for testing or special effects)"""
        if hand_type in self.hand_levels:
            self.hand_levels[hand_type] = max(1, level)

    def get_all_hand_levels(self) -> Dict[str, int]:
        """Get all hand levels as a readable dict"""
        return {
            hand_type.name: self.hand_levels[hand_type]
            for hand_type in HandType
            if hand_type != HandType.NONE
        }


# ---------------------------------------------------------------------------
# Example modifier functions for common jokers
# ---------------------------------------------------------------------------

def create_flat_mult_modifier(mult_bonus: int):
    """Create a modifier that adds flat mult"""
    def modifier(score: float, hand: List[int], engine: ScoreEngine) -> float:
        # In Balatro, flat mult is added before multiplication
        # So we need to reverse engineer: score = chips * (base_mult + bonus)
        # This is a simplification - ideally track chips/mult separately
        return score * (1 + mult_bonus / 10)  # Approximation
    return modifier

def create_mult_multiplier(multiplier: float):
    """Create a modifier that multiplies the score"""
    def modifier(score: float, hand: List[int], engine: ScoreEngine) -> float:
        return score * multiplier
    return modifier

def create_chip_bonus_modifier(chip_bonus: int):
    """Create a modifier that adds chips"""
    def modifier(score: float, hand: List[int], engine: ScoreEngine) -> float:
        # Chips are added before mult, so the bonus gets multiplied
        # This is approximate without tracking chips/mult separately
        return score + chip_bonus * 2  # Assume average mult of 2
    return modifier


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    engine = ScoreEngine()
    
    # Test base scoring
    print("=== Base Hand Values ===")
    for hand_type in HandType:
        if hand_type != HandType.NONE:
            chips, mult = engine.get_hand_chips_mult(hand_type)
            print(f"{hand_type.name:15} Level 1: {chips:3} chips × {mult} mult = {chips * mult}")
    
    # Test planet upgrades
    print("\n=== After Planet Upgrades ===")
    engine.apply_planet(HandType.FLUSH)
    engine.apply_planet(HandType.FLUSH)
    engine.apply_planet(HandType.FLUSH)
    
    chips, mult = engine.get_hand_chips_mult(HandType.FLUSH)
    print(f"Flush Level 4: {chips} chips × {mult} mult = {chips * mult}")
    
    # Test with joker modifier
    print("\n=== With Joker Modifier ===")
    engine.register_modifier(create_mult_multiplier(2.0))  # Like a 2x mult joker
    
    hand = [0, 13, 26, 39, 1]  # Example flush hand
    score, mult = engine.score(hand, HandType.FLUSH, card_chips=30)
    print(f"Flush with 2x joker: {score}")
