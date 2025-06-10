"""balatro_gym/scoring_engine_accurate.py - Complete scoring engine with all required methods"""

from __future__ import annotations

from enum import IntEnum
from typing import List, Dict, Tuple, Callable, Optional
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
    FIVE_KIND       = 9  # Added for Balatro
    FLUSH_HOUSE     = 10 # Added for Balatro
    FLUSH_FIVE      = 11 # Added for Balatro

# Base scoring values for each hand type
BASE_HAND_VALUES = {
    HandType.HIGH_CARD: (5, 1),
    HandType.ONE_PAIR: (10, 2),
    HandType.TWO_PAIR: (20, 2),
    HandType.THREE_KIND: (30, 3),
    HandType.STRAIGHT: (30, 4),
    HandType.FLUSH: (35, 4),
    HandType.FULL_HOUSE: (40, 4),
    HandType.FOUR_KIND: (60, 7),
    HandType.STRAIGHT_FLUSH: (100, 8),
    HandType.FIVE_KIND: (120, 12),
    HandType.FLUSH_HOUSE: (140, 14),
    HandType.FLUSH_FIVE: (160, 16),
}

# Planet effects on hand levels
PLANET_HAND_MAP = {
    'Mercury': HandType.ONE_PAIR,
    'Venus': HandType.TWO_PAIR,
    'Earth': HandType.THREE_KIND,
    'Mars': HandType.STRAIGHT,
    'Jupiter': HandType.FLUSH,
    'Saturn': HandType.FULL_HOUSE,
    'Uranus': HandType.FOUR_KIND,
    'Neptune': HandType.STRAIGHT_FLUSH,
    'Pluto': HandType.HIGH_CARD,
    'Planet X': HandType.FIVE_KIND,
    'Ceres': HandType.FLUSH_HOUSE,
    'Eris': HandType.FLUSH_FIVE,
}

# ---------------------------------------------------------------------------
# ScoreEngine
# ---------------------------------------------------------------------------
class ScoreEngine:
    """Complete scoring engine for Balatro with hand levels and planet support"""
    
    def __init__(self):
        # Initialize hand levels (all start at 1)
        self.hand_levels = {hand_type: 1 for hand_type in HandType}
        
        # Track play counts for jokers like Obelisk
        self.hand_play_counts = {hand_type: 0 for hand_type in HandType}
        
        # Modifier functions
        self.modifiers: List[Callable] = []
    
    def get_hand_level(self, hand_type: HandType) -> int:
        """Get the current level for a hand type"""
        return self.hand_levels.get(hand_type, 1)
    
    def set_hand_level(self, hand_type: HandType, level: int):
        """Set the level for a hand type"""
        self.hand_levels[hand_type] = max(1, min(level, 15))  # Clamp between 1-15
    
    def apply_planet(self, hand_type: HandType):
        """Apply a planet card to increase hand level"""
        current_level = self.hand_levels.get(hand_type, 1)
        self.hand_levels[hand_type] = min(current_level + 1, 15)
    
    def get_hand_chips_mult(self, hand_type: HandType) -> Tuple[int, int]:
        """Get base chips and mult for a hand type at current level"""
        base_chips, base_mult = BASE_HAND_VALUES.get(hand_type, (5, 1))
        level = self.get_hand_level(hand_type)
        
        # Each level adds to both chips and mult
        # Level 1: base values
        # Level 2: +10 chips, +1 mult
        # Level 3: +20 chips, +2 mult, etc.
        
        level_bonus = level - 1
        final_chips = base_chips + (level_bonus * 10)
        final_mult = base_mult + level_bonus
        
        return final_chips, final_mult
    
    def score_hand(self, cards: List[int], hand_type: HandType) -> int:
        """Score a hand (simplified version)"""
        base_chips, base_mult = self.get_hand_chips_mult(hand_type)
        
        # Add card values
        card_chips = 0
        for card_id in cards:
            # Convert card ID to rank (0-51 system)
            rank = (card_id % 13) + 2  # 2-14
            if rank == 14:  # Ace
                card_chips += 11
            elif rank >= 10:  # Face cards
                card_chips += 10
            else:
                card_chips += rank
        
        total_chips = base_chips + card_chips
        
        # Calculate final score
        score = total_chips * base_mult
        
        # Apply modifiers
        for modifier in self.modifiers:
            score = modifier(score, cards, self)
        
        return int(score)
    
    def register_modifier(self, fn: Callable):
        """Register a scoring modifier (for joker effects)"""
        self.modifiers.append(fn)
    
    def reset(self):
        """Reset all hand levels to 1"""
        self.hand_levels = {hand_type: 1 for hand_type in HandType}
        self.hand_play_counts = {hand_type: 0 for hand_type in HandType}
    
    def get_all_hand_levels(self) -> Dict[HandType, int]:
        """Get all current hand levels"""
        return self.hand_levels.copy()
    
    def get_play_count(self, hand_type: HandType) -> int:
        """Get how many times a hand has been played"""
        return self.hand_play_counts.get(hand_type, 0)