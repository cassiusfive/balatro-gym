"""balatro_gym/unified_scoring.py - Unified scoring system that properly integrates joker effects

This fixes the mismatch between CompleteJokerEffects and ScoreEngine by:
1. Tracking chips and mult separately throughout the calculation
2. Applying effects in the correct order (base -> additions -> multipliers)
3. Using a consistent effect format across all systems
"""

from __future__ import annotations
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from enum import IntEnum

from balatro_gym.scoring_engine import ScoreEngine, HandType
from .complete_joker_effects import CompleteJokerEffects

# ---------------------------------------------------------------------------
# Unified Effect Format
# ---------------------------------------------------------------------------

@dataclass
class ScoringEffect:
    """Unified format for all scoring effects"""
    chips_add: int = 0          # Add to chips (before mult)
    mult_add: int = 0           # Add to mult (before multiplication)
    chips_mult: float = 1.0     # Multiply chips
    mult_mult: float = 1.0      # Multiply mult
    x_mult: float = 1.0         # Final score multiplier
    money: int = 0              # Money gained
    retriggers: int = 0         # Card retriggers
    message: str = ""           # Debug message

    def combine(self, other: 'ScoringEffect') -> 'ScoringEffect':
        """Combine two effects"""
        return ScoringEffect(
            chips_add=self.chips_add + other.chips_add,
            mult_add=self.mult_add + other.mult_add,
            chips_mult=self.chips_mult * other.chips_mult,
            mult_mult=self.mult_mult * other.mult_mult,
            x_mult=self.x_mult * other.x_mult,
            money=self.money + other.money,
            retriggers=self.retriggers + other.retriggers,
            message=f"{self.message}; {other.message}" if self.message and other.message else self.message or other.message
        )

# ---------------------------------------------------------------------------
# Effect Converter
# ---------------------------------------------------------------------------

class EffectConverter:
    """Converts CompleteJokerEffects output to unified format"""
    
    @staticmethod
    def convert_joker_effect(effect_dict: Optional[Dict]) -> ScoringEffect:
        """Convert joker effect dictionary to ScoringEffect"""
        if not effect_dict:
            return ScoringEffect()
        
        # Handle different effect formats from CompleteJokerEffects
        
        # Basic format: {'chips': 50, 'mult': 4, 'x_mult': 2.0}
        if isinstance(effect_dict, dict):
            return ScoringEffect(
                chips_add=effect_dict.get('chips', 0),
                mult_add=effect_dict.get('mult', 0),
                x_mult=effect_dict.get('x_mult', 1.0),
                money=effect_dict.get('money', 0),
                message=effect_dict.get('message', '')
            )
        
        # Handle numeric returns (some jokers return just a multiplier)
        elif isinstance(effect_dict, (int, float)):
            # Assume it's a mult addition
            return ScoringEffect(mult_add=int(effect_dict))
        
        return ScoringEffect()

# ---------------------------------------------------------------------------
# Unified Scoring Context
# ---------------------------------------------------------------------------

@dataclass
class ScoringContext:
    """Complete context for scoring a hand"""
    cards: List[Any]              # Card objects
    scoring_cards: List[Any]      # Cards that count for the hand
    hand_type: HandType           # Poker hand type
    hand_type_name: str          # Human-readable name
    game_state: Dict             # Full game state
    
    # Scoring components
    base_chips: int = 0
    base_mult: int = 0
    card_chips: int = 0          # Chips from card values
    
    # Phases for effects
    phase: str = 'scoring'       # Current phase

# ---------------------------------------------------------------------------
# Unified Scorer
# ---------------------------------------------------------------------------

class UnifiedScorer:
    """Unified scoring system that properly integrates all effects"""
    
    def __init__(self, score_engine: ScoreEngine, joker_effects: CompleteJokerEffects):
        self.engine = score_engine
        self.joker_effects = joker_effects
        self.effect_converter = EffectConverter()
    
    def score_hand(self, context: ScoringContext) -> Tuple[int, Dict[str, Any]]:
        """
        Score a hand with all effects properly applied
        
        Returns:
            (final_score, scoring_breakdown)
        """
        
        # 1. Get base hand values from engine
        base_chips, base_mult = self.engine.get_hand_chips_mult(context.hand_type)
        
        # 2. Initialize scoring components
        chips = base_chips
        mult = base_mult
        x_mult = 1.0
        money_gained = 0
        
        # Track individual components for debugging
        breakdown = {
            'base_chips': base_chips,
            'base_mult': base_mult,
            'card_chips': 0,
            'joker_chips': 0,
            'joker_mult': 0,
            'joker_x_mult': 1.0,
            'effects_applied': []
        }
        
        # 3. Add card base values
        card_chip_total = 0
        for card in context.scoring_cards:
            if hasattr(card, 'chip_value'):
                card_chip_total += card.chip_value()
            elif hasattr(card, 'base_value'):
                card_chip_total += card.base_value
            else:
                # Estimate based on rank
                rank = getattr(card, 'rank', 0)
                if isinstance(rank, IntEnum):
                    rank = rank.value
                card_chip_total += 11 if rank == 14 else min(rank, 10)
        
        chips += card_chip_total
        breakdown['card_chips'] = card_chip_total
        
        # 4. Apply BEFORE scoring joker effects
        before_context = {
            'phase': 'before_scoring',
            'cards': context.cards,
            'scoring_cards': context.scoring_cards,
            'hand_type': context.hand_type_name
        }
        
        for joker_name in context.game_state.get('jokers', []):
            if isinstance(joker_name, str):
                joker = type('Joker', (), {'name': joker_name})
                raw_effect = self.joker_effects.apply_joker_effect(joker, before_context, context.game_state)
                effect = self.effect_converter.convert_joker_effect(raw_effect)
                
                if effect.chips_add or effect.mult_add or effect.x_mult != 1.0:
                    breakdown['effects_applied'].append(f"{joker_name} (before): {effect.message}")
        
        # 5. Apply INDIVIDUAL card scoring effects
        individual_chips = 0
        individual_mult = 0
        individual_x_mult = 1.0
        
        for card in context.scoring_cards:
            card_context = {
                'phase': 'individual_scoring',
                'card': card,
                'cards': context.cards,
                'scoring_cards': context.scoring_cards,
                'hand_type': context.hand_type_name
            }
            
            for joker_name in context.game_state.get('jokers', []):
                if isinstance(joker_name, str):
                    joker = type('Joker', (), {'name': joker_name})
                    raw_effect = self.joker_effects.apply_joker_effect(joker, card_context, context.game_state)
                    effect = self.effect_converter.convert_joker_effect(raw_effect)
                    
                    individual_chips += effect.chips_add
                    individual_mult += effect.mult_add
                    individual_x_mult *= effect.x_mult
                    money_gained += effect.money
                    
                    if effect.chips_add or effect.mult_add or effect.x_mult != 1.0:
                        card_str = f"{getattr(card, 'rank', '?')} of {getattr(card, 'suit', '?')}"
                        breakdown['effects_applied'].append(f"{joker_name} on {card_str}: +{effect.chips_add}c +{effect.mult_add}m x{effect.x_mult}")
        
        # Apply individual effects
        chips += individual_chips
        mult += individual_mult
        x_mult *= individual_x_mult
        
        breakdown['joker_chips'] += individual_chips
        breakdown['joker_mult'] += individual_mult
        breakdown['joker_x_mult'] *= individual_x_mult
        
        # 6. Apply MAIN scoring effects
        scoring_context = {
            'phase': 'scoring',
            'cards': context.cards,
            'scoring_cards': context.scoring_cards,
            'hand_type': context.hand_type_name
        }
        
        for joker_name in context.game_state.get('jokers', []):
            if isinstance(joker_name, str):
                joker = type('Joker', (), {'name': joker_name})
                raw_effect = self.joker_effects.apply_joker_effect(joker, scoring_context, context.game_state)
                effect = self.effect_converter.convert_joker_effect(raw_effect)
                
                # Apply additive effects
                chips += effect.chips_add
                mult += effect.mult_add
                
                # Apply multiplicative effects
                chips = int(chips * effect.chips_mult)
                mult = int(mult * effect.mult_mult)
                
                # Track final multiplier
                x_mult *= effect.x_mult
                money_gained += effect.money
                
                if effect.chips_add or effect.mult_add or effect.x_mult != 1.0:
                    breakdown['effects_applied'].append(
                        f"{joker_name}: +{effect.chips_add}c +{effect.mult_add}m x{effect.x_mult}"
                    )
                
                breakdown['joker_chips'] += effect.chips_add
                breakdown['joker_mult'] += effect.mult_add
                breakdown['joker_x_mult'] *= effect.x_mult
        
        # 7. Apply card enhancements and editions
        enhancement_chips = 0
        enhancement_mult = 0
        enhancement_x_mult = 1.0
        
        for card in context.scoring_cards:
            # Enhancement effects
            if hasattr(card, 'enhancement'):
                if card.enhancement == 'bonus':
                    enhancement_chips += 30
                elif card.enhancement == 'mult':
                    enhancement_mult += 4
                elif card.enhancement == 'glass':
                    enhancement_x_mult *= 2.0
                elif card.enhancement == 'steel':
                    enhancement_x_mult *= 1.5
                elif card.enhancement == 'stone':
                    enhancement_chips += 50
                elif card.enhancement == 'gold':
                    money_gained += 3
                elif card.enhancement == 'lucky':
                    import random
                    if random.random() < 0.2:
                        money_gained += 1
            
            # Edition effects
            if hasattr(card, 'edition'):
                if card.edition == 'foil':
                    enhancement_chips += 50
                elif card.edition == 'holographic':
                    enhancement_mult += 10
                elif card.edition == 'polychrome':
                    enhancement_x_mult *= 1.5
        
        chips += enhancement_chips
        mult += enhancement_mult
        x_mult *= enhancement_x_mult
        
        # 8. Calculate final score
        # Order is: (base_chips + additions) * (base_mult + additions) * x_mult
        final_score = int(chips * mult * x_mult)
        
        # 9. Update game state with money
        if money_gained > 0:
            context.game_state['money'] = context.game_state.get('money', 0) + money_gained
        
        # Final breakdown
        breakdown['final_chips'] = chips
        breakdown['final_mult'] = mult
        breakdown['final_x_mult'] = x_mult
        breakdown['final_score'] = final_score
        breakdown['money_gained'] = money_gained
        
        return final_score, breakdown

# ---------------------------------------------------------------------------
# Integration Helper
# ---------------------------------------------------------------------------

def create_unified_scorer(engine: ScoreEngine, joker_effects: CompleteJokerEffects) -> UnifiedScorer:
    """Create a unified scorer instance"""
    return UnifiedScorer(engine, joker_effects)

# ---------------------------------------------------------------------------
# Usage Example
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Create components
    engine = ScoreEngine()
    joker_effects = CompleteJokerEffects()
    scorer = UnifiedScorer(engine, joker_effects)
    
    # Example hand
    cards = [
        type('Card', (), {'rank': 14, 'suit': 'Hearts', 'base_value': 11, 'enhancement': None}),
        type('Card', (), {'rank': 14, 'suit': 'Spades', 'base_value': 11, 'enhancement': 'mult'}),
        type('Card', (), {'rank': 2, 'suit': 'Hearts', 'base_value': 2, 'enhancement': None}),
        type('Card', (), {'rank': 3, 'suit': 'Clubs', 'base_value': 3, 'enhancement': None}),
        type('Card', (), {'rank': 5, 'suit': 'Diamonds', 'base_value': 5, 'enhancement': None}),
    ]
    
    # Create context
    context = ScoringContext(
        cards=cards,
        scoring_cards=cards[:2],  # Just the pair of aces
        hand_type=HandType.ONE_PAIR,
        hand_type_name='Pair',
        game_state={
            'jokers': ['Joker', 'Greedy Joker', 'Fibonacci'],
            'money': 10
        }
    )
    
    # Score the hand
    score, breakdown = scorer.score_hand(context)
    
    print(f"Final Score: {score}")
    print("\nBreakdown:")
    for key, value in breakdown.items():
        if key != 'effects_applied':
            print(f"  {key}: {value}")
    
    print("\nEffects Applied:")
    for effect in breakdown['effects_applied']:
        print(f"  - {effect}")
