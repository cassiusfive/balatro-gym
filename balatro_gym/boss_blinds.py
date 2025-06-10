"""balatro_gym/boss_blinds.py - Complete Boss Blind System

Implements all boss blinds from Balatro with their unique debuff effects.
Boss blinds appear every 3rd round (ante X.3) and have special abilities that
make the round more challenging.
"""

from __future__ import annotations
from enum import IntEnum, auto
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import random

# ---------------------------------------------------------------------------
# Boss Blind Types
# ---------------------------------------------------------------------------

class BossBlindType(IntEnum):
    # Common Boss Blinds
    THE_HOOK = auto()      # Discards 2 random cards per hand
    THE_WALL = auto()      # Extra large blind (2x base chips)
    THE_WHEEL = auto()     # 1 in 7 cards get drawn face down
    THE_HOUSE = auto()     # First hand drawn face down
    THE_MARK = auto()      # All face cards drawn face down
    THE_FISH = auto()      # Cards drawn face down after each hand played
    THE_PSYCHIC = auto()   # Must play 5 cards
    THE_GOAD = auto()      # All Spades are debuffed
    THE_WATER = auto()     # Start with 0 discards
    THE_WINDOW = auto()    # All Diamonds are debuffed
    THE_MANACLE = auto()   # -1 Hand Size
    THE_EYE = auto()       # No repeat hand types this round
    THE_MOUTH = auto()     # Play only 1 hand type this round
    THE_PLANT = auto()     # All face cards are debuffed
    THE_SERPENT = auto()   # After each hand, always draw 3 cards
    THE_PILLAR = auto()    # Cards played previously are debuffed
    THE_NEEDLE = auto()    # Play only 1 hand
    THE_HEAD = auto()      # All Hearts are debuffed
    THE_CLUB = auto()      # All Clubs are debuffed
    THE_TOOTH = auto()     # Lose $1 per card played
    THE_FLINT = auto()     # Base chips and mult halved
    THE_OXIDE = auto()     # All cards played give no chips
    THE_ARM = auto()       # Decrease level of played poker hand
    THE_VIOLET = auto()    # All cards are debuffed
    THE_VERDANT = auto()   # Required cards scale up by 1 per hand until 7
    THE_AMBER = auto()     # -1 active joker slot
    THE_CRIMSON = auto()   # All Heart cards are flipped
    THE_CERULEAN = auto()  # All cards in deck are flipped

@dataclass
class BossBlind:
    """Boss blind configuration"""
    blind_type: BossBlindType
    name: str
    description: str
    mult: float = 1.0  # Chip requirement multiplier
    money_reward: int = 5  # Extra money for defeating
    
    # Effect functions
    on_round_start: Optional[Callable] = None
    on_hand_drawn: Optional[Callable] = None
    on_card_played: Optional[Callable] = None
    on_hand_scored: Optional[Callable] = None
    on_discard: Optional[Callable] = None
    can_play_hand: Optional[Callable] = None
    modify_card_score: Optional[Callable] = None

# ---------------------------------------------------------------------------
# Boss Blind Definitions
# ---------------------------------------------------------------------------

BOSS_BLINDS: Dict[BossBlindType, BossBlind] = {
    BossBlindType.THE_HOOK: BossBlind(
        blind_type=BossBlindType.THE_HOOK,
        name="The Hook",
        description="Discards 2 random cards per hand",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_WALL: BossBlind(
        blind_type=BossBlindType.THE_WALL,
        name="The Wall",
        description="Extra large blind",
        mult=2.0,  # Double chip requirement
        money_reward=5
    ),
    
    BossBlindType.THE_WHEEL: BossBlind(
        blind_type=BossBlindType.THE_WHEEL,
        name="The Wheel",
        description="1 in 7 cards get drawn face down",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_HOUSE: BossBlind(
        blind_type=BossBlindType.THE_HOUSE,
        name="The House",
        description="First hand drawn face down",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_MARK: BossBlind(
        blind_type=BossBlindType.THE_MARK,
        name="The Mark",
        description="All face cards drawn face down",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_FISH: BossBlind(
        blind_type=BossBlindType.THE_FISH,
        name="The Fish",
        description="Cards drawn face down after each hand played",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_PSYCHIC: BossBlind(
        blind_type=BossBlindType.THE_PSYCHIC,
        name="The Psychic",
        description="Must play 5 cards",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_GOAD: BossBlind(
        blind_type=BossBlindType.THE_GOAD,
        name="The Goad",
        description="All Spades are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_WATER: BossBlind(
        blind_type=BossBlindType.THE_WATER,
        name="The Water",
        description="Start with 0 discards",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_WINDOW: BossBlind(
        blind_type=BossBlindType.THE_WINDOW,
        name="The Window",
        description="All Diamonds are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_MANACLE: BossBlind(
        blind_type=BossBlindType.THE_MANACLE,
        name="The Manacle",
        description="-1 Hand Size",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_EYE: BossBlind(
        blind_type=BossBlindType.THE_EYE,
        name="The Eye",
        description="No repeat hand types this round",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_MOUTH: BossBlind(
        blind_type=BossBlindType.THE_MOUTH,
        name="The Mouth",
        description="Play only 1 hand type this round",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_PLANT: BossBlind(
        blind_type=BossBlindType.THE_PLANT,
        name="The Plant",
        description="All face cards are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_SERPENT: BossBlind(
        blind_type=BossBlindType.THE_SERPENT,
        name="The Serpent",
        description="After each hand, always draw 3 cards",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_PILLAR: BossBlind(
        blind_type=BossBlindType.THE_PILLAR,
        name="The Pillar",
        description="Cards played previously are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_NEEDLE: BossBlind(
        blind_type=BossBlindType.THE_NEEDLE,
        name="The Needle",
        description="Play only 1 hand",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_HEAD: BossBlind(
        blind_type=BossBlindType.THE_HEAD,
        name="The Head",
        description="All Hearts are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_CLUB: BossBlind(
        blind_type=BossBlindType.THE_CLUB,
        name="The Club",
        description="All Clubs are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_TOOTH: BossBlind(
        blind_type=BossBlindType.THE_TOOTH,
        name="The Tooth",
        description="Lose $1 per card played",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_FLINT: BossBlind(
        blind_type=BossBlindType.THE_FLINT,
        name="The Flint",
        description="Base chips and mult halved",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_OXIDE: BossBlind(
        blind_type=BossBlindType.THE_OXIDE,
        name="The Oxide",
        description="All cards played give no chips",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_ARM: BossBlind(
        blind_type=BossBlindType.THE_ARM,
        name="The Arm",
        description="Decrease level of played poker hand",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_VIOLET: BossBlind(
        blind_type=BossBlindType.THE_VIOLET,
        name="The Violet",
        description="All cards are debuffed",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_VERDANT: BossBlind(
        blind_type=BossBlindType.THE_VERDANT,
        name="The Verdant",
        description="Required cards scale up by 1 per hand until 7",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_AMBER: BossBlind(
        blind_type=BossBlindType.THE_AMBER,
        name="The Amber",
        description="-1 active joker slot",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_CRIMSON: BossBlind(
        blind_type=BossBlindType.THE_CRIMSON,
        name="The Crimson",
        description="All Heart cards are flipped",
        mult=1.0,
        money_reward=5
    ),
    
    BossBlindType.THE_CERULEAN: BossBlind(
        blind_type=BossBlindType.THE_CERULEAN,
        name="The Cerulean",
        description="All cards in deck are flipped",
        mult=1.0,
        money_reward=5
    ),
}

# ---------------------------------------------------------------------------
# Boss Blind Manager
# ---------------------------------------------------------------------------

class BossBlindManager:
    """Manages boss blind effects and state"""
    
    def __init__(self):
        self.active_blind: Optional[BossBlind] = None
        self.blind_state: Dict[str, Any] = {}
        
    def activate_boss_blind(self, blind_type: BossBlindType, game_state: Dict) -> Dict[str, Any]:
        """Activate a boss blind and apply initial effects"""
        self.active_blind = BOSS_BLINDS[blind_type]
        self.blind_state = {
            'played_hand_types': set(),
            'played_cards': set(),
            'first_hand': True,
            'hands_played': 0,
            'cards_required': 5,  # For The Verdant
            'disabled_joker_slots': 0,
            'face_down_cards': set(),
        }
        
        effects = {
            'message': f"{self.active_blind.name}: {self.active_blind.description}",
            'chip_mult': self.active_blind.mult,
            'modifications': {}
        }
        
        # Apply initial effects based on blind type
        if blind_type == BossBlindType.THE_WATER:
            effects['modifications']['discards'] = 0
            
        elif blind_type == BossBlindType.THE_MANACLE:
            effects['modifications']['hand_size'] = -1
            
        elif blind_type == BossBlindType.THE_NEEDLE:
            effects['modifications']['hands'] = 1
            
        elif blind_type == BossBlindType.THE_AMBER:
            self.blind_state['disabled_joker_slots'] = 1
            effects['modifications']['active_jokers'] = -1
            
        return effects
    
    def on_hand_drawn(self, hand_cards: List[Any], game_state: Dict) -> Dict[str, Any]:
        """Apply effects when hand is drawn"""
        if not self.active_blind:
            return {}
            
        effects = {'face_down_cards': [], 'discarded_cards': []}
        
        if self.active_blind.blind_type == BossBlindType.THE_HOOK:
            # Discard 2 random cards
            if len(hand_cards) >= 2:
                to_discard = random.sample(range(len(hand_cards)), 2)
                effects['discarded_cards'] = to_discard
                
        elif self.active_blind.blind_type == BossBlindType.THE_WHEEL:
            # 1 in 7 cards face down
            for i, card in enumerate(hand_cards):
                if random.random() < 1/7:
                    effects['face_down_cards'].append(i)
                    
        elif self.active_blind.blind_type == BossBlindType.THE_HOUSE:
            # First hand all face down
            if self.blind_state['first_hand']:
                effects['face_down_cards'] = list(range(len(hand_cards)))
                
        elif self.active_blind.blind_type == BossBlindType.THE_MARK:
            # Face cards face down
            for i, card in enumerate(hand_cards):
                if hasattr(card, 'rank') and card.rank in [11, 12, 13]:  # J, Q, K
                    effects['face_down_cards'].append(i)
                    
        elif self.active_blind.blind_type == BossBlindType.THE_FISH:
            # All face down after first hand
            if not self.blind_state['first_hand']:
                effects['face_down_cards'] = list(range(len(hand_cards)))
        
        return effects
    
    def can_play_hand(self, selected_cards: List[Any], hand_type: str) -> Tuple[bool, str]:
        """Check if hand can be played given boss blind restrictions"""
        if not self.active_blind:
            return True, ""
            
        if self.active_blind.blind_type == BossBlindType.THE_PSYCHIC:
            # Must play exactly 5 cards
            if len(selected_cards) != 5:
                return False, "Must play exactly 5 cards"
                
        elif self.active_blind.blind_type == BossBlindType.THE_EYE:
            # No repeat hand types
            if hand_type in self.blind_state['played_hand_types']:
                return False, f"Cannot play {hand_type} again"
                
        elif self.active_blind.blind_type == BossBlindType.THE_MOUTH:
            # Only one hand type allowed
            if self.blind_state['played_hand_types'] and hand_type not in self.blind_state['played_hand_types']:
                allowed = list(self.blind_state['played_hand_types'])[0]
                return False, f"Can only play {allowed}"
                
        elif self.active_blind.blind_type == BossBlindType.THE_VERDANT:
            # Requires specific number of cards
            required = self.blind_state['cards_required']
            if len(selected_cards) < required:
                return False, f"Must play at least {required} cards"
        
        return True, ""
    
    def modify_scoring(self, base_chips: int, base_mult: int, 
                      played_cards: List[Any], hand_type: str) -> Tuple[int, int]:
        """Modify scoring based on boss blind effects"""
        if not self.active_blind:
            return base_chips, base_mult
            
        chips = base_chips
        mult = base_mult
        
        # Debuff effects
        if self.active_blind.blind_type == BossBlindType.THE_FLINT:
            # Halve base values
            chips = chips // 2
            mult = mult // 2
            
        elif self.active_blind.blind_type == BossBlindType.THE_OXIDE:
            # No chips from cards
            chips = 0
            
        elif self.active_blind.blind_type == BossBlindType.THE_ARM:
            # Decrease hand level effect (simulate by reducing base values)
            chips = int(chips * 0.75)
            mult = int(mult * 0.75)
            
        # Card-specific debuffs
        debuffed_cards = 0
        for card in played_cards:
            if self._is_card_debuffed(card):
                debuffed_cards += 1
        
        if debuffed_cards > 0:
            # Each debuffed card reduces effectiveness
            penalty = 0.8 ** debuffed_cards
            chips = int(chips * penalty)
            mult = int(mult * penalty)
        
        return chips, mult
    
    def _is_card_debuffed(self, card) -> bool:
        """Check if a card is debuffed by current boss blind"""
        if not self.active_blind:
            return False
            
        # Suit debuffs
        if hasattr(card, 'suit'):
            if self.active_blind.blind_type == BossBlindType.THE_GOAD and card.suit == 'Spades':
                return True
            elif self.active_blind.blind_type == BossBlindType.THE_WINDOW and card.suit == 'Diamonds':
                return True
            elif self.active_blind.blind_type == BossBlindType.THE_HEAD and card.suit == 'Hearts':
                return True
            elif self.active_blind.blind_type == BossBlindType.THE_CLUB and card.suit == 'Clubs':
                return True
                
        # Rank debuffs
        if hasattr(card, 'rank'):
            if self.active_blind.blind_type == BossBlindType.THE_PLANT and card.rank in [11, 12, 13]:
                return True
                
        # Universal debuffs
        if self.active_blind.blind_type == BossBlindType.THE_VIOLET:
            return True
            
        # Previously played cards
        if self.active_blind.blind_type == BossBlindType.THE_PILLAR:
            card_id = getattr(card, 'id', None) or id(card)
            if card_id in self.blind_state.get('played_cards', set()):
                return True
        
        return False
    
    def on_hand_scored(self, played_cards: List[Any], hand_type: str, game_state: Dict):
        """Update state after hand is scored"""
        if not self.active_blind:
            return
            
        # Update played hand types
        self.blind_state['played_hand_types'].add(hand_type)
        self.blind_state['first_hand'] = False
        self.blind_state['hands_played'] += 1
        
        # Track played cards for The Pillar
        if self.active_blind.blind_type == BossBlindType.THE_PILLAR:
            for card in played_cards:
                card_id = getattr(card, 'id', None) or id(card)
                self.blind_state['played_cards'].add(card_id)
                
        # Update required cards for The Verdant
        if self.active_blind.blind_type == BossBlindType.THE_VERDANT:
            self.blind_state['cards_required'] = min(7, self.blind_state['cards_required'] + 1)
            
        # Money penalty for The Tooth
        if self.active_blind.blind_type == BossBlindType.THE_TOOTH:
            game_state['money'] = max(0, game_state.get('money', 0) - len(played_cards))
            
        # Special draw for The Serpent
        if self.active_blind.blind_type == BossBlindType.THE_SERPENT:
            # Force draw exactly 3 cards next hand
            game_state['force_draw_count'] = 3
    
    def get_disabled_joker_count(self) -> int:
        """Get number of disabled joker slots"""
        return self.blind_state.get('disabled_joker_slots', 0)
    
    def deactivate(self):
        """Clear boss blind effects"""
        self.active_blind = None
        self.blind_state = {}

# ---------------------------------------------------------------------------
# Integration with Environment
# ---------------------------------------------------------------------------

def select_boss_blind(ante: int, exclude: Optional[List[BossBlindType]] = None) -> BossBlindType:
    """Select a random boss blind for the given ante"""
    # All blinds available from the start in standard Balatro
    available_blinds = list(BossBlindType)
    
    # Exclude specific blinds if requested (e.g., Chicot joker effect)
    if exclude:
        available_blinds = [b for b in available_blinds if b not in exclude]
    
    # Random selection
    return random.choice(available_blinds)

# ---------------------------------------------------------------------------
# Example Usage in Environment
# ---------------------------------------------------------------------------

"""
# In your BalatroEnv class:

def __init__(self):
    # ... existing init code ...
    self.boss_blind_manager = BossBlindManager()
    self.active_boss_blind = None

def _step_blind_select(self, action: int):
    # ... existing code ...
    
    if blind_type == 2:  # Boss blind
        # Select and activate boss blind
        boss_type = select_boss_blind(self.ante)
        effects = self.boss_blind_manager.activate_boss_blind(boss_type, self.game_state)
        
        # Apply chip multiplier
        self.chips_needed = int(self.chips_needed * effects['chip_mult'])
        
        # Apply modifications
        if 'discards' in effects['modifications']:
            self.game.discards = effects['modifications']['discards']
        if 'hand_size' in effects['modifications']:
            self.game.hand_size += effects['modifications']['hand_size']
        if 'hands' in effects['modifications']:
            self.game.hands = effects['modifications']['hands']
            
        self.active_boss_blind = boss_type
        info['boss_blind'] = self.boss_blind_manager.active_blind.name
        info['boss_effect'] = self.boss_blind_manager.active_blind.description

def _step_play(self, action: int):
    # When drawing cards
    if need_to_draw:
        effects = self.boss_blind_manager.on_hand_drawn(hand_cards, self.game_state)
        # Apply face down effects
        # Apply discard effects
    
    # Before playing hand
    if action == ACTION_PLAY_HAND:
        can_play, message = self.boss_blind_manager.can_play_hand(selected_cards, hand_type)
        if not can_play:
            return self._get_observation(), -1.0, False, False, {'error': message}
    
    # When scoring
    base_chips, base_mult = self.engine.get_hand_chips_mult(hand_type)
    chips, mult = self.boss_blind_manager.modify_scoring(base_chips, base_mult, played_cards, hand_type)
    
    # After scoring
    self.boss_blind_manager.on_hand_scored(played_cards, hand_type, self.game_state)

def _advance_round(self):
    # ... existing code ...
    
    # Deactivate boss blind when round ends
    if self.boss_blind_manager.active_blind:
        # Award extra money for beating boss
        self.player.chips += self.boss_blind_manager.active_blind.money_reward
        self.boss_blind_manager.deactivate()
        self.active_boss_blind = None
"""
