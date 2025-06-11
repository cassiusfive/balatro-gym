"""State management for Balatro RL environment.

This module contains all state-related classes including the unified game state
that serves as the single source of truth for the entire game, and card state
tracking for enhancements, editions, and seals.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum

from balatro_gym.cards import Card, Enhancement, Edition, Seal
from balatro_gym.constants import Phase
from balatro_gym.scoring_engine import HandType
from balatro_gym.jokers import JokerInfo
from balatro_gym.boss_blinds import BossBlindType


@dataclass
class CardState:
    """Tracks per-card state including enhancements, editions, and seals."""
    
    card_index: int
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    
    # Tracking for effects
    times_played: int = 0
    times_scored: int = 0
    times_discarded: int = 0
    times_held: int = 0
    
    # Special flags
    is_debuffed: bool = False
    is_face_down: bool = False
    is_destroyed: bool = False
    
    def calculate_chip_bonus(self, base_chips: int) -> int:
        """Calculate modified chip value based on enhancements."""
        if self.enhancement == Enhancement.BONUS:
            return base_chips + 30
        elif self.enhancement == Enhancement.STONE:
            return 50  # Stone cards always give +50 chips
        else:
            return base_chips
    
    def calculate_mult_bonus(self, base_mult: int) -> int:
        """Calculate modified mult value based on enhancements."""
        if self.enhancement == Enhancement.MULT:
            return base_mult + 4
        else:
            return base_mult
    
    def get_x_mult(self) -> float:
        """Get multiplicative mult from enhancements."""
        if self.enhancement == Enhancement.GLASS:
            return 2.0
        return 1.0
    
    def copy(self) -> 'CardState':
        """Create a deep copy of the card state."""
        return CardState(
            card_index=self.card_index,
            enhancement=self.enhancement,
            edition=self.edition,
            seal=self.seal,
            times_played=self.times_played,
            times_scored=self.times_scored,
            times_discarded=self.times_discarded,
            times_held=self.times_held,
            is_debuffed=self.is_debuffed,
            is_face_down=self.is_face_down,
            is_destroyed=self.is_destroyed
        )


@dataclass
class UnifiedGameState:
    """Single source of truth for all game state.
    
    This class contains all game state that needs to be tracked across
    different systems. It serves as the central state store that all
    game systems read from and write to.
    """
    
    # Core game state
    ante: int = 1
    round: int = 1  # 1=small, 2=big, 3=boss
    phase: Phase = Phase.BLIND_SELECT
    chips_needed: int = 300
    chips_scored: int = 0  # Total career chips scored
    round_chips_scored: int = 0  # Chips scored in current round only
    money: int = 4
    
    # Cards and hands
    deck: List[Card] = field(default_factory=list)
    hand_indexes: List[int] = field(default_factory=list)  # Indexes into deck
    selected_cards: List[int] = field(default_factory=list)  # Indexes into hand_indexes
    hands_left: int = 4
    discards_left: int = 3
    hand_size: int = 8
    
    # Collections
    jokers: List[JokerInfo] = field(default_factory=list)
    consumables: List[str] = field(default_factory=list)  # Consumable names
    vouchers: List[str] = field(default_factory=list)  # Voucher names
    joker_slots: int = 5
    consumable_slots: int = 2
    
    # Shop state
    shop_inventory: List[Any] = field(default_factory=list)
    shop_reroll_cost: int = 5
    
    # Statistics
    hands_played_total: int = 0
    hands_played_ante: int = 0
    best_hand_this_ante: int = 0
    jokers_sold: int = 0
    cards_discarded_total: int = 0
    rerolls_used: int = 0
    shop_visits: int = 0
    
    # Hand levels (HandType -> level)
    hand_levels: Dict[HandType, int] = field(default_factory=dict)
    
    # Card states (card index -> CardState)
    card_states: Dict[int, CardState] = field(default_factory=dict)
    
    # Boss blind state
    active_boss_blind: Optional[BossBlindType] = None
    boss_blind_active: bool = False
    face_down_cards: List[int] = field(default_factory=list)  # Indexes into hand_indexes
    force_draw_count: Optional[int] = None  # For The Serpent boss
    disabled_joker_slots: int = 0  # For The Plant boss
    
    # Special game modes/effects
    eternal_jokers: List[int] = field(default_factory=list)  # Indexes of eternal jokers
    perishable_counters: Dict[int, int] = field(default_factory=dict)  # Joker index -> rounds left
    rental_jokers: List[int] = field(default_factory=list)  # Indexes of rental jokers
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for joker effects and other systems.
        
        This is used by various game systems that expect state in dictionary format.
        """
        # Get cards in hand
        hand_cards = []
        for idx in self.hand_indexes:
            if 0 <= idx < len(self.deck):
                hand_cards.append(self.deck[idx])
        
        return {
            # Core state
            'deck': self.deck,
            'hand': hand_cards,
            'jokers': [{'name': j.name, 'id': j.id} for j in self.jokers],
            'consumables': self.consumables,
            'vouchers': self.vouchers,
            'money': self.money,
            'ante': self.ante,
            'round': self.round,
            'phase': self.phase.value,
            
            # Hand/discard state
            'hands_left': self.hands_left,
            'discards_left': self.discards_left,
            'hand_size': self.hand_size,
            'joker_slots': self.joker_slots,
            'consumable_slots': self.consumable_slots,
            
            # Statistics
            'hands_played': self.hands_played_total,
            'hands_played_ante': self.hands_played_ante,
            'round_chips_scored': self.round_chips_scored,
            'chips_scored': self.chips_scored,
            'chips_needed': self.chips_needed,
            
            # Special states
            'boss_blind_active': self.boss_blind_active,
            'active_boss_blind': self.active_boss_blind.name if self.active_boss_blind else None,
            'face_down_cards': self.face_down_cards,
            
            # Collections info
            'joker_count': len(self.jokers),
            'consumable_count': len(self.consumables),
            'voucher_count': len(self.vouchers),
        }
    
    def get_card_state(self, card_index: int) -> CardState:
        """Get or create card state for a given card index."""
        if card_index not in self.card_states:
            self.card_states[card_index] = CardState(card_index)
        return self.card_states[card_index]
    
    def copy(self) -> 'UnifiedGameState':
        """Create a deep copy of the entire game state."""
        return UnifiedGameState(
            # Core state
            ante=self.ante,
            round=self.round,
            phase=self.phase,
            chips_needed=self.chips_needed,
            chips_scored=self.chips_scored,
            round_chips_scored=self.round_chips_scored,
            money=self.money,
            
            # Cards - need to copy the list but Card objects are immutable
            deck=self.deck.copy() if self.deck else [],
            hand_indexes=self.hand_indexes.copy(),
            selected_cards=self.selected_cards.copy(),
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            hand_size=self.hand_size,
            
            # Collections - JokerInfo objects are immutable
            jokers=self.jokers.copy(),
            consumables=self.consumables.copy(),
            vouchers=self.vouchers.copy(),
            joker_slots=self.joker_slots,
            consumable_slots=self.consumable_slots,
            
            # Shop
            shop_inventory=self.shop_inventory.copy(),
            shop_reroll_cost=self.shop_reroll_cost,
            
            # Statistics
            hands_played_total=self.hands_played_total,
            hands_played_ante=self.hands_played_ante,
            best_hand_this_ante=self.best_hand_this_ante,
            jokers_sold=self.jokers_sold,
            cards_discarded_total=self.cards_discarded_total,
            rerolls_used=self.rerolls_used,
            shop_visits=self.shop_visits,
            
            # Levels and states - need deep copies
            hand_levels=self.hand_levels.copy(),
            card_states={k: v.copy() for k, v in self.card_states.items()},
            
            # Boss blind
            active_boss_blind=self.active_boss_blind,
            boss_blind_active=self.boss_blind_active,
            face_down_cards=self.face_down_cards.copy(),
            force_draw_count=self.force_draw_count,
            disabled_joker_slots=self.disabled_joker_slots,
            
            # Special modes
            eternal_jokers=self.eternal_jokers.copy(),
            perishable_counters=self.perishable_counters.copy(),
            rental_jokers=self.rental_jokers.copy(),
        )
    
    def reset_round_state(self):
        """Reset state for a new round (but not a new ante)."""
        self.round_chips_scored = 0
        self.face_down_cards = []
        self.force_draw_count = None
        
        # Reset per-round card tracking
        for card_state in self.card_states.values():
            card_state.is_face_down = False
    
    def reset_ante_state(self):
        """Reset state for a new ante."""
        self.reset_round_state()
        self.hands_played_ante = 0
        self.best_hand_this_ante = 0
        self.boss_blind_active = False
        self.active_boss_blind = None
        self.disabled_joker_slots = 0
        
        # Update perishable counters
        expired_jokers = []
        for joker_idx, rounds_left in self.perishable_counters.items():
            self.perishable_counters[joker_idx] = rounds_left - 1
            if self.perishable_counters[joker_idx] <= 0:
                expired_jokers.append(joker_idx)
        
        # Remove expired perishable jokers
        for joker_idx in expired_jokers:
            if 0 <= joker_idx < len(self.jokers):
                self.jokers.pop(joker_idx)
            del self.perishable_counters[joker_idx]
    
    def add_joker(self, joker: JokerInfo, eternal: bool = False, 
                  perishable: bool = False, rental: bool = False) -> bool:
        """Add a joker with optional modifiers."""
        if len(self.jokers) >= self.joker_slots:
            return False
        
        self.jokers.append(joker)
        joker_idx = len(self.jokers) - 1
        
        if eternal:
            self.eternal_jokers.append(joker_idx)
        if perishable:
            self.perishable_counters[joker_idx] = 5  # 5 rounds before perishing
        if rental:
            self.rental_jokers.append(joker_idx)
        
        return True
    
    def remove_joker(self, joker_idx: int) -> Optional[JokerInfo]:
        """Remove a joker by index, handling all associated state."""
        if not (0 <= joker_idx < len(self.jokers)):
            return None
        
        # Check if joker is eternal
        if joker_idx in self.eternal_jokers:
            return None  # Can't remove eternal jokers
        
        removed = self.jokers.pop(joker_idx)
        
        # Clean up associated state
        if joker_idx in self.eternal_jokers:
            self.eternal_jokers.remove(joker_idx)
        if joker_idx in self.perishable_counters:
            del self.perishable_counters[joker_idx]
        if joker_idx in self.rental_jokers:
            self.rental_jokers.remove(joker_idx)
        
        # Adjust indices for remaining jokers
        self.eternal_jokers = [idx - 1 if idx > joker_idx else idx 
                               for idx in self.eternal_jokers]
        self.rental_jokers = [idx - 1 if idx > joker_idx else idx 
                             for idx in self.rental_jokers]
        
        # Adjust perishable counters
        new_perishable = {}
        for idx, count in self.perishable_counters.items():
            new_idx = idx - 1 if idx > joker_idx else idx
            if new_idx >= 0:
                new_perishable[new_idx] = count
        self.perishable_counters = new_perishable
        
        return removed
    
    def get_active_joker_count(self) -> int:
        """Get number of active (non-disabled) joker slots."""
        return max(0, len(self.jokers) - self.disabled_joker_slots)
    
    def get_hand_cards(self) -> List[Card]:
        """Get the actual Card objects currently in hand."""
        cards = []
        for idx in self.hand_indexes:
            if 0 <= idx < len(self.deck):
                cards.append(self.deck[idx])
        return cards
    
    def get_selected_cards(self) -> List[Card]:
        """Get the actual Card objects currently selected."""
        cards = []
        for hand_idx in self.selected_cards:
            if 0 <= hand_idx < len(self.hand_indexes):
                deck_idx = self.hand_indexes[hand_idx]
                if 0 <= deck_idx < len(self.deck):
                    cards.append(self.deck[deck_idx])
        return cards
