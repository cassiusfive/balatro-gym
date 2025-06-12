"""balatro_gym/envs/balatro_env_complete.py - Complete Balatro RL Environment

This is the complete, production-ready Balatro environment with all systems integrated:
- BalatroGame with ScoreEngine for game mechanics
- Shop system with full joker/consumable support
- CompleteJokerEffects for all 150+ joker abilities
- Unified scoring system that properly calculates scores
- Tarot/Spectral card effects
- Complete action space and observations
- Proper reward shaping for RL
- Full RNG control and reproducibility
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum
from dataclasses import dataclass, field
import random
import pickle
import numpy as np

import gymnasium as gym
from gymnasium import spaces

# Import card primitives first
from balatro_gym.core.cards import (
    Card, Suit, Rank,
    Enhancement, Edition, Seal,
    EnhancementEffects, EditionEffects, SealEffects, CardState
)

# Import constants
from balatro_gym.core.constants import Phase, Action

# Import all game modules
from balatro_gym.core.balatro_game import BalatroGame
from balatro_gym.core.scoring_engine import ScoreEngine, HandType
from balatro_gym.core.shop import Shop, ShopAction, PlayerState, ItemType
from balatro_gym.core.jokers import JokerInfo, JOKER_LIBRARY
from balatro_gym.core.planets import Planet
from balatro_gym.core.consumables import (
    TarotCard, SpectralCard, ConsumableManager
)
from balatro_gym.core.unified_scoring import UnifiedScorer, ScoringContext
from balatro_gym.core.complete_joker_effects import CompleteJokerEffects

# Import boss blind system
from balatro_gym.core.boss_blinds import (
    BossBlindType, BossBlind, BOSS_BLINDS, BossBlindManager, select_boss_blind
)

# Blind scaling
BLIND_CHIPS = {
    1: {'small': 300, 'big': 450, 'boss': 600},
    2: {'small': 450, 'big': 675, 'boss': 900},
    3: {'small': 600, 'big': 900, 'boss': 1200},
    4: {'small': 900, 'big': 1350, 'boss': 1800},
    5: {'small': 1350, 'big': 2025, 'boss': 2700},
    6: {'small': 2100, 'big': 3150, 'boss': 4200},
    7: {'small': 3300, 'big': 4950, 'boss': 6600},
    8: {'small': 5250, 'big': 7875, 'boss': 10500},
}

def get_blind_chips(ante: int, blind_type: str) -> int:
    """Get blind chip requirement for any ante level"""
    if ante <= 8:
        return BLIND_CHIPS[ante][blind_type]
    else:
        # Exponential scaling after ante 8
        base = BLIND_CHIPS[8][blind_type]
        multiplier = 1.5 ** (ante - 8)
        return int(base * multiplier)

# Create joker mappings from JOKER_LIBRARY
JOKER_ID_TO_NAME = {joker.id: joker.name for joker in JOKER_LIBRARY}
JOKER_NAME_TO_ID = {joker.name: joker.id for joker in JOKER_LIBRARY}

# ---------------------------------------------------------------------------
# RNG System for Full Reproducibility
# ---------------------------------------------------------------------------

class DeterministicRNG:
    """Centralized RNG system with separate streams for each subsystem"""

    def __init__(self, master_seed: Optional[int] = None):
        self.master_seed = master_seed or random.randint(0, 2**32 - 1)
        self.streams = {}
        self.history = []
        self._initialize_streams()

    def _initialize_streams(self):
        """Create separate RNG streams for each game subsystem"""
        stream_names = [
            'deck_shuffle', 'card_draw', 'shop_generation', 'shop_reroll',
            'joker_effects', 'blind_selection', 'skip_rewards', 'pack_opening',
            'voucher_appearance', 'boss_abilities', 'random_events',
            'card_enhancement', 'edition_rolls', 'seal_applications',
            'consumable_effects', 'score_variance'
        ]

        for i, name in enumerate(stream_names):
            # Each stream gets a unique seed derived from master seed
            stream_seed = (self.master_seed + i * 1000) % (2**32)
            self.streams[name] = random.Random(stream_seed)

    def get_float(self, stream: str, low: float = 0.0, high: float = 1.0) -> float:
        """Get random float from a specific stream"""
        if stream not in self.streams:
            raise ValueError(f"Unknown RNG stream: {stream}")

        value = self.streams[stream].uniform(low, high)
        self.history.append((stream, 'float', value))
        return value

    def get_int(self, stream: str, low: int, high: int) -> int:
        """Get random integer from a specific stream (inclusive)"""
        if stream not in self.streams:
            raise ValueError(f"Unknown RNG stream: {stream}")

        value = self.streams[stream].randint(low, high)
        self.history.append((stream, 'int', value))
        return value

    def choice(self, stream: str, sequence: List[Any]) -> Any:
        """Make a random choice from a sequence"""
        if stream not in self.streams:
            raise ValueError(f"Unknown RNG stream: {stream}")

        if not sequence:
            raise ValueError("Cannot choose from empty sequence")

        value = self.streams[stream].choice(sequence)
        self.history.append((stream, 'choice', value))
        return value

    def shuffle(self, stream: str, sequence: List[Any]) -> None:
        """Shuffle a sequence in-place"""
        if stream not in self.streams:
            raise ValueError(f"Unknown RNG stream: {stream}")

        self.streams[stream].shuffle(sequence)
        self.history.append((stream, 'shuffle', len(sequence)))

    def get_state(self) -> Dict[str, Any]:
        """Get complete RNG state for saving"""
        return {
            'master_seed': self.master_seed,
            'streams': {name: rng.getstate() for name, rng in self.streams.items()},
            'history_length': len(self.history)
        }

    def set_state(self, state: Dict[str, Any]) -> None:
        """Restore complete RNG state"""
        self.master_seed = state['master_seed']
        for name, stream_state in state['streams'].items():
            if name in self.streams:
                self.streams[name].setstate(stream_state)

# ---------------------------------------------------------------------------
# Unified Game State
# ---------------------------------------------------------------------------

@dataclass
class UnifiedGameState:
    """Single source of truth for all game state"""
    # Core game state
    ante: int = 1
    round: int = 1  # 1=small, 2=big, 3=boss
    phase: Phase = Phase.BLIND_SELECT
    chips_needed: int = 300
    chips_scored: int = 0
    round_chips_scored: int = 0  # Chips scored in current round only
    money: int = 4

    # Cards and hands
    deck: List[Card] = field(default_factory=list)
    hand_indexes: List[int] = field(default_factory=list)
    selected_cards: List[int] = field(default_factory=list)
    hands_left: int = 4
    discards_left: int = 3
    hand_size: int = 8

    # Collections
    jokers: List[JokerInfo] = field(default_factory=list)
    consumables: List[str] = field(default_factory=list)
    vouchers: List[str] = field(default_factory=list)
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

    # Hand levels
    hand_levels: Dict[HandType, int] = field(default_factory=dict)

    # Card states (card index -> CardState)
    card_states: Dict[int, CardState] = field(default_factory=dict)

    # Boss blind state
    active_boss_blind: Optional[BossBlindType] = None
    boss_blind_active: bool = False
    face_down_cards: List[int] = field(default_factory=list)
    force_draw_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict for joker effects and other systems"""
        return {
            'deck': self.deck,
            'hand': [self.deck[i] for i in self.hand_indexes if i < len(self.deck)],
            'jokers': [{'name': j.name, 'id': j.id} for j in self.jokers],
            'consumables': self.consumables,
            'vouchers': self.vouchers,
            'money': self.money,
            'ante': self.ante,
            'round': self.round,
            'hands_left': self.hands_left,
            'discards_left': self.discards_left,
            'hand_size': self.hand_size,
            'joker_slots': self.joker_slots,
            'consumable_slots': self.consumable_slots,
            'hands_played': self.hands_played_total,
            'hands_played_ante': self.hands_played_ante,
            'round_chips_scored': self.round_chips_scored,
        }

    def copy(self) -> 'UnifiedGameState':
        """Create a deep copy of the state"""
        return UnifiedGameState(
            ante=self.ante,
            round=self.round,
            phase=self.phase,
            chips_needed=self.chips_needed,
            chips_scored=self.chips_scored,
            round_chips_scored=self.round_chips_scored,
            money=self.money,
            deck=self.deck.copy() if self.deck else [],
            hand_indexes=self.hand_indexes.copy(),
            selected_cards=self.selected_cards.copy(),
            hands_left=self.hands_left,
            discards_left=self.discards_left,
            hand_size=self.hand_size,
            jokers=self.jokers.copy(),
            consumables=self.consumables.copy(),
            vouchers=self.vouchers.copy(),
            joker_slots=self.joker_slots,
            consumable_slots=self.consumable_slots,
            shop_inventory=self.shop_inventory.copy(),
            shop_reroll_cost=self.shop_reroll_cost,
            hands_played_total=self.hands_played_total,
            hands_played_ante=self.hands_played_ante,
            best_hand_this_ante=self.best_hand_this_ante,
            jokers_sold=self.jokers_sold,
            hand_levels=self.hand_levels.copy(),
            card_states=self.card_states.copy(),
            active_boss_blind=self.active_boss_blind,
            boss_blind_active=self.boss_blind_active,
            face_down_cards=self.face_down_cards.copy(),
            force_draw_count=self.force_draw_count,
        )

# ---------------------------------------------------------------------------
# Card Adapter for Converting Between Representations
# ---------------------------------------------------------------------------

class CardAdapter:
    """Convert between different card representations used by various modules"""

    @staticmethod
    def from_game_card(game_card: Any) -> Card:
        """Convert old game card format to new Card primitive"""
        # Handle old game card format (may have rank/suit as attributes)
        if hasattr(game_card, 'rank') and hasattr(game_card, 'suit'):
            rank_value = game_card.rank.value + 2 if hasattr(game_card.rank, 'value') else game_card.rank
            suit_value = game_card.suit.value if hasattr(game_card.suit, 'value') else game_card.suit
            return Card(rank=Rank(rank_value), suit=Suit(suit_value))
        return game_card  # Already in new format

    @staticmethod
    def to_scoring_format(card: Card, card_idx: Optional[int] = None, state: Optional[UnifiedGameState] = None) -> Any:
        """Convert Card to scoring engine format with enhancements"""
        # Get base chip value
        base_chips = card.rank.base_chips

        # Get card state if available
        card_state = None
        if state and card_idx is not None:
            card_state = state.card_states.get(card_idx, CardState(card_idx))

        # Calculate modified values
        if card_state:
            # Apply enhancement effects
            chip_value = card_state.calculate_chip_bonus(base_chips)

            # Check if card is stone (no rank/suit)
            if card_state.enhancement == Enhancement.STONE:
                rank_value = 0  # Stone cards have no rank
                suit_name = 'Stone'
            else:
                rank_value = card.rank.value
                suit_name = card.suit.name.title()
        else:
            chip_value = base_chips
            rank_value = card.rank.value
            suit_name = card.suit.name.title()

        return type('ScoringCard', (), {
            'rank': rank_value,
            'suit': suit_name,
            'base_value': base_chips,
            'chip_value': lambda: chip_value,
            'enhancement': card_state.enhancement if card_state else Enhancement.NONE,
            'edition': card_state.edition if card_state else Edition.NONE,
            'seal': card_state.seal if card_state else Seal.NONE,
            'card_state': card_state,
            'original_card': card
        })

    @staticmethod
    def to_consumable_format(card: Card, card_idx: Optional[int] = None, state: Optional[UnifiedGameState] = None) -> Any:
        """Convert Card to consumable module format"""
        # Get card state
        card_state = None
        if state and card_idx is not None:
            card_state = state.card_states.get(card_idx, CardState(card_idx))

        # Create a card-like object that consumable system expects
        return type('ConsumableCard', (), {
            'rank': card.rank,
            'suit': card.suit,
            'enhancement': card_state.enhancement if card_state else Enhancement.NONE,
            'edition': card_state.edition if card_state else Edition.NONE,
            'seal': card_state.seal if card_state else Seal.NONE,
            'card_idx': card_idx
        })

    @staticmethod
    def encode_to_int(card: Card) -> int:
        """Encode card to 0-51 integer for observations"""
        return int(card)  # Uses Card.__int__ method

# ---------------------------------------------------------------------------
# Complete Balatro RL Environment
# ---------------------------------------------------------------------------

class BalatroEnv(gym.Env):
    """Complete Balatro environment with all game systems integrated"""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, *, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed

        # Initialize RNG system
        self.rng = DeterministicRNG(seed)

        # Action and observation spaces
        self.action_space = spaces.Discrete(Action.ACTION_SPACE_SIZE)
        self.observation_space = self._create_observation_space()

        # Initialize state
        self.state = UnifiedGameState()

        # Game systems (initialized in reset)
        self.engine: Optional[ScoreEngine] = None
        self.game: Optional[BalatroGame] = None
        self.joker_effects_engine: Optional[CompleteJokerEffects] = None
        self.consumable_manager: Optional[ConsumableManager] = None
        self.unified_scorer: Optional[UnifiedScorer] = None
        self.shop: Optional[Shop] = None
        self.boss_blind_manager: Optional[BossBlindManager] = None

        # Initialize environment
        self.reset()

    def _create_observation_space(self):
        """Create the complete observation space"""
        return spaces.Dict({
            # Hand and card state
            'hand': spaces.Box(-1, 51, (8,), dtype=np.int8),
            'hand_size': spaces.Box(0, 12, (), dtype=np.int8),
            'deck_size': spaces.Box(0, 52, (), dtype=np.int8),
            'selected_cards': spaces.MultiBinary(8),

            # Scoring state - FIXED: Use int64 for large scores
            'chips_scored': spaces.Box(0, 10_000_000_000, (), dtype=np.int64),  # Changed to int64
            'round_chips_scored': spaces.Box(0, 10_000_000, (), dtype=np.int32),
            'progress_ratio': spaces.Box(0.0, 2.0, (), dtype=np.float32),
            'mult': spaces.Box(0, 10_000, (), dtype=np.int32),
            'chips_needed': spaces.Box(0, 10_000_000, (), dtype=np.int32),  # Increased range
            'money': spaces.Box(-20, 999, (), dtype=np.int32),

            # Round state - FIXED: Use larger types
            'ante': spaces.Box(1, 1000, (), dtype=np.int16),  # Changed to int16
            'round': spaces.Box(1, 3, (), dtype=np.int8),
            'hands_left': spaces.Box(0, 12, (), dtype=np.int8),
            'discards_left': spaces.Box(0, 10, (), dtype=np.int8),

            # Jokers
            'joker_count': spaces.Box(0, 10, (), dtype=np.int8),
            'joker_ids': spaces.Box(0, 200, (10,), dtype=np.int16),
            'joker_slots': spaces.Box(0, 10, (), dtype=np.int8),

            # Consumables
            'consumable_count': spaces.Box(0, 5, (), dtype=np.int8),
            'consumables': spaces.Box(0, 100, (5,), dtype=np.int16),
            'consumable_slots': spaces.Box(0, 5, (), dtype=np.int8),

            # Shop
            'shop_items': spaces.Box(0, 300, (10,), dtype=np.int16),
            'shop_costs': spaces.Box(0, 5000, (10,), dtype=np.int16),
            'shop_rerolls': spaces.Box(0, 999, (), dtype=np.int16),

            # Hand levels (for each poker hand type)
            'hand_levels': spaces.Box(0, 15, (12,), dtype=np.int8),

            # Phase and action validity
            'phase': spaces.Box(0, 3, (), dtype=np.int8),
            'action_mask': spaces.MultiBinary(Action.ACTION_SPACE_SIZE),

            # Stats for reward shaping
            'hands_played': spaces.Box(0, 10000, (), dtype=np.int32),  # Increased range
            'best_hand_this_ante': spaces.Box(0, 10_000_000, (), dtype=np.int32),  # Increased range

            # Boss blind info
            'boss_blind_active': spaces.Box(0, 1, (), dtype=np.int8),
            'boss_blind_type': spaces.Box(0, 30, (), dtype=np.int8),
            'face_down_cards': spaces.MultiBinary(8),

            # # ADD: Better hand representation
            # 'hand_one_hot': spaces.Box(0, 1, (8, 52), dtype=np.float32),  # One-hot encoding of cards
            # 'hand_suits': spaces.Box(0, 4, (8,), dtype=np.int8),  # Suit counts per position
            # 'hand_ranks': spaces.Box(0, 13, (8,), dtype=np.int8),  # Rank values per position

            # # ADD: Hand potential analysis
            # 'rank_counts': spaces.Box(0, 4, (13,), dtype=np.int8),  # How many of each rank in hand
            # 'suit_counts': spaces.Box(0, 8, (4,), dtype=np.int8),   # How many of each suit in hand
            # 'straight_potential': spaces.Box(0, 1, (), dtype=np.float32),  # Probability of making straight
            # 'flush_potential': spaces.Box(0, 1, (), dtype=np.float32),     # Probability of making flush

            # # ADD: Historical context
            # 'avg_score_per_hand': spaces.Box(0, 10000, (), dtype=np.float32),
            # 'hands_until_shop': spaces.Box(0, 20, (), dtype=np.int8),
            # 'rounds_until_boss': spaces.Box(0, 3, (), dtype=np.int8),

            # # ADD: Joker synergy indicators
            # 'has_mult_jokers': spaces.Box(0, 1, (), dtype=np.int8),
            # 'has_chip_jokers': spaces.Box(0, 1, (), dtype=np.int8),
            # 'has_xmult_jokers': spaces.Box(0, 1, (), dtype=np.int8),
            # 'has_economy_jokers': spaces.Box(0, 1, (), dtype=np.int8),
            # 'hand_potential_scores': spaces.Box(0, 10000, (12,), dtype=np.int32),  # Expected score for each hand type
            # 'joker_synergy_score': spaces.Box(0, 10, (), dtype=np.float32),
            # 'risk_level': spaces.Box(0, 1, (), dtype=np.float32),  # How close to losing
            # 'economy_health': spaces.Box(0, 1, (), dtype=np.float32),  # Money relative to needs


            # # ADD: Risk indicators
            # 'blind_difficulty': spaces.Box(0, 1, (), dtype=np.float32),  # Normalized difficulty
            # 'win_probability': spaces.Box(0, 1, (), dtype=np.float32),   # Estimated win chance

        })
    def _calculate_hand_features(self, hand_cards: List[Card]) -> Dict[str, np.ndarray]:
        """Calculate advanced hand features for better decision making"""
        features = {}

        rank_counts = np.zeros(13, dtype=np.int8)
        suit_counts = np.zeros(4, dtype=np.int8)

        for card in hand_cards:
            if card:
                rank_counts[card.rank.value - 2] += 1
                suit_counts[card.suit.value] += 1

        features['rank_counts'] = rank_counts
        features['suit_counts'] = suit_counts

        # Straight potential (simplified)
        consecutive = 0
        max_consecutive = 0
        sorted_ranks = sorted(set(card.rank.value for card in hand_cards if card))

        for i in range(1, len(sorted_ranks)):
            if sorted_ranks[i] - sorted_ranks[i-1] == 1:
                consecutive += 1
                max_consecutive = max(max_consecutive, consecutive)
            else:
                consecutive = 0

        features['straight_potential'] = min(1.0, max_consecutive / 4.0)

        # Flush potential
        max_suit = max(suit_counts)
        features['flush_potential'] = min(1.0, max_suit / 5.0)

        return features
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment to initial state"""
        if seed is not None:
            self._seed = seed
            self.rng = DeterministicRNG(seed)

        # Reset state
        self.state = UnifiedGameState()
        self.state.round_chips_scored = 0  # Initialize round score

        # Initialize core game systems with RNG
        self.engine = ScoreEngine()

        # Create initial deck using new Card primitives
        initial_deck = []
        for suit in Suit:
            for rank in Rank:
                initial_deck.append(Card(rank=rank, suit=suit))

        # Shuffle deck
        self.rng.shuffle('deck_shuffle', initial_deck)

        # Initialize game with our deck
        self.game = BalatroGame(engine=self.engine)
        # Replace game's deck with our Card primitives
        self.game.deck = initial_deck
        self.state.deck = initial_deck

        self.joker_effects_engine = CompleteJokerEffects()
        self.consumable_manager = ConsumableManager()
        self.boss_blind_manager = BossBlindManager()

        # Initialize unified scorer
        self.unified_scorer = UnifiedScorer(self.engine, self.joker_effects_engine)

        # Initialize hand levels
        for hand_type in HandType:
            # Check if it's a valid playable hand type
            # You might need to check the actual HandType enum definition
            try:
                level = self.engine.get_hand_level(hand_type)
                self.state.hand_levels[hand_type] = level
            except (KeyError, ValueError):
                # Skip hand types that don't have levels
                continue

        # Sync state with game
        self._sync_state_from_game()

        # If starting in play phase, apply initial boss blind effects
        if self.state.phase == Phase.PLAY and self.state.boss_blind_active:
            self._apply_boss_blind_to_hand()

        return self._get_observation(), {}

    def _calculate_steel_bonus(self) -> float:
        """Calculate mult multiplier from steel cards remaining in hand"""
        steel_mult = 1.0
        selected_hand_indexes = {self.state.hand_indexes[i] for i in self.state.selected_cards
                                 if i < len(self.state.hand_indexes)}
        for idx in self.state.hand_indexes:
            if idx not in selected_hand_indexes:
                card_state = self.state.card_states.get(idx)
                if card_state and card_state.enhancement == Enhancement.STEEL:
                    steel_mult *= EnhancementEffects.get_mult_multiplier(Enhancement.STEEL, in_hand=True)
        return steel_mult

    def _apply_boss_blind_to_hand(self):
        """Apply boss blind effects when drawing a new hand"""
        if not self.state.boss_blind_active or not self.boss_blind_manager.active_blind:
            return

        hand_cards = [self.state.deck[i] for i in self.state.hand_indexes]
        effects = self.boss_blind_manager.on_hand_drawn(hand_cards, self.state.to_dict())

        # Apply face down effects
        if 'face_down_cards' in effects:
            self.state.face_down_cards = effects['face_down_cards']

        # Apply forced discards (The Hook)
        if 'discarded_cards' in effects:
            for idx in sorted(effects['discarded_cards'], reverse=True):
                if idx < len(self.state.hand_indexes):
                    self.state.hand_indexes.pop(idx)
            self._sync_state_to_game()

    def _sync_state_from_game(self):
        """Sync unified state from game systems"""
        if self.game:
            # Preserve scoring state
            current_total_score = self.state.chips_scored
            current_round_score = self.state.round_chips_scored

            self.state.deck = self.game.deck
            self.state.hand_indexes = self.game.hand_indexes
            self.state.hands_left = self.game.round_hands
            self.state.discards_left = self.game.round_discards

            # Restore scoring state
            self.state.chips_scored = current_total_score
            self.state.round_chips_scored = current_round_score

    def _sync_state_to_game(self):
        """Sync game systems from unified state"""
        if self.game:
            self.game.deck = self.state.deck
            self.game.hand_indexes = self.state.hand_indexes
            self.game.round_hands = self.state.hands_left
            self.game.round_discards = self.state.discards_left
            self.game.round_score = self.state.chips_scored

    def step(self, action: int):
        """Execute action and return step results"""
        # Check for termination conditions
        if self.state.ante > 100:  # Terminate after ante 100
            return self._get_observation(), 0.0, True, False, {'terminated': 'max_ante_reached'}

        if self.state.chips_scored > 1_000_000_000:  # Terminate if score gets too high
            return self._get_observation(), 0.0, True, False, {'terminated': 'max_score_reached'}

        # Validate action
        if not self._is_valid_action(action):
            return self._get_observation(), -1.0, False, False, {'error': 'Invalid action'}

        # Route to appropriate handler
        if self.state.phase == Phase.PLAY:
            return self._step_play(action)
        elif self.state.phase == Phase.SHOP:
            return self._step_shop(action)
        elif self.state.phase == Phase.BLIND_SELECT:
            return self._step_blind_select(action)
        elif self.state.phase == Phase.PACK_OPEN:
            return self._step_pack_open(action)

    def _step_play(self, action: int):
        """Handle playing phase actions with unified scoring and enhanced reward shaping"""
        reward = 0.0
        terminated = False
        info = {}

        if action == Action.PLAY_HAND:
            if len(self.state.selected_cards) == 0:
                return self._get_observation(), -1.0, False, False, {'error': 'No cards selected'}

            # Get the actual cards for scoring
            selected_cards = []
            selected_game_cards = []
            for idx in self.state.selected_cards:
                if idx < len(self.state.hand_indexes):
                    card_idx = self.state.hand_indexes[idx]
                    if card_idx < len(self.state.deck):
                        card = self.state.deck[card_idx]
                        selected_game_cards.append(card)
                        # Convert to scoring format with enhancements
                        scoring_card = CardAdapter.to_scoring_format(card, card_idx, self.state)
                        selected_cards.append(scoring_card)

            # Highlight cards in the game
            self._sync_state_to_game()
            for idx in self.state.selected_cards:
                if idx < len(self.game.hand_indexes):
                    self.game.highlight_card(idx)

            # Classify the hand
            hand_type, _ = self.game._classify_hand(
                [self.game.deck[i] for i in self.game.highlighted_indexes]
            )

            # Get hand type name
            hand_type_name = hand_type.name.replace('_', ' ').title()

            # Check if boss blind allows playing this hand
            if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
                can_play, message = self.boss_blind_manager.can_play_hand(selected_game_cards, hand_type_name)
                if not can_play:
                    return self._get_observation(), -1.0, False, False, {'error': message}

            # Create scoring context
            scoring_context = ScoringContext(
                cards=selected_cards,
                scoring_cards=selected_cards,
                hand_type=hand_type,
                hand_type_name=hand_type_name,
                game_state=self.state.to_dict()
            )

            # Use unified scorer
            base_score, breakdown = self.unified_scorer.score_hand(scoring_context)

            # Apply card enhancement/edition/seal effects
            final_chips = base_score
            final_mult = 1
            extra_money = 0
            cards_to_destroy = []
            cards_to_retrigger = []
            consumables_to_create = []

            # Process each played card for special effects
            for i, (idx, card, scoring_card) in enumerate(zip(self.state.selected_cards, selected_game_cards, selected_cards)):
                if idx < len(self.state.hand_indexes):
                    card_idx = self.state.hand_indexes[idx]
                    card_state = self.state.card_states.get(card_idx)

                    if card_state:
                        # Apply enhancement effects
                        if card_state.enhancement == Enhancement.GLASS:
                            # Check if glass breaks
                            if self.rng.get_float('card_enhancement') < 0.25:
                                cards_to_destroy.append(card_idx)
                        elif card_state.enhancement == Enhancement.GOLD:
                            # Gold cards will earn money at end of round
                            pass  # Handled in round end
                        elif card_state.enhancement == Enhancement.LUCKY:
                            mult_roll = self.rng.get_float('card_enhancement')
                            money_roll = self.rng.get_float('card_enhancement')
                            lucky_mult, lucky_money = EnhancementEffects.get_lucky_bonus(mult_roll, money_roll)
                            if lucky_mult > 0:
                                final_mult += lucky_mult
                            if lucky_money > 0:
                                extra_money += lucky_money

                        # Apply seal effects
                        if card_state.seal == Seal.GOLD:
                            extra_money += SealEffects.get_money_bonus(card_state.seal)
                        elif card_state.seal == Seal.RED:
                            cards_to_retrigger.append(i)
                        elif card_state.seal == Seal.BLUE:
                            planet = SealEffects.get_planet_created(card_state.seal, hand_type_name)
                            if planet and len(self.state.consumables) < self.state.consumable_slots:
                                consumables_to_create.append(planet)

            # Calculate final score with enhancement/edition multipliers
            # This is simplified - in reality, the unified scorer should handle these
            final_score = base_score

            # Apply steel card bonus for cards remaining in hand
            steel_mult = self._calculate_steel_bonus()
            final_score = int(final_score * steel_mult)

            # Apply boss blind scoring modifications
            if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
                # Get base chips and mult for modification
                base_chips, base_mult = self.engine.get_hand_chips_mult(hand_type)
                modified_chips, modified_mult = self.boss_blind_manager.modify_scoring(
                    base_chips, base_mult, selected_game_cards, hand_type_name
                )
                # Apply the modification ratio to the final score
                if base_chips > 0 and base_mult > 0:
                    chip_ratio = modified_chips / base_chips
                    mult_ratio = modified_mult / base_mult
                    final_score = int(final_score * chip_ratio * mult_ratio)

            # Apply retriggers (simplified - just add 50% more score per retrigger)
            retrigger_bonus = len(cards_to_retrigger) * 0.5
            final_score = int(final_score * (1 + retrigger_bonus))

            # Add extra money
            self.state.money += extra_money

            # Create consumables
            for consumable in consumables_to_create:
                if len(self.state.consumables) < self.state.consumable_slots:
                    self.state.consumables.append(consumable)

            # Destroy glass cards that broke
            for card_idx in cards_to_destroy:
                # Mark card as destroyed (would need proper implementation)
                pass

            # Store old values for reward calculation
            old_round_score = self.state.round_chips_scored
            old_progress = min(1.0, old_round_score / max(1, self.state.chips_needed))

            # Update state - add to ROUND score
            self.state.round_chips_scored += final_score
            self.state.chips_scored += final_score
            self.state.hands_played_total += 1
            self.state.hands_played_ante += 1
            self.state.best_hand_this_ante = max(self.state.best_hand_this_ante, final_score)

            # Track hand usage for jokers like Obelisk
            self.engine.hand_play_counts[hand_type] += 1

            # Boss blind post-scoring effects
            if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
                self.boss_blind_manager.on_hand_scored(selected_game_cards, hand_type_name, self.state.to_dict())

                # Apply any state changes from boss blind (e.g., The Tooth money penalty)
                if 'money' in self.state.to_dict():
                    self.state.money = self.state.to_dict()['money']

            # Clear selection
            self.state.selected_cards = []

            # ===== ENHANCED REWARD SHAPING =====
            # Calculate progress
            new_progress = min(1.0, self.state.round_chips_scored / max(1, self.state.chips_needed))

            # 1. Progress-based reward (most important)
            progress_reward = 15.0 * new_progress

            # 2. Milestone bonuses for crossing thresholds
            milestone_reward = 0.0
            if old_progress < 0.25 <= new_progress:
                milestone_reward = 5.0
            elif old_progress < 0.5 <= new_progress:
                milestone_reward = 10.0
            elif old_progress < 0.75 <= new_progress:
                milestone_reward = 15.0
            elif old_progress < 1.0 <= new_progress:
                milestone_reward = 25.0

            # 3. Score component with ante-based scaling
            if self.state.ante <= 3:
                score_reward = min(10.0, final_score / 100.0)  # Early game: linear scaling
            else:
                score_reward = min(10.0, 3.0 * np.log10(max(1, final_score)))  # Late game: log scaling

            # 4. Hand quality bonus
            hand_quality_values = {
                HandType.HIGH_CARD: 0.1,
                HandType.ONE_PAIR: 0.5,
                HandType.TWO_PAIR: 1.0,
                HandType.THREE_KIND: 2.0,
                HandType.STRAIGHT: 2.5,
                HandType.FLUSH: 2.5,
                HandType.FULL_HOUSE: 3.5,
                HandType.FOUR_KIND: 5.0,
                HandType.STRAIGHT_FLUSH: 7.0,
                HandType.FIVE_KIND: 10.0
            }
            hand_quality_reward = hand_quality_values.get(hand_type, 0.0)

            # 5. Efficiency bonus (using fewer cards for strong hands)
            efficiency_reward = 0.0
            cards_played = len(selected_game_cards)
            if hand_type >= HandType.THREE_KIND and cards_played <= 3:
                efficiency_reward = 2.0
            elif hand_type >= HandType.FLUSH and cards_played == 5:
                efficiency_reward = 1.0
            elif cards_played <= 4 and self.state.hands_left <= 2:
                efficiency_reward = 1.5  # Conservative play when low on hands

            # 6. Joker synergy bonus
            synergy_reward = 0.0
            joker_names = [j.name for j in self.state.jokers]

            # Check for flush synergies
            if hand_type == HandType.FLUSH and any(name in ['Smeared Joker', 'Four Fingers', 'Shortcut'] for name in joker_names):
                synergy_reward += 2.0

            # Check for pair/set synergies
            if hand_type in [HandType.ONE_PAIR, HandType.TWO_PAIR, HandType.THREE_KIND]:
                if any(name in ['Odd Todd', 'Even Steven', 'Jolly Joker', 'Zany Joker'] for name in joker_names):
                    synergy_reward += 1.5

            # Check for face card synergies
            face_cards = sum(1 for c in selected_game_cards if c.rank.value >= 11)
            if face_cards > 0 and any(name in ['Scary Face', 'Smiley Face', 'Business Card'] for name in joker_names):
                synergy_reward += 0.5 * face_cards

            # 7. Strategic play bonus
            strategy_reward = 0.0
            if new_progress > 0.7 and self.state.hands_left >= 3:
                strategy_reward = 2.0  # Saving resources when ahead
            elif new_progress < 0.3 and hand_type >= HandType.FLUSH:
                strategy_reward = 3.0  # Using strong hands when behind

            # 8. Ante progression bonus
            if self.state.ante >= 4:
                ante_bonus = min(5.0, (self.state.ante - 3) * 0.5)
            else:
                ante_bonus = 0.0

            # Combine all reward components
            reward = (
                progress_reward +
                milestone_reward +
                score_reward +
                hand_quality_reward * 2.0 +  # Weight hand quality
                efficiency_reward * 1.5 +      # Weight efficiency
                synergy_reward * 3.0 +         # Weight synergies highly
                strategy_reward * 2.0 +        # Weight strategy
                ante_bonus
            )

            # Cap total reward
            reward = min(reward, 100.0)

            # Add detailed breakdown to info
            info['reward_breakdown'] = {
                'progress': progress_reward,
                'milestone': milestone_reward,
                'score': score_reward,
                'hand_quality': hand_quality_reward,
                'efficiency': efficiency_reward,
                'synergy': synergy_reward,
                'strategy': strategy_reward,
                'ante_bonus': ante_bonus,
                'total': reward
            }

            # Add scoring breakdown to info
            info['score_breakdown'] = breakdown
            info['final_score'] = final_score
            info['hand_type'] = hand_type
            info['cards_played'] = len(selected_game_cards)

            # Check round end conditions using ROUND score
            if self.state.round_chips_scored >= self.state.chips_needed:
                # Beat the blind! Big bonus
                blind_clear_bonus = 25.0 + (10.0 * self.state.ante)
                reward += min(50.0, blind_clear_bonus)  # Cap blind clear bonus
                self._advance_round()
                info['beat_blind'] = True
            elif self.state.hands_left <= 1:
                # Failed the blind - penalty based on how close we got
                failure_penalty = -50.0 * (1.0 - new_progress)
                reward += failure_penalty
                terminated = True
                info['failed'] = True
            else:
                self.state.hands_left -= 1
                self.game.round_hands = self.state.hands_left

                # Draw new cards
                self.game._draw_cards()
                # Don't call _sync_state_from_game() - manually sync only what's needed
                self.state.hand_indexes = self.game.hand_indexes.copy()

                # Apply boss blind effects to newly drawn hand
                if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
                    hand_cards = [self.state.deck[i] for i in self.state.hand_indexes]
                    effects = self.boss_blind_manager.on_hand_drawn(hand_cards, self.state.to_dict())

                    # Apply face down effects
                    if 'face_down_cards' in effects:
                        self.state.face_down_cards = effects['face_down_cards']

                    # Apply forced discards (The Hook)
                    if 'discarded_cards' in effects:
                        for idx in sorted(effects['discarded_cards'], reverse=True):
                            if idx < len(self.state.hand_indexes):
                                self.state.hand_indexes.pop(idx)

                # Handle forced draw count (The Serpent)
                if self.state.force_draw_count is not None:
                    # Ensure we have exactly the forced number of cards
                    while len(self.state.hand_indexes) > self.state.force_draw_count:
                        self.state.hand_indexes.pop()
                    while len(self.state.hand_indexes) < self.state.force_draw_count:
                        # Draw more cards if needed
                        available = [i for i, card in enumerate(self.state.deck) if i not in self.state.hand_indexes]
                        if available:
                            self.state.hand_indexes.append(self.rng.choice('card_draw', available))
                    self.state.force_draw_count = None

        elif action == Action.DISCARD:
            if self.state.discards_left <= 0:
                return self._get_observation(), -1.0, False, False, {'error': 'No discards left'}

            # Get discarded cards
            discarded = []
            purple_seal_count = 0
            discard_joker_names = []

            for idx in self.state.selected_cards:
                if idx < len(self.state.hand_indexes):
                    card_idx = self.state.hand_indexes[idx]
                    if card_idx < len(self.state.deck):
                        card = self.state.deck[card_idx]
                        # Check for purple seal
                        card_state = self.state.card_states.get(card_idx)
                        if card_state and card_state.seal == Seal.PURPLE:
                            purple_seal_count += 1
                        # Create simple card format for discard effects
                        discard_card = type('Card', (), {
                            'rank': card.rank.value,
                            'suit': card.suit.name.title()
                        })
                        discarded.append(discard_card)

            # Apply discard effects
            discard_context = {
                'phase': 'discard',
                'discarded_cards': discarded,
                'last_discarded_card': discarded[-1] if discarded else None,
                'is_first_discard': self.state.discards_left == self.game.discards
            }

            money_from_discards = 0
            for joker in self.state.jokers:
                effect = self.joker_effects_engine.apply_joker_effect(
                    type('Joker', (), {'name': joker.name}),
                    discard_context,
                    self.state.to_dict()
                )
                if effect and 'money' in effect:
                    money_from_discards += effect['money']
                    self.state.money += effect['money']
                # Track discard jokers
                if joker.name in ['Faceless Joker', 'Hit the Road', 'Reserved Parking', 'Luchador']:
                    discard_joker_names.append(joker.name)

            # Execute discard
            self._sync_state_to_game()
            for idx in sorted(self.state.selected_cards, reverse=True):
                if idx < len(self.game.hand_indexes):
                    self.game.highlight_card(idx)

            self.game.discard_hand()
            self.state.discards_left -= 1
            self.state.selected_cards = []
            self._sync_state_from_game()

            # Create tarot cards from purple seals
            if purple_seal_count > 0:
                tarots = ['The Fool', 'The Magician', 'The High Priestess', 'The Empress',
                        'The Emperor', 'The Hierophant', 'The Lovers', 'The Chariot',
                        'Strength', 'The Hermit', 'Wheel of Fortune', 'Justice',
                        'The Hanged Man', 'Death', 'Temperance', 'The Devil',
                        'The Tower', 'The Star', 'The Moon', 'The Sun', 'Judgement', 'The World']

                for _ in range(purple_seal_count):
                    if len(self.state.consumables) < self.state.consumable_slots:
                        tarot = self.rng.choice('seal_applications', tarots)
                        self.state.consumables.append(tarot)
                        info['created_tarot'] = tarot

            # Enhanced discard reward
            reward = 0.2  # Base discard value

            # Bonus for discard synergies
            if discard_joker_names:
                reward += 0.5 * len(discard_joker_names)

            # Money earned bonus
            if money_from_discards > 0:
                reward += money_from_discards / 5.0

            # Strategic discard bonus (when we have good cards to draw)
            progress = self.state.round_chips_scored / max(1, self.state.chips_needed)
            if progress < 0.5 and self.state.discards_left > 1:
                reward += 0.5  # Encourage discarding when behind
            elif progress > 0.8 and self.state.discards_left > 1:
                reward -= 0.3  # Discourage wasteful discards when ahead

        elif Action.SELECT_CARD_BASE <= action < Action.SELECT_CARD_BASE + Action.SELECT_CARD_COUNT:
            card_idx = action - Action.SELECT_CARD_BASE
            if card_idx < len(self.state.hand_indexes):
                if card_idx in self.state.selected_cards:
                    self.state.selected_cards.remove(card_idx)
                else:
                    self.state.selected_cards.append(card_idx)

        elif Action.USE_CONSUMABLE_BASE <= action < Action.USE_CONSUMABLE_BASE + Action.USE_CONSUMABLE_COUNT:
            consumable_idx = action - Action.USE_CONSUMABLE_BASE
            reward, info = self._use_consumable(consumable_idx)

        return self._get_observation(), reward, terminated, False, info

    def _use_consumable(self, consumable_idx: int) -> Tuple[float, Dict]:
        """Use a consumable with full tarot/spectral effects"""
        if consumable_idx >= len(self.state.consumables):
            return -1.0, {'error': 'Invalid consumable'}

        consumable_name = self.state.consumables[consumable_idx]

        # Get target cards if needed
        target_cards = []
        if len(self.state.selected_cards) > 0:
            for idx in self.state.selected_cards:
                if idx < len(self.state.hand_indexes):
                    card_idx = self.state.hand_indexes[idx]
                    if card_idx < len(self.state.deck):
                        card = self.state.deck[card_idx]
                        # Convert to consumable format
                        consumable_card = CardAdapter.to_consumable_format(card, card_idx, self.state)
                        target_cards.append(consumable_card)

        # Apply consumable effect
        result = self.consumable_manager.use_consumable(
            consumable_name, self.state.to_dict(), target_cards
        )

        reward = 0.0
        info = {'consumable_used': consumable_name}

        if result['success']:
            self.state.consumables.pop(consumable_idx)

            # Handle different effect types
            if result.get('money_gained', 0) > 0:
                self.state.money += result['money_gained']
                reward += result['money_gained'] / 10.0

            if result.get('planet_used'):
                # Apply planet to score engine
                planet_map = {
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
                    'Eris': HandType.FLUSH_FIVE
                }
                if result['planet_used'] in planet_map:
                    self.engine.apply_planet(planet_map[result['planet_used']])
                    self.state.hand_levels[planet_map[result['planet_used']]] += 1
                    reward += 10.0

            if result.get('cards_affected'):
                # Update card enhancements/editions/seals based on effects
                for affected in result['cards_affected']:
                    if hasattr(affected, 'card_idx'):
                        card_idx = affected.card_idx
                        # Get or create card state
                        if card_idx not in self.state.card_states:
                            self.state.card_states[card_idx] = CardState(card_idx)

                        # Update properties
                        if hasattr(affected, 'enhancement'):
                            self.state.card_states[card_idx].enhancement = affected.enhancement
                        if hasattr(affected, 'edition'):
                            self.state.card_states[card_idx].edition = affected.edition
                        if hasattr(affected, 'seal'):
                            self.state.card_states[card_idx].seal = affected.seal
                reward += len(result['cards_affected']) * 2.0

            if result.get('cards_created'):
                reward += len(result['cards_created']) * 3.0

            if result.get('cards_destroyed'):
                reward += len(result['cards_destroyed']) * 1.0

            if result.get('jokers_created'):
                for joker_name in result['jokers_created']:
                    if len(self.state.jokers) < self.state.joker_slots:
                        # Find joker info
                        for joker_info in JOKER_LIBRARY:
                            if joker_info.name == joker_name:
                                self.state.jokers.append(joker_info)
                                break
                reward += len(result['jokers_created']) * 15.0

            if result.get('items_created'):
                for item in result['items_created']:
                    if len(self.state.consumables) < self.state.consumable_slots:
                        self.state.consumables.append(item)
                reward += len(result['items_created']) * 5.0

            if result.get('hand_size_change'):
                self.state.hand_size += result['hand_size_change']
                self.game.hand_size = self.state.hand_size

            info['result'] = result['message']
        else:
            reward = -1.0
            info['error'] = result.get('message', 'Failed to use consumable')

        self.state.selected_cards = []
        return reward, info

    def _step_shop(self, action: int):
        """Handle shop phase actions using the Shop module"""
        if self.shop is None:
            return self._get_observation(), -1.0, False, False, {'error': 'No shop available'}

        info = {}

        # Map our action IDs to Shop action IDs
        if action == Action.SHOP_END:
            shop_action = ShopAction.SKIP
        elif action == Action.SHOP_REROLL:
            shop_action = ShopAction.REROLL
        elif Action.SHOP_BUY_BASE <= action < Action.SHOP_BUY_BASE + Action.SHOP_BUY_COUNT:
            item_idx = action - Action.SHOP_BUY_BASE
            if not (0 <= item_idx < len(self.shop.inventory)):
                return self._get_observation(), -1.0, False, False, {'error': 'Invalid item index'}

            item = self.shop.inventory[item_idx]
            if item.item_type == ItemType.PACK:
                shop_action = ShopAction.BUY_PACK_BASE + item_idx
            elif item.item_type == ItemType.JOKER:
                shop_action = ShopAction.BUY_JOKER_BASE + item_idx
            elif item.item_type == ItemType.CARD:
                shop_action = ShopAction.BUY_CARD_BASE + item_idx
            elif item.item_type == ItemType.VOUCHER:
                shop_action = ShopAction.BUY_VOUCHER_BASE + item_idx
            else:
                return self._get_observation(), -1.0, False, False, {'error': 'Unknown item type'}
        elif Action.SELL_JOKER_BASE <= action < Action.SELL_JOKER_BASE + Action.SELL_JOKER_COUNT:
            joker_idx = action - Action.SELL_JOKER_BASE
            if 0 <= joker_idx < len(self.state.jokers):
                sold_joker = self.state.jokers.pop(joker_idx)
                sell_value = max(3, sold_joker.base_cost // 2)
                self.state.money += sell_value
                self.state.jokers_sold += 1

                # Sync with player state
                self._sync_player_state()

                return self._get_observation(), sell_value / 5.0, False, False, {'sold_joker': sold_joker.name}
            else:
                return self._get_observation(), -1.0, False, False, {'error': 'Invalid joker index'}
        else:
            return self._get_observation(), -1.0, False, False, {'error': 'Invalid shop action'}

        # Execute shop action
        self._sync_player_state()
        reward, done_shopping, shop_info = self.shop.step(shop_action)
        info.update(shop_info)

        # Handle successful purchases
        if 'new_cards' in shop_info:
            reward = 5.0
            info['opened_pack'] = True
        elif 'error' not in shop_info and Action.SHOP_BUY_BASE <= action < Action.SHOP_BUY_BASE + Action.SHOP_BUY_COUNT:
            verb, _ = ShopAction.decode(shop_action)
            if verb == "buy_joker":
                # Sync joker from player state
                self._sync_jokers_from_player()
                reward = 15.0
            elif verb == "buy_card":
                reward = 3.0
                info['bought_card'] = True
            elif verb == "buy_voucher":
                # Sync vouchers
                self.state.vouchers = self.shop.player.vouchers.copy()
                reward = 10.0
                info['bought_voucher'] = self.state.vouchers[-1] if self.state.vouchers else None

        # Sync money
        self.state.money = self.shop.player.chips

        # Check if shopping is done
        if done_shopping:
            self.state.phase = Phase.PLAY
            self._sync_state_to_game()
            self.game._draw_cards()
            self._sync_state_from_game()

        return self._get_observation(), reward, False, False, info

    def _step_blind_select(self, action: int):
        """Handle blind selection phase"""
        reward = 0.0
        info = {}

        if Action.SELECT_BLIND_BASE <= action < Action.SELECT_BLIND_BASE + Action.SELECT_BLIND_COUNT:
            blind_type = action - Action.SELECT_BLIND_BASE  # 0=small, 1=big, 2=boss
            self.state.round = blind_type + 1
            blind_key = ['small', 'big', 'boss'][blind_type]

            # Use the new scaling function
            self.state.chips_needed = get_blind_chips(self.state.ante, blind_key)

            # Handle boss blind activation
            if blind_type == 2:  # Boss blind
                # Select and activate boss blind
                boss_type = select_boss_blind(self.state.ante)
                effects = self.boss_blind_manager.activate_boss_blind(boss_type, self.state.to_dict())

                # Apply chip multiplier
                self.state.chips_needed = int(self.state.chips_needed * effects['chip_mult'])

                # Apply modifications
                if 'discards' in effects['modifications']:
                    self.state.discards_left = effects['modifications']['discards']
                    self.game.round_discards = effects['modifications']['discards']
                if 'hand_size' in effects['modifications']:
                    self.state.hand_size += effects['modifications']['hand_size']
                    self.game.hand_size = self.state.hand_size
                if 'hands' in effects['modifications']:
                    self.state.hands_left = effects['modifications']['hands']
                    self.game.round_hands = effects['modifications']['hands']

                self.state.active_boss_blind = boss_type
                self.state.boss_blind_active = True
                info['boss_blind'] = self.boss_blind_manager.active_blind.name
                info['boss_effect'] = self.boss_blind_manager.active_blind.description

                # Boss blind bonus reward
                reward = 10.0

            # Update game blind
            self.game.blinds[self.game.blind_index] = self.state.chips_needed

            # Transition to play
            self.state.phase = Phase.PLAY
            self._sync_state_from_game()
            self.game._draw_cards()       # Draw the initial hand
            self._sync_state_from_game()  # Sync FROM game to get the drawn cards

        elif action == Action.SKIP_BLIND:
            # Skip blind - trigger skip effects
            for joker in self.state.jokers:
                self.joker_effects_engine.apply_joker_effect(
                    type('Joker', (), {'name': joker.name}),
                    {'phase': 'skip_blind'},
                    self.state.to_dict()
                )

            reward = -5.0
            self._advance_round()
            info['skipped_blind'] = True

        return self._get_observation(), reward, False, False, info

    def _step_pack_open(self, action: int):
        """Handle pack opening - simplified for now"""
        self.state.phase = Phase.SHOP
        self._generate_shop()
        return self._get_observation(), 0.0, False, False, {}

    def _advance_round(self):
        """Advance to next round/ante"""
        # Apply end-of-round effects
        end_effects = self.joker_effects_engine.end_of_round_effects(self.state.to_dict())

        # Handle joker destruction
        for effect in end_effects:
            if 'destroy_joker' in effect:
                joker_name = effect['destroy_joker']
                self.state.jokers = [j for j in self.state.jokers if j.name != joker_name]

        # Award money from gold cards held in hand
        gold_money = 0
        for idx in self.state.hand_indexes:
            card_state = self.state.card_states.get(idx)
            if card_state and card_state.enhancement == Enhancement.GOLD:
                gold_money += EnhancementEffects.get_gold_value(card_state.enhancement)
        self.state.money += gold_money

        # Deactivate boss blind if active
        if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
            # Award extra money for beating boss
            self.state.money += self.boss_blind_manager.active_blind.money_reward
            self.boss_blind_manager.deactivate()
            self.state.active_boss_blind = None
            self.state.boss_blind_active = False
            self.state.face_down_cards = []

        # Reset ROUND stats (not total stats)
        self.state.round_chips_scored = 0  # Reset round score
        self.state.best_hand_this_ante = 0
        self.state.hands_played_ante = 0
        # DO NOT reset chips_scored - that's career total

        # Progress ante/round
        if self.state.round == 3:
            self.state.ante += 1
            self.state.round = 1

            # Add termination check
            if self.state.ante > 100:
                return  # Don't continue if we've hit max ante
        else:
            self.state.round += 1

        # Award money
        money_earned = 25 * self.state.round + (10 if self.state.round == 3 else 0)
        self.state.money += money_earned

        # Reset hands/discards
        self.state.hands_left = 4
        self.state.discards_left = 3

        # Go to shop
        self.state.phase = Phase.SHOP
        self._generate_shop()

    def _generate_shop(self):
        """Create shop with items using the Shop module"""
        # Sync player state first
        self._sync_player_state()

        # Use shop RNG stream
        shop_seed = self.rng.get_int('shop_generation', 0, 2**31 - 1)
        self.shop = Shop(self.state.ante, self.shop.player if self.shop else self._create_player_state(), seed=shop_seed)
        self.state.shop_inventory = self.shop.inventory.copy()
        self.state.shop_reroll_cost = int(self.shop.reroll_cost * self.shop._cost_mult())

    def _create_player_state(self) -> PlayerState:
        """Create player state from unified state"""
        player = PlayerState(chips=self.state.money)
        player.jokers = [j.id for j in self.state.jokers]
        player.vouchers = self.state.vouchers.copy()
        return player

    def _sync_player_state(self):
        """Sync player state with unified state"""
        if self.shop and self.shop.player:
            self.shop.player.chips = self.state.money
            self.shop.player.jokers = [j.id for j in self.state.jokers]
            self.shop.player.vouchers = self.state.vouchers.copy()

    def _sync_jokers_from_player(self):
        """Sync jokers from player state back to unified state"""
        if self.shop and self.shop.player:
            # Get new jokers that were added
            current_joker_ids = {j.id for j in self.state.jokers}
            for joker_id in self.shop.player.jokers:
                if joker_id not in current_joker_ids:
                    # Find joker info and add it
                    for joker_info in JOKER_LIBRARY:
                        if joker_info.id == joker_id:
                            self.state.jokers.append(joker_info)
                            break

    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid"""
        mask = self._get_action_mask()
        return bool(mask[action])

    def _get_action_mask(self):
        """Get valid actions for current state - FIXED VERSION"""
        mask = np.zeros(Action.ACTION_SPACE_SIZE, dtype=np.int8)

        if self.state.phase == Phase.PLAY:
            # Card selection - allow selecting any card in hand
            for i in range(min(8, len(self.state.hand_indexes))):
                mask[Action.SELECT_CARD_BASE + i] = 1

            # Play hand if cards selected
            if len(self.state.selected_cards) > 0:
                mask[Action.PLAY_HAND] = 1

            # Discard if cards selected and discards left
            if len(self.state.selected_cards) > 0 and self.state.discards_left > 0:
                mask[Action.DISCARD] = 1

            # Use consumables
            for i in range(len(self.state.consumables)):
                mask[Action.USE_CONSUMABLE_BASE + i] = 1

        elif self.state.phase == Phase.SHOP:
            if self.shop:
                # Buy items if can afford
                for i in range(len(self.shop.inventory)):
                    if self.state.money >= self.shop.inventory[i].cost:
                        mask[Action.SHOP_BUY_BASE + i] = 1

                # Reroll if can afford
                if self.state.money >= self.state.shop_reroll_cost:
                    mask[Action.SHOP_REROLL] = 1

            # Always can end shop
            mask[Action.SHOP_END] = 1

            # Sell jokers
            for i in range(len(self.state.jokers)):
                mask[Action.SELL_JOKER_BASE + i] = 1

        elif self.state.phase == Phase.BLIND_SELECT:
            # Select any blind
            for i in range(Action.SELECT_BLIND_COUNT):
                mask[Action.SELECT_BLIND_BASE + i] = 1
            mask[Action.SKIP_BLIND] = 1

        return mask

    def _get_observation(self):
        """Build observation dict"""
        # Build hand array
        hand_array = np.full(8, -1, dtype=np.int8)
        for i, idx in enumerate(self.state.hand_indexes[:8]):
            if idx < len(self.state.deck):
                hand_array[i] = CardAdapter.encode_to_int(self.state.deck[idx])

        # Get hand levels
        for hand_type in HandType:
                self.state.hand_levels[hand_type] = self.state.hand_levels.get(hand_type, 0)

        # Get consumable IDs
        consumable_ids = self._get_consumable_ids()

        obs = {
            'hand': hand_array,
            'hand_size': np.int8(len(self.state.hand_indexes)),
            'deck_size': np.int8(sum(1 for _ in self.state.deck)),
            'selected_cards': np.array([1 if i in self.state.selected_cards else 0 for i in range(8)]),

            # FIXED: Use proper data types
            'chips_scored': np.int64(self.state.chips_scored),  # Changed to int64
            'round_chips_scored': np.int32(self.state.round_chips_scored),
            'progress_ratio': np.float32(min(2.0, self.state.round_chips_scored / max(1, self.state.chips_needed))),
            'mult': np.int32(1),
            'chips_needed': np.int32(self.state.chips_needed),
            'money': np.int32(self.state.money),

            'ante': np.int16(self.state.ante),  # Changed to int16
            'round': np.int8(self.state.round),
            'hands_left': np.int8(self.state.hands_left),
            'discards_left': np.int8(self.state.discards_left),

            'joker_count': np.int8(len(self.state.jokers)),
            'joker_ids': np.array([j.id for j in self.state.jokers] +
                                 [0] * (10 - len(self.state.jokers)), dtype=np.int16),
            'joker_slots': np.int8(self.state.joker_slots),

            'consumable_count': np.int8(len(self.state.consumables)),
            'consumables': np.array(consumable_ids, dtype=np.int16),
            'consumable_slots': np.int8(self.state.consumable_slots),

            'shop_items': np.zeros(10, dtype=np.int16),
            'shop_costs': np.zeros(10, dtype=np.int16),
            'shop_rerolls': np.int16(self.state.shop_reroll_cost),

            'hand_levels': np.array(list(self.state.hand_levels.values())[:12], dtype=np.int8),
            'phase': np.int8(self.state.phase),
            'action_mask': self._get_action_mask(),

            'hands_played': np.int32(self.state.hands_played_total),  # Changed to int32
            'best_hand_this_ante': np.int32(self.state.best_hand_this_ante),

            # Boss blind info
            'boss_blind_active': np.int8(1 if self.state.boss_blind_active else 0),
            'boss_blind_type': np.int8(self.state.active_boss_blind.value if self.state.active_boss_blind else 0),
            'face_down_cards': np.array([1 if i in self.state.face_down_cards else 0 for i in range(8)]),
        }

        # Add shop info if in shop
        if self.state.phase == Phase.SHOP and self.shop:
            shop_obs = self.shop.get_observation()
            for i, (item_type, cost) in enumerate(zip(shop_obs['shop_item_type'][:10],
                                                      shop_obs['shop_cost'][:10])):
                obs['shop_items'][i] = item_type
                obs['shop_costs'][i] = cost

        return obs

    def _get_consumable_ids(self):
        """Convert consumable names to IDs for observation"""
        consumable_id_map = {
            # Tarots
            'The Fool': 1, 'The Magician': 2, 'The High Priestess': 3,
            'The Empress': 4, 'The Emperor': 5, 'The Hierophant': 6,
            'The Lovers': 7, 'The Chariot': 8, 'Strength': 9,
            'The Hermit': 10, 'Wheel of Fortune': 11, 'Justice': 12,
            'The Hanged Man': 13, 'Death': 14, 'Temperance': 15,
            'The Devil': 16, 'The Tower': 17, 'The Star': 18,
            'The Moon': 19, 'The Sun': 20, 'Judgement': 21,
            'The World': 22,

            # Planets
            'Mercury': 30, 'Venus': 31, 'Earth': 32, 'Mars': 33,
            'Jupiter': 34, 'Saturn': 35, 'Uranus': 36, 'Neptune': 37,
            'Pluto': 38, 'Planet X': 39, 'Ceres': 40, 'Eris': 41,

            # Spectrals
            'Familiar': 50, 'Grim': 51, 'Incantation': 52, 'Talisman': 53,
            'Aura': 54, 'Wraith': 55, 'Sigil': 56, 'Ouija': 57,
            'Ectoplasm': 58, 'Immolate': 59, 'Ankh': 60, 'Deja Vu': 61,
            'Hex': 62, 'Trance': 63, 'Medium': 64, 'Cryptid': 65,
            'The Soul': 66, 'Black Hole': 67
        }

        ids = []
        for consumable in self.state.consumables:
            ids.append(consumable_id_map.get(consumable, 0))

        return ids + [0] * (5 - len(ids))

    def save_state(self) -> Dict[str, Any]:
        """Save complete environment state for checkpointing"""
        return {
            'state': self.state.copy(),
            'rng_state': self.rng.get_state(),
            'engine_state': {
                'hand_levels': self.engine.hand_levels.copy(),
                'hand_play_counts': self.engine.hand_play_counts.copy(),
            },
            'game_state': {
                'deck': self.game.deck.copy(),
                'state': self.game.state,
                'blind_index': self.game.blind_index,
            },
            'boss_blind_state': {
                'active_blind': self.boss_blind_manager.active_blind,
                'blind_state': self.boss_blind_manager.blind_state.copy() if self.boss_blind_manager.blind_state else {}
            }
        }

    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Load complete environment state from checkpoint"""
        self.state = saved_state['state'].copy()
        self.rng.set_state(saved_state['rng_state'])

        # Restore engine state
        self.engine.hand_levels = saved_state['engine_state']['hand_levels'].copy()
        self.engine.hand_play_counts = saved_state['engine_state']['hand_play_counts'].copy()

        # Restore game state
        self.game.deck = saved_state['game_state']['deck'].copy()
        self.game.state = saved_state['game_state']['state']
        self.game.blind_index = saved_state['game_state']['blind_index']

        # Restore boss blind state
        if 'boss_blind_state' in saved_state:
            self.boss_blind_manager.active_blind = saved_state['boss_blind_state']['active_blind']
            self.boss_blind_manager.blind_state = saved_state['boss_blind_state']['blind_state'].copy()

        # Sync states
        self._sync_state_from_game()

    def render(self):
        """Render the game state"""
        if self.render_mode != "human":
            return

        print(f"\n{'='*50}")
        print(f"Ante {self.state.ante} - Round {self.state.round} - Phase: {Phase(self.state.phase).name}")

        # Show boss blind info if active
        if self.state.boss_blind_active and self.boss_blind_manager.active_blind:
            print(f"BOSS BLIND: {self.boss_blind_manager.active_blind.name}")
            print(f"Effect: {self.boss_blind_manager.active_blind.description}")

        # FIXED: Show round score, not total score
        print(f"Score: {self.state.round_chips_scored}/{self.state.chips_needed} | Total: {self.state.chips_scored} | Money: ${self.state.money}")
        print(f"Hands: {self.state.hands_left} | Discards: {self.state.discards_left}")

        if self.state.phase == Phase.PLAY:
            # Show hand with face down indicators and enhancements
            hand_str = ""
            for i, idx in enumerate(self.state.hand_indexes):
                if idx < len(self.state.deck):
                    if i in self.state.face_down_cards:
                        hand_str += "[??] "
                    else:
                        card = self.state.deck[idx]
                        card_str = str(card)

                        # Add enhancement/edition/seal indicators
                        card_state = self.state.card_states.get(idx)
                        if card_state:
                            modifiers = []
                            if card_state.enhancement != Enhancement.NONE:
                                enhancement_symbols = {
                                    Enhancement.BONUS: '+',
                                    Enhancement.MULT: '×',
                                    Enhancement.WILD: 'W',
                                    Enhancement.GLASS: 'G',
                                    Enhancement.STEEL: 'S',
                                    Enhancement.STONE: '◊',
                                    Enhancement.GOLD: '$',
                                    Enhancement.LUCKY: '?'
                                }
                                modifiers.append(enhancement_symbols.get(card_state.enhancement, '!'))

                            if card_state.edition != Edition.NONE:
                                edition_symbols = {
                                    Edition.FOIL: 'F',
                                    Edition.HOLOGRAPHIC: 'H',
                                    Edition.POLYCHROME: 'P',
                                    Edition.NEGATIVE: '-'
                                }
                                modifiers.append(edition_symbols.get(card_state.edition, '!'))

                            if card_state.seal != Seal.NONE:
                                seal_symbols = {
                                    Seal.GOLD: '●G',
                                    Seal.RED: '●R',
                                    Seal.BLUE: '●B',
                                    Seal.PURPLE: '●P'
                                }
                                modifiers.append(seal_symbols.get(card_state.seal, '●'))

                            if modifiers:
                                card_str = f"{card_str}[{''.join(modifiers)}]"

                        hand_str += f"{card_str} "

            print(f"\nHand: {hand_str}")
            if self.state.selected_cards:
                print(f"Selected: {self.state.selected_cards}")

            # Show legend if any cards have modifiers
            if any(idx in self.state.card_states for idx in self.state.hand_indexes):
                print("Legend: + Bonus | × Mult | W Wild | G Glass | S Steel | ◊ Stone | $ Gold | ? Lucky")
                print("        F Foil | H Holo | P Polychrome | - Negative | ● Seal (G/R/B/P)")

        elif self.state.phase == Phase.SHOP and self.shop:
            print("\n=== SHOP ===")
            for i, item in enumerate(self.shop.inventory):
                affordable = "✓" if self.state.money >= item.cost else "✗"
                print(f"[{i}] {affordable} {item.name:<25} ${item.cost}")
            print(f"\nReroll cost: ${self.state.shop_reroll_cost}")

        elif self.state.phase == Phase.BLIND_SELECT:
            print("\n=== SELECT BLIND ===")
            print(f"[0] Small Blind: {get_blind_chips(self.state.ante, 'small')} chips")
            print(f"[1] Big Blind: {get_blind_chips(self.state.ante, 'big')} chips")
            print(f"[2] Boss Blind: {get_blind_chips(self.state.ante, 'boss')} chips")
            print(f"[S] Skip Blind")

        if self.state.jokers:
            joker_display = []
            active_slots = self.state.joker_slots
            if self.state.boss_blind_active:
                active_slots -= self.boss_blind_manager.get_disabled_joker_count()

            for i, joker in enumerate(self.state.jokers[:active_slots]):
                joker_display.append(joker.name)

            # Show disabled jokers
            for i in range(active_slots, len(self.state.jokers)):
                joker_display.append(f"[DISABLED: {self.state.jokers[i].name}]")

            print(f"\nJokers ({len(self.state.jokers)}/{self.state.joker_slots}): {', '.join(joker_display)}")

        if self.state.consumables:
            print(f"Consumables ({len(self.state.consumables)}/{self.state.consumable_slots}): {', '.join(self.state.consumables)}")
    def close(self):
        pass


# ---------------------------------------------------------------------------
# Environment Validator for Testing
# ---------------------------------------------------------------------------

class BalatroEnvValidator:
    """Validate environment behavior matches Balatro"""

    @staticmethod
    def validate_determinism(env_class, seed: int = 42, steps: int = 100):
        """Check if environment is deterministic with same seed"""
        env1 = env_class(seed=seed)
        env2 = env_class(seed=seed)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        # Check initial observations match
        for key in obs1:
            if not np.array_equal(obs1[key], obs2[key]):
                raise AssertionError(f"Initial observations differ for key: {key}")

        # Run same actions
        for i in range(steps):
            # Get valid action
            valid_actions = np.where(obs1['action_mask'])[0]
            if len(valid_actions) == 0:
                break

            action = valid_actions[i % len(valid_actions)]

            obs1, r1, t1, tr1, info1 = env1.step(action)
            obs2, r2, t2, tr2, info2 = env2.step(action)

            # Check all outputs match
            if r1 != r2:
                raise AssertionError(f"Rewards differ at step {i}: {r1} vs {r2}")

            if t1 != t2 or tr1 != tr2:
                raise AssertionError(f"Termination differs at step {i}")

            for key in obs1:
                if not np.array_equal(obs1[key], obs2[key]):
                    raise AssertionError(f"Observations differ at step {i} for key: {key}")

        print(f"✓ Determinism validated over {steps} steps")

    @staticmethod
    def validate_action_masking(env):
        """Check that invalid actions are properly masked"""
        obs, _ = env.reset()

        # Try all actions
        for action in range(env.action_space.n):
            if obs['action_mask'][action]:
                # Should succeed
                _, _, _, _, info = env.step(action)
                if 'error' in info and info['error'] == 'Invalid action':
                    raise AssertionError(f"Valid action {action} was rejected")
            else:
                # Should fail
                obs_before = obs.copy()
                _, reward, _, _, info = env.step(action)
                if 'error' not in info:
                    raise AssertionError(f"Invalid action {action} was accepted")
                if reward != -1.0:
                    raise AssertionError(f"Invalid action {action} gave reward {reward}")

        print("✓ Action masking validated")


# ---------------------------------------------------------------------------
# Factory function for creating environments
# ---------------------------------------------------------------------------

def make_balatro_env(**kwargs):
    """Factory function for creating environments"""
    def _init():
        return BalatroEnv(**kwargs)
    return _init


# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test the complete environment
    env = BalatroEnv(render_mode="human", seed=42)

    # Validate determinism
    print("Testing determinism...")
    BalatroEnvValidator.validate_determinism(BalatroEnv, seed=42, steps=50)

    # Test basic gameplay
    print("\nTesting basic gameplay...")
    obs, _ = env.reset()

    print("Complete Balatro Environment initialized!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"\nStarting interactive test...")

    # Run a test episode
    done = False
    step = 0

    while not done and step < 100:
        env.render()
        valid_actions = np.where(obs['action_mask'])[0]
        print(f"\nValid actions: {valid_actions}")

        # Simple policy for testing
        if env.state.phase == Phase.BLIND_SELECT:
            # Always select small blind
            action = Action.SELECT_BLIND_BASE
        elif env.state.phase == Phase.SHOP:
            # End shop immediately
            action = Action.SHOP_END
        else:
            # Random valid action
            action = np.random.choice(valid_actions) if len(valid_actions) > 0 else 0

        print(f"Taking action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        print(f"Reward: {reward:.2f}")
        if info:
            print(f"Info: {info}")

        step += 1

    print(f"\nEpisode finished after {step} steps")
    print(f"Final score: {env.state.chips_scored}")
    print(f"Reached ante: {env.state.ante}")

    # Test save/load
    print("\nTesting save/load...")
    saved = env.save_state()

    # Take some actions
    for _ in range(5):
        valid = np.where(env._get_action_mask())[0]
        if len(valid) > 0:
            env.step(np.random.choice(valid))

    # Save current observation
    obs_after = env._get_observation()

    # Restore
    env.load_state(saved)
    obs_restored = env._get_observation()

    # Check they match
    for key in obs_restored:
        if not np.array_equal(obs_after[key], obs_restored[key]):
            print(f"✓ State properly changed for key: {key}")

    print("✓ Save/load working correctly")
