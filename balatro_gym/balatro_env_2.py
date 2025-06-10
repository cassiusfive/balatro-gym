"""balatro_gym/envs/balatro_env_complete.py - Complete Balatro RL Environment

This is the complete, production-ready Balatro environment with all systems integrated:
- BalatroGame with ScoreEngine for game mechanics
- Shop system with full joker/consumable support
- CompleteJokerEffects for all 150+ joker abilities
- Unified scoring system that properly calculates scores
- Tarot/Spectral card effects
- Complete action space and observations
- Proper reward shaping for RL
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Any, Tuple
from enum import Enum, IntEnum
import random

import gymnasium as gym
from gymnasium import spaces

# Import all game modules
from balatro_gym.balatro_game_v2 import BalatroGame, Card
from balatro_gym.scoring_engine_accurate import ScoreEngine, HandType
from balatro_gym.shop import Shop, ShopAction, PlayerState, ItemType
from balatro_gym.jokers import JokerInfo, JOKER_LIBRARY
from balatro_gym.planets import Planet
from balatro_gym.consumables import (
    Card as ConsumableCard, Suit, Rank, Enhancement, Edition, Seal,
    TarotCard, SpectralCard, ConsumableManager
)
from balatro_gym.unified_scoring import UnifiedScorer, ScoringContext
from complete_joker_effects import CompleteJokerEffects

# ---------------------------------------------------------------------------
# Action Space Constants
# ---------------------------------------------------------------------------

# Playing phase actions
ACTION_PLAY_HAND = 0
ACTION_DISCARD = 1
ACTION_SELECT_CARDS = range(2, 10)      # Select cards 0-7
ACTION_USE_CONSUMABLE = range(10, 15)   # Use consumables 0-4

# Shop phase actions  
ACTION_SHOP_BUY = range(20, 30)         # Buy shop items 0-9
ACTION_SHOP_REROLL = 30
ACTION_SHOP_END = 31
ACTION_SELL_JOKER = range(32, 37)      # Sell jokers 0-4
ACTION_SELL_CONSUMABLE = range(37, 42)  # Sell consumables 0-4

# Blind selection actions
ACTION_SELECT_BLIND = range(45, 48)     # Select small/big/boss
ACTION_SKIP_BLIND = 48

# Pack opening actions
ACTION_SELECT_FROM_PACK = range(50, 55) # Select from pack 0-4
ACTION_SKIP_PACK = 55

ACTION_SPACE_SIZE = 60

# Game phases
class Phase(IntEnum):
    PLAY = 0
    SHOP = 1
    BLIND_SELECT = 2
    PACK_OPEN = 3

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

# Create joker mappings from JOKER_LIBRARY
JOKER_ID_TO_NAME = {joker.id: joker.name for joker in JOKER_LIBRARY}
JOKER_NAME_TO_ID = {joker.name: joker.id for joker in JOKER_LIBRARY}

# ---------------------------------------------------------------------------
# Complete Balatro RL Environment
# ---------------------------------------------------------------------------

class BalatroEnv(gym.Env):
    """Complete Balatro environment with all game systems integrated"""
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, *, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.render_mode = render_mode
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Action and observation spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = self._create_observation_space()

        # Initialize all systems
        self.reset()

    def _create_observation_space(self):
        """Create the complete observation space"""
        return spaces.Dict({
            # Hand and card state
            'hand': spaces.Box(-1, 51, (8,), dtype=np.int8),
            'hand_size': spaces.Box(0, 12, (), dtype=np.int8),
            'deck_size': spaces.Box(0, 52, (), dtype=np.int8),
            'selected_cards': spaces.MultiBinary(8),
            
            # Scoring state
            'chips_scored': spaces.Box(0, 1_000_000, (), dtype=np.int32),
            'mult': spaces.Box(0, 10_000, (), dtype=np.int32),
            'chips_needed': spaces.Box(0, 100_000, (), dtype=np.int32),
            'money': spaces.Box(-20, 999, (), dtype=np.int32),
            
            # Round state
            'ante': spaces.Box(1, 20, (), dtype=np.int8),
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
            'action_mask': spaces.MultiBinary(ACTION_SPACE_SIZE),
            
            # Stats for reward shaping
            'hands_played': spaces.Box(0, 1000, (), dtype=np.int16),
            'best_hand_this_ante': spaces.Box(0, 1_000_000, (), dtype=np.int32),
        })

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """Reset the environment to initial state"""
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        # Initialize core game systems
        self.engine = ScoreEngine()
        self.game = BalatroGame(engine=self.engine)
        self.joker_effects_engine = CompleteJokerEffects()
        self.consumable_manager = ConsumableManager()
        self.player = PlayerState(chips=4)
        
        # Initialize unified scorer
        self.unified_scorer = UnifiedScorer(self.engine, self.joker_effects_engine)
        
        # Game state
        self.ante = 1
        self.round = 1  # 1=small, 2=big, 3=boss
        self.phase = Phase.BLIND_SELECT
        self.chips_needed = BLIND_CHIPS[1]['small']
        
        # Collections
        self.jokers = []  # List of owned joker names
        self.consumables = []  # List of consumable names
        self.joker_slots = 5
        self.consumable_slots = 2
        
        # Card state
        self.selected_cards = []
        self.hand_array = np.full(8, -1, dtype=np.int8)
        
        # Shop
        self.shop: Optional[Shop] = None
        
        # Statistics
        self.hands_played = 0
        self.total_hands_played = 0
        self.hands_played_this_ante = 0
        self.best_hand_this_ante = 0
        self.jokers_sold = 0
        
        # Create game state dict for joker effects
        self._update_game_state()
        
        return self._get_observation(), {}

    def _update_game_state(self):
        """Update game state dict for joker effects"""
        self.game_state = {
            'deck': self.game.deck,
            'hand': [self.game.deck[i] for i in self.game.hand_indexes],
            'jokers': [{'name': name} for name in self.jokers],
            'consumables': self.consumables,
            'money': self.player.chips,
            'ante': self.ante,
            'hands_left': self.game.round_hands,
            'discards_left': self.game.round_discards,
            'hand_stats': {},
            'joker_slots': self.joker_slots,
            'consumable_slots': self.consumable_slots,
        }

    def step(self, action: int):
        """Execute action and return step results"""
        # Validate action
        if not self._is_valid_action(action):
            return self._get_observation(), -1.0, False, False, {'error': 'Invalid action'}
        
        # Route to appropriate handler
        if self.phase == Phase.PLAY:
            return self._step_play(action)
        elif self.phase == Phase.SHOP:
            return self._step_shop(action)
        elif self.phase == Phase.BLIND_SELECT:
            return self._step_blind_select(action)
        elif self.phase == Phase.PACK_OPEN:
            return self._step_pack_open(action)

    def _step_play(self, action: int):
        """Handle playing phase actions with unified scoring"""
        reward = 0.0
        terminated = False
        info = {}
        
        if action == ACTION_PLAY_HAND:
            if len(self.selected_cards) == 0:
                return self._get_observation(), -1.0, False, False, {'error': 'No cards selected'}
            
            # Get the actual cards for scoring
            selected_cards = []
            for idx in self.selected_cards:
                if idx < len(self.game.hand_indexes):
                    card_idx = self.game.hand_indexes[idx]
                    deck_card = self.game.deck[card_idx]
                    
                    # Create enhanced card object with all properties
                    card = type('Card', (), {
                        'rank': deck_card.rank.value + 2,  # Convert to 2-14 range
                        'suit': ['Spades', 'Clubs', 'Hearts', 'Diamonds'][deck_card.suit.value],
                        'base_value': deck_card.chip_value(),
                        'chip_value': deck_card.chip_value,
                        'enhancement': getattr(deck_card, 'enhancement', None),
                        'edition': getattr(deck_card, 'edition', None),
                        'seal': getattr(deck_card, 'seal', None)
                    })
                    selected_cards.append(card)
            
            # Highlight cards in the game
            for idx in self.selected_cards:
                if idx < len(self.game.hand_indexes):
                    self.game.highlight_card(idx)
            
            # Classify the hand
            hand_type, _ = self.game._classify_hand(
                [self.game.deck[i] for i in self.game.highlighted_indexes]
            )
            
            # Get hand type name
            hand_type_name = hand_type.name.replace('_', ' ').title()
            
            # Create scoring context
            scoring_context = ScoringContext(
                cards=selected_cards,
                scoring_cards=selected_cards,
                hand_type=hand_type,
                hand_type_name=hand_type_name,
                game_state=self.game_state
            )
            
            # Use unified scorer
            final_score, breakdown = self.unified_scorer.score_hand(scoring_context)
            
            # Update game score
            self.game.round_score += final_score
            
            # Update statistics
            self.hands_played += 1
            self.total_hands_played += 1
            self.hands_played_this_ante += 1
            self.best_hand_this_ante = max(self.best_hand_this_ante, final_score)
            
            # Track hand usage for jokers like Obelisk
            self.engine.hand_play_counts[hand_type] += 1
            
            # Clear selection
            self.selected_cards = []
            
            # Calculate reward
            progress = self.game.round_score / self.chips_needed
            reward = 10 * progress + (final_score / 1000)
            
            # Add scoring breakdown to info
            info['score_breakdown'] = breakdown
            info['final_score'] = final_score
            
            # Check round end conditions
            if self.game.state == BalatroGame.State.WIN:
                reward += 1000
                terminated = True
                info['won'] = True
            elif self.game.state == BalatroGame.State.LOSS:
                reward -= 100
                terminated = True
                info['failed'] = True
            elif self.game.round_score >= self.chips_needed:
                reward += 50
                self._advance_round()
                info['beat_blind'] = True
            elif self.game.round_hands == 0:
                self.game.state = BalatroGame.State.LOSS
                reward -= 100
                terminated = True
                info['failed'] = True
            else:
                self.game._draw_cards()
                self._update_hand_array()
                
        elif action == ACTION_DISCARD:
            if self.game.round_discards <= 0:
                return self._get_observation(), -1.0, False, False, {'error': 'No discards left'}
            
            # Get discarded cards
            discarded = []
            for idx in self.selected_cards:
                if idx < len(self.game.hand_indexes):
                    card_idx = self.game.hand_indexes[idx]
                    deck_card = self.game.deck[card_idx]
                    card = type('Card', (), {
                        'rank': deck_card.rank.value + 2,
                        'suit': ['Spades', 'Clubs', 'Hearts', 'Diamonds'][deck_card.suit.value]
                    })
                    discarded.append(card)
            
            # Apply discard effects
            discard_context = {
                'phase': 'discard',
                'discarded_cards': discarded,
                'last_discarded_card': discarded[-1] if discarded else None,
                'is_first_discard': self.game.round_discards == self.game.discards
            }
            
            for joker_name in self.jokers:
                joker = type('Joker', (), {'name': joker_name})
                effect = self.joker_effects_engine.apply_joker_effect(
                    joker, discard_context, self.game_state
                )
                if effect and 'money' in effect:
                    self.player.chips += effect['money']
            
            # Execute discard
            for idx in sorted(self.selected_cards, reverse=True):
                if idx < len(self.game.hand_indexes):
                    self.game.highlight_card(idx)
            
            self.game.discard_hand()
            self.selected_cards = []
            self._update_hand_array()
            reward = 0.5
            
        elif action in ACTION_SELECT_CARDS:
            card_idx = action - 2
            if card_idx < len(self.game.hand_indexes):
                if card_idx in self.selected_cards:
                    self.selected_cards.remove(card_idx)
                else:
                    self.selected_cards.append(card_idx)
                    
        elif action in ACTION_USE_CONSUMABLE:
            reward, info = self._use_consumable(action - 10)
        
        self._update_game_state()
        return self._get_observation(), reward, terminated, False, info

    def _use_consumable(self, consumable_idx: int) -> Tuple[float, Dict]:
        """Use a consumable with full tarot/spectral effects"""
        if consumable_idx >= len(self.consumables):
            return -1.0, {'error': 'Invalid consumable'}
        
        consumable_name = self.consumables[consumable_idx]
        
        # Get target cards if needed
        target_cards = []
        if len(self.selected_cards) > 0:
            for idx in self.selected_cards:
                if idx < len(self.game.hand_indexes):
                    card_idx = self.game.hand_indexes[idx]
                    deck_card = self.game.deck[card_idx]
                    card = ConsumableCard(
                        rank=Rank(deck_card.rank.value + 2),
                        suit=Suit(deck_card.suit.value),
                        enhancement=Enhancement(getattr(deck_card, 'enhancement', Enhancement.NONE)),
                        edition=Edition(getattr(deck_card, 'edition', Edition.NONE)),
                        seal=Seal(getattr(deck_card, 'seal', Seal.NONE))
                    )
                    target_cards.append(card)
        
        # Apply consumable effect
        result = self.consumable_manager.use_consumable(
            consumable_name, self.game_state, target_cards
        )
        
        reward = 0.0
        info = {'consumable_used': consumable_name}
        
        if result['success']:
            self.consumables.pop(consumable_idx)
            
            # Handle different effect types
            if result.get('money_gained', 0) > 0:
                self.player.chips += result['money_gained']
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
                    reward += 10.0
            
            if result.get('cards_affected'):
                reward += len(result['cards_affected']) * 2.0
                
            if result.get('cards_created'):
                reward += len(result['cards_created']) * 3.0
                
            if result.get('cards_destroyed'):
                reward += len(result['cards_destroyed']) * 1.0
                
            if result.get('jokers_created'):
                for joker in result['jokers_created']:
                    if len(self.jokers) < self.joker_slots:
                        self.jokers.append(joker)
                reward += len(result['jokers_created']) * 15.0
                
            if result.get('items_created'):
                for item in result['items_created']:
                    if len(self.consumables) < self.consumable_slots:
                        self.consumables.append(item)
                reward += len(result['items_created']) * 5.0
                
            if result.get('hand_size_change'):
                self.game.hand_size += result['hand_size_change']
                
            info['result'] = result['message']
        else:
            reward = -1.0
            info['error'] = result.get('message', 'Failed to use consumable')
        
        self.selected_cards = []
        return reward, info

    def _step_shop(self, action: int):
        """Handle shop phase actions using the Shop module"""
        info = {}
        
        # Map our action IDs to Shop action IDs
        if action == ACTION_SHOP_END:
            shop_action = ShopAction.SKIP
        elif action == ACTION_SHOP_REROLL:
            shop_action = ShopAction.REROLL
        elif action in ACTION_SHOP_BUY:
            item_idx = action - 20
            if item_idx >= len(self.shop.inventory):
                return self._get_observation(), -1.0, False, False, {'error': 'Invalid item'}
            
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
        elif action in ACTION_SELL_JOKER:
            joker_idx = action - 32
            if joker_idx < len(self.jokers):
                sold_joker_name = self.jokers.pop(joker_idx)
                sold_joker_id = JOKER_NAME_TO_ID.get(sold_joker_name, 0)
                
                # Get sell value from JOKER_LIBRARY
                sell_value = 3  # Default
                for joker_info in JOKER_LIBRARY:
                    if joker_info.id == sold_joker_id:
                        sell_value = max(3, joker_info.base_cost // 2)
                        break
                
                self.player.chips += sell_value
                
                # Remove from player's joker list too
                if sold_joker_id in self.player.jokers:
                    self.player.jokers.remove(sold_joker_id)
                
                self._update_game_state()
                return self._get_observation(), sell_value / 5.0, False, False, {'sold_joker': sold_joker_name}
        else:
            return self._get_observation(), -1.0, False, False, {'error': 'Invalid shop action'}
        
        # Execute shop action
        reward, done_shopping, shop_info = self.shop.step(shop_action)
        info.update(shop_info)
        
        # Handle successful purchases
        if 'new_cards' in shop_info:
            reward = 5.0
            info['opened_pack'] = True
        elif 'error' not in shop_info and action in ACTION_SHOP_BUY:
            verb, _ = ShopAction.decode(shop_action)
            if verb == "buy_joker":
                joker_id = self.player.jokers[-1]
                joker_name = JOKER_ID_TO_NAME.get(joker_id, f"Unknown_Joker_{joker_id}")
                if joker_name not in self.jokers:
                    self.jokers.append(joker_name)
                reward = 15.0
                info['bought_joker'] = joker_name
            elif verb == "buy_card":
                reward = 3.0
                info['bought_card'] = True
            elif verb == "buy_voucher":
                reward = 10.0
                info['bought_voucher'] = self.player.vouchers[-1]
        
        # Check if shopping is done
        if done_shopping:
            self.phase = Phase.PLAY
            self.game._draw_cards()
            self._update_hand_array()
        
        self._update_game_state()
        return self._get_observation(), reward, False, False, info

    def _step_blind_select(self, action: int):
        """Handle blind selection phase"""
        reward = 0.0
        info = {}
        
        if action in ACTION_SELECT_BLIND:
            blind_type = action - 45  # 0=small, 1=big, 2=boss
            self.round = blind_type + 1
            blind_key = ['small', 'big', 'boss'][blind_type]
            self.chips_needed = BLIND_CHIPS[min(self.ante, 8)][blind_key]
            
            # Update game blind
            self.game.blinds[self.game.blind_index] = self.chips_needed
            
            # Boss blind bonus reward
            if blind_type == 2:
                reward = 10.0
            
            # Transition to play
            self.phase = Phase.PLAY
            self._update_hand_array()
            
        elif action == ACTION_SKIP_BLIND:
            # Skip blind - trigger skip effects
            for joker_name in self.jokers:
                joker = type('Joker', (), {'name': joker_name})
                self.joker_effects_engine.apply_joker_effect(
                    joker, {'phase': 'skip_blind'}, self.game_state
                )
            
            reward = -5.0
            self._advance_round()
            info['skipped_blind'] = True
        
        self._update_game_state()
        return self._get_observation(), reward, False, False, info

    def _step_pack_open(self, action: int):
        """Handle pack opening - simplified for now"""
        self.phase = Phase.SHOP
        self._generate_shop()
        return self._get_observation(), 0.0, False, False, {}

    def _advance_round(self):
        """Advance to next round/ante"""
        # Apply end-of-round effects
        end_effects = self.joker_effects_engine.end_of_round_effects(self.game_state)
        
        # Handle joker destruction
        for effect in end_effects:
            if 'destroy_joker' in effect:
                joker_name = effect['destroy_joker']
                if joker_name in self.jokers:
                    self.jokers.remove(joker_name)
        
        # Reset round stats
        self.best_hand_this_ante = 0
        self.hands_played_this_ante = 0
        
        # Progress ante/round
        if self.round == 3:
            self.ante += 1
            self.round = 1
        else:
            self.round += 1
        
        # Award money
        money_earned = 25 * self.round + (10 if self.round == 3 else 0)
        self.player.chips += money_earned
        
        # Go to shop
        self.phase = Phase.SHOP
        self._generate_shop()

    def _generate_shop(self):
        """Create shop with items using the Shop module"""
        # Make sure player state is synced
        self.player.jokers = [JOKER_NAME_TO_ID.get(name, 0) for name in self.jokers]
        self.shop = Shop(self.ante, self.player, seed=int(np.random.randint(1<<31)))

    def _update_hand_array(self):
        """Update hand array for observation"""
        self.hand_array = np.full(8, -1, dtype=np.int8)
        for i, idx in enumerate(self.game.hand_indexes[:8]):
            self.hand_array[i] = self.game.deck[idx].encode()

    def _is_valid_action(self, action: int) -> bool:
        """Check if action is valid"""
        mask = self._get_action_mask()
        return bool(mask[action])

    def _get_action_mask(self):
        """Get valid actions for current state"""
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        
        if self.phase == Phase.PLAY:
            # Card selection
            for i in range(min(8, len(self.game.hand_indexes))):
                mask[2 + i] = 1
            
            # Play hand if cards selected
            if len(self.selected_cards) > 0:
                mask[ACTION_PLAY_HAND] = 1
            
            # Discard if cards selected and discards left
            if len(self.selected_cards) > 0 and self.game.round_discards > 0:
                mask[ACTION_DISCARD] = 1
            
            # Use consumables
            for i in range(len(self.consumables)):
                mask[10 + i] = 1
                
        elif self.phase == Phase.SHOP:
            if self.shop:
                # Buy items
                for i in range(len(self.shop.inventory)):
                    if self.player.chips >= self.shop.inventory[i].cost:
                        mask[20 + i] = 1
                
                # Reroll
                if self.player.chips >= int(self.shop.reroll_cost * self.shop._cost_mult()):
                    mask[ACTION_SHOP_REROLL] = 1
            
            # Always can end shop
            mask[ACTION_SHOP_END] = 1
            
            # Sell jokers
            for i in range(len(self.jokers)):
                mask[32 + i] = 1
                
        elif self.phase == Phase.BLIND_SELECT:
            # Select any blind
            mask[45:48] = 1
            mask[ACTION_SKIP_BLIND] = 1
        
        return mask

    def _get_observation(self):
        """Build observation dict"""
        # Get hand levels from engine
        hand_levels = []
        for hand_type in HandType:
            if hand_type != HandType.NONE:
                level = self.engine.get_hand_level(hand_type)
                hand_levels.append(level)
        
        # Get consumable IDs
        consumable_ids = self._get_consumable_ids()
        
        obs = {
            'hand': self.hand_array.copy(),
            'hand_size': np.int8(len(self.game.hand_indexes)),
            'deck_size': np.int8(sum(not c.played for c in self.game.deck)),
            'selected_cards': np.array([1 if i in self.selected_cards else 0 for i in range(8)]),
            
            'chips_scored': np.int32(self.game.round_score),
            'mult': np.int32(1),  # Base mult
            'chips_needed': np.int32(self.chips_needed),
            'money': np.int32(self.player.chips),
            
            'ante': np.int8(self.ante),
            'round': np.int8(self.round),
            'hands_left': np.int8(self.game.round_hands),
            'discards_left': np.int8(self.game.round_discards),
            
            'joker_count': np.int8(len(self.jokers)),
            'joker_ids': np.array([JOKER_NAME_TO_ID.get(j, 0) for j in self.jokers] + 
                                 [0] * (10 - len(self.jokers)), dtype=np.int16),
            'joker_slots': np.int8(self.joker_slots),
            
            'consumable_count': np.int8(len(self.consumables)),
            'consumables': np.array(consumable_ids, dtype=np.int16),
            'consumable_slots': np.int8(self.consumable_slots),
            
            'shop_items': np.zeros(10, dtype=np.int16),
            'shop_costs': np.zeros(10, dtype=np.int16),
            'shop_rerolls': np.int16(self.shop.reroll_cost if self.phase == Phase.SHOP and self.shop else 0),
            
            'hand_levels': np.array(hand_levels[:12], dtype=np.int8),
            
            'phase': np.int8(self.phase),
            'action_mask': self._get_action_mask(),
            
            'hands_played': np.int16(self.total_hands_played),
            'best_hand_this_ante': np.int32(self.best_hand_this_ante),
        }
        
        # Add shop info if in shop
        if self.phase == Phase.SHOP and self.shop:
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
        for consumable in self.consumables:
            ids.append(consumable_id_map.get(consumable, 0))
        
        return ids + [0] * (5 - len(ids))

    def render(self):
        """Render the game state"""
        if self.render_mode != "human":
            return
        
        print(f"\n{'='*50}")
        print(f"Ante {self.ante} - Round {self.round} - Phase: {Phase(self.phase).name}")
        print(f"Score: {self.game.round_score}/{self.chips_needed} | Money: ${self.player.chips}")
        print(f"Hands: {self.game.round_hands} | Discards: {self.game.round_discards}")
        
        if self.phase == Phase.PLAY:
            print(f"\nHand: {self.game.hand_to_string()}")
            if self.selected_cards:
                print(f"Selected: {self.selected_cards}")
        
        elif self.phase == Phase.SHOP and self.shop:
            print("\n=== SHOP ===")
            for i, item in enumerate(self.shop.inventory):
                affordable = "✓" if self.player.chips >= item.cost else "✗"
                print(f"[{i}] {affordable} {item.name:<25} ${item.cost}")
            print(f"\nReroll cost: ${int(self.shop.reroll_cost * self.shop._cost_mult())}")
        
        elif self.phase == Phase.BLIND_SELECT:
            print("\n=== SELECT BLIND ===")
            print(f"[0] Small Blind: {BLIND_CHIPS[min(self.ante, 8)]['small']} chips")
            print(f"[1] Big Blind: {BLIND_CHIPS[min(self.ante, 8)]['big']} chips")
            print(f"[2] Boss Blind: {BLIND_CHIPS[min(self.ante, 8)]['boss']} chips")
            print(f"[S] Skip Blind")
        
        if self.jokers:
            print(f"\nJokers ({len(self.jokers)}/{self.joker_slots}): {', '.join(self.jokers)}")
        
        if self.consumables:
            print(f"Consumables ({len(self.consumables)}/{self.consumable_slots}): {', '.join(self.consumables)}")

    def close(self):
        pass


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
    env = BalatroEnv(render_mode="human")
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
        if env.phase == Phase.BLIND_SELECT:
            # Always select small blind
            action = 45
        elif env.phase == Phase.SHOP:
            # End shop immediately
            action = ACTION_SHOP_END
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
    print(f"Final score: {env.game.round_score}")
    print(f"Reached ante: {env.ante}")
