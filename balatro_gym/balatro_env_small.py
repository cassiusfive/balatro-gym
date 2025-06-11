"""Main Balatro RL Environment.

This is the main entry point for the Balatro gym environment. It coordinates
between all the different subsystems and provides the gym.Env interface.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import gymnasium as gym
from gymnasium import spaces

from balatro_gym.envs.state import UnifiedGameState
from balatro_gym.envs.rng import DeterministicRNG
from balatro_gym.envs.observation_builder import ObservationBuilder
from balatro_gym.envs.action_handler import ActionHandler
from balatro_gym.envs.phase_handlers import (
    PlayPhaseHandler, ShopPhaseHandler, 
    BlindSelectHandler, PackOpenHandler
)

from balatro_gym.constants import Phase
from balatro_gym.cards import Card, Suit, Rank
from balatro_gym.balatro_game import BalatroGame
from balatro_gym.scoring_engine import ScoreEngine
from balatro_gym.jokers import JOKER_LIBRARY
from balatro_gym.consumables import ConsumableManager
from balatro_gym.unified_scoring import UnifiedScorer
from balatro_gym.complete_joker_effects import CompleteJokerEffects
from balatro_gym.boss_blinds import BossBlindManager


class BalatroEnv(gym.Env):
    """Complete Balatro environment with all game systems integrated.
    
    This environment provides a gym interface to the Balatro card game,
    supporting all core mechanics including:
    - Playing poker hands with scoring
    - Shop system with jokers/consumables/vouchers
    - Boss blinds with special effects
    - Full RNG control for reproducibility
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, *, render_mode: str | None = None, seed: int | None = None):
        """Initialize the Balatro environment.
        
        Args:
            render_mode: How to render the game ('human' for text, 'rgb_array' for images)
            seed: Random seed for reproducibility
        """
        super().__init__()
        self.render_mode = render_mode
        self._seed = seed
        
        # Initialize RNG system
        self.rng = DeterministicRNG(seed)
        
        # Initialize observation builder
        self.obs_builder = ObservationBuilder()
        
        # Action and observation spaces
        self.action_space = spaces.Discrete(ActionHandler.get_action_space_size())
        self.observation_space = self.obs_builder.create_observation_space()
        
        # Initialize state
        self.state = UnifiedGameState()
        
        # Initialize game systems (will be set up in reset)
        self.engine: Optional[ScoreEngine] = None
        self.game: Optional[BalatroGame] = None
        self.joker_effects_engine: Optional[CompleteJokerEffects] = None
        self.consumable_manager: Optional[ConsumableManager] = None
        self.unified_scorer: Optional[UnifiedScorer] = None
        self.boss_blind_manager: Optional[BossBlindManager] = None
        
        # Initialize action handler
        self.action_handler = ActionHandler(self.state, self.rng)
        
        # Initialize phase handlers
        self.play_handler = None
        self.shop_handler = None
        self.blind_select_handler = None
        self.pack_open_handler = None
        
        # Initialize environment
        self.reset()

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state.
        
        Args:
            seed: Optional seed to reset RNG
            options: Additional reset options
            
        Returns:
            Tuple of (observation, info)
        """
        if seed is not None:
            self._seed = seed
            self.rng = DeterministicRNG(seed)
        
        # Reset state
        self.state = UnifiedGameState()
        
        # Initialize core game systems
        self.engine = ScoreEngine()
        self.game = BalatroGame(engine=self.engine)
        self.joker_effects_engine = CompleteJokerEffects()
        self.consumable_manager = ConsumableManager()
        self.boss_blind_manager = BossBlindManager()
        self.unified_scorer = UnifiedScorer(self.engine, self.joker_effects_engine)
        
        # Create initial deck
        initial_deck = self._create_standard_deck()
        self.rng.shuffle('deck_shuffle', initial_deck)
        
        # Set up game with deck
        self.game.deck = initial_deck
        self.state.deck = initial_deck
        
        # Initialize hand levels
        self._initialize_hand_levels()
        
        # Update action handler with new state
        self.action_handler.state = self.state
        self.action_handler.rng = self.rng
        
        # Initialize phase handlers with dependencies
        self.play_handler = PlayPhaseHandler(
            self.state, self.game, self.engine, 
            self.unified_scorer, self.joker_effects_engine,
            self.consumable_manager, self.boss_blind_manager, 
            self.rng
        )
        
        self.shop_handler = ShopPhaseHandler(self.state, self.rng)
        
        self.blind_select_handler = BlindSelectHandler(
            self.state, self.game, self.boss_blind_manager, 
            self.joker_effects_engine, self.rng
        )
        
        self.pack_open_handler = PackOpenHandler(self.state, self.shop_handler)
        
        # Sync initial state
        self._sync_state_from_game()
        
        # Apply initial boss blind effects if needed
        if self.state.phase == Phase.PLAY and self.state.boss_blind_active:
            self.play_handler.apply_boss_blind_to_hand()
        
        return self.obs_builder.build_observation(self.state), {}

    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute an action and return the results.
        
        Args:
            action: Action to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Check for termination conditions
        if self.state.ante > 100:
            return self.obs_builder.build_observation(self.state), 0.0, True, False, {
                'terminated': 'max_ante_reached'
            }
        
        if self.state.chips_scored > 1_000_000_000:
            return self.obs_builder.build_observation(self.state), 0.0, True, False, {
                'terminated': 'max_score_reached'
            }
        
        # Validate action
        if not self.action_handler.is_valid_action(action):
            return self.obs_builder.build_observation(self.state), -1.0, False, False, {
                'error': 'Invalid action'
            }
        
        # Route to appropriate phase handler
        if self.state.phase == Phase.PLAY:
            reward, terminated, info = self.play_handler.step(action)
        elif self.state.phase == Phase.SHOP:
            reward, terminated, info = self.shop_handler.step(action)
        elif self.state.phase == Phase.BLIND_SELECT:
            reward, terminated, info = self.blind_select_handler.step(action)
        elif self.state.phase == Phase.PACK_OPEN:
            reward, terminated, info = self.pack_open_handler.step(action)
        else:
            raise ValueError(f"Unknown phase: {self.state.phase}")
        
        # Build observation
        observation = self.obs_builder.build_observation(self.state)
        
        return observation, reward, terminated, False, info

    def render(self):
        """Render the game state."""
        if self.render_mode != "human":
            return
        
        from balatro_gym.envs.renderer import ConsoleRenderer
        renderer = ConsoleRenderer()
        renderer.render(self.state, self.boss_blind_manager)

    def close(self):
        """Clean up resources."""
        pass

    def save_state(self) -> Dict[str, Any]:
        """Save complete environment state for checkpointing.
        
        Returns:
            Dictionary containing all state needed to restore environment
        """
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
                'blind_state': self.boss_blind_manager.blind_state.copy() 
                    if self.boss_blind_manager.blind_state else {}
            }
        }
    
    def load_state(self, saved_state: Dict[str, Any]) -> None:
        """Load environment state from checkpoint.
        
        Args:
            saved_state: State dictionary from save_state()
        """
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
        
        # Update phase handlers with restored state
        self._update_phase_handlers()
        
        # Sync states
        self._sync_state_from_game()

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------
    
    def _create_standard_deck(self) -> list[Card]:
        """Create a standard 52-card deck."""
        deck = []
        for suit in Suit:
            for rank in Rank:
                deck.append(Card(rank=rank, suit=suit))
        return deck
    
    def _initialize_hand_levels(self):
        """Initialize hand levels from the scoring engine."""
        from balatro_gym.scoring_engine import HandType
        
        for hand_type in HandType:
            try:
                level = self.engine.get_hand_level(hand_type)
                self.state.hand_levels[hand_type] = level
            except (KeyError, ValueError):
                # Skip hand types that don't have levels
                continue
    
    def _sync_state_from_game(self):
        """Sync unified state from game systems."""
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
        """Sync game systems from unified state."""
        if self.game:
            self.game.deck = self.state.deck
            self.game.hand_indexes = self.state.hand_indexes
            self.game.round_hands = self.state.hands_left
            self.game.round_discards = self.state.discards_left
            self.game.round_score = self.state.chips_scored
    
    def _update_phase_handlers(self):
        """Update phase handlers after state restoration."""
        # Phase handlers need references to current state
        self.action_handler.state = self.state
        
        if self.play_handler:
            self.play_handler.state = self.state
        if self.shop_handler:
            self.shop_handler.state = self.state
        if self.blind_select_handler:
            self.blind_select_handler.state = self.state
        if self.pack_open_handler:
            self.pack_open_handler.state = self.state


# Factory function for creating environments
def make_balatro_env(**kwargs):
    """Factory function for creating Balatro environments.
    
    Args:
        **kwargs: Arguments to pass to BalatroEnv
        
    Returns:
        Function that creates a BalatroEnv instance
    """
    def _init():
        return BalatroEnv(**kwargs)
    return _init
