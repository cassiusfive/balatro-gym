"""Blind selection phase handler for Balatro RL environment.

This module handles the blind selection phase where players choose
between small blind, big blind, boss blind, or skipping.
"""

from typing import Tuple, Dict

from balatro_gym.envs.state import UnifiedGameState
from balatro_gym.envs.rng import DeterministicRNG
from balatro_gym.envs.utils.blind_scaling import get_blind_chips
from balatro_gym.constants import Action, Phase
from balatro_gym.boss_blinds import BossBlindManager, select_boss_blind
from balatro_gym.complete_joker_effects import CompleteJokerEffects
from balatro_gym.balatro_game import BalatroGame


class BlindSelectHandler:
    """Handles blind selection phase."""
    
    def __init__(self, 
                 state: UnifiedGameState,
                 game: BalatroGame,
                 boss_blind_manager: BossBlindManager,
                 joker_effects_engine: CompleteJokerEffects,
                 rng: DeterministicRNG):
        """Initialize the blind select handler.
        
        Args:
            state: Game state
            game: Core game instance
            boss_blind_manager: Boss blind manager
            joker_effects_engine: Joker effects processor
            rng: RNG system
        """
        self.state = state
        self.game = game
        self.boss_blind_manager = boss_blind_manager
        self.joker_effects_engine = joker_effects_engine
        self.rng = rng
    
    def step(self, action: int) -> Tuple[float, bool, Dict]:
        """Process an action during blind select phase.
        
        Args:
            action: Action to execute
            
        Returns:
            Tuple of (reward, terminated, info)
        """
        if Action.SELECT_BLIND_BASE <= action < Action.SELECT_BLIND_BASE + Action.SELECT_BLIND_COUNT:
            return self._handle_select_blind(action)
        elif action == Action.SKIP_BLIND:
            return self._handle_skip_blind()
        else:
            return -1.0, False, {'error': 'Invalid blind select action'}
    
    def _handle_select_blind(self, action: int) -> Tuple[float, bool, Dict]:
        """Handle selecting a specific blind."""
        blind_type = action - Action.SELECT_BLIND_BASE  # 0=small, 1=big, 2=boss
        blind_names = ['small', 'big', 'boss']
        blind_name = blind_names[blind_type]
        
        # Set round
        self.state.round = blind_type + 1
        
        # Calculate chip requirement
        base_chips = get_blind_chips(self.state.ante, blind_name)
        self.state.chips_needed = base_chips
        
        info = {
            'action': 'selected_blind',
            'blind_type': blind_name,
            'ante': self.state.ante,
            'round': self.state.round,
            'chips_needed': self.state.chips_needed
        }
        
        # Initialize reward
        reward = 0.0
        
        # Handle boss blind activation
        if blind_type == 2:  # Boss blind
            reward, boss_info = self._activate_boss_blind(base_chips)
            info.update(boss_info)
        
        # Update game blind requirement
        if hasattr(self.game, 'blinds') and hasattr(self.game, 'blind_index'):
            self.game.blinds[self.game.blind_index] = self.state.chips_needed
        
        # Reset round-specific state
        self.state.reset_round_state()
        
        # Apply any blind selection joker effects
        selection_effects = self._apply_selection_effects(blind_name)
        info.update(selection_effects)
        
        # Transition to play phase
        self.state.phase = Phase.PLAY
        
        # Draw initial hand (handled by main env)
        info['transition_to'] = 'play'
        
        return reward, False, info
    
    def _handle_skip_blind(self) -> Tuple[float, bool, Dict]:
        """Handle skipping the current blind."""
        # Get skip reward/penalty based on blind type
        skip_penalty = -5.0
        skip_tag_reward = 0
        
        # Apply skip blind joker effects
        skip_effects = []
        for joker in self.state.jokers:
            effect = self.joker_effects_engine.apply_joker_effect(
                type('Joker', (), {'name': joker.name}), 
                {'phase': 'skip_blind'}, 
                self.state.to_dict()
            )
            if effect:
                skip_effects.append(effect)
        
        # Process skip effects (tags, money, etc.)
        total_money_gained = 0
        tags_gained = []
        
        for effect in skip_effects:
            if 'money' in effect:
                total_money_gained += effect['money']
                self.state.money += effect['money']
            if 'tag' in effect:
                tags_gained.append(effect['tag'])
        
        # Some jokers give rewards for skipping
        skip_jokers = ['Throwback', 'Vagabond']
        skip_joker_count = sum(1 for j in self.state.jokers if j.name in skip_jokers)
        if skip_joker_count > 0:
            skip_tag_reward = 15.0 * skip_joker_count
        
        # Calculate total reward
        reward = skip_penalty + skip_tag_reward + (total_money_gained / 5.0)
        
        # Advance round without playing
        self._advance_round_after_skip()
        
        info = {
            'action': 'skipped_blind',
            'money_gained': total_money_gained,
            'tags_gained': tags_gained,
            'new_ante': self.state.ante,
            'new_round': self.state.round
        }
        
        return reward, False, info
    
    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------
    
    def _activate_boss_blind(self, base_chips: int) -> Tuple[float, Dict]:
        """Activate a boss blind and apply its effects."""
        # Select boss blind based on ante
        boss_type = select_boss_blind(self.state.ante)
        
        # Activate the boss blind
        effects = self.boss_blind_manager.activate_boss_blind(boss_type, self.state.to_dict())
        
        # Apply chip multiplier
        self.state.chips_needed = int(base_chips * effects['chip_mult'])
        
        # Apply game modifications
        modifications = effects.get('modifications', {})
        
        if 'discards' in modifications:
            self.state.discards_left = modifications['discards']
            if hasattr(self.game, 'round_discards'):
                self.game.round_discards = modifications['discards']
        
        if 'hand_size' in modifications:
            self.state.hand_size += modifications['hand_size']
            if hasattr(self.game, 'hand_size'):
                self.game.hand_size = self.state.hand_size
        
        if 'hands' in modifications:
            self.state.hands_left = modifications['hands']
            if hasattr(self.game, 'round_hands'):
                self.game.round_hands = modifications['hands']
        
        if 'joker_slots' in modifications:
            # The Plant boss disables joker slots
            self.state.disabled_joker_slots = modifications.get('disabled_slots', 0)
        
        # Set boss blind state
        self.state.active_boss_blind = boss_type
        self.state.boss_blind_active = True
        
        # Boss blind selection gives bonus reward
        reward = 10.0 + (2.0 * self.state.ante)  # Scale with ante
        
        info = {
            'boss_blind': self.boss_blind_manager.active_blind.name,
            'boss_effect': self.boss_blind_manager.active_blind.description,
            'chip_multiplier': effects['chip_mult'],
            'modifications': modifications
        }
        
        return reward, info
    
    def _apply_selection_effects(self, blind_type: str) -> Dict:
        """Apply any joker effects that trigger on blind selection."""
        effects = {}
        
        # Some jokers care about which blind is selected
        if blind_type == 'small':
            # Matador gains X mult when selecting small blind
            if any(j.name == 'Matador' for j in self.state.jokers):
                effects['matador_activated'] = True
        
        elif blind_type == 'big':
            # Some effects for big blind
            pass
        
        elif blind_type == 'boss':
            # Stuntman gives extra chips/mult for boss blinds
            if any(j.name == 'Stuntman' for j in self.state.jokers):
                effects['stuntman_activated'] = True
        
        return effects
    
    def _advance_round_after_skip(self):
        """Advance to the next round after skipping."""
        # Progress round/ante
        if self.state.round == 3:
            # Move to next ante
            self.state.ante += 1
            self.state.round = 1
            self.state.reset_ante_state()
        else:
            # Move to next round in same ante
            self.state.round += 1
        
        # Award skip money (less than playing)
        skip_money = 15  # Base skip reward
        self.state.money += skip_money
        
        # Transition to shop
        self.state.phase = Phase.SHOP
