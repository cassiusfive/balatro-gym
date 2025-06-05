# balatro_gym/envs/balatro_env_v2.py
import gym
import numpy as np
from typing import Dict, List, Tuple, Optional
from balatro_probability_engine import (
    BalatroOptimalPlayer, 
    Card, 
    Rank, 
    Suit,
    HandType
)

class BalatroEnvWithExpert(gym.Env):
    """Balatro environment with integrated expert player for trajectory generation"""
    
    def __init__(self, use_expert_for_actions=False):
        super().__init__()
        
        # Original env setup
        self.setup_action_space()
        self.setup_observation_space()
        
        # Add expert player
        self.expert = BalatroOptimalPlayer()
        self.use_expert_for_actions = use_expert_for_actions
        
        # Trajectory recording
        self.current_trajectory = []
        self.trajectories = []
        
    def reset(self):
        """Reset and optionally start trajectory recording"""
        obs = super().reset()
        
        if self.use_expert_for_actions:
            self.current_trajectory = [{
                'observation': obs,
                'game_state': self.get_full_state()
            }]
            
        return obs
    
    def step(self, action):
        """Execute action (from RL or expert) and record trajectory"""
        
        # If using expert, override action
        if self.use_expert_for_actions:
            expert_action = self.get_expert_action()
            action = expert_action
            
        # Execute action
        obs, reward, done, info = super().step(action)
        
        # Record trajectory
        if self.use_expert_for_actions:
            self.current_trajectory.append({
                'action': action,
                'reward': reward,
                'observation': obs,
                'game_state': self.get_full_state(),
                'info': info
            })
            
            if done:
                self.trajectories.append(self.current_trajectory)
                self.current_trajectory = []
                
        return obs, reward, done, info
    
    def get_expert_action(self):
        """Get optimal action from probability engine"""
        state = self.get_full_state()
        
        # Convert gym state to cards
        hand = self._state_to_cards(state['hand'])
        jokers = state['jokers']
        
        # Get expert decision
        decision = self.expert.decide_play(hand, jokers, state)
        
        # Convert decision to gym action
        return self._decision_to_action(decision)
    
    def _state_to_cards(self, hand_array):
        """Convert numpy array to Card objects"""
        cards = []
        for card_encoding in hand_array:
            if card_encoding > 0:  # Not empty slot
                rank, suit = self._decode_card(card_encoding)
                cards.append(Card(Rank(rank), Suit(suit)))
        return cards
    
    def _decision_to_action(self, decision):
        """Convert expert decision to gym action space"""
        if decision['action'] == 'play':
            # Convert card selection to action
            return self._cards_to_play_action(decision['cards'])
        else:  # discard
            return self._discards_to_action(decision['discard_indices'])
