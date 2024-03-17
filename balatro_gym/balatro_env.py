import numpy as np
from enum import Enum

from .balatro_game import BalatroGame

import gymnasium as gym
from gymnasium import spaces

class BalatroEnv(gym.Env):
    metadata = {}

    BAD_ACTION_PENALTY = -0.1

    MAX_DECK_SIZE = 52
    MAX_HAND_SIZE = 8

    # ACTIONS = Enum('Action', ['PLAY_HAND', 'DISCARD_HAND'], start=0)
    MAX_ACTIONS = 2 + MAX_HAND_SIZE

    def __init__(self, render_mode=None):

        self.action_space = spaces.Discrete(10)

        deck_space = spaces.MultiDiscrete([53] * 52)
        hand_space = spaces.MultiDiscrete([52] * 8)
        selected_space = spaces.MultiDiscrete([52] * 5)

        self.observation_space = spaces.Tuple(deck_space, hand_space, selected_space)

    def step(self, action):
        if action not in self._get_legal_moves:
            pass
        pass

    def resolve_action(self):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def _get_legal_moves(self):

        [self.ACTIONS.PLAY_HAND, self.ACTIONS.DISCARD_HAND]
        return 
    
    
