import numpy as np

from .balatro_game import BalatroGame

import gymnasium as gym
from gymnasium import spaces

class BalatroSmallEnv(gym.Env):
    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    MAX_DECK_SIZE = 52
    MAX_HAND_SIZE = 8
    MAX_ACTIONS = 2 + MAX_HAND_SIZE

    MAX_HANDS = 10
    MAX_DISCARDS = 8

    def __init__(self, render_mode=None, chip_threshold=500):
        self.action_space = spaces.Discrete(self.MAX_ACTIONS)

        self.observation_space = spaces.Dict({
            "deck": spaces.Dict({
                "cards": spaces.Box(0, 52, shape=(self.MAX_DECK_SIZE,), dtype=int), 
                "cards_played": spaces.MultiBinary(self.MAX_DECK_SIZE)
            }),
            "hand": spaces.Box(0, 51, shape=(self.MAX_HAND_SIZE,), dtype=int),
            "highlighted": spaces.Box(0, 51, shape=(5,), dtype=int),
            "round_score": spaces.Discrete(100000),
            "round_hands": spaces.Discrete(self.MAX_HANDS),
            "round_discards": spaces.Discrete(self.MAX_DISCARDS),
        })

        self.chip_threshold = chip_threshold

        self.game = BalatroGame()
        self.game.blinds[0] = chip_threshold

        self.render_mode = render_mode

    def step(self, action):
        if action not in self.valid_actions():
            raise RuntimeError("Environment tried to take an invalid action.")
        self.resolve_action(action)
         
        reward = 1 if self.game.blind_index == 1 else 0
        done = self.game.state != BalatroGame.State.IN_PROGRESS or reward == 1
        return self._get_observation(), reward, done, False, {}

    def resolve_action(self, action):
        if action == 0:
            self.game.play_hand()
        elif action == 1: 
            self.game.discard_hand()
        else:
            self.game.highlight_card(action - 2)

    def reset(self, seed=None, options=None):
        self.game = BalatroGame()
        self.game.blinds[0] = self.chip_threshold
        return self._get_observation(), {}

    def render(self):
        if self.render_mode == "ansi":
            res = f"Ante: {self.game.ante}, Blind: {self.game.blind_index + 1}/3\n"
            res += f"Score: {self.game.round_score}/{self.game.blinds[self.game.blind_index]}\n\n"
            res += f"Highlighted: {self.game.highlighted_to_string()}\n"
            res += f"Hand: {self.game.hand_to_string()}\n"
            return res

    def _get_observation(self):

        cards = np.zeros((self.MAX_DECK_SIZE,), dtype=int)
        cards_played = np.zeros((self.MAX_DECK_SIZE,), dtype=int)

        for i in range(min(len(self.game.deck), self.MAX_DECK_SIZE)):
            cards[i] = self.game.deck[i].encode() + 1
            cards_played[i] = int(self.game.deck[i].played)

        hand = self._normalize_array(self.game.hand_indexes, self.MAX_HAND_SIZE)
        highlighted = self._normalize_array(self.game.highlighted_indexes, 5)

        return {
            "deck": {
                "cards": cards, 
                "cards_played": cards_played
            },
            "hand": hand,
            "highlighted": highlighted,
            "round_score": self.game.round_score,
            "round_hands": self.game.round_hands,
            "round_discards": self.game.round_discards,
        }
    
    @staticmethod
    def _normalize_array(arr, size):
        normalized_array = np.zeros((size,), dtype=int)
        normalized_array[:len(arr)] = arr
        return normalized_array

    def valid_actions(self):
        actions = []
        if len(self.game.highlighted_indexes) > 0:
            if self.game.round_hands > 0:
                actions.append(0)
            if self.game.round_discards > 0:
                actions.append(1)
        if len(self.game.highlighted_indexes) < 5:
            for i in range(len(self.game.hand_indexes)):
                actions.append(i + 2)
        return actions

    def action_masks(self):  
        return [action in self.valid_actions() for action in np.arange(self.MAX_ACTIONS, dtype=int)]
    
