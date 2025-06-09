"""balatro_gym/envs/balatro_env_v2.py – now applies Planet consumables.

After purchasing a Planet Pack the env checks `info["planet"]` and calls
`self.game.engine.apply_consumable(planet)`, making the truth-table mutation
immediately active.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

import gymnasium as gym
from gymnasium import spaces

from balatro_gym.shop import Shop, ShopAction, PlayerState
from balatro_gym.scoring_engine import ScoreEngine, Planet
from balatro_gym.balatro_game_v2 import BalatroGame

PLAY_MAX_ID = 9
SHOP_MIN_ID = 10
ACTION_SPACE_SIZE = 70

PHASE_PLAY = 0
PHASE_SHOP = 1

class BalatroEnvWithExpert(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            {
                "hand": spaces.Box(-1, 51, (8,), dtype=np.int8),
                "chips": spaces.Box(0, 1_000_000_000, (), dtype=np.int32),
                "phase": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
            }
        )

        # core state
        self.engine = ScoreEngine()
        self.game = BalatroGame(engine=self.engine)
        self.player = PlayerState(chips=1_000)
        self.ante = 1
        self.phase = PHASE_PLAY
        self.shop: Optional[Shop] = None
        self._deal()

    # ---------------- reset ----------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.engine = ScoreEngine()
        self.game = BalatroGame(engine=self.engine)
        self.player.chips = 1_000
        self.ante = 1
        self.phase = PHASE_PLAY
        self.shop = None
        self._deal()
        return self._obs(), {}

    def _deal(self):
        self.game._draw_cards()
        self.hand = np.array(self.game.hand_indexes[:8], dtype=np.int8)

    # ---------------- step ---------------
    def step(self, action: int):
        if self.phase == PHASE_SHOP:
            return self._step_shop(action)
        return self._step_play(action)

    def _step_shop(self, action: int):
        reward, done_shop, info = self.shop.step(action, self.player)

        # apply planet consumable immediately
        if "planet" in info:
            self.engine.apply_consumable(info["planet"])

        if done_shop:
            self.phase = PHASE_PLAY
            self._deal()
        return self._obs(), reward, False, False, info

    def _step_play(self, action: int):
        reward = 0.0
        if action == 0:  # simple play first 5
            self.game.highlighted_indexes = self.game.hand_indexes[:5]
            reward = self.game.play_hand()
            self.player.chips += int(reward)
            self._end_round()
        elif action == 1:  # discard all
            self.player.chips = max(0, self.player.chips - 1)
            self.game.discard_hand()
            self._deal()
        return self._obs(), reward, False, False, {}

    def _end_round(self):
        self.phase = PHASE_SHOP
        self.ante += 1
        self.player.chips += 50 * self.ante
        self.shop = Shop(self.ante, self.player, seed=int(np.random.randint(1<<31)))

    # ---------------- obs/mask ------------
    def _mask(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == PHASE_PLAY:
            mask[:PLAY_MAX_ID+1] = 1
        else:
            mask[ShopAction.SKIP] = mask[ShopAction.REROLL] = 1
            for i in range(len(self.shop.inventory)):
                mask[SHOP_MIN_ID + i] = 1
        return mask

    def _obs(self):
        obs = {
            "hand": self.hand.copy(),
            "chips": np.int32(self.player.chips),
            "phase": np.int8(self.phase),
            "action_mask": self._mask(),
        }
        if self.phase == PHASE_SHOP:
            obs.update(self.shop.get_observation())
        return obs

    # ------------ render -------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.phase == PHASE_SHOP:
            print("-- SHOP -- Chips:", self.player.chips)
            for idx, item in enumerate(self.shop.inventory):
                print(f"[{idx}] {item.name} cost:{item.cost}")
        else:
            suit = "♠♣♥♦"; rank="23456789TJQKA"
            print("Hand:", " ".join(f"{rank[c%13]}{suit[c//13]}" for c in self.hand))

    def close(self):
        pass
