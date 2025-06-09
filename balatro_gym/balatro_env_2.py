"""balatro_gym/envs/balatro_env_v2.py

Rewired: PLAY-phase rewards now come from **ScoreEngine** instead of a dummy
constant.  The env keeps a `BalatroGame` instance that already embeds
`ScoreEngine`, so calling `game.play_hand()` yields real chip payouts.

Action mapping (temporary, until full controller is implemented):
    0   → play the first 5 cards of the current 8‑card hand
    1   → discard entire hand and draw eight new cards (costs 1 chip)
    2‑9 → no‑ops for now

Shop phase (IDs ≥10) is unchanged.

This is “good enough” for early RL experiments; refine the action mapping later.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional

import gymnasium as gym
from gymnasium import spaces

from balatro_gym.shop import Shop, ShopAction, PlayerState
from balatro_gym.scoring_engine import ScoreEngine
from balatro_gym.balatro_game_v2 import BalatroGame, Card  # Card for encode helper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PLAY_MAX_ID   = 9      # 0-9 = play-phase actions (stub)
SHOP_MIN_ID   = 10     # 10‑69 handled by ShopAction
ACTION_SPACE_SIZE = 70

PHASE_PLAY = 0
PHASE_SHOP = 1

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class BalatroEnvWithExpert(gym.Env):
    """Balatro environment with Shop + ScoreEngine‑backed rewards."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: Optional[str] = None):
        super().__init__()
        self.render_mode = render_mode

        # Action / observation spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            {
                "hand": spaces.Box(low=-1, high=51, shape=(8,), dtype=np.int8),
                "chips": spaces.Box(low=0, high=1_000_000_000, shape=(), dtype=np.int32),
                "phase": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
                # shop_* keys injected on the fly during shop phase
            }
        )

        # Core game & scoring
        self.engine = ScoreEngine()
        self.game   = BalatroGame(engine=self.engine)

        # Player economy mirrored for shop purchases
        self.player = PlayerState(chips=1_000)
        self.ante   = 1

        # Phase control
        self.phase      = PHASE_PLAY
        self.shop: Optional[Shop] = None
        self.round_done = False

    # ------------------------------------------------------------------
    # Reset / Round helpers
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self.engine = ScoreEngine()
        self.game   = BalatroGame(engine=self.engine)
        self.player.chips = 1_000
        self.ante   = 1
        self.phase  = PHASE_PLAY
        self._deal_new_hand()
        return self._get_obs(), {}

    def _deal_new_hand(self):
        self.game._draw_cards()  # refresh BalatroGame’s internal hand
        # copy first 8 Card indices for observation
        self.hand = np.array(self.game.hand_indexes[:8], dtype=np.int8)

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.phase == PHASE_SHOP:
            return self._step_shop(action)
        return self._step_play(action)

    # -------------------- SHOP phase -----------------------
    def _step_shop(self, action: int):
        if not ShopAction.is_shop_action(action):
            raise ValueError("Non‑shop action during SHOP phase")
        reward, done_shop, info = self.shop.step(action, self.player)
        if done_shop:
            self.phase = PHASE_PLAY
            self._deal_new_hand()
        return self._get_obs(), reward, False, False, info

    # -------------------- PLAY phase -----------------------
    def _step_play(self, action: int):
        reward = 0.0
        info: Dict = {}

        # Action 0 – play first 5 cards
        if action == 0:
            # Highlight first five cards in BalatroGame
            self.game.highlighted_indexes = self.game.hand_indexes[:5]
            reward = self.game.play_hand()  # returns chip payout via ScoreEngine
            self.player.chips += int(reward)
            self.round_done = True
        # Action 1 – discard hand (simple demo)
        elif action == 1:
            self.player.chips = max(0, self.player.chips - 1)  # discard cost
            self.game.discard_hand()
            self._deal_new_hand()
        # Actions 2‑9 no‑op for now
        else:
            pass

        if self.round_done:
            self._end_of_round(won=True)

        return self._get_obs(), reward, False, False, info

    # -------------------- end of round ▸ shop --------------
    def _end_of_round(self, *, won: bool):
        if won:
            self.ante += 1
            self.player.chips += 50 * self.ante
        self.phase = PHASE_SHOP
        self.shop  = Shop(self.ante, self.player, seed=int(np.random.randint(1<<31)))
        self.round_done = False

    # ------------------------------------------------------------------
    # Observation & masks
    # ------------------------------------------------------------------
    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == PHASE_PLAY:
            mask[:PLAY_MAX_ID+1] = 1
        else:
            inv_len = len(self.shop.inventory) if self.shop else 0
            shop_ids = SHOP_MIN_ID + np.arange(inv_len, dtype=np.int8)
            mask[shop_ids] = 1
            mask[ShopAction.SKIP]   = 1
            mask[ShopAction.REROLL] = 1
        return mask

    def _get_obs(self) -> Dict:
        obs = {
            "hand":  np.array(self.hand, copy=True),
            "chips": np.int32(self.player.chips),
            "phase": np.int8(self.phase),
            "action_mask": self._action_mask(),
        }
        if self.phase == PHASE_SHOP and self.shop is not None:
            obs.update(self.shop.get_observation())
        return obs

    # ------------------------------------------------------------------
    # Render helpers
    # ------------------------------------------------------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.phase == PHASE_SHOP:
            print("-- SHOP -- Chips:", self.player.chips)
            for i, itm in enumerate(self.shop.inventory):
                print(f"[{i}] {itm.name:<20} cost:{itm.cost}")
        else:
            suit = "♠♣♥♦"
            rank = "23456789TJQKA"
            pretty = " ".join(f"{rank[c%13]}{suit[c//13]}" for c in self.hand)
            print(f"Ante {self.ante} | Chips {self.player.chips} | {pretty}")

    def close(self):
        pass
