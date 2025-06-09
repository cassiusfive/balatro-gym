"""balatro_gym/envs/balatro_env_v2.py

Full‑round Balatro environment **with shop phase and expert player**.
-------------------------------------------------------------------
* PLAY phase  – agent (or expert) plays/discards as in vanilla Balatro.
* SHOP phase  – uses `shop.Shop`, letting the agent buy packs/jokers/vouchers,
                reroll, or skip.

Action ID layout
----------------
0‑9     : original in‑round actions (play, discard, etc.) – unchanged
10‑69   : shop actions (see `shop.ShopAction`)

Observation keys (superset)
---------------------------
hand            : (8,)      int8   – card indices 0‑51 (‑1 = empty)
chips           : ()        int32
phase           : ()        int8    – 0 = PLAY, 1 = SHOP
shop_item_type  : (≤10,)    int8    – only when in shop
shop_cost       : (≤10,)    int32   –        "        "
shop_payload    : (≤10,)    object  –        "        "
action_mask     : (70,)     int8    – legal moves this step

Dependencies
------------
* gymnasium
* numpy
* balatro_probability_engine (for expert; optional)
* shop.py (in the same package)
"""
from __future__ import annotations

import random
from typing import Dict, Optional, List, Tuple

import gymnasium as gym
import numpy as np

from shop import Shop, ShopAction, PlayerState

# Optional expert imports  (wrap in try/except so env is usable without them)
try:
    from balatro_probability_engine import BalatroOptimalPlayer, Card, Rank, Suit
except ImportError:  # pragma: no cover
    BalatroOptimalPlayer = None  # type: ignore

# ---------------------------------------------------------------------------
# Constants – action IDs
# ---------------------------------------------------------------------------
PLAY_MAX_ID = 9          # 0‑9 reserved for gameplay (whatever your original env used)
SHOP_MIN_ID = 10         # 10‑69 handled by ShopAction
ACTION_SPACE_SIZE = 70

# Phase enum
PHASE_PLAY = 0
PHASE_SHOP = 1


class BalatroEnvWithExpert(gym.Env):
    """Balatro RL environment with optional optimal‑player trajectories."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: Optional[str] = None, use_expert_for_actions: bool = False):
        super().__init__()
        self.render_mode = render_mode

        # ------------- action / observation spaces ------------
        self.action_space = gym.spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = gym.spaces.Dict(
            {
                "hand": gym.spaces.Box(low=-1, high=51, shape=(8,), dtype=np.int8),
                "chips": gym.spaces.Box(low=0, high=1_000_000_000, shape=(), dtype=np.int32),
                "phase": gym.spaces.Discrete(2),
                "action_mask": gym.spaces.MultiBinary(ACTION_SPACE_SIZE),
                # Shop keys are appended on‑the‑fly when phase==SHOP.
            }
        )

        # ------------------- game state -----------------------
        self.rng = np.random.default_rng()
        self.ante: int = 1
        self.player = PlayerState(chips=1_000)
        self.hand: np.ndarray = np.full(8, -1, dtype=np.int8)  # -1 means empty

        # phase control
        self.phase: int = PHASE_PLAY
        self.shop: Optional[Shop] = None
        self.round_done: bool = False  # set when blind beaten or bust

        # ------------------- expert player --------------------
        self.use_expert_for_actions = use_expert_for_actions and BalatroOptimalPlayer is not None
        self.expert = BalatroOptimalPlayer() if self.use_expert_for_actions else None

        # trajectory recording
        self.current_trajectory: List[Dict] = []
        self.trajectories: List[List[Dict]] = []

    # ======================================================================
    # Reset / round management
    # ======================================================================

    def reset(self, *, seed: Optional[int] = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.ante = 1
        self.player.chips = 1_000
        self.player.deck.clear()
        self.player.jokers.clear()
        self.player.vouchers.clear()
        self._start_new_round()

        obs = self._get_obs()
        if self.use_expert_for_actions:
            self.current_trajectory = [{"observation": obs.copy(), "game_state": self._get_full_state()}]
        return obs, {}

    # ----------------------------------------------------------------------
    # Core loop – step
    # ----------------------------------------------------------------------

    def step(self, action: int):
        if self.phase == PHASE_SHOP:
            return self._step_shop(action)
        else:
            return self._step_play(action)

    # ------------------- shop phase --------------------

    def _step_shop(self, action: int):
        if not ShopAction.is_shop_action(action):
            raise ValueError("Action not valid during shop phase")

        reward, shop_done, info = self.shop.step(action, self.player)  # type: ignore
        if shop_done:
            self.phase = PHASE_PLAY
            self._start_new_round()

        obs = self._get_obs()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    # ------------------- play phase (simplified) --------------------

    def _step_play(self, action: int):
        # In a real implementation you'd route 0‑9 to play/discard/etc.
        # Here we stub out a dummy reward and escalate ante every 5 actions.
        reward = 0.0
        info = {}

        # Dummy: every action costs 1 chip
        self.player.chips -= 1

        # Terminal condition example
        self.round_done = self.player.chips <= 0 or action == 0  # pretend action 0 clears blind
        if self.round_done:
            self._end_of_round(won=True)

        obs = self._get_obs()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, info

    # ==================== Round transitions =========================

    def _start_new_round(self):
        # deal new hand
        deck = np.arange(52, dtype=np.int8)
        self.rng.shuffle(deck)
        self.hand = deck[:8]
        self.round_done = False

    def _end_of_round(self, *, won: bool):
        if won:
            self.ante += 1
            self.player.chips += 50 * self.ante  # dummy prize
        if self.ante > 8 or not won:
            # episode end – Gym termination not implemented for brevity
            pass

        # enter shop
        self.phase = PHASE_SHOP
        self.shop = Shop(self.ante, self.player, seed=int(self.rng.integers(2**31)))

    # ======================================================================
    # Expert helpers (placeholder – call your probability engine here)
    # ======================================================================

    def get_expert_action(self) -> int:
        if self.phase == PHASE_SHOP:
            return ShopAction.SKIP  # expert skips shop by default
        return 0  # stub – always choose play‑action 0

    # ======================================================================
    # Observation / state helpers
    # ======================================================================

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == PHASE_SHOP:
            mask[SHOP_MIN_ID:ACTION_SPACE_SIZE] = 1  # all shop actions allowed by default
            # could refine using self.shop.inventory length
        else:
            mask[:PLAY_MAX_ID + 1] = 1
        return mask

    def _get_obs(self) -> Dict:
        obs: Dict = {
            "hand": self.hand.copy(),
            "chips": np.int32(self.player.chips),
            "phase": np.int8(self.phase),
            "action_mask": self._action_mask(),
        }
        if self.phase == PHASE_SHOP:
            obs.update(self.shop.get_observation())  # type: ignore
        return obs

    # full state dump for expert / trajectory recording
    def _get_full_state(self) -> Dict:
        return {
            "ante": self.ante,
            "chips": self.player.chips,
            "jokers": self.player.jokers.copy(),
            "vouchers": self.player.vouchers.copy(),
            "hand": self.hand.copy(),
            "phase": self.phase,
        }

    # ======================================================================
    # Render (basic)
    # ======================================================================

    def render(self):
        if self.render_mode != "human":
            return
        if self.phase == PHASE_SHOP:
            print("-- SHOP --")
            for idx, itm in enumerate(self.shop.inventory):  # type: ignore
                print(f"[{idx}] {itm.name}  Cost:{itm.cost}  Payload:{itm.payload}")
        else:
            suit_symbols = "♠♥♦♣"
            rank_symbols = "23456789TJQKA"
            hand_pretty = " ".join(f"{rank_symbols[c%13]}{suit_symbols[c//13]}" for c in self.hand)
            print(f"(Ante {self.ante}) Chips:{self.player.chips}  Hand:{hand_pretty}")

    def close(self):
        pass
