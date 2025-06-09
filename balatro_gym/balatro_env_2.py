"""balatro_gym/envs/balatro_env_v2.py – now registers dynamic joker modifiers.

Adds two common jokers:
* **Ride the Bus** (ID 44) – gains ×1 Mult per consecutive hand with no face
  card. Resets on face card.
* **Green Joker** (ID 58) – starts ×1; +1 Mult after each hand scored, −1 Mult
  per discard action.

When the agent buys one of these jokers during the Shop phase, the environment
instantiates the corresponding modifier class and registers it with
`ScoreEngine`.  Discard actions trigger `GreenJoker.on_discard()` so its Mult is
reduced correctly.
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional, List, Callable

import gymnasium as gym
from gymnasium import spaces

from balatro_gym.shop import Shop, ShopAction, PlayerState
from balatro_gym.scoring_engine import ScoreEngine, Planet
from balatro_gym.balatro_game_v2 import BalatroGame

# ---------------------------------------------------------------------------
# Dynamic Joker Modifier classes
# ---------------------------------------------------------------------------

ModifierFn = Callable[[float, List[int], ScoreEngine], float]

class RideTheBus:
    """ID 44 – +1 Mult per consecutive face‑less scoring hand."""
    def __init__(self):
        self.streak = 0
    def __call__(self, score: float, hand: List[int], eng: ScoreEngine) -> float:
        has_face = any((c % 13) >= 9 for c in hand)  # J,Q,K,A indexes 9‑12
        if has_face:
            self.streak = 0
        else:
            self.streak += 1
        return score * (1 + self.streak)

class GreenJoker:
    """ID 58 – +1 Mult per hand, −1 per discard."""
    def __init__(self):
        self.mult = 1.0
    def on_discard(self):
        self.mult = max(0.0, self.mult - 1)
    def __call__(self, score: float, hand: List[int], eng: ScoreEngine) -> float:
        res = score * self.mult
        self.mult += 1.0
        return res

# ---------------------------------------------------------------------------
# Env constants
# ---------------------------------------------------------------------------
PLAY_MAX_ID   = 9
SHOP_MIN_ID   = 10
ACTION_SPACE_SIZE = 70
PHASE_PLAY = 0
PHASE_SHOP = 1

# Mapping joker_id → constructor
JOKER_FACTORY = {44: RideTheBus, 58: GreenJoker}

# ---------------------------------------------------------------------------
class BalatroEnvWithExpert(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            hand=spaces.Box(-1, 51, (8,), dtype=np.int8),
            chips=spaces.Box(0, 1_000_000_000, (), dtype=np.int32),
            phase=spaces.Discrete(2),
            action_mask=spaces.MultiBinary(ACTION_SPACE_SIZE),
        )

        # core objects
        self.engine = ScoreEngine()
        self.game   = BalatroGame(engine=self.engine)
        self.player = PlayerState(chips=1_000)
        self.ante   = 1

        # dynamic joker instances needing per‑step callbacks (e.g., GreenJoker)
        self.dynamic_jokers: List[object] = []

        self.phase = PHASE_PLAY
        self.shop: Optional[Shop] = None
        self._deal()

    # ------------------------------------------------------------------
    # Reset & helpers
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.engine = ScoreEngine()
        self.game   = BalatroGame(engine=self.engine)
        self.player.chips = 1_000
        self.ante = 1
        self.dynamic_jokers.clear()
        self.phase = PHASE_PLAY
        self.shop  = None
        self._deal()
        return self._obs(), {}

    def _deal(self):
        self.game._draw_cards()
        self.hand = np.array(self.game.hand_indexes[:8], dtype=np.int8)

    # ------------------------------------------------------------------
    # Step dispatcher
    # ------------------------------------------------------------------
    def step(self, action: int):
        if self.phase == PHASE_SHOP:
            return self._step_shop(action)
        return self._step_play(action)

    # ---------------- SHOP phase ----------------
    def _step_shop(self, action: int):
        verb, idx = ShopAction.decode(action)
        joker_id_purchased = None
        if verb == "buy_joker" and idx < len(self.shop.inventory):
            joker_id_purchased = self.shop.inventory[idx].payload["joker_id"]
        reward, done_shop, info = self.shop.step(action, self.player)

        # register planet consumable
        if "planet" in info:
            self.engine.apply_consumable(info["planet"])

        # register dynamic joker modifier
        if joker_id_purchased in JOKER_FACTORY:
            modifier = JOKER_FACTORY[joker_id_purchased]()
            self.engine.register_modifier(modifier)  # type: ignore
            if isinstance(modifier, GreenJoker):
                self.dynamic_jokers.append(modifier)

        if done_shop:
            self.phase = PHASE_PLAY
            self._deal()
        return self._obs(), reward, False, False, info

    # ---------------- PLAY phase ----------------
    def _step_play(self, action: int):
        reward = 0.0
        if action == 0:  # play first 5
            self.game.highlighted_indexes = self.game.hand_indexes[:5]
            reward = self.game.play_hand()
            self.player.chips += int(reward)
            self._end_round()
        elif action == 1:  # discard all (penalise chips, update GreenJoker)
            self.player.chips = max(0, self.player.chips - 1)
            for j in self.dynamic_jokers:
                if isinstance(j, GreenJoker):
                    j.on_discard()
            self.game.discard_hand()
            self._deal()
        return self._obs(), reward, False, False, {}

    # -------------- round → shop --------------
    def _end_round(self):
        self.phase = PHASE_SHOP
        self.ante += 1
        self.player.chips += 50 * self.ante
        self.shop = Shop(self.ante, self.player, seed=int(np.random.randint(1<<31)))

    # -------------- observation ---------------
    def _mask(self):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == PHASE_PLAY:
            mask[:PLAY_MAX_ID+1] = 1
        else:
            mask[ShopAction.SKIP] = mask[ShopAction.REROLL] = 1
            for i in range(len(self.shop.inventory)):
                mask[SHOP_MIN_ID+i] = 1
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

    # -------------- render --------------------
    def render(self):
        if self.render_mode != "human":
            return
        if self.phase == PHASE_SHOP:
            print("-- SHOP -- Chips:", self.player.chips)
            for i,it in enumerate(self.shop.inventory):
                print(f"[{i}] {it.name:<20} cost:{it.cost}")
        else:
            suit="♠♣♥♦"; rank="23456789TJQKA"
            print("Hand:"," ".join(f"{rank[c%13]}{suit[c//13]}" for c in self.hand))

    def close(self):
        pass
