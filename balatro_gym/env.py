"""
balatro_gym/env.py
==================

A slimmed-down Gymnasium environment for **8-card draw poker**:

Gameplay
--------
* **Phase 0 – Discard**:
    Agent submits an 8-bit discard mask (0–255). Each bit i=1 indicates card i will be replaced.
* **Phase 1 – Select-Five**:
    After drawing replacements, agent chooses *exactly* 5 out of 8 cards to score.
    There are `C(8,5) = 56` combinations (action IDs 256–311).

The unified action space is **Discrete(312)**.
At each step, `action_mask()` exposes only the legal subset of that space.

Observation
-----------
Dict(
    cards       : MultiBinary(8×52)  – one-hot encoding of the 8-card hand
    phase       : Discrete(2)        – 0 = discard, 1 = select-five
    action_mask : MultiBinary(312)   – mask of currently valid actions
)

Reward
------
Returns a normalized poker hand score in the range [0, 1].

Dependencies
------------
* `gymnasium`
* `numpy`
* [`treys`](https://github.com/worldveil/deuces) (MIT, optional) for scoring
"""


from __future__ import annotations
from balatro_gym.core.balatro_game import Card, BalatroGame

import random
from itertools import combinations
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

# --------------------------------------------------------------------------- #
# Action-space helpers                                                        #
# --------------------------------------------------------------------------- #

NUM_DISCARD_ACTIONS = 256            # 2**8
DISCARD_OFFSET = 0                   # IDs 0-255

FIVE_CARD_COMBOS: List[Tuple[int, ...]] = list(combinations(range(8), 5))
NUM_SELECT_ACTIONS = len(FIVE_CARD_COMBOS)  # 56
SELECT_OFFSET = NUM_DISCARD_ACTIONS        # 256

ACTION_SPACE_SIZE = NUM_DISCARD_ACTIONS + NUM_SELECT_ACTIONS  # 312


def decode_discard(action_id: int) -> List[int]:
    """Return indices (0-7) to throw away for a *discard* action."""
    return [i for i in range(8) if (action_id >> i) & 1]


def decode_select(action_id: int) -> Tuple[int, ...]:
    """Return the 5 kept indices for a *select-five* action."""
    return FIVE_CARD_COMBOS[action_id - SELECT_OFFSET]


# --------------------------------------------------------------------------- #
# Optional – Poker evaluator (Treys)                                          #
# --------------------------------------------------------------------------- #

try:
    from treys import Card as _TreysCard, Evaluator as _TreysEval

    _evaluator = _TreysEval()

    SUITS = "shdc"          # Treys order: spades, hearts, diamonds, clubs
    RANKS = "23456789TJQKA"

    def _int_to_card(idx: int) -> Card:
    	rank = Card.Ranks(idx % 13)
    	suit = Card.Suits(idx // 13)
    	return Card(rank, suit)

    def score_five(cards: np.ndarray) -> float:
    	card_objs = [_int_to_card(int(c)) for c in cards]
    	chips = BalatroGame._evaluate_hand(card_objs)
    	return chips / 1000.0  # normalize (optional, tune upper bound as needed)

except ModuleNotFoundError:
    _evaluator = None

    def score_five(cards: np.ndarray) -> float:  # pragma: no cover
        """Fallback: deterministic hash for scoring (non-deterministic order-safe)."""
        return hash(tuple(sorted(map(int, cards)))) % 7463 / 7462.0


# --------------------------------------------------------------------------- #
# Environment                                                                 #
# --------------------------------------------------------------------------- #


class EightCardDrawEnv(gym.Env):
    """
    8-card draw-poker environment (single-hand episode).

    Observation
    -----------
    Dict(
        cards       : MultiBinary(8×52)  – one-hot representation of current hand
        phase       : Discrete(2)        – 0 = discard phase, 1 = select-five phase
        action_mask : MultiBinary(312)   – legal actions at this step
    )

    Action
    ------
    Discrete(312)
        0-255   : discard mask (bit i=1 ⇒ throw card i)
        256-311 : choose combo index from ``FIVE_CARD_COMBOS``

    Reward
    ------
    Normalised poker score in [0, 1].
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, *, render_mode: str | None = None):
        super().__init__()
        self.render_mode = render_mode

        # Static spaces
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Dict(
            {
                "cards": spaces.MultiBinary((8, 52)),
                "phase": spaces.Discrete(2),
                "action_mask": spaces.MultiBinary(ACTION_SPACE_SIZE),
            }
        )

        # Internal state
        self.deck: np.ndarray | None = None
        self.hand: np.ndarray | None = None
        self.phase: int = 0       # 0 = discard, 1 = select-five
        self._terminated: bool = False

    # ------------------------------- Helpers -------------------------------- #

    def _deal_hand(self) -> None:
        self.deck = np.arange(52, dtype=np.int8)
        np.random.shuffle(self.deck)
        self.hand = self.deck[:8].copy()        # shape (8,)

    def _encode_cards(self) -> np.ndarray:
        one_hot = np.zeros((8, 52), dtype=np.int8)
        one_hot[np.arange(8), self.hand] = 1
        return one_hot

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=np.int8)
        if self.phase == 0:
            mask[:NUM_DISCARD_ACTIONS] = 1
        else:
            mask[SELECT_OFFSET : SELECT_OFFSET + NUM_SELECT_ACTIONS] = 1
        return mask

    # ---------------------------- Gym interface ----------------------------- #

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._deal_hand()
        self.phase = 0
        self._terminated = False

        obs = {
            "cards": self._encode_cards(),
            "phase": np.array(self.phase, dtype=np.int8),
            "action_mask": self._action_mask(),
        }
        return obs, {}

    def step(self, action: int):
        if self._terminated:
            raise RuntimeError("`step()` called on terminated episode")

        reward = 0.0
        info = {}

        # ----------------------- Phase 0 → discard ------------------------ #
        if self.phase == 0:
            discards = decode_discard(action)
            n_draw = len(discards)
            if n_draw:
                draw = self.deck[8 : 8 + n_draw]
                self.hand[discards] = draw
            self.phase = 1
            terminated = False

        # ----------------------- Phase 1 → score -------------------------- #
        else:
            keep_idx = decode_select(action)
            keep_cards = self.hand[list(keep_idx)]
            reward = score_five(keep_cards)
            terminated = True
            self._terminated = True

        obs = {
            "cards": self._encode_cards(),
            "phase": np.array(self.phase, dtype=np.int8),
            "action_mask": self._action_mask(),
        }
        return obs, reward, terminated, False, info

    # ------------------------------- Render -------------------------------- #

    def render(self):
        if self.render_mode != "human":
            return
        SUITS = "♠♥♦♣"
        RANKS = "23456789TJQKA"
        pretty = [f"{RANKS[c % 13]}{SUITS[c // 13]}" for c in self.hand]
        phase_name = "Discard" if self.phase == 0 else "Select-5"
        print(f"[{phase_name}] Hand: {' '.join(pretty)}")

    # --------------------------- Close (noop) ------------------------------- #

    def close(self):
        pass


# --------------------------------------------------------------------------- #
# Factory helper (optional convenience)                                       #
# --------------------------------------------------------------------------- #


def make(id: str = "EightCardDraw-v0", **kwargs):
    if id != "EightCardDraw-v0":
        raise ValueError(f"Unknown env id: {id}")
    return EightCardDrawEnv(**kwargs)
