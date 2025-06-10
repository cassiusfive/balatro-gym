"""Centralised enumerations and helper utilities for Balatro Gym
================================================================
This module replaces scattered *magic integers* with explicit `IntEnum`
classes. Import these enums everywhere instead of raw numbers to gain:

1. **Safety** – your IDE & `mypy` will scream when you pass the wrong kind of
   integer.
2. **Readability** – `Action.PLAY_HAND` is self‑explanatory; `0` is not.
3. **Refactor‑friendliness** – change a value here, *all* call‑sites update.

Usage (env excerpt)
-------------------
```python
from balatro_gym.constants import Action, Phase

if Phase(self.state.phase) is Phase.PLAY:
    if Action(action) is Action.PLAY_HAND:
        ...
    elif Action.SELECT_CARD_BASE <= action < Action.SELECT_CARD_BASE + Action.SELECT_CARD_COUNT:
        card_idx = action - Action.SELECT_CARD_BASE
        ...
```

Tip: wrap the integer `action` into `Action(action)` once at the top of
`step()` and pattern‑match from there.
"""

from __future__ import annotations

from enum import IntEnum, unique


@unique
class Phase(IntEnum):
    """High‑level game phases."""

    PLAY = 0
    SHOP = 1
    BLIND_SELECT = 2
    PACK_OPEN = 3


@unique
class Action(IntEnum):
    """Flat action space (0‑59) with *base* offsets for parameterised actions.

    For ranges (e.g. *select card 0‑7*) we expose both **BASE** and **COUNT**
    constants so you can compute the concrete integer or reverse‑map from it.
    """

    # === Play‑phase basics ===
    PLAY_HAND: int = 0
    DISCARD: int = 1

    # --- Card selection (8 options) ---
    SELECT_CARD_BASE: int = 2
    SELECT_CARD_COUNT: int = 8  # → 2‑9

    # --- Consumables (5 slots) ---
    USE_CONSUMABLE_BASE: int = 10
    USE_CONSUMABLE_COUNT: int = 5  # → 10‑14

    # === Shop‑phase ===
    SHOP_BUY_BASE: int = 20  # 10 item slots → 20‑29
    SHOP_BUY_COUNT: int = 10

    SHOP_REROLL: int = 30
    SHOP_END: int = 31

    # Selling
    SELL_JOKER_BASE: int = 32  # 5 joker slots → 32‑36
    SELL_JOKER_COUNT: int = 5

    SELL_CONSUMABLE_BASE: int = 37  # 5 consumable slots → 37‑41
    SELL_CONSUMABLE_COUNT: int = 5

    # === Blind selection ===
    SELECT_BLIND_BASE: int = 45  # small/big/boss → 45‑47
    SELECT_BLIND_COUNT: int = 3
    SKIP_BLIND: int = 48

    # === Pack opening ===
    SELECT_FROM_PACK_BASE: int = 50  # 5 choices → 50‑54
    SELECT_FROM_PACK_COUNT: int = 5
    SKIP_PACK: int = 55

    # === Meta ===
    ACTION_SPACE_SIZE: int = 60

    # ------------------------------------------------------------------
    # Helper methods
    # ------------------------------------------------------------------
    def offset(self, index: int) -> int:  # noqa: D401 – simple helper
        """Return the concrete *integer* ID for an indexed variant.

        Example::
            Action.SELECT_CARD_BASE.offset(3)  # → 5
        """
        return int(self) + index

    @classmethod
    def from_offset(cls, base: "Action", id_: int) -> int:
        """Reverse‑map: given a *concrete* action ID, return its **index**.

        Example::
            idx = Action.from_offset(Action.SELECT_CARD_BASE, action_id)
        """
        return id_ - int(base)


__all__ = [
    "Phase",
    "Action",
]
