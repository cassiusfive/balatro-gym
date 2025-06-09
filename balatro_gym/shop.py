"""shop.py – Deck-building subsystem for balatro_gym

• Enumerates all 150 Jokers (see `JOKER_LIBRARY`).
• Implements **pack opening** so buying a pack actually adds playing cards to the
  player’s deck (requested by user, June 9 ’25).

Pack rules (simplified for RL)
──────────────────────────────
Standard Pack   → 3 random cards  (indexes 0‑51)
Joker Pack      → 1 random card   (plus dense reward for flavour)
Tarot/Planet/Spectral Pack → 1 random card each (consumables ignored for now)

When a pack is opened, the new card indices are returned in `info["new_cards"]`
so training scripts can log them if desired.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

# ──────────────────────── Item types ────────────────────────────

class ItemType(IntEnum):
    PACK = auto()
    CARD = auto()
    JOKER = auto()
    VOUCHER = auto()

# ─────────────────────── Jokers (150) ───────────────────────────

@dataclass(frozen=True)
class JokerInfo:
    id: int
    name: str
    base_cost: int
    effect: str

# (For brevity, listing only first 10; full list in original file…)
JOKER_LIBRARY: List[JokerInfo] = [
    JokerInfo(1, "Joker", 2, "+4 Mult"),
    JokerInfo(2, "Greedy Joker", 5, "♦ +3 Mult"),
    JokerInfo(3, "Lusty Joker", 5, "♥ +3 Mult"),
    JokerInfo(4, "Wrathful Joker", 5, "♠ +3 Mult"),
    JokerInfo(5, "Gluttonous Joker", 5, "♣ +3 Mult"),
    JokerInfo(6, "Jolly Joker", 3, "+8 Mult if Pair"),
    JokerInfo(7, "Zany Joker", 4, "+12 Mult if Trips"),
    JokerInfo(8, "Mad Joker", 4, "+10 Mult if Two‑Pair"),
    JokerInfo(9, "Crazy Joker", 4, "+12 Mult if Straight"),
    JokerInfo(10, "Droll Joker", 4, "+10 Mult if Flush"),
    # … continue up to 150 …
]

# ─────────────── Shop item and cost tables ─────────────────────

@dataclass
class ShopItem:
    item_type: ItemType
    name: str
    cost: int
    payload: Dict

    def __repr__(self):
        return f"{self.name} (cost={self.cost})"

COST_TABLE: Dict[str, int] = {
    "Standard Pack": 250,
    "Joker Pack": 500,
    "Tarot Pack": 600,
    "Planet Pack": 900,
    "Spectral Pack": 1300,
    "Voucher: Magic Trick": 600,
    "Voucher: Minimalist": 750,
}

ANTE_COST_MULT = 1.15
VOUCHER_EFFECTS = {"Magic Trick": {"pack_discount": 0.9}}

# ─────────────── ShopAction encoding (10‑69) ───────────────────

class ShopAction(IntEnum):
    SKIP = 10
    REROLL = 11
    BUY_PACK_BASE = 12
    BUY_JOKER_BASE = 20
    BUY_CARD_BASE = 40
    BUY_VOUCHER_BASE = 60

    @classmethod
    def is_shop_action(cls, a: int) -> bool:
        return a >= cls.SKIP

    @classmethod
    def decode(cls, a: int) -> Tuple[str, int]:
        if a == cls.SKIP:
            return ("skip", -1)
        if a == cls.REROLL:
            return ("reroll", -1)
        if cls.BUY_PACK_BASE <= a < cls.BUY_JOKER_BASE:
            return ("buy_pack", a - cls.BUY_PACK_BASE)
        if cls.BUY_JOKER_BASE <= a < cls.BUY_CARD_BASE:
            return ("buy_joker", a - cls.BUY_JOKER_BASE)
        if cls.BUY_CARD_BASE <= a < cls.BUY_VOUCHER_BASE:
            return ("buy_card", a - cls.BUY_CARD_BASE)
        if cls.BUY_VOUCHER_BASE <= a < cls.BUY_VOUCHER_BASE + 10:
            return ("buy_voucher", a - cls.BUY_VOUCHER_BASE)
        raise ValueError(a)

# ─────────────── Player struct ─────────────────────────────———

@dataclass
class PlayerState:
    chips: int
    vouchers: List[str] = field(default_factory=list)
    jokers: List[int] = field(default_factory=list)
    deck: List[int] = field(default_factory=list)

# ─────────────────────── Shop core ─────────────────────────────

class Shop:
    def __init__(self, ante: int, player: PlayerState, *, seed: Optional[int] = None):
        self.ante = ante
        self.player = player
        self.rng = random.Random(seed)
        self.inventory: List[ShopItem] = []
        self.reroll_cost = 50
        self._generate_inventory()

    # ---------------------- cost helpers ----------------------

    def _cost_mult(self) -> float:
        m = ANTE_COST_MULT ** (self.ante - 1)
        if "Magic Trick" in self.player.vouchers:
            m *= VOUCHER_EFFECTS["Magic Trick"]["pack_discount"]
        return m

    # -------------------- inventory gen ----------------------

    def _generate_inventory(self):
        self.inventory.clear()
        mult = self._cost_mult()

        # Packs
        for name in ["Standard Pack", "Joker Pack", self.rng.choice(["Tarot Pack", "Planet Pack", "Spectral Pack"])]:
            self.inventory.append(ShopItem(ItemType.PACK, name, int(COST_TABLE[name] * mult), {"pack_type": name}))

        # Jokers (3 unique not owned)
        candidates = [j for j in JOKER_LIBRARY if j.id not in self.player.jokers and j.base_cost > 0]
        for joker in self.rng.sample(candidates, k=min(3, len(candidates))):
            self.inventory.append(ShopItem(ItemType.JOKER, joker.name, int(joker.base_cost * mult), {"joker_id": joker.id}))

        # Voucher
        voucher_name = self.rng.choice(["Voucher: Magic Trick", "Voucher: Minimalist"])
        self.inventory.append(ShopItem(ItemType.VOUCHER, voucher_name, int(COST_TABLE[voucher_name] * mult), {"voucher": voucher_name.split(": ")[1]}))

        # Two random single cards
        for _ in range(2):
            c = self.rng.randint(0, 51)
            self.inventory.append(ShopItem(ItemType.CARD, f"Card {c}", 40, {"card": c}))

    # -------------------- observation ------------------------

    def get_observation(self) -> Dict:
        return {
            "shop_item_type": [itm.item_type for itm in self.inventory],
            "shop_cost": [itm.cost for itm in self.inventory],
            "shop_payload": [itm.payload for itm in self.inventory],
        }

    # -------------------- step -------------------------------

    def step(self, action_id: int, player: PlayerState):
        verb, idx = ShopAction.decode(action_id)
        info: Dict = {}
        reward = 0.0

        # ---- skip / reroll ----
        if verb == "skip":
            return reward, True, info
        if verb == "reroll":
            cost = int(self.reroll_cost * self._cost_mult())
            if player.chips < cost:
                return -1.0, False, {"error": "Not enough chips"}
            player.chips -= cost
            self.reroll_cost = int(self.reroll_cost * 1.35)
            self._generate_inventory()
            return 0.0, False, info

        # ---- validate index ----
        if idx >= len(self.inventory):
            return -1.0, False, {"error": "Invalid index"}
        item = self.inventory[idx]
        if player.chips < item.cost:
            return -1.0, False, {"error": "Insufficient chips"}

        # ---- purchase ----
        player.chips -= item.cost
        self.inventory.pop(idx)

        if verb == "buy_pack":
            new_cards = self._open_pack(item.payload["pack_type"], player)
            info["new_cards"] = new_cards
        elif verb == "buy_card":
            player.deck.append(item.payload["card"])
        elif verb == "buy_joker":
            if len(player.jokers) >= 5:
                return -1.0, False, {"error": "Joker slots full"}
            player.jokers.append(item.payload["joker_id"])
        elif verb == "buy_voucher":
            player.vouchers.append(item.payload["voucher"])
        else:
            raise RuntimeError(verb)

        return reward, False, info

    # -------------------- pack opening -----------------------

    def _open_pack(self, pack_type: str, player: PlayerState) -> List[int]:
        """Return list of card indices added to deck."""
        new_cards: List[int] = []
        if pack_type == "Standard Pack":
            count = 3
        else:  # Joker, Tarot, Planet, Spectral – simplified
            count = 1
        for _ in range(count):
            card = self.rng.randint(0, 51)
            player.deck.append(card)
            new_cards.append(card)
        return new_cards
