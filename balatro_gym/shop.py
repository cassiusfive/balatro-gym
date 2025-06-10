# balatro_gym/shop.py
# ---------------------------------------------------------------------------
# Deck-building subsystem: packs, jokers, vouchers, rerolls, skip-shop
# ---------------------------------------------------------------------------
from __future__ import annotations
from .jokers import JokerInfo, JOKER_LIBRARY

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Dict, List, Optional, Tuple

# --------------------------------------------------------------------------- #
#  Item / Action enums                                                        #
# --------------------------------------------------------------------------- #

class ItemType(IntEnum):
    PACK    = auto()
    CARD    = auto()
    JOKER   = auto()
    VOUCHER = auto()


# --------------------------------------------------------------------------- #
# Shop cost tables & vouchers                                                 #
# --------------------------------------------------------------------------- #
COST_TABLE: Dict[str, int] = {
    "Standard Pack": 250,
    "Joker Pack":     500,
    "Tarot Pack":     600,
    "Planet Pack":    900,
    "Spectral Pack":  1300,
    "Voucher: Magic Trick": 600,
    "Voucher: Minimalist":  750,
}

ANTE_COST_MULT = 1.15  # 15 % cost increase per ante level

VOUCHER_EFFECTS = {"Magic Trick": {"pack_discount": 0.9}}

# --------------------------------------------------------------------------- #
# ShopAction encoding (IDs 10-69)                                             #
# --------------------------------------------------------------------------- #
class ShopAction(IntEnum):
    SKIP           = 10
    REROLL         = 11
    BUY_PACK_BASE  = 12
    BUY_JOKER_BASE = 20
    BUY_CARD_BASE  = 40
    BUY_VOUCHER_BASE = 60

    @classmethod
    def is_shop_action(cls, a: int) -> bool:
        return a >= cls.SKIP

    @classmethod
    def decode(cls, a: int) -> Tuple[str, int]:
        if a == cls.SKIP:   return ("skip", -1)
        if a == cls.REROLL: return ("reroll", -1)
        if cls.BUY_PACK_BASE <= a < cls.BUY_JOKER_BASE:
            return ("buy_pack",  a - cls.BUY_PACK_BASE)
        if cls.BUY_JOKER_BASE <= a < cls.BUY_CARD_BASE:
            return ("buy_joker", a - cls.BUY_JOKER_BASE)
        if cls.BUY_CARD_BASE <= a < cls.BUY_VOUCHER_BASE:
            return ("buy_card",  a - cls.BUY_CARD_BASE)
        if cls.BUY_VOUCHER_BASE <= a < cls.BUY_VOUCHER_BASE + 10:
            return ("buy_voucher", a - cls.BUY_VOUCHER_BASE)
        raise ValueError(a)

# --------------------------------------------------------------------------- #
# PlayerState struct                                                          #
# --------------------------------------------------------------------------- #
@dataclass
class PlayerState:
    chips: int
    vouchers: List[str] = field(default_factory=list)
    jokers: List[int] = field(default_factory=list)
    deck: List[int] = field(default_factory=list)

# --------------------------------------------------------------------------- #
# ShopItem data-class                                                         #
# --------------------------------------------------------------------------- #
@dataclass
class ShopItem:
    item_type: ItemType
    name: str
    cost: int
    payload: Dict

# --------------------------------------------------------------------------- #
# Shop core                                                                   #
# --------------------------------------------------------------------------- #
class Shop:
    """Generates inventory, processes purchases, handles rerolls & skip."""

    def __init__(self, ante: int, player: PlayerState, *, seed: Optional[int] = None):
        self.ante = ante
        self.player = player
        self.rng = random.Random(seed)
        self.inventory: List[ShopItem] = []
        self.reroll_cost = 50
        self._generate_inventory()

    # ---------------- cost / discount helpers ----------------
    def _cost_mult(self) -> float:
        m = ANTE_COST_MULT ** (self.ante - 1)
        if "Magic Trick" in self.player.vouchers:
            m *= VOUCHER_EFFECTS["Magic Trick"]["pack_discount"]
        return m

    # ---------------- inventory generation -------------------
    def _generate_inventory(self):
        self.inventory.clear()
        mult = self._cost_mult()

        # three packs
        for pname in ["Standard Pack", "Joker Pack",
                      self.rng.choice(["Tarot Pack", "Planet Pack", "Spectral Pack"])]:
            self.inventory.append(ShopItem(ItemType.PACK, pname,
                                           int(COST_TABLE[pname] * mult),
                                           {"pack_type": pname}))

        # three unique jokers not already owned
        candid = [j for j in JOKER_LIBRARY if j.base_cost > 0 and j.id not in self.player.jokers]
        for joker in self.rng.sample(candid, k=min(3, len(candid))):
            self.inventory.append(ShopItem(ItemType.JOKER, joker.name,
                                           int(joker.base_cost * mult),
                                           {"joker_id": joker.id}))

        # one voucher
        vname = self.rng.choice(["Voucher: Magic Trick", "Voucher: Minimalist"])
        self.inventory.append(ShopItem(ItemType.VOUCHER, vname,
                                       int(COST_TABLE[vname] * mult),
                                       {"voucher": vname.split(": ")[1]}))

        # two random single cards
        for _ in range(2):
            c = self.rng.randint(0, 51)
            self.inventory.append(ShopItem(ItemType.CARD, f"Card {c}", 40, {"card": c}))

    # ---------------- observation helper ---------------------
    def get_observation(self) -> Dict:
        return {
            "shop_item_type": [i.item_type for i in self.inventory],
            "shop_cost":      [i.cost      for i in self.inventory],
            "shop_payload":   [i.payload   for i in self.inventory],
        }

    # ---------------- pack opening ---------------------------
    def _open_pack(self, pack_type: str) -> List[int]:
        new_cards = []
        count = 3 if pack_type == "Standard Pack" else 1
        for _ in range(count):
            card = self.rng.randint(0, 51)
            self.player.deck.append(card)
            new_cards.append(card)
        return new_cards

    # ---------------- main step ------------------------------
    def step(self, action_id: int):
        verb, idx = ShopAction.decode(action_id)
        info: Dict = {}
        reward = 0.0

        # skip shop
        if verb == "skip":
            return reward, True, info

        # reroll inventory
        if verb == "reroll":
            cost = int(self.reroll_cost * self._cost_mult())
            if self.player.chips < cost:
                return -1.0, False, {"error": "Insufficient chips for reroll"}
            self.player.chips -= cost
            self.reroll_cost = int(self.reroll_cost * 1.35)
            self._generate_inventory()
            return 0.0, False, info

        # validate index
        if idx >= len(self.inventory):
            return -1.0, False, {"error": "Invalid shop index"}
        item = self.inventory[idx]
        if self.player.chips < item.cost:
            return -1.0, False, {"error": "Insufficient chips"}

        # deduct chips & remove from inventory
        self.player.chips -= item.cost
        self.inventory.pop(idx)

        # apply purchase
        if verb == "buy_pack":
            pack_type = item.payload.get("pack_type", item.name)
            info["new_cards"] = self._open_pack(pack_type)
        elif verb == "buy_card":
            self.player.deck.append(item.payload["card"])
        elif verb == "buy_joker":
            if len(self.player.jokers) >= 5:
                return -1.0, False, {"error": "Joker slots full"}
            self.player.jokers.append(item.payload["joker_id"])
        elif verb == "buy_voucher":
            self.player.vouchers.append(item.payload["voucher"])
        else:
            raise RuntimeError(verb)

        return reward, False, info
