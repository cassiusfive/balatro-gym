"""shop.py – Deck-building subsystem for balatro_gym
────────────────────────────────────────────────────────
Full enumeration of **all 150 Jokers** with ID, base_shop_cost and concise effect stub.
Only four fields are stored (id, name, base_cost, effect) to keep the data model
minimal; rarity / unlock requirements can be inferred later from the same ID if
needed.

The sampling logic is unchanged: the shop will choose jokers the player does NOT
own, using each joker’s `base_cost` scaled by ante and vouchers.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import List, Dict, Tuple, Optional

# ──────────────────────── Constants & Enums ──────────────────────────────

class ItemType(IntEnum):
    PACK = auto()
    CARD = auto()
    JOKER = auto()
    VOUCHER = auto()


@dataclass(frozen=True)
class JokerInfo:
    id: int
    name: str
    base_cost: int  # cost at Ante‑1 before scaling; 0 = not purchasable (legendary)
    effect: str     # short effect description (engine interprets)


# ──────────────────────── Joker Library (150) ───────────────────────────

JOKER_LIBRARY: List[JokerInfo] = [
    JokerInfo(1,   "Joker",              2,  "+4 Mult"),
    JokerInfo(2,   "Greedy Joker",       5,  "♦ cards +3 Mult on score"),
    JokerInfo(3,   "Lusty Joker",        5,  "♥ cards +3 Mult on score"),
    JokerInfo(4,   "Wrathful Joker",     5,  "♠ cards +3 Mult on score"),
    JokerInfo(5,   "Gluttonous Joker",   5,  "♣ cards +3 Mult on score"),
    JokerInfo(6,   "Jolly Joker",        3,  "+8 Mult if hand has Pair"),
    JokerInfo(7,   "Zany Joker",         4,  "+12 Mult if hand has Trips"),
    JokerInfo(8,   "Mad Joker",          4,  "+10 Mult if hand has Two‑Pair"),
    JokerInfo(9,   "Crazy Joker",        4,  "+12 Mult if hand has Straight"),
    JokerInfo(10,  "Droll Joker",        4,  "+10 Mult if hand has Flush"),
    JokerInfo(11,  "Sly Joker",          3,  "+50 Chips if Pair"),
    JokerInfo(12,  "Wily Joker",         4,  "+100 Chips if Trips"),
    JokerInfo(13,  "Clever Joker",       4,  "+80 Chips if Two‑Pair"),
    JokerInfo(14,  "Devious Joker",      4,  "+100 Chips if Straight"),
    JokerInfo(15,  "Crafty Joker",       4,  "+80 Chips if Flush"),
    JokerInfo(16,  "Half Joker",         5,  "+20 Mult if ≤3 cards"),
    JokerInfo(17,  "Joker Stencil",      8,  "×1 Mult per empty slot"),
    JokerInfo(18,  "Four Fingers",       7,  "Flush & Straight need 4 cards"),
    JokerInfo(19,  "Mime",               5,  "Retrigger held‑card abilities"),
    JokerInfo(20,  "Credit Card",        1,  "Go to −$20 debt"),
    JokerInfo(21,  "Ceremonial Dagger",  6,  "Sacrifice neighbor on blind for permanent Mult"),
    JokerInfo(22,  "Banner",             5,  "+30 Chips / remaining discard"),
    JokerInfo(23,  "Mystic Summit",      5,  "+15 Mult at 0 discards"),
    JokerInfo(24,  "Marble Joker",       6,  "Add Stone card on blind pick"),
    JokerInfo(25,  "Loyalty Card",       5,  "×4 every 6 hands"),
    JokerInfo(26,  "8 Ball",             5,  "1/4 chance for Tarot per played 8"),
    JokerInfo(27,  "Misprint",           4,  "+0‑23 Mult"),
    JokerInfo(28,  "Dusk",               5,  "Retrigger played cards in final hand"),
    JokerInfo(29,  "Raised Fist",        5,  "+rank×2 Mult from lowest held"),
    JokerInfo(30,  "Chaos the Clown",    4,  "One free reroll / shop"),
    JokerInfo(31,  "Fibonacci",          8,  "+8 Mult per A,2,3,5,8"),
    JokerInfo(32,  "Steel Joker",        7,  "×0.2 Mult per Steel card"),
    JokerInfo(33,  "Scary Face",         4,  "+30 Chips per face card"),
    JokerInfo(34,  "Abstract Joker",     4,  "+3 Mult per Joker"),
    JokerInfo(35,  "Delayed Gratification", 4,  "Earn $2 per discard if none used"),
    JokerInfo(36,  "Hack",               6,  "Retrigger 2‑5s"),
    JokerInfo(37,  "Pareidolia",         5,  "All cards count as face cards"),
    JokerInfo(38,  "Gros Michel",        5,  "+15 Mult, 1/6 self‑destruct"),
    JokerInfo(39,  "Even Steven",        4,  "+4 Mult per even rank"),
    JokerInfo(40,  "Odd Todd",           4,  "+31 Chips per odd rank"),
    JokerInfo(41,  "Scholar",            4,  "+20 Chips & +4 Mult per Ace"),
    JokerInfo(42,  "Business Card",      4,  "Face cards ½ chance give $2"),
    JokerInfo(43,  "Supernova",          5,  "+hands‑played counter to Mult"),
    JokerInfo(44,  "Ride the Bus",       6,  "+1 Mult per hand w/o face"),
    JokerInfo(45,  "Space Joker",        5,  "¼ chance upgrade hand level"),
    JokerInfo(46,  "Egg",                4,  "+$3 sell value each round"),
    JokerInfo(47,  "Burglar",            6,  "+3 hands, lose discards on blind"),
    JokerInfo(48,  "Blackboard",         6,  "×3 if all ♠/♣"),
    JokerInfo(49,  "Runner",             5,  "+15 Chips if Straight (stacks)"),
    JokerInfo(50,  "Ice Cream",          5,  "+100 Chips −5 per hand"),
    JokerInfo(51,  "DNA",                8,  "Clone 1‑card first hand"),
    JokerInfo(52,  "Splash",             3,  "Every card counts in scoring"),
    JokerInfo(53,  "Blue Joker",         5,  "+2 Chips per card in deck"),
    JokerInfo(54,  "Sixth Sense",        6,  "Single 6 -> Spectral card"),
    JokerInfo(55,  "Constellation",      6,  "+0.1× per Planet used"),
    JokerInfo(56,  "Hiker",              5,  "+5 Chips perm to played cards"),
    JokerInfo(57,  "Faceless Joker",     4,  "+$5 for 3+ face discarded"),
    JokerInfo(58,  "Green Joker",        4,  "+1 Mult per hand, −1 per discard"),
    JokerInfo(59,  "Superposition",      4,  "Create Tarot if Ace+Straight"),
    JokerInfo(60,  "To Do List",         4,  "Earn $4 if specific hand"),
    JokerInfo(61,  "Cavendish",          4,  "×3, 1/1000 self‑destruct"),
    JokerInfo(62,  "Card Sharp",         6,  "×3 if hand already played"),
    JokerInfo(63,  "Red Card",           5,  "+3 Mult per pack skipped"),
    JokerInfo(64,  "Madness",            7,  "Gain ×0.5 Mult, destroy random"),
    JokerInfo(65,  "Square Joker",       4,  "+4 Chips if 4‑card hand"),
    JokerInfo(66,  "Séance",             6,  "Straight Flush -> Spectral"),
    JokerInfo(67,  "Riff‑Raff",          6,  "Create 2 common jokers on blind"),
    JokerInfo(68,  "Vampire",            7,  "+0.1× per Enhanced card"),
    JokerInfo(69,  "Shortcut",           7,  "Straights allow gaps 1"),
    JokerInfo(70,  "Hologram",           7,  "+0.25× per card added"),
    JokerInfo(71,  "Vagabond",           8,  "Tarot if hand ≤$4"),
    JokerInfo(72,  "Baron",              8,  "×1.5 per King held"),
    JokerInfo(73,  "Cloud 9",            7,  "Earn $ per 9 in deck"),
    JokerInfo(74,  "Rocket",             6,  "Earn $ each round, increases"),
    JokerInfo(75,  "Obelisk",            8,  "+0.2× per hand avoiding top hand"),
    JokerInfo(76,  "Midas Mask",         7,  "Face cards turn to Gold"),
    JokerInfo(77,  "Luchador",           5,  "Sell to disable Boss"),
    JokerInfo(78,  "Photograph",         5,  "First face card ×2 Mult"),
    JokerInfo(79,  "Gift Card",          6,  "+$1 sell value all items"),
    JokerInfo(80,  "Turtle Bean",        6,  "+5 hand size, −1/round"),
    JokerInfo(81,  "Erosion",            6,  "+4 Mult per card below start"),
    JokerInfo(82,  "Reserved Parking",   6,  "Face cards ½ chance $1 held"),
    JokerInfo(83,  "Mail‑In Rebate",     4,  "Earn $5 per rank discarded"),
    JokerInfo(84,  "To the Moon",        5,  "Extra interest"),
    JokerInfo(85,  "Hallucination",      4,  "½ chance Tarot on pack open"),
    JokerInfo(86,  "Fortune Teller",     6,  "+1 Mult per Tarot used"),
    JokerInfo(87,  "Juggler",            4,  "+1 hand size"),
    JokerInfo(88,  "Drunkard",           4,  "+1 discard each round"),
    JokerInfo(89,  "Stone Joker",        6,  "+25 Chips per Stone card"),
    JokerInfo(90,  "Golden Joker",       6,  "+$4 each round"),
    JokerInfo(91,  "Lucky Cat",          6,  "+0.25× per lucky trigger"),
    JokerInfo(92,  "Baseball Card",      8,  "Uncommon jokers ×1.5"),
    JokerInfo(93,  "Bull",               6,  "+2 Chips per $5"),
    JokerInfo(94,  "Diet Cola",          6,  "Sell -> free Double Tag"),
    JokerInfo(95,  "Trading Card",       6,  "Discard 1 card -> $3"),
    JokerInfo(96,  "Flash Card",         5,  "+2 Mult per reroll"),
    JokerInfo(97,  "Popcorn",            5,  "+20 Mult, −4 per round"),
    JokerInfo(98,  "Spare Trousers",     6,  "+2 Mult if Two Pair"),
    JokerInfo(99,  "Ancient Joker",      8,  "×1.5 per specified suit"),
    JokerInfo(100, "Ramen",              6,  "×2, loses 0.01× per discard"),
    JokerInfo(101, "Walkie Talkie",      4,  "+10 Chips +4 Mult per 10 /4"),
    JokerInfo(102, "Seltzer",            6,  "Retrigger next 10 hands"),
    JokerInfo(103, "Castle",             6,  "+3 Chips per discarded suit"),
    JokerInfo(104, "Smiley Face",        4,  "+5 Mult per face card"),
    JokerInfo(105, "Campfire",           9,  "+0.25× per card sold"),
    JokerInfo(106, "Golden Ticket",      5,  "+$4 per Gold card scored"),
    JokerInfo(107, "Mr. Bones",          5,  "Prevent death if 25% chips"),
    JokerInfo(108, "Acrobat",            6,  "×3 final hand"),
    JokerInfo(109, "Sock and Buskin",    6,  "Retrigger face cards"),
    JokerInfo(110, "Swashbuckler",       4,  "+Mult equal to sell values"),
    JokerInfo(111, "Troubadour",         6,  "+2 hand, −1 hand/round"),
    JokerInfo(112, "Certificate",        6,  "Start round with random sealed card"),
    JokerInfo(113, "Smeared Joker",      7,  "♥/♦ same suit, ♠/♣ same"),
    JokerInfo(114, "Throwback",          6,  "+0.25× per blind skipped"),
    JokerInfo(115, "Hanging Chad",       4,  "Retrigger first scored card ×2"),
    JokerInfo(116, "Rough Gem",          7,  "+$1 per ♦ scored"),
    JokerInfo(117, "Bloodstone",         7,  "50% chance ×1.5 ♥"),
    JokerInfo(118, "Arrowhead",          7,  "+50 Chips per ♠ scored"),
    JokerInfo(119, "Onyx Agate",         7,  "+7 Mult per ♣ scored"),
    JokerInfo(120, "Glass Joker",        6,  "+0.75× per Glass destroyed"),
    JokerInfo(121, "Showman",            5,  "Duplicates consumables"),
    JokerInfo(122, "Flower Pot",         6,  "×3 if hand has all suits"),
    JokerInfo(123, "Blueprint",         10,  "Copy right Joker ability"),
    JokerInfo(124, "Wee Joker",          8,  "+8 Chips per 2 scored"),
    JokerInfo(125, "Merry Andy",         7,  "+3 discards, −1 hand size"),
    JokerInfo(126, "Oops! All 6s",       4,  "Double all probabilities"),
    JokerInfo(127, "The Idol",           6,  "×2 per changing rank suit"),
    JokerInfo(128, "Seeing Double",      6,  "×2 if hand has ♣ + any suit"),
    JokerInfo(129, "Matador",            7,  "+$8 on Boss trigger"),
    JokerInfo(130, "Hit the Road",       8,  "+0.5× per Jack discarded"),
    JokerInfo(131, "The Duo",            8,  "×2 if Pair"),
    JokerInfo(132, "The Trio",           8,  "×3 if Trips"),
    JokerInfo(133, "The Family",         8,  "×4 if Quads"),
    JokerInfo(134, "The Order",          8,  "×3 if Straight"),
    JokerInfo(135, "The Tribe",          8,  "×2 if Flush"),
    JokerInfo(136, "Stuntman",           7,  "+250 Chips, −2 hand size"),
    JokerInfo(137, "Invisible Joker",    8,  "After 2 rounds, duplicate Joker"),
    JokerInfo(138, "Brainstorm",        10,  "Copy leftmost Joker"),
    JokerInfo(139, "Satellite",          6,  "Earn $ per unique Planet"),
    JokerInfo(140, "Shoot the Moon",     5,  "+13 Mult per Queen held"),
    JokerInfo(141, "Driver's License",   7,  "×3 if 16 Enhanced cards"),
    JokerInfo(142, "Cartomancer",        6,  "Create Tarot on blind"),
    JokerInfo(143, "Astronomer",         8,  "Planet cards free"),
    JokerInfo(144, "Burnt Joker",        8,  "Upgrade first discarded hand"),
    JokerInfo(145, "Bootstraps",         7,  "+2 Mult per $5"),
    JokerInfo(146, "Canio",              0,  "Legendary: +1× per face destroyed"),
    JokerInfo(147, "Triboulet",          0,  "Legendary: Kings & Queens ×2"),
    JokerInfo(148, "Yorick",             0,  "Legendary: +1× every 23 discards"),
    JokerInfo(149, "Chicot",             0,  "Legendary: disable Boss"),
    JokerInfo(150, "Perkeo",             0,  "Legendary: negative copy consumable"),
]

# ───────────────────── Base costs for item types ────────────────────────
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

# ShopAction etc. remain identical
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
        if cls.BUY_V
