"""Card-level primitives shared across the simulator.
Goals
-----
* **Single source of truth** – eliminates duplicate `Card` definitions.
* **Low memory footprint** – `@dataclass(slots=True, frozen=True)` avoids per‑instance `__dict__`.
* **Interoperable with NumPy** – `IntEnum` values are integers so you can store them in `np.uint8` arrays without casting.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Final, Optional, Dict, Tuple

@unique
class Suit(IntEnum):
    """Card suit (♣ ♦ ♥ ♠) encoded as uint8."""
    CLUBS: int = 0
    DIAMONDS: int = 1
    HEARTS: int = 2
    SPADES: int = 3
    
    def symbol(self) -> str:  # → "♣" / "♦" / "♥" / "♠"
        return "♣♦♥♠"[self]

@unique
class Rank(IntEnum):
    """Card rank (2‑A) encoded as its *face value* so maths just works."""
    TWO: int = 2
    THREE: int = 3
    FOUR: int = 4
    FIVE: int = 5
    SIX: int = 6
    SEVEN: int = 7
    EIGHT: int = 8
    NINE: int = 9
    TEN: int = 10
    JACK: int = 11
    QUEEN: int = 12
    KING: int = 13
    ACE: int = 14
    
    @property
    def short(self) -> str:  # → "2" … "A"
        lookup: Final = {
            Rank.TEN: "T",
            Rank.JACK: "J",
            Rank.QUEEN: "Q",
            Rank.KING: "K",
            Rank.ACE: "A",
        }
        return lookup.get(self, str(self.value))
    
    @property
    def base_chips(self) -> int:
        """Base chip value for this rank"""
        if self <= Rank.TEN:
            return self.value  # 2-10 worth face value
        elif self == Rank.ACE:
            return 11
        else:  # J, Q, K
            return 10

@unique
class Enhancement(IntEnum):
    """Card enhancements that modify scoring behavior"""
    NONE: int = 0
    BONUS: int = 1      # +30 chips
    MULT: int = 2       # +4 mult
    WILD: int = 3       # Any suit
    GLASS: int = 4      # ×2 mult, 1/4 chance to destroy
    STEEL: int = 5      # ×1.5 mult while in hand
    STONE: int = 6      # +50 chips, no rank/suit
    GOLD: int = 7       # $3 when held at end of round
    LUCKY: int = 8      # 1/5 chance for +20 mult, 1/15 for $20

@unique
class Edition(IntEnum):
    """Card editions that add visual flair and bonuses"""
    NONE: int = 0
    FOIL: int = 1         # +50 chips
    HOLOGRAPHIC: int = 2  # +10 mult
    POLYCHROME: int = 3   # ×1.5 mult
    NEGATIVE: int = 4     # +1 joker slot (jokers only)

@unique
class Seal(IntEnum):
    """Card seals that trigger special effects"""
    NONE: int = 0
    GOLD: int = 1    # Earn $3 when played and scored
    RED: int = 2     # Retrigger card
    BLUE: int = 3    # Creates Planet card for played hand
    PURPLE: int = 4  # Creates Tarot card when discarded

@dataclass(frozen=True, slots=True)
class Card:
    """Immutable playing card (≈32 B per instance)"""
    rank: Rank
    suit: Suit
    
    def __str__(self) -> str:
        return f"{self.rank.short}{self.suit.symbol()}"
    
    # Handy helpers so you can sort()/hash() cards effortlessly
    def __int__(self) -> int:  # unique 0–51 mapping → (rank‑2) * 4 + suit
        return (self.rank - 2) * 4 + self.suit
    
    def __lt__(self, other: "Card") -> bool:  # sort by rank then suit
        return (self.rank, self.suit) < (other.rank, other.suit)
    
    def __hash__(self) -> int:
        return hash((self.rank, self.suit))
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Card):
            return NotImplemented
        return self.rank == other.rank and self.suit == other.suit

# Enhancement effect functions
class EnhancementEffects:
    """Static methods for applying enhancement effects"""
    
    @staticmethod
    def get_chip_bonus(enhancement: Enhancement, base_chips: int) -> int:
        """Get additional chips from enhancement"""
        if enhancement == Enhancement.BONUS:
            return 30
        elif enhancement == Enhancement.STONE:
            return 50  # Stone cards always give 50 chips
        return 0
    
    @staticmethod
    def get_mult_bonus(enhancement: Enhancement) -> int:
        """Get additional mult from enhancement"""
        if enhancement == Enhancement.MULT:
            return 4
        return 0
    
    @staticmethod
    def get_mult_multiplier(enhancement: Enhancement, in_hand: bool = False) -> float:
        """Get mult multiplier from enhancement"""
        if enhancement == Enhancement.GLASS:
            return 2.0
        elif enhancement == Enhancement.STEEL and in_hand:
            return 1.5
        return 1.0
    
    @staticmethod
    def is_wild(enhancement: Enhancement) -> bool:
        """Check if card acts as wild (any suit)"""
        return enhancement == Enhancement.WILD
    
    @staticmethod
    def is_stone(enhancement: Enhancement) -> bool:
        """Check if card is stone (no suit/rank)"""
        return enhancement == Enhancement.STONE
    
    @staticmethod
    def get_gold_value(enhancement: Enhancement) -> int:
        """Get money earned from gold cards"""
        if enhancement == Enhancement.GOLD:
            return 3
        return 0
    
    @staticmethod
    def should_break_glass(rng_value: float) -> bool:
        """Check if glass card breaks (1/4 chance)"""
        return rng_value < 0.25
    
    @staticmethod
    def get_lucky_bonus(mult_rng: float, money_rng: float) -> Tuple[int, int]:
        """Get lucky card bonus (mult, money)"""
        mult_bonus = 20 if mult_rng < 0.2 else 0  # 1/5 chance
        money_bonus = 20 if money_rng < 0.0667 else 0  # 1/15 chance
        return mult_bonus, money_bonus

# Edition effect functions
class EditionEffects:
    """Static methods for applying edition effects"""
    
    @staticmethod
    def get_chip_bonus(edition: Edition) -> int:
        """Get additional chips from edition"""
        if edition == Edition.FOIL:
            return 50
        return 0
    
    @staticmethod
    def get_mult_bonus(edition: Edition) -> int:
        """Get additional mult from edition"""
        if edition == Edition.HOLOGRAPHIC:
            return 10
        return 0
    
    @staticmethod
    def get_mult_multiplier(edition: Edition) -> float:
        """Get mult multiplier from edition"""
        if edition == Edition.POLYCHROME:
            return 1.5
        return 1.0
    
    @staticmethod
    def get_joker_slots(edition: Edition) -> int:
        """Get extra joker slots from edition (jokers only)"""
        if edition == Edition.NEGATIVE:
            return 1
        return 0

# Seal effect functions  
class SealEffects:
    """Static methods for applying seal effects"""
    
    @staticmethod
    def get_money_bonus(seal: Seal) -> int:
        """Get money earned when card is played"""
        if seal == Seal.GOLD:
            return 3
        return 0
    
    @staticmethod
    def should_retrigger(seal: Seal) -> bool:
        """Check if card should retrigger"""
        return seal == Seal.RED
    
    @staticmethod
    def get_planet_created(seal: Seal, hand_type: str) -> Optional[str]:
        """Get planet card created for hand type"""
        if seal == Seal.BLUE:
            # Map hand types to planets
            planet_map = {
                'High Card': 'Pluto',
                'One Pair': 'Mercury', 
                'Two Pair': 'Venus',
                'Three Kind': 'Earth',
                'Straight': 'Mars',
                'Flush': 'Jupiter',
                'Full House': 'Saturn',
                'Four Kind': 'Uranus',
                'Straight Flush': 'Neptune',
                'Five Kind': 'Planet X',
                'Flush House': 'Ceres',
                'Flush Five': 'Eris'
            }
            return planet_map.get(hand_type)
        return None
    
    @staticmethod
    def get_tarot_created(seal: Seal) -> Optional[str]:
        """Get tarot card created when discarded"""
        if seal == Seal.PURPLE:
            # Return a random tarot (would need RNG in real implementation)
            return 'The Fool'  # Placeholder
        return None

# Card state tracking for mutable properties
@dataclass
class CardState:
    """Mutable card state (enhancement, edition, seal)"""
    card_id: int  # Index in deck
    enhancement: Enhancement = Enhancement.NONE
    edition: Edition = Edition.NONE
    seal: Seal = Seal.NONE
    
    def calculate_chip_bonus(self, base_chips: int) -> int:
        """Calculate total chip bonus from all modifiers"""
        total = base_chips
        total += EnhancementEffects.get_chip_bonus(self.enhancement, base_chips)
        total += EditionEffects.get_chip_bonus(self.edition)
        return total
    
    def calculate_mult_bonus(self) -> int:
        """Calculate total mult bonus from all modifiers"""
        total = 0
        total += EnhancementEffects.get_mult_bonus(self.enhancement)
        total += EditionEffects.get_mult_bonus(self.edition)
        return total
    
    def calculate_mult_multiplier(self, in_hand: bool = False) -> float:
        """Calculate total mult multiplier from all modifiers"""
        mult = 1.0
        mult *= EnhancementEffects.get_mult_multiplier(self.enhancement, in_hand)
        mult *= EditionEffects.get_mult_multiplier(self.edition)
        return mult

__all__ = [
    "Suit",
    "Rank", 
    "Card",
    "Enhancement",
    "Edition",
    "Seal",
    "EnhancementEffects",
    "EditionEffects",
    "SealEffects",
    "CardState",
]
