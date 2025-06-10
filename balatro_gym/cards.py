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
from typing import Final


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


@dataclass(frozen=True, slots=True)
class Card:
    """Immutable playing card (≈32 B per instance)"""

    rank: Rank
    suit: Suit

    def __str__(self) -> str:
        return f"{self.rank.short}{self.suit.symbol()}"

    # Handy helpers so you can sort()/hash() cards effortlessly
    def __int__(self) -> int:  # unique 0–51 mapping → (rank‑2) * 4 + suit
        return (self.rank - 2) * 4 + self.suit

    def __lt__(self, other: "Card") -> bool:  # sort by rank then suit
        return (self.rank, self.suit) < (other.rank, other.suit)


__all__ = [
    "Suit",
    "Rank",
    "Card",
]

