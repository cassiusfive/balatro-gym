# balatro_gym/actions.py
from itertools import combinations
from typing import Tuple, List

# ---------- 1. Discard actions (bitmask 0–255) ----------
NUM_DISCARD_ACTIONS = 256     # 2^8
DISCARD_OFFSET      = 0       # IDs 0–255

# ---------- 2. Select-five actions (56 combos) ----------
FIVE_CARD_COMBOS: List[Tuple[int, ...]] = list(combinations(range(8), 5))  # len == 56
NUM_SELECT_ACTIONS = len(FIVE_CARD_COMBOS)     # 56
SELECT_OFFSET      = NUM_DISCARD_ACTIONS       # 256
ACTION_SPACE_SIZE  = NUM_DISCARD_ACTIONS + NUM_SELECT_ACTIONS  # 312

def encode_discard(mask: int) -> int:
    """mask is an int 0-255, already the action id"""
    return mask  # convenience alias

def decode_discard(action_id: int) -> List[int]:
    """return list of card indices (0-7) to throw away"""
    return [i for i in range(8) if (action_id >> i) & 1]

def encode_select(indices: Tuple[int, ...]) -> int:
    """indices is a 5-tuple of sorted ints 0-7"""
    return SELECT_OFFSET + FIVE_CARD_COMBOS.index(tuple(indices))

def decode_select(action_id: int) -> Tuple[int, ...]:
    """return the 5 indices kept (for scoring)"""
    return FIVE_CARD_COMBOS[action_id - SELECT_OFFSET]

