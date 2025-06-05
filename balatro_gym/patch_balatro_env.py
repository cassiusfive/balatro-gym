# patch_balatro_env.py

import numpy as np
from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import decode_discard, decode_select
from balatro_gym.score_with_balatro import score_five_balatro

# Keep a reference to the original step() method
_orig_step = EightCardDrawEnv.step

def _patched_step(self, action: int):
    """
    Exactly the same as the original EightCardDrawEnv.step(), except:
    - In Phase 1, we call score_five_balatro(...) instead of the default score_five(...)
    """
    if self._terminated:
        raise RuntimeError("`step()` called on terminated episode")

    reward = 0.0
    info = {}

    # Phase 0: Discard
    if self.phase == 0:
        discards = decode_discard(action)
        n_draw = len(discards)
        if n_draw:
            draw = self.deck[8 : 8 + n_draw]
            self.hand[discards] = draw
        self.phase = 1
        terminated = False

    # Phase 1: Select‐Five
    else:
        keep_idx = decode_select(action)
        keep_cards = self.hand[list(keep_idx)]  # array of 5 ints
        # Replace Treys/hash scoring with BalatroGame scoring:
        reward = score_five_balatro(keep_cards)
        terminated = True
        self._terminated = True

    # Build next observation (same as original)
    obs = {
        "cards": self._encode_cards(),
        "phase": np.array(self.phase, dtype=np.int8),
        "action_mask": self._action_mask(),
    }
    return obs, reward, terminated, False, info

# Apply the monkey‐patch
EightCardDrawEnv.step = _patched_step

