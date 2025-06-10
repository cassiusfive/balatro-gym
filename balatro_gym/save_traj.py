import numpy as np
import pickle
from datetime import datetime
from typing import List, Tuple

# Ensure our patch is applied before we import EightCardDrawEnv
import balatro_gym.patch_balatro_env  # applies the BalatroGame scoring patch

from balatro_gym.env import EightCardDrawEnv
from balatro_gym.actions import decode_discard, decode_select
from balatro_gym.score_with_balatro import int_to_card
from balatro_gym.balatro_game import Card, BalatroGame

# Import the heuristic functions you defined in heuristic_baseline.py
from heuristic_baseline import (
    THRESHOLD_RANK,
    make_discard_action,
    make_select_action,
)

# ──────────────────────────────────────────────────────────────────────── #
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────── #
NUM_ROUNDS = 500_000             # how many rounds (each round = up to 4 hands) to collect
HANDS_PER_ROUND = 4              # define a round as exactly 4 hands
DISCARDS_PER_ROUND = 4           # total discard budget across HANDS_PER_ROUND hands
PASS_THRESHOLD = 300             # raw‐chip score threshold for pass/fail at the end of a round

# By default, every 10th round uses the heuristic policy; others use random:
HEURISTIC_EVERY_N_ROUNDS = 10


# ──────────────────────────────────────────────────────────────────────── #
# HELPER FUNCTION FOR RAW‐CHIP COMPUTATION
# ──────────────────────────────────────────────────────────────────────── #
def compute_raw_chip_value(card_ids: np.ndarray) -> int:
    """
    Given a length‐5 array of card IDs (0..51), convert each to a Card object
    and call BalatroGame._evaluate_hand(...) to get the raw chip integer.
    """
    cards = [int_to_card(int(idx)) for idx in card_ids]
    return BalatroGame._evaluate_hand(cards)


# ──────────────────────────────────────────────────────────────────────── #
# RANDOM‐POLICY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────── #
def random_policy_discard(obs_mask: np.ndarray, remaining_discards: int) -> Tuple[int, int]:
    """
    Given the current Phase 0 `obs_mask` (shape (312,)) and how many
    discards remain, pick a truly random legal discard action that
    does not exceed `remaining_discards`. Returns (action, num_discarded).
    """
    valid_actions = np.flatnonzero(obs_mask == 1)
    filtered = []
    for a in valid_actions:
        discard_indices = decode_discard(int(a))
        if len(discard_indices) <= remaining_discards:
            filtered.append(a)

    # If nothing fits the budget, force pick “discard 0 cards”
    if not filtered:
        for a in valid_actions:
            if len(decode_discard(int(a))) == 0:
                filtered = [a]
                break

    choice = int(np.random.choice(filtered))
    num_discarded = len(decode_discard(choice))
    return choice, num_discarded


def random_policy_select(obs_mask: np.ndarray) -> int:
    """
    Given the current Phase 1 `obs_mask` (shape (312,)), pick a random
    valid select action [256..311] uniformly.
    """
    valid_actions = np.flatnonzero(obs_mask == 1)
    return int(np.random.choice(valid_actions))


# ──────────────────────────────────────────────────────────────────────── #
# MAIN TRAJECTORY‐COLLECTION FUNCTION
# ──────────────────────────────────────────────────────────────────────── #
def collect_trajectories(num_rounds: int,
                         hands_per_round: int,
                         discard_budget: int) -> List[List[List[dict]]]:
    """
    Run `num_rounds` rounds, each consisting of `hands_per_round` independent hands.
    Some rounds (every HEURISTIC_EVERY_N_ROUNDS‐th) will use the heuristic policy
    (from heuristic_baseline.py), otherwise default to random‐policy. Each hand has:
      - Phase 0: discard (budgeted by remaining_discards)
      - Phase 1: select‐five (no discard cost)

    We re‐instantiate a fresh EightCardDrawEnv() at the start of each hand so that
    the deck is reset. We also enforce that across the entire round you may discard
    at most `discard_budget` cards total.

    Returns:
        A nested list of shape [num_rounds][hands_per_round][2]:
          Each innermost element is a transition dict with keys:
            "hand_before":         np.ndarray(8,)   (card IDs 0..51)
            "phase":               int (0 or 1)
            "action":              int (0..311)
            "reward":              float (normalized [0,1])
            "hand_after":          np.ndarray(8,) or None
            "keep_indices":        tuple of int (length=5) or None
            "balatro_raw_score":   int or None (only in Phase 1)
            "num_discards":        int (cards discarded in Phase 0) or 0
            "done":                bool
            "round_score_so_far":  int (only in Phase 1)
            "round_pass":          bool (only on final hand’s Phase 1)
    """
    all_rounds: List[List[List[dict]]] = []

    for rnd in range(num_rounds):
        # Decide policy for this round:
        use_heuristic = (rnd % HEURISTIC_EVERY_N_ROUNDS == 0)

        remaining_discards = discard_budget
        round_raw_score = 0
        round_trajectories: List[List[dict]] = []

        for hand_idx in range(hands_per_round):
            # Re‐instantiate a fresh env for each hand (deck reset)
            env = EightCardDrawEnv()
            obs, _ = env.reset()
            hand_transitions: List[dict] = []
            done = False

            while not done:
                phase = int(obs["phase"])
                mask = obs["action_mask"]  # shape (312,)

                if phase == 0:
                    # ── Phase 0: Discard step ──
                    if use_heuristic:
                        # Use the heuristic_baseline’s discard action function:
                        discard_action = make_discard_action(env.hand, THRESHOLD_RANK)
                        discard_positions = decode_discard(discard_action)
                        if len(discard_positions) > remaining_discards:
                            # Fallback to “discard 0 cards”
                            discard_action = 0
                            num_discarded = 0
                        else:
                            num_discarded = len(discard_positions)
                        action = discard_action
                    else:
                        # Random policy: pick a random legal discard ≤ remaining_discards
                        action, num_discarded = random_policy_discard(mask, remaining_discards)

                    remaining_discards -= num_discarded

                else:
                    # ── Phase 1: Select‐five step ──
                    if use_heuristic:
                        # Use the heuristic_baseline’s select action function:
                        action = make_select_action(env.hand)
                    else:
                        action = random_policy_select(mask)
                    num_discarded = 0

                # ── Step the environment ─────────────────────────────────────────────
                hand_before = env.hand.copy()  # np.ndarray(8,)
                next_obs, reward, done, truncated, info = env.step(action)
                hand_after = env.hand.copy() if (phase == 0) else None

                keep_indices = None
                balatro_raw_score = None
                if phase == 1:
                    keep_indices = decode_select(action)
                    kept_ids = env.hand[list(keep_indices)]
                    balatro_raw_score = compute_raw_chip_value(kept_ids)
                    round_raw_score += balatro_raw_score

                transition = {
                    "hand_before":       hand_before,        # np.ndarray(8,)
                    "phase":             phase,              # 0 or 1
                    "action":            action,             # int 0..311
                    "reward":            reward,             # float normalized [0,1]
                    "hand_after":        hand_after,         # np.ndarray(8,) or None
                    "keep_indices":      keep_indices,       # tuple length=5 or None
                    "balatro_raw_score": balatro_raw_score,  # int or None
                    "num_discards":      num_discarded,      # how many cards dropped
                    "done":              done                # bool
                }

                hand_transitions.append(transition)
                obs = next_obs

            # ── End of one hand (Phases 0 & 1) ──
            # Annotate the Phase 1 transition with running total and pass/fail
            phase1_transition = hand_transitions[1]
            phase1_transition["round_score_so_far"] = round_raw_score
            if hand_idx == hands_per_round - 1:
                phase1_transition["round_pass"] = (round_raw_score > PASS_THRESHOLD)

            round_trajectories.append(hand_transitions)

        # ── End of this round (HANDS_PER_ROUND hands) ──
        all_rounds.append(round_trajectories)

        # (Optional) Progress output
        if (rnd + 1) % 50_000 == 0:
            print(f"Collected {rnd + 1}/{num_rounds} rounds...")

    return all_rounds


# ──────────────────────────────────────────────────────────────────────── #
# MAIN & SAVE
# ──────────────────────────────────────────────────────────────────────── #
def main():
    trajectories = collect_trajectories(NUM_ROUNDS, HANDS_PER_ROUND, DISCARDS_PER_ROUND)
    print(f"Collected {len(trajectories)} rounds; each round has {HANDS_PER_ROUND} hands.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pickles/rounds_{timestamp}.pkl"

    with open(filename, "wb") as f:
        pickle.dump(trajectories, f)
    print(f"Saved rounds to {filename}")


if __name__ == "__main__":
    main()
