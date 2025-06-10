"""run_trajectories.py – quick‑start script to gather Balatro trajectories
===========================================================================
Collects a user‑configurable number of *rounds* with `BalatroEnvWithExpert`
(using the current hybrid ScoreEngine, Planet hooks, and two dynamic jokers).

For simplicity:
* PLAY phase: choose `action=0` (play hand) 80 % of the time, otherwise
  `action=1` (discard all) – tweak as needed.
* SHOP phase: buy the first affordable **Planet Pack** or dynamic Joker
  (RideTheBus / GreenJoker); otherwise skip.

Results are saved as a pickle list of per‑round dictionaries:
    [ {"transitions": [...], "planets": [...], "chips_final": int}, ... ]

Usage
-----
```bash
python -m balatro_gym.run_trajectories  --rounds 2000  --outfile data.pkl
```
"""
from __future__ import annotations

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np

from balatro_gym.envs.balatro_env_v2 import BalatroEnvWithExpert, PLAY_MAX_ID, SHOP_MIN_ID, JOKER_FACTORY
from balatro_gym.shop import ShopAction, ItemType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def choose_play_action(obs) -> int:
    """Simple stochastic policy: 80 % play, 20 % discard."""
    if np.random.rand() < 0.8:
        return 0  # play first five
    return 1      # discard all


def choose_shop_action(obs) -> int:
    """Buy first affordable Planet Pack or supported Joker; else skip."""
    chips = int(obs["chips"])
    types = obs["shop_item_type"]
    costs = obs["shop_cost"]
    payloads = obs["shop_payload"]

    for idx, (itype, cost, payload) in enumerate(zip(types, costs, payloads)):
        if cost > chips:
            continue
        if itype == ItemType.PACK and payload.get("pack_type") == "Planet Pack":
            return ShopAction.BUY_PACK_BASE + idx
        if itype == ItemType.JOKER and payload.get("joker_id") in JOKER_FACTORY:
            return ShopAction.BUY_JOKER_BASE + idx
    return int(ShopAction.SKIP)


def collect(rounds: int, seed: int | None = None) -> List[Dict]:
    if seed is not None:
        np.random.seed(seed)
    env = BalatroEnvWithExpert()
    out: List[Dict] = []

    for r in range(rounds):
        obs, _ = env.reset()
        round_dict = {"transitions": [], "planets": [], "chips_final": 0}
        done_round = False
        while not done_round:
            phase = int(obs["phase"])
            if phase == 0:
                action = choose_play_action(obs)
            else:
                action = choose_shop_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            round_dict["transitions"].append({
                "phase": phase,
                "action": action,
                "reward": reward,
                "chips": int(obs["chips"]),
            })
            if "planet" in info:
                round_dict["planets"].append(info["planet"].name)

            # detect round end: SHOP → PLAY transition
            done_round = (phase == 1 and next_obs["phase"] == 0)
            obs = next_obs
        round_dict["chips_final"] = int(obs["chips"])
        out.append(round_dict)
        if (r+1) % 100 == 0:
            print(f"Collected {r+1}/{rounds} rounds")
    env.close()
    return out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=1000)
    parser.add_argument("--outfile", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    traj = collect(args.rounds, seed=args.seed)

    outfile: Path
    if args.outfile is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfile = Path(f"trajectories_{ts}.pkl")
    else:
        outfile = args.outfile

    with outfile.open("wb") as f:
        pickle.dump(traj, f)
    print("Saved", outfile)
