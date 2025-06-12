from gymnasium.envs.registration import register
from .env import EightCardDrawEnv

register(
    id="balatro-gym/Balatro-v1",
    entry_point="balatro_gym.envs.balatro_env:BalatroEnv",
)

def make(id: str):
    if id == "EightCardDraw-v0":
        return EightCardDrawEnv()
    raise ValueError(f"Unknown id {id}")
