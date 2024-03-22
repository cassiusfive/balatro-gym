from balatro_gym.balatro_env import BalatroEnv # noqa
from balatro_gym.balatro_small_env import BalatroSmallEnv # noqa
from gymnasium.envs.registration import register

register(
    id="Balatro-v0",
    entry_point="balatro_gym:BalatroEnv",
)

register(
    id="BalatroSmall-v0",
    entry_point="balatro_gym:BalatroSmallEnv",
)