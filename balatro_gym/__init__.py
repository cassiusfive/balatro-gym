from gymnasium.envs.registration import register

register(
    id="Balatro-v0",
    entry_point="balatro_gym:BalatroEnv"
)