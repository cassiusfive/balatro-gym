from .env import EightCardDrawEnv

def make(id: str):
    if id == "EightCardDraw-v0":
        return EightCardDrawEnv()
    raise ValueError(f"Unknown id {id}")

