# Balatro Gym

`balatro-gym` provides a [Gymnasium](https://gymnasium.farama.org/) environment for the poker-themed rougelike deck-builder [Balatro](https://www.playbalatro.com/). This project provides a standard interface to train reinforcement learning models for Balatro v1.0.0.

## Action Space

| Value | Meaning         |
| ----- | --------------- |
| 0     | Play hand       |
| 1     | Discard hand    |
| 2-9   | Select card     |
| WIP   | Use consumable  |
| WIP   | Sell consumable |
| WIP   | Reorder jokers  |
| WIP   | Sell joker      |
| WIP   | Select blind    |
| WIP   | Skip blind      |
| WIP   | Buy pack        |
| WIP   | Buy card        |
| WIP   | Buy voucher     |
| WIP   | Reroll shop     |
