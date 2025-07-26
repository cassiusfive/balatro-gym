# Balatro Gym

[![discord](https://img.shields.io/badge/discord-7289da.svg?style=flat-square&logo=discord)](https://amplication.com/discord)
![GitHub License](https://img.shields.io/github/license/cassiusfive/balatro-gym?style=flat-square)


`balatro-gym` provides a [Gymnasium](https://gymnasium.farama.org/) environment for the poker-themed rougelike deck-builder [Balatro](https://www.playbalatro.com/). This project provides a standard interface to train reinforcement learning models for Balatro v1.0.0.

## Install

```bash
git clone https://github.com/cassiusfive/balatro-gym
cd balatro-gym
pip install -e .
```

## MDP - Observations, Actions and Rewards

### Observation

WIP

### Action

WIP

### Reward

WIP

## Usage

```python3
import gymnasium as gym
import balatro_gym

env = gym.make("BalatroGym-v0")
```
