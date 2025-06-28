#!/usr/bin/env python3
"""Training script with heavy progression rewards to fix conservative play"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import gymnasium as gym
import torch
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback

from train_balatro_fixed import BalatroEnvFixed
from robust_training import SafeBalatroEnv


class ProgressionRewardWrapper(gym.Wrapper):
    """Wrapper that heavily rewards ante progression and penalizes conservative play"""
    
    def __init__(self, env):
        super().__init__(env)
        self.ante_history = []
        self.max_ante_reached = 1
        self.steps_on_current_ante = 0
        self.total_steps = 0
        self.last_ante = 1
        self.last_round = 1
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.ante_history = [1]
        self.max_ante_reached = 1
        self.steps_on_current_ante = 0
        self.total_steps = 0
        self.last_ante = 1
        self.last_round = 1
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1
        
        # Get current game state
        current_ante = 1
        current_round = 1
        if hasattr(self.env, 'env') and hasattr(self.env.env, 'state'):
            state = self.env.env.state
            current_ante = state.ante
            current_round = state.round
        elif hasattr(self.env, 'state'):
            state = self.env.state
            current_ante = state.ante
            current_round = state.round
        
        # Track ante progression
        if current_ante == self.last_ante:
            self.steps_on_current_ante += 1
        else:
            self.steps_on_current_ante = 0
        
        # HEAVY PROGRESSION REWARDS
        if current_ante > self.last_ante:
            # Massive bonus for reaching new antes
            progression_bonus = 200 * (current_ante - self.last_ante)
            reward += progression_bonus
            info['ante_progression_bonus'] = progression_bonus
            print(f"\nðŸŽ‰ ANTE {self.last_ante} â†’ {current_ante}! Bonus: +{progression_bonus}")
            self.last_ante = current_ante
            
        # Round progression bonus (smaller)
        if current_round > self.last_round and current_ante == self.last_ante:
            round_bonus = 20
            reward += round_bonus
            self.last_round = current_round
            
        # Update max ante
        if current_ante > self.max_ante_reached:
            self.max_ante_reached = current_ante
            # Extra bonus for personal best
            best_bonus = 100
            reward += best_bonus
            info['new_best_ante'] = current_ante
        
        # PENALTIES FOR CONSERVATIVE PLAY
        
        # Penalty for staying on Ante 1 too long
        if current_ante == 1 and self.steps_on_current_ante > 150:
            penalty = -0.5 * (self.steps_on_current_ante - 150)
            reward += penalty
            
            # Force termination if stuck too long
            if self.steps_on_current_ante > 300:
                terminated = True
                reward -= 50  # Big penalty
                info['stuck_on_ante_1'] = True
                print(f"\nâŒ Terminated for being stuck on Ante 1 for {self.steps_on_current_ante} steps!")
        
        # General penalty for not progressing
        if self.total_steps > 200 and current_ante < 2:
            reward -= 0.2
        elif self.total_steps > 400 and current_ante < 3:
            reward -= 0.3
        elif self.total_steps > 600 and current_ante < 4:
            reward -= 0.4
        
        # Bonus for efficient progression
        if current_ante >= 3 and self.total_steps < 300:
            efficiency_bonus = 50
            reward += efficiency_bonus
            info['efficiency_bonus'] = efficiency_bonus
        
        # Update round tracking
        if current_ante != self.last_ante:
            self.last_round = current_round
        
        return obs, reward, terminated, truncated, info


def make_progression_env(seed=0, rank=0):
    """Create environment with progression rewards"""
    def _init():
        env = BalatroEnvFixed(seed=seed + rank)
        env = ProgressionRewardWrapper(env)
        env = SafeBalatroEnv(env, max_invalid_actions=50, max_episode_steps=1000)  # Shorter episodes
        env = Monitor(env)
        return env
    return _init


def train_with_progression(args):
    """Train with progression-focused rewards"""
    
    print("ðŸš€ BALATRO TRAINING - PROGRESSION FOCUSED")
    print("=" * 60)
    print("Rewards:")
    print("  - Ante progression: +200 per ante")
    print("  - New best ante: +100 bonus")
    print("  - Stuck on Ante 1 > 150 steps: increasing penalty")
    print("  - Stuck on Ante 1 > 300 steps: termination with -50 penalty")
    print("=" * 60)
    
    # Create environments
    print(f"\nðŸ“¦ Creating {args.n_envs} progression-focused environments...")
    env = SubprocVecEnv([make_progression_env(args.seed, i) for i in range(args.n_envs)])
    
    run_dir = Path(f"run_{args.run_name}")
    run_dir.mkdir(exist_ok=True)
    
    # Load the conservative model as starting point
    print(f"\nðŸ“‚ Loading conservative model from {args.load_path}")
    
    # First load the model to check its architecture
    conservative_model = PPO.load(args.load_path, device="cuda")
    
    # Create new model with updated hyperparameters and matching architecture
    model = PPO(
        policy="MultiInputPolicy",  # Changed to MultiInputPolicy for dict obs
        env=env,
        learning_rate=args.learning_rate,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.3,  # Larger clip range for bigger policy changes
        clip_range_vf=None,
        ent_coef=0.02,   # More exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=str(run_dir / "tb_logs"),
        policy_kwargs={
            "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),  # Match conservative model architecture exactly
            "activation_fn": torch.nn.ReLU
        },
        device="cuda",
        verbose=1
    )
    
    # Load the weights from the conservative model
    print("\nðŸ“¥ Loading weights from conservative model...")
    model.policy.load_state_dict(conservative_model.policy.state_dict())
    
    # Progress tracking callback
    class ProgressionCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.ante_reaches = []
            self.episode_count = 0
            
        def _on_step(self):
            for info in self.locals['infos']:
                if 'episode' in info:
                    self.episode_count += 1
                    
                if 'new_best_ante' in info:
                    ante = info['new_best_ante']
                    self.ante_reaches.append(ante)
                    print(f"\nðŸ† Episode {self.episode_count} reached Ante {ante}!")
                    
                if 'stuck_on_ante_1' in info:
                    print(f"\nâš ï¸ Episode {self.episode_count} stuck on Ante 1")
            
            if self.num_timesteps % 100000 == 0 and self.ante_reaches:
                avg_ante = np.mean(self.ante_reaches[-100:]) if len(self.ante_reaches) >= 100 else np.mean(self.ante_reaches)
                max_ante = max(self.ante_reaches)
                print(f"\n[Step {self.num_timesteps:,}] "
                      f"Avg ante (last 100): {avg_ante:.2f}, "
                      f"Max ante ever: {max_ante}")
            
            return True
    
    # Callbacks
    callbacks = [
        ProgressionCallback(),
        CheckpointCallback(
            save_freq=max(100000 // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="balatro_progression"
        )
    ]
    
    print(f"\nðŸŽ® Starting progression-focused training...")
    print(f"ðŸ“ˆ Monitor with: tensorboard --logdir {run_dir}/tb_logs\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False,  # Continue from 50M
            tb_log_name="PPO_progression"
        )
        print("\nâœ… Training completed!")
        
    except KeyboardInterrupt:
        print("\nâš ï¸ Training interrupted")
    finally:
        model.save(str(run_dir / "final_model"))
        env.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                       help="Additional timesteps to train")
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-4,
                       help="Higher LR for faster adaptation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="balatro_progression")
    parser.add_argument("--load-path", type=str, 
                       default="run_balatro_a40_fixed/final_model.zip",
                       help="Path to conservative model")
    
    args = parser.parse_args()
    train_with_progression(args)


if __name__ == "__main__":
    main()
