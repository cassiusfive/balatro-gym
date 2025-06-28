#!/usr/bin/env python3
"""Robust training script with error handling and single-process fallback"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.utils import set_random_seed

# Import the fixed environment and safe wrapper from train_balatro_fixed
from train_balatro_fixed import BalatroEnvFixed as BalatroEnvSB3, SafeBalatroEnv


def make_env(seed=0, rank=0):
    """Create a safe environment"""
    def _init():
        set_random_seed(seed + rank)
        try:
            # Create base environment
            env = BalatroEnvSB3(seed=seed + rank)
            # Add safety wrapper
            env = SafeBalatroEnv(env, max_invalid_actions=50, max_episode_steps=1000)
            # Add monitoring
            env = Monitor(env)
            return env
        except Exception as e:
            print(f"[ERROR] Failed to create environment: {e}")
            raise
    return _init


class RobustProgressCallback(BaseCallback):
    """Progress callback with error handling"""
    
    def __init__(self, check_freq=10000):
        super().__init__()
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.errors = 0
        self.invalid_terminations = 0
        
    def _on_step(self) -> bool:
        try:
            # Collect episode info
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    
                    if info.get("invalid_action_termination", False):
                        self.invalid_terminations += 1
                    if "error" in info:
                        self.errors += 1
            
            # Print progress
            if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]
                print(f"\n[Step {self.num_timesteps:,}]")
                print(f"  Episodes: {len(self.episode_rewards)}")
                print(f"  Avg Reward: {np.mean(recent_rewards):.2f} Â± {np.std(recent_rewards):.2f}")
                print(f"  Avg Length: {np.mean(self.episode_lengths[-100:]):.0f}")
                if self.invalid_terminations > 0:
                    print(f"  Invalid terminations: {self.invalid_terminations} ({self.invalid_terminations/len(self.episode_rewards)*100:.1f}%)")
                if self.errors > 0:
                    print(f"  Errors: {self.errors}")
                    
        except Exception as e:
            print(f"[ERROR] Callback error: {e}")
            
        return True


def train_robust(args):
    """Robust training with fallback options"""
    
    # Setup directories
    run_dir = Path(f"run_{args.run_name}")
    run_dir.mkdir(exist_ok=True)
    model_dir = run_dir / "models"
    model_dir.mkdir(exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    
    print("ðŸŽ° BALATRO RL - ROBUST TRAINING")
    print("=" * 60)
    print(f"ðŸ“Š Timesteps: {args.timesteps:,}")
    print(f"ðŸ–¥ï¸  Environments: {args.n_envs}")
    print(f"ðŸ’¾ Checkpoint every: {args.checkpoint_freq:,} steps")
    print("=" * 60)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() and args.use_gpu else "cpu"
    print(f"ðŸŽ® Using device: {device}")
    
    # Try to create environments
    print("\nðŸ“¦ Creating environments...")
    env = None
    
    # First try SubprocVecEnv if multiple envs requested
    if args.n_envs > 1 and not args.force_dummy:
        try:
            print(f"Attempting SubprocVecEnv with {args.n_envs} processes...")
            env = SubprocVecEnv([make_env(args.seed, i) for i in range(args.n_envs)])
            print("âœ… SubprocVecEnv created successfully")
        except Exception as e:
            print(f"âš ï¸  SubprocVecEnv failed: {e}")
            print("Falling back to DummyVecEnv...")
    
    # Fallback to DummyVecEnv
    if env is None:
        try:
            env = DummyVecEnv([make_env(args.seed, i) for i in range(args.n_envs)])
            print(f"âœ… DummyVecEnv created with {args.n_envs} environments")
        except Exception as e:
            print(f"âŒ Failed to create environments: {e}")
            return
    
    # Create evaluation environment (always use DummyVecEnv for stability)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv([make_env(args.seed + 1000, 0)])
    
    # Create model
    print("\nðŸ§  Initializing PPO model...")
    try:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=min(2048, 128 * args.n_envs),  # Scale with environments
            batch_size=args.batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(log_dir / "tb_logs"),
            device=device,
            policy_kwargs={
                "net_arch": dict(pi=[512, 512], vf=[512, 512]),
                "activation_fn": torch.nn.ReLU,
                "ortho_init": True,
            }
        )
        print("âœ… Model created successfully")
    except Exception as e:
        print(f"âŒ Failed to create model: {e}")
        env.close()
        eval_env.close()
        return
    
    # Setup callbacks
    callbacks = []
    
    # Progress monitoring
    progress_callback = RobustProgressCallback(check_freq=10000)
    callbacks.append(progress_callback)
    
    # Checkpoints
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="balatro",
        verbose=0
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation (optional)
    if not args.skip_eval:
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=str(model_dir / "best_model"),
            log_path=str(log_dir / "eval"),
            eval_freq=max(args.eval_freq // args.n_envs, 1),
            n_eval_episodes=3,
            deterministic=True,
            verbose=0
        )
        callbacks.append(eval_callback)
    
    # Training
    print(f"\nðŸš€ Starting training...")
    print(f"ðŸ“ˆ Monitor with: tensorboard --logdir {log_dir}/tb_logs\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True,
            reset_num_timesteps=True
        )
        print("\nâœ… Training completed!")
        
    except KeyboardInterrupt:
        print("\nâš ï¸  Training interrupted by user")
    except Exception as e:
        print(f"\nâŒ Training error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always save the model
        try:
            final_path = model_dir / "final_model"
            model.save(str(final_path))
            print(f"\nðŸ’¾ Model saved to {final_path}")
        except:
            print("âš ï¸  Failed to save model")
        
        # Print statistics
        if hasattr(progress_callback, 'episode_rewards') and len(progress_callback.episode_rewards) > 0:
            print(f"\nðŸ“Š Training Statistics:")
            print(f"  Total episodes: {len(progress_callback.episode_rewards)}")
            print(f"  Average reward: {np.mean(progress_callback.episode_rewards):.2f}")
            print(f"  Best reward: {max(progress_callback.episode_rewards):.2f}")
            print(f"  Worst reward: {min(progress_callback.episode_rewards):.2f}")
        
        # Clean up
        try:
            env.close()
            eval_env.close()
        except:
            pass
        
        print("\nâœ¨ Done!")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Robust Balatro Training")
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    parser.add_argument("--run-name", type=str, default="balatro_robust")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation during training")
    parser.add_argument("--force-dummy", action="store_true", help="Force DummyVecEnv instead of SubprocVecEnv")
    parser.add_argument("--use-gpu", action="store_true", help="Force GPU usage")
    
    args = parser.parse_args()
    train_robust(args)


if __name__ == "__main__":
    main()
