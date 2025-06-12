#!/usr/bin/env python3
"""Simple HPC training script for Balatro RL - Single GPU version"""

import os
import argparse
import time
from pathlib import Path
from datetime import datetime

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

# Import your Balatro environment
import gymnasium
import balatro_gym

def make_env(rank: int, seed: int = 0):
    """Create a single environment instance"""
    def _init():
        env = gymnasium.make("balatro-gym/Balatro-v1", seed=seed + rank)
        env = Monitor(env)
        return env
    return _init


def create_vec_normalize_with_auto_keys(env, norm_reward=True, clip_obs=10.0):
    """
    Create VecNormalize that automatically detects Box observation spaces
    and only normalizes those keys.
    """
    obs_space = env.observation_space

    print(f"Observation space: {obs_space}")

    # Check if observation space is a Dict/MultiDict with multiple components
    if hasattr(obs_space, 'spaces') and hasattr(obs_space.spaces, 'items'):
        # Find all Box observation keys
        box_keys = []
        for key, space in obs_space.spaces.items():
            if isinstance(space, gymnasium.spaces.Box):
                box_keys.append(key)
                print(f"  Found Box space to normalize: '{key}' {space}")
            else:
                print(f"  Skipping non-Box space: '{key}' {space}")

        if box_keys:
            print(f"Auto-detected observation keys for normalization: {box_keys}")
            return VecNormalize(
                env,
                norm_obs=True,
                norm_reward=norm_reward,
                clip_obs=clip_obs,
                norm_obs_keys=box_keys
            )
        else:
            print("No Box observation spaces found - disabling observation normalization")
            return VecNormalize(
                env,
                norm_obs=False,
                norm_reward=norm_reward,
                clip_obs=clip_obs
            )

    # Single observation space (not Dict)
    elif isinstance(obs_space, gymnasium.spaces.Box):
        print("Single Box observation space - using default normalization")
        return VecNormalize(
            env,
            norm_obs=True,
            norm_reward=norm_reward,
            clip_obs=clip_obs
        )

    else:
        print(f"Non-Box observation space ({type(obs_space).__name__}) - disabling observation normalization")
        return VecNormalize(
            env,
            norm_obs=False,
            norm_reward=norm_reward,
            clip_obs=clip_obs
        )


def train_balatro_hpc(args):
    """Main training function for HPC"""

    # Setup directories
    job_id = os.environ.get('SLURM_JOB_ID', 'local')
    run_dir = Path(f"run_{job_id}")
    run_dir.mkdir(exist_ok=True)

    model_dir = run_dir / "models"
    model_dir.mkdir(exist_ok=True)

    log_dir = run_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    print(f"Starting Balatro RL Training")
    print(f"Job ID: {job_id}")
    print(f"Run directory: {run_dir}")

    # Check GPU availability
    if torch.cuda.is_available():
        print(f"GPU available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("No GPU available, using CPU")
        device = 'cpu'

    # Create environments
    print(f"\nCreating {args.n_envs} parallel environments...")

    if args.n_envs > 1:
        # Use SubprocVecEnv for true parallelism
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        # Use DummyVecEnv for single environment
        env = DummyVecEnv([make_env(0, args.seed)])

    # Add normalization wrapper with auto-detection
    print("\nSetting up training environment normalization:")
    env = create_vec_normalize_with_auto_keys(env, norm_reward=True, clip_obs=10.0)

    # Create evaluation environment
    eval_env = DummyVecEnv([make_env(1000, args.seed)])
    print("\nSetting up evaluation environment normalization:")
    eval_env = create_vec_normalize_with_auto_keys(eval_env, norm_reward=False, clip_obs=10.0)

    # Create PPO model
    print("\nInitializing PPO model...")
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=2048,
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
            "net_arch": [dict(pi=[512, 512], vf=[512, 512])]
        }
    )

    # Setup callbacks
    callbacks = []

    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.n_envs, 1),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="balatro_ppo",
        save_vecnormalize=True
    )
    callbacks.append(checkpoint_callback)

    # Evaluation callback - evaluate and save best model
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        eval_freq=max(args.eval_freq // args.n_envs, 1),
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )
    callbacks.append(eval_callback)

    # Start training
    print(f"\nStarting training for {args.timesteps:,} timesteps...")
    print(f"This will generate approximately {args.timesteps // args.n_envs:,} episodes")

    start_time = time.time()

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
            progress_bar=True,
            reset_num_timesteps=False
        )

        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/3600:.2f} hours")
        print(f"Throughput: {args.timesteps/training_time:.2f} steps/second")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        training_time = time.time() - start_time

    except Exception as e:
        print(f"\nError during training: {e}")
        raise

    finally:
        # Save final model
        print("\nSaving final model...")
        model.save(str(model_dir / "final_model"))
        env.save(str(model_dir / "vec_normalize.pkl"))

        # Save training summary
        summary = {
            "job_id": job_id,
            "total_timesteps": args.timesteps,
            "n_envs": args.n_envs,
            "training_time_hours": training_time / 3600,
            "throughput_steps_per_sec": args.timesteps / training_time if training_time > 0 else 0,
            "device": device,
            "completed": datetime.now().isoformat()
        }

        import json
        with open(model_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print(f"\nAll models saved to: {model_dir}")
        print("Training complete!")

        # Close environments
        env.close()
        eval_env.close()


def main():
    parser = argparse.ArgumentParser(description="Train Balatro RL agent on HPC")
    parser.add_argument("--timesteps", type=int, default=10_000_000,
                       help="Total timesteps to train")
    parser.add_argument("--n-envs", type=int, default=16,
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                       help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=256,
                       help="Batch size for PPO")
    parser.add_argument("--checkpoint-freq", type=int, default=100_000,
                       help="Save checkpoint every N steps")
    parser.add_argument("--eval-freq", type=int, default=50_000,
                       help="Evaluate every N steps")

    args = parser.parse_args()

    # Print configuration
    print("Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")

    train_balatro_hpc(args)


if __name__ == "__main__":
    main()
