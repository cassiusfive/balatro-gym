#!/usr/bin/env python3
"""Complete training script with comprehensive observation space fixes"""

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

from balatro_gym.envs.balatro_env_2 import BalatroEnv as OriginalBalatroEnv


class BalatroEnvFixed(gym.Env):
    """Fixed environment that handles all observation space issues"""
    
    def __init__(self, seed=None):
        self.env = OriginalBalatroEnv(seed=seed)
        self.action_space = self.env.action_space
        
        # Track observation space transformations
        self.space_transforms = {}
        
        # Fix observation space
        spaces_dict = {}
        for key, space in self.env.observation_space.spaces.items():
            try:
                if isinstance(space, gym.spaces.Box):
                    # Ensure all Box spaces have at least 1 dimension
                    if len(space.shape) == 0 or space.shape == ():
                        # Convert scalar to 1D array
                        new_shape = (1,)
                        
                        # Handle low/high bounds
                        if hasattr(space.low, 'shape') and len(space.low.shape) > 0:
                            low_val = np.array([space.low.item()])
                            high_val = np.array([space.high.item()])
                        else:
                            low_val = np.array([float(space.low)])
                            high_val = np.array([float(space.high)])
                        
                        spaces_dict[key] = gym.spaces.Box(
                            low=low_val,
                            high=high_val,
                            shape=new_shape,
                            dtype=space.dtype
                        )
                        self.space_transforms[key] = 'scalar_to_array'
                    
                    # Fix int16 overflow for large value fields
                    elif space.dtype in [np.int16, np.int8] and key in [
                        'chips_scored', 'round_chips_scored', 'chips_needed', 
                        'shop_costs', 'shop_items', 'joker_ids', 'shop_rerolls',
                        'hand_potential_scores', 'best_hand_this_ante', 'money',
                        'hands_played', 'ante', 'round_chips_scored_rank'
                    ]:
                        # Upgrade to int32
                        spaces_dict[key] = gym.spaces.Box(
                            low=np.full(space.shape, np.iinfo(np.int32).min, dtype=np.int32),
                            high=np.full(space.shape, np.iinfo(np.int32).max, dtype=np.int32),
                            shape=space.shape,
                            dtype=np.int32
                        )
                        self.space_transforms[key] = 'int_upgrade'
                    
                    else:
                        # Keep as is, but ensure proper shape
                        spaces_dict[key] = space
                        
                elif isinstance(space, gym.spaces.Discrete):
                    # Convert Discrete to Box for stable baselines compatibility
                    spaces_dict[key] = gym.spaces.Box(
                        low=0,
                        high=space.n - 1,
                        shape=(1,),
                        dtype=np.int32
                    )
                    self.space_transforms[key] = 'discrete_to_box'
                    
                elif isinstance(space, gym.spaces.MultiBinary):
                    # Ensure MultiBinary has proper shape
                    if hasattr(space, 'n'):
                        shape = (space.n,)
                    else:
                        shape = space.shape
                    spaces_dict[key] = gym.spaces.Box(
                        low=0,
                        high=1,
                        shape=shape,
                        dtype=np.int8
                    )
                    self.space_transforms[key] = 'multibinary_to_box'
                    
                else:
                    # Unknown space type - convert to Box
                    print(f"Warning: Unknown space type for {key}: {type(space)}")
                    spaces_dict[key] = gym.spaces.Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(1,),
                        dtype=np.float32
                    )
                    self.space_transforms[key] = 'unknown_to_box'
                    
            except Exception as e:
                print(f"Error processing space '{key}': {e}")
                # Fallback: create a simple Box space
                spaces_dict[key] = gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32
                )
                self.space_transforms[key] = 'error_fallback'
        
        self.observation_space = gym.spaces.Dict(spaces_dict)
        print(f"Fixed observation space with {len(self.space_transforms)} transformations")
    
    def _fix_observation(self, obs):
        """Transform observation to match fixed observation space"""
        fixed_obs = {}
        
        for key, space in self.observation_space.spaces.items():
            if key not in obs:
                # Missing key - create default value
                if isinstance(space, gym.spaces.Box):
                    fixed_obs[key] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    fixed_obs[key] = space.sample()
                continue
            
            value = obs[key]
            transform = self.space_transforms.get(key, None)
            
            try:
                # Apply transformations based on what we did to the space
                if transform == 'scalar_to_array':
                    # Convert scalar to array
                    if not isinstance(value, np.ndarray):
                        value = np.array([value])
                    elif value.shape == ():
                        value = np.array([value.item()])
                    elif len(value.shape) == 0:
                        value = value.reshape((1,))
                        
                elif transform == 'discrete_to_box':
                    # Convert discrete value to array
                    if not isinstance(value, np.ndarray):
                        value = np.array([int(value)], dtype=np.int32)
                    else:
                        value = value.reshape((1,)).astype(np.int32)
                        
                elif transform == 'multibinary_to_box':
                    # Ensure proper shape for multibinary
                    if not isinstance(value, np.ndarray):
                        value = np.array(value, dtype=np.int8)
                    else:
                        value = value.astype(np.int8)
                    if value.shape != space.shape:
                        value = value.reshape(space.shape)
                        
                elif transform == 'int_upgrade':
                    # Upgrade int type and clip values
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                    value = np.clip(value, np.iinfo(np.int32).min, np.iinfo(np.int32).max)
                    value = value.astype(np.int32)
                    
                else:
                    # No transform or unknown - ensure numpy array with correct dtype
                    if not isinstance(value, np.ndarray):
                        value = np.array(value)
                    if hasattr(space, 'dtype'):
                        value = value.astype(space.dtype)
                
                # Final shape check
                if hasattr(space, 'shape') and value.shape != space.shape:
                    # Try to reshape
                    try:
                        value = value.reshape(space.shape)
                    except:
                        # Create correct shape filled with zeros
                        correct_value = np.zeros(space.shape, dtype=space.dtype)
                        # Copy as much data as possible
                        flat_value = value.flatten()
                        flat_correct = correct_value.flatten()
                        copy_size = min(len(flat_value), len(flat_correct))
                        flat_correct[:copy_size] = flat_value[:copy_size]
                        value = correct_value.reshape(space.shape)
                
                fixed_obs[key] = value
                
            except Exception as e:
                print(f"Warning: Error fixing observation for '{key}': {e}")
                # Fallback to zeros
                if isinstance(space, gym.spaces.Box):
                    fixed_obs[key] = np.zeros(space.shape, dtype=space.dtype)
                else:
                    fixed_obs[key] = space.sample()
        
        return fixed_obs
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._fix_observation(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._fix_observation(obs), reward, terminated, truncated, info
    
    def render(self):
        return self.env.render()
    
    def close(self):
        return self.env.close()


# Alias for compatibility
BalatroEnvSB3 = BalatroEnvFixed


class SafeBalatroEnv(gym.Wrapper):
    """Wrapper that adds safety features to prevent crashes"""
    
    def __init__(self, env, max_invalid_actions=50, max_episode_steps=1000):
        super().__init__(env)
        self.max_invalid_actions = max_invalid_actions
        self.max_episode_steps = max_episode_steps
        self.consecutive_invalid = 0
        self.episode_steps = 0
        self.last_obs = None
        
    def step(self, action):
        try:
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.episode_steps += 1
            
            # Track invalid actions
            if reward == -1.0 and not terminated and not truncated:
                self.consecutive_invalid += 1
                if self.consecutive_invalid >= self.max_invalid_actions:
                    terminated = True
                    reward = -50.0  # Penalty
                    info['invalid_action_termination'] = True
            else:
                self.consecutive_invalid = 0
            
            # Force termination if episode too long
            if self.episode_steps >= self.max_episode_steps:
                truncated = True
                info['max_steps_reached'] = True
            
            self.last_obs = obs
            return obs, reward, terminated, truncated, info
            
        except Exception as e:
            print(f"[ERROR] Environment step failed: {e}")
            # Return a valid observation and terminate
            if self.last_obs is not None:
                obs = self.last_obs
            else:
                obs = self.env.observation_space.sample()
            return obs, -100.0, True, False, {'error': str(e)}
    
    def reset(self, **kwargs):
        try:
            self.consecutive_invalid = 0
            self.episode_steps = 0
            obs, info = self.env.reset(**kwargs)
            self.last_obs = obs
            return obs, info
        except Exception as e:
            print(f"[ERROR] Environment reset failed: {e}")
            # Return a valid observation
            obs = self.env.observation_space.sample()
            return obs, {'error': str(e)}


def make_env_fixed(seed=0, rank=0):
    """Create environment with all fixes"""
    def _init():
        env = BalatroEnvFixed(seed=seed + rank)
        env = SafeBalatroEnv(env, max_invalid_actions=50, max_episode_steps=1000)
        env = Monitor(env)
        return env
    return _init


def train_fixed(args):
    """Main training function with fixes"""
    
    print("ðŸ”§ BALATRO TRAINING - COMPREHENSIVE FIX")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Environments: {args.n_envs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Timesteps: {args.timesteps:,}")
    print("=" * 60)
    
    # Test environment first
    print("\nTesting environment...")
    try:
        test_env = BalatroEnvFixed(seed=42)
        obs, _ = test_env.reset()
        print(f"âœ… Reset successful - {len(obs)} observations")
        
        # Test a few steps
        for i in range(10):
            action = test_env.action_space.sample()
            obs, reward, terminated, truncated, info = test_env.step(action)
            if i == 0:
                print(f"âœ… Step successful - reward: {reward}")
        
        test_env.close()
        print("âœ… Environment test passed!")
    except Exception as e:
        print(f"âŒ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Create environments
    print(f"\nðŸ“¦ Creating {args.n_envs} environments...")
    env = SubprocVecEnv([make_env_fixed(args.seed, i) for i in range(args.n_envs)])
    
    # Setup directories
    run_dir = Path(f"run_{args.run_name}")
    run_dir.mkdir(exist_ok=True)
    
    # Calculate n_steps
    n_steps = max(256, args.batch_size // args.n_envs)
    n_steps = (n_steps // 64) * 64
    
    print(f"\nðŸ§  Creating PPO model...")
    print(f"  n_steps: {n_steps}")
    print(f"  Total batch: {n_steps * args.n_envs}")
    
    # Create model
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=str(run_dir / "tb_logs"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        policy_kwargs={
            "net_arch": dict(pi=[512, 512, 256], vf=[512, 512, 256]),
            "activation_fn": torch.nn.ReLU
        }
    )
    
    # Progress callback
    class ProgressCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.episode_rewards = []
            self.episode_lengths = []
            
        def _on_step(self):
            # Collect episode statistics
            for info in self.locals.get("infos", []):
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
            
            # Print progress every 100k steps
            if self.num_timesteps % 100000 == 0:
                if len(self.episode_rewards) > 0:
                    recent_rewards = self.episode_rewards[-100:] if len(self.episode_rewards) >= 100 else self.episode_rewards
                    print(f"\n[Step {self.num_timesteps:,}]")
                    print(f"  Episodes: {len(self.episode_rewards)}")
                    print(f"  Mean reward: {np.mean(recent_rewards):.2f} Â± {np.std(recent_rewards):.2f}")
                    print(f"  Mean length: {np.mean(self.episode_lengths[-100:] if len(self.episode_lengths) >= 100 else self.episode_lengths):.0f}")
                else:
                    print(f"\n[Step {self.num_timesteps:,}] Training...")
            return True
    
    # Callbacks
    callbacks = [
        ProgressCallback(),
        CheckpointCallback(
            save_freq=max(500000 // args.n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="balatro"
        )
    ]
    
    print(f"\nðŸŽ® Starting training...")
    print(f"ðŸ“ˆ Monitor with: tensorboard --logdir {run_dir}/tb_logs\n")
    
    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            log_interval=50,
            progress_bar=True
        )
        print("\nâœ… Training completed!")
        
    except KeyboardInterrupt:
        print("\nâš ï¸ Training interrupted")
    except Exception as e:
        print(f"\nâŒ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        model.save(str(run_dir / "final_model"))
        env.close()
        torch.cuda.empty_cache()
        print("\nðŸ’¾ Model saved")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=100_000_000)
    parser.add_argument("--n-envs", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", type=str, default="balatro_fixed")
    
    args = parser.parse_args()
    train_fixed(args)


if __name__ == "__main__":
    main()
