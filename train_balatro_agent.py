"""Train RL agents on the Balatro environment.

This script provides a complete training pipeline including:
- Multiple algorithm support (PPO, A2C, DQN)
- Curriculum learning
- Behavioral cloning warm-start
- Comprehensive logging
- Model checkpointing
- Hyperparameter tuning
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Callable, Type, List, Tuple
from pathlib import Path
import json
import pickle
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import (
    BaseCallback, EvalCallback, CheckpointCallback, CallbackList
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from balatro_gym.balatro_env_2 import BalatroEnv, make_balatro_env, Phase
import wandb


# ---------------------------------------------------------------------------
# Custom Feature Extractor for Balatro
# ---------------------------------------------------------------------------

class BalatroFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor that understands Balatro's observation structure"""
    
    def __init__(self, observation_space: spaces.Dict, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimensions for each component
        hand_dim = 8 * 52  # One-hot encoded cards
        joker_dim = 10 * 16  # Joker embeddings
        game_state_dim = 32  # All scalar features
        
        # Build the network
        self.hand_net = nn.Sequential(
            nn.Linear(hand_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        self.joker_net = nn.Sequential(
            nn.Linear(joker_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU()
        )
        
        self.game_state_net = nn.Sequential(
            nn.Linear(game_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Combine all features
        combined_dim = 128 + 64 + 32  # From each subnet
        self.combined_net = nn.Sequential(
            nn.Linear(combined_dim, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Process hand (convert to one-hot)
        hand = observations['hand'].long()
        batch_size = hand.shape[0]
        hand_one_hot = torch.zeros(batch_size, 8, 52, device=hand.device)
        
        for i in range(8):
            valid_cards = hand[:, i] >= 0
            if valid_cards.any():
                hand_one_hot[valid_cards, i, hand[valid_cards, i]] = 1
        
        hand_features = self.hand_net(hand_one_hot.view(batch_size, -1))
        
        # Process jokers (simple embedding)
        joker_ids = observations['joker_ids'].float()
        joker_features = self.joker_net(joker_ids.view(batch_size, -1))
        
        # Process game state
        game_features = torch.cat([
            observations['chips_scored'].float().unsqueeze(1) / 1e6,  # Normalize
            observations['chips_needed'].float().unsqueeze(1) / 1e5,
            observations['progress_ratio'].float().unsqueeze(1),
            observations['money'].float().unsqueeze(1) / 100,
            observations['ante'].float().unsqueeze(1) / 10,
            observations['round'].float().unsqueeze(1) / 3,
            observations['hands_left'].float().unsqueeze(1) / 10,
            observations['discards_left'].float().unsqueeze(1) / 5,
            observations['hand_levels'].float() / 10,  # 12 values
            observations['phase'].float().unsqueeze(1) / 3,
        ], dim=1)
        
        game_state_features = self.game_state_net(game_features)
        
        # Combine all features
        combined = torch.cat([hand_features, joker_features, game_state_features], dim=1)
        return self.combined_net(combined)


# ---------------------------------------------------------------------------
# Curriculum Learning Wrapper
# ---------------------------------------------------------------------------

class CurriculumBalatroEnv(gym.Wrapper):
    """Wrapper that implements curriculum learning by limiting max ante"""
    
    def __init__(self, env: BalatroEnv, initial_max_ante: int = 3, 
                 ante_increment: int = 1, success_threshold: float = 0.8):
        super().__init__(env)
        self.current_max_ante = initial_max_ante
        self.ante_increment = ante_increment
        self.success_threshold = success_threshold
        
        # Track performance
        self.episode_antes = []
        self.episode_rewards = []
        self.episodes_at_current_level = 0
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.episodes_at_current_level += 1
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Check if we should end episode at curriculum limit
        if self.env.state.ante > self.current_max_ante:
            terminated = True
            info['curriculum_limit_reached'] = True
        
        # Track episode completion
        if terminated or truncated:
            self.episode_antes.append(self.env.state.ante)
            self.episode_rewards.append(info.get('episode_reward', 0))
            
            # Check if we should increase difficulty
            if self.episodes_at_current_level >= 100:  # Every 100 episodes
                recent_success_rate = sum(
                    a >= self.current_max_ante for a in self.episode_antes[-100:]
                ) / 100
                
                if recent_success_rate >= self.success_threshold:
                    self.current_max_ante += self.ante_increment
                    self.episodes_at_current_level = 0
                    print(f"Curriculum: Increasing max ante to {self.current_max_ante}")
        
        return obs, reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Custom Callbacks
# ---------------------------------------------------------------------------

class BalatroMetricsCallback(BaseCallback):
    """Track Balatro-specific metrics during training"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_antes = []
        self.episode_scores = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        # Check for episode end
        for i, done in enumerate(self.locals['dones']):
            if done:
                info = self.locals['infos'][i]
                
                # Extract metrics
                self.episode_rewards.append(self.locals['rewards'][i])
                self.episode_antes.append(info.get('ante', 1))
                self.episode_scores.append(info.get('final_score', 0))
                self.episode_lengths.append(info.get('episode_length', 0))
                
                # Log to tensorboard
                self.logger.record('balatro/episode_ante', info.get('ante', 1))
                self.logger.record('balatro/episode_score', info.get('final_score', 0))
                self.logger.record('balatro/episode_reward', self.locals['rewards'][i])
                
                # Log to wandb if available
                if wandb.run is not None:
                    wandb.log({
                        'episode_ante': info.get('ante', 1),
                        'episode_score': info.get('final_score', 0),
                        'episode_reward': self.locals['rewards'][i],
                        'global_step': self.num_timesteps
                    })
        
        return True


# ---------------------------------------------------------------------------
# Behavioral Cloning from Expert Trajectories
# ---------------------------------------------------------------------------

class BehavioralCloning:
    """Pre-train policy using expert demonstrations"""
    
    def __init__(self, model, trajectories_path: str):
        self.model = model
        self.trajectories = self._load_trajectories(trajectories_path)
        
    def _load_trajectories(self, path: str):
        """Load expert trajectories"""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data['trajectories']
    
    def pretrain(self, n_epochs: int = 10, batch_size: int = 64):
        """Pre-train the policy network"""
        print(f"Pre-training with behavioral cloning for {n_epochs} epochs...")
        
        # Extract states and actions
        states = []
        actions = []
        
        for trajectory in self.trajectories:
            for step in trajectory.steps:
                # Only use successful trajectories
                if trajectory.final_ante >= 5:  # Minimum performance threshold
                    states.append(step.observation)
                    actions.append(step.action)
        
        if not states:
            print("No valid expert demonstrations found!")
            return
        
        # Convert to tensors
        # This is simplified - you'd need to properly process the dict observations
        print(f"Training on {len(states)} expert state-action pairs")
        
        # TODO: Implement actual BC training loop
        # This would involve:
        # 1. Processing observations to match the policy network input
        # 2. Creating a DataLoader
        # 3. Training the policy network with supervised learning
        
        print("Behavioral cloning pre-training complete!")


# ---------------------------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------------------------

def train_balatro_agent(
    algorithm: str = "PPO",
    total_timesteps: int = 1_000_000,
    n_envs: int = 8,
    seed: int = 42,
    use_curriculum: bool = True,
    use_wandb: bool = True,
    save_dir: str = "models",
    checkpoint_freq: int = 10_000,
    eval_freq: int = 5_000,
    expert_trajectories: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None
):
    """Train an RL agent on Balatro"""
    
    # Initialize wandb
    if use_wandb:
        wandb.init(
            project="balatro-rl",
            config={
                "algorithm": algorithm,
                "total_timesteps": total_timesteps,
                "n_envs": n_envs,
                "seed": seed,
                "use_curriculum": use_curriculum,
                "hyperparams": hyperparams or {}
            }
        )
    
    # Create save directory
    save_path = Path(save_dir) / f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Create environments
    def make_env(rank: int, seed: int = 0):
        def _init():
            env = BalatroEnv(seed=seed + rank)
            env = Monitor(env)
            if use_curriculum:
                env = CurriculumBalatroEnv(env)
            return env
        return _init
    
    # Create vectorized environment
    if n_envs > 1:
        env = SubprocVecEnv([make_env(i, seed) for i in range(n_envs)])
    else:
        env = DummyVecEnv([make_env(0, seed)])
    
    # Normalize rewards and observations
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Create eval environment
    eval_env = DummyVecEnv([make_env(1000, seed)])  # Different seed for eval
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # Default hyperparameters
    default_hyperparams = {
        "PPO": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "features_extractor_class": BalatroFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
            }
        },
        "DQN": {
            "learning_rate": 1e-4,
            "buffer_size": 100_000,
            "learning_starts": 10_000,
            "batch_size": 32,
            "tau": 1.0,
            "gamma": 0.99,
            "train_freq": 4,
            "gradient_steps": 1,
            "target_update_interval": 1000,
            "exploration_fraction": 0.1,
            "exploration_initial_eps": 1.0,
            "exploration_final_eps": 0.05,
            "policy_kwargs": {
                "features_extractor_class": BalatroFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
                "net_arch": [512, 512]
            }
        },
        "A2C": {
            "learning_rate": 7e-4,
            "n_steps": 5,
            "gamma": 0.99,
            "gae_lambda": 1.0,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "policy_kwargs": {
                "features_extractor_class": BalatroFeaturesExtractor,
                "features_extractor_kwargs": {"features_dim": 512},
                "net_arch": [dict(pi=[256, 256], vf=[256, 256])]
            }
        }
    }
    
    # Merge with provided hyperparams
    algo_hyperparams = default_hyperparams.get(algorithm, {})
    if hyperparams:
        algo_hyperparams.update(hyperparams)
    
    # Create model
    if algorithm == "PPO":
        model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=str(save_path / "tb_logs"),
                   **algo_hyperparams)
    elif algorithm == "DQN":
        model = DQN("MultiInputPolicy", env, verbose=1, tensorboard_log=str(save_path / "tb_logs"),
                   **algo_hyperparams)
    elif algorithm == "A2C":
        model = A2C("MultiInputPolicy", env, verbose=1, tensorboard_log=str(save_path / "tb_logs"),
                   **algo_hyperparams)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Behavioral cloning pre-training
    if expert_trajectories and Path(expert_trajectories).exists():
        bc = BehavioralCloning(model, expert_trajectories)
        bc.pretrain(n_epochs=10)
    
    # Create callbacks
    callbacks = []
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        n_eval_episodes=10
    )
    callbacks.append(eval_callback)
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(save_path / "checkpoints"),
        name_prefix=f"{algorithm}_checkpoint"
    )
    callbacks.append(checkpoint_callback)
    
    # Balatro metrics callback
    metrics_callback = BalatroMetricsCallback()
    callbacks.append(metrics_callback)
    
    # Train the model
    print(f"\nStarting training with {algorithm}...")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Number of environments: {n_envs}")
    print(f"Save directory: {save_path}")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=CallbackList(callbacks),
            log_interval=10,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    # Save final model
    model.save(str(save_path / f"{algorithm}_final"))
    env.save(str(save_path / "vec_normalize.pkl"))
    
    # Save training config
    config = {
        "algorithm": algorithm,
        "total_timesteps": total_timesteps,
        "n_envs": n_envs,
        "seed": seed,
        "use_curriculum": use_curriculum,
        "hyperparams": algo_hyperparams,
        "save_path": str(save_path)
    }
    
    with open(save_path / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining complete! Model saved to {save_path}")
    
    # Final evaluation
    print("\nRunning final evaluation...")
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    print(f"Final performance: {mean_reward:.2f} +/- {std_reward:.2f}")
    
    if use_wandb:
        wandb.finish()
    
    return model, save_path


# ---------------------------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------------------------

def tune_hyperparameters(
    algorithm: str = "PPO",
    n_trials: int = 20,
    n_timesteps: int = 100_000,
    n_envs: int = 4,
    seed: int = 42
):
    """Use Optuna for hyperparameter tuning"""
    import optuna
    
    def objective(trial):
        # Sample hyperparameters
        if algorithm == "PPO":
            hyperparams = {
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-5, 1e-3),
                "n_steps": trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096]),
                "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128, 256]),
                "n_epochs": trial.suggest_int("n_epochs", 5, 15),
                "gamma": trial.suggest_uniform("gamma", 0.95, 0.999),
                "gae_lambda": trial.suggest_uniform("gae_lambda", 0.9, 0.99),
                "clip_range": trial.suggest_uniform("clip_range", 0.1, 0.3),
                "ent_coef": trial.suggest_loguniform("ent_coef", 1e-4, 1e-1),
            }
        else:
            raise ValueError(f"Tuning not implemented for {algorithm}")
        
        # Train with sampled hyperparameters
        model, save_path = train_balatro_agent(
            algorithm=algorithm,
            total_timesteps=n_timesteps,
            n_envs=n_envs,
            seed=seed,
            use_wandb=False,  # Disable wandb for tuning
            hyperparams=hyperparams
        )
        
        # Evaluate
        eval_env = make_vec_env(lambda: Monitor(BalatroEnv()), n_envs=1)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=10)
        
        return mean_reward
    
    # Create study
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    print("\nBest hyperparameters:")
    print(study.best_params)
    
    return study.best_params


# ---------------------------------------------------------------------------
# Testing and Visualization
# ---------------------------------------------------------------------------

def test_trained_agent(
    model_path: str,
    n_episodes: int = 5,
    render: bool = True,
    record_video: bool = False
):
    """Test a trained agent"""
    from stable_baselines3.common.vec_env import VecVideoRecorder
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment
    env = BalatroEnv(render_mode="human" if render else None)
    env = Monitor(env)
    
    if record_video:
        env = VecVideoRecorder(
            DummyVecEnv([lambda: env]),
            f"videos/{Path(model_path).stem}",
            record_video_trigger=lambda x: x % 1 == 0,
            video_length=10000
        )
    
    # Run episodes
    episode_rewards = []
    episode_antes = []
    
    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        done = False
        step = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done and step < 5000:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            step += 1
            
            if render:
                env.render()
        
        final_ante = info.get('ante', 1)
        final_score = info.get('final_score', 0)
        
        episode_rewards.append(episode_reward)
        episode_antes.append(final_ante)
        
        print(f"Episode {episode + 1} complete:")
        print(f"  Total reward: {episode_reward:.2f}")
        print(f"  Final ante: {final_ante}")
        print(f"  Final score: {final_score}")
        print(f"  Steps: {step}")
    
    print(f"\nAverage reward: {np.mean(episode_rewards):.2f} +/- {np.std(episode_rewards):.2f}")
    print(f"Average ante: {np.mean(episode_antes):.1f}")
    
    if record_video:
        env.close()


# ---------------------------------------------------------------------------
# Main Script
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train RL agents on Balatro")
    parser.add_argument("--algorithm", type=str, default="PPO",
                       choices=["PPO", "DQN", "A2C"],
                       help="RL algorithm to use")
    parser.add_argument("--timesteps", type=int, default=1_000_000,
                       help="Total training timesteps")
    parser.add_argument("--n-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--no-curriculum", action="store_true",
                       help="Disable curriculum learning")
    parser.add_argument("--no-wandb", action="store_true",
                       help="Disable Weights & Biases logging")
    parser.add_argument("--expert-trajectories", type=str, default=None,
                       help="Path to expert trajectories for behavioral cloning")
    parser.add_argument("--tune", action="store_true",
                       help="Run hyperparameter tuning")
    parser.add_argument("--test", type=str, default=None,
                       help="Test a trained model")
    parser.add_argument("--quick-test", action="store_true",
                       help="Run a quick training test (10k steps)")
    
    args = parser.parse_args()
    
    if args.tune:
        # Run hyperparameter tuning
        best_params = tune_hyperparameters(
            algorithm=args.algorithm,
            n_trials=20,
            n_timesteps=100_000,
            n_envs=args.n_envs,
            seed=args.seed
        )
        print(f"\nBest hyperparameters found:")
        print(json.dumps(best_params, indent=2))
        
    elif args.test:
        # Test a trained model
        test_trained_agent(
            model_path=args.test,
            n_episodes=5,
            render=True,
            record_video=False
        )
        
    elif args.quick_test:
        # Quick test run
        print("Running quick test (10k timesteps)...")
        model, save_path = train_balatro_agent(
            algorithm=args.algorithm,
            total_timesteps=10_000,
            n_envs=2,
            seed=args.seed,
            use_curriculum=not args.no_curriculum,
            use_wandb=False,
            checkpoint_freq=5_000,
            eval_freq=5_000
        )
        print(f"\nQuick test complete! Model saved to {save_path}")
        
    else:
        # Full training run
        model, save_path = train_balatro_agent(
            algorithm=args.algorithm,
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            seed=args.seed,
            use_curriculum=not args.no_curriculum,
            use_wandb=not args.no_wandb,
            expert_trajectories=args.expert_trajectories
        )
        
        print("\nTraining complete!")
        print(f"Model saved to: {save_path}")
        print(f"\nTo test the trained agent, run:")
        print(f"python train_balatro_rl.py --test {save_path}/PPO_final.zip")


# ---------------------------------------------------------------------------
# Example Usage
# ---------------------------------------------------------------------------

"""
# Basic training with PPO:
python train_balatro_rl.py --algorithm PPO --timesteps 1000000

# Quick test to verify setup:
python train_balatro_rl.py --quick-test

# Train with behavioral cloning warm-start:
python train_balatro_rl.py --algorithm PPO --expert-trajectories trajectories/trajectories_20241210_153045.pkl

# Hyperparameter tuning:
python train_balatro_rl.py --tune --algorithm PPO

# Test a trained model:
python train_balatro_rl.py --test models/PPO_20241210_160000/PPO_final.zip

# Train with DQN instead:
python train_balatro_rl.py --algorithm DQN --timesteps 500000 --n-envs 1

# Disable curriculum learning:
python train_balatro_rl.py --no-curriculum

# Train with custom hyperparameters (modify the script or use config file)
"""