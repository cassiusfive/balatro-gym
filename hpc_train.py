"""Advanced training script optimized for HPC clusters with multi-GPU support"""

import os
import sys
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

from balatro_gym.balatro_env_2 import BalatroEnv, make_balatro_env


class HPCTrainer:
    """Optimized trainer for HPC environments"""
    
    def __init__(self, args):
        self.args = args
        self.setup_distributed()
        self.setup_directories()
        
    def setup_distributed(self):
        """Setup distributed training if multiple GPUs available"""
        if 'SLURM_PROCID' in os.environ:
            self.rank = int(os.environ['SLURM_PROCID'])
            self.world_size = int(os.environ['SLURM_NTASKS'])
            self.gpu = self.rank % torch.cuda.device_count()
            
            # Initialize distributed backend
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=self.world_size,
                rank=self.rank
            )
            
            torch.cuda.set_device(self.gpu)
            print(f"Process {self.rank}/{self.world_size} using GPU {self.gpu}")
        else:
            self.rank = 0
            self.world_size = 1
            self.gpu = 0 if torch.cuda.is_available() else None
    
    def setup_directories(self):
        """Create necessary directories"""
        self.run_dir = Path(f"run_{os.environ.get('SLURM_JOB_ID', 'local')}")
        self.run_dir.mkdir(exist_ok=True)
        
        self.model_dir = self.run_dir / "models"
        self.model_dir.mkdir(exist_ok=True)
        
        self.log_dir = self.run_dir / "logs"
        self.log_dir.mkdir(exist_ok=True)
    
    def create_envs(self, n_envs: int, seed: int, rank_offset: int = 0):
        """Create parallel environments optimized for HPC"""
        def make_env(rank: int):
            def _init():
                env = BalatroEnv(seed=seed + rank + rank_offset)
                # Add monitors and wrappers
                return env
            return _init
        
        # Use SubprocVecEnv for true parallelism
        envs = SubprocVecEnv([make_env(i) for i in range(n_envs)])
        envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
        
        return envs
    
    def train(self):
        """Main training loop optimized for HPC"""
        if self.rank == 0:
            print(f"Starting training on {self.world_size} processes")
            print(f"Total environments: {self.args.n_envs * self.world_size}")
            print(f"Total timesteps: {self.args.timesteps}")
        
        # Set seeds for reproducibility
        seed = self.args.seed + self.rank * 1000
        set_random_seed(seed)
        
        # Create environments
        envs = self.create_envs(
            self.args.n_envs, 
            seed, 
            rank_offset=self.rank * self.args.n_envs
        )
        
        # Create model with optimized settings for HPC
        device = f'cuda:{self.gpu}' if self.gpu is not None else 'cpu'
        
        model = PPO(
            "MultiInputPolicy",
            envs,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,  # Larger batch for GPU
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            tensorboard_log=str(self.log_dir) if self.rank == 0 else None,
            verbose=1 if self.rank == 0 else 0,
            device=device,
            policy_kwargs={
                "net_arch": [dict(pi=[512, 512], vf=[512, 512])],
                "activation_fn": torch.nn.ReLU
            }
        )
        
        # Setup callbacks (only on rank 0)
        callbacks = []
        if self.rank == 0:
            checkpoint_callback = CheckpointCallback(
                save_freq=50000,
                save_path=str(self.model_dir),
                name_prefix="balatro_ppo"
            )
            callbacks.append(checkpoint_callback)
            
            # Save best model
            eval_env = self.create_envs(4, seed + 10000)
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(self.model_dir / "best"),
                log_path=str(self.log_dir / "eval"),
                eval_freq=10000,
                deterministic=True,
                render=False,
                n_eval_episodes=10
            )
            callbacks.append(eval_callback)
        
        # Training loop with periodic synchronization
        timesteps_per_process = self.args.timesteps // self.world_size
        
        try:
            start_time = time.time()
            
            model.learn(
                total_timesteps=timesteps_per_process,
                callback=CallbackList(callbacks) if callbacks else None,
                log_interval=10 if self.rank == 0 else None,
                progress_bar=self.rank == 0
            )
            
            training_time = time.time() - start_time
            
            if self.rank == 0:
                print(f"\nTraining completed in {training_time/3600:.2f} hours")
                print(f"Throughput: {self.args.timesteps / training_time:.2f} steps/second")
                
                # Save final model
                model.save(str(self.model_dir / "final_model"))
                envs.save(str(self.model_dir / "vec_normalize.pkl"))
                
                # Save training config
                config = {
                    "args": vars(self.args),
                    "world_size": self.world_size,
                    "training_time": training_time,
                    "throughput": self.args.timesteps / training_time,
                    "timestamp": datetime.now().isoformat()
                }
                
                with open(self.model_dir / "config.json", "w") as f:
                    json.dump(config, f, indent=2)
        
        finally:
            envs.close()
            if self.world_size > 1:
                dist.destroy_process_group()


def run_hpc_training():
    """Entry point for HPC training"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=10_000_000)
    parser.add_argument("--n-envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-freq", type=int, default=100_000)
    parser.add_argument("--eval-freq", type=int, default=50_000)
    
    args = parser.parse_args()
    
    # Detect HPC environment
    if 'SLURM_JOB_ID' in os.environ:
        print(f"Running on SLURM job {os.environ['SLURM_JOB_ID']}")
        print(f"Node: {os.environ.get('SLURMD_NODENAME', 'unknown')}")
        
        # Log GPU info
        if torch.cuda.is_available():
            print(f"GPUs available: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    
    trainer = HPCTrainer(args)
    trainer.train()


# ---------------------------------------------------------------------------
# Batch Job Submission Scripts
# ---------------------------------------------------------------------------

def generate_experiment_scripts():
    """Generate multiple experiment configurations for batch submission"""
    
    experiments = [
        # Baseline PPO
        {
            "name": "ppo_baseline",
            "algorithm": "PPO",
            "timesteps": 10_000_000,
            "n_envs": 16,
            "lr": 3e-4,
            "batch_size": 64
        },
        # PPO with larger batch
        {
            "name": "ppo_large_batch",
            "algorithm": "PPO", 
            "timesteps": 10_000_000,
            "n_envs": 32,
            "lr": 3e-4,
            "batch_size": 256
        },
        # PPO with curriculum
        {
            "name": "ppo_curriculum",
            "algorithm": "PPO",
            "timesteps": 10_000_000,
            "n_envs": 16,
            "lr": 3e-4,
            "use_curriculum": True
        },
        # Different learning rates
        {
            "name": "ppo_lr_1e4",
            "algorithm": "PPO",
            "timesteps": 5_000_000,
            "n_envs": 16,
            "lr": 1e-4
        },
        {
            "name": "ppo_lr_1e3",
            "algorithm": "PPO",
            "timesteps": 5_000_000,
            "n_envs": 16,
            "lr": 1e-3
        }
    ]
    
    # Create experiment directory
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    
    # Generate individual job scripts
    for exp in experiments:
        script_content = f"""#!/bin/bash
#SBATCH --job-name=balatro_{exp['name']}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/{exp['name']}_%j.out
#SBATCH --error=logs/{exp['name']}_%j.err

source $HOME/balatro_env/bin/activate
cd $HOME/balatro_rl_experiments

python train_balatro_rl.py \\
    --algorithm {exp['algorithm']} \\
    --timesteps {exp['timesteps']} \\
    --n-envs {exp['n_envs']} \\
    --learning-rate {exp.get('lr', 3e-4)} \\
    --save-name {exp['name']}
"""
        
        script_path = exp_dir / f"{exp['name']}.sbatch"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Generated: {script_path}")
    
    # Generate batch submission script
    batch_script = """#!/bin/bash
# Submit all experiments

for script in experiments/*.sbatch; do
    echo "Submitting $script"
    sbatch $script
    sleep 2
done

echo "All experiments submitted!"
squeue -u $USER
"""
    
    with open("submit_all_experiments.sh", "w") as f:
        f.write(batch_script)
    
    os.chmod("submit_all_experiments.sh", 0o755)
    print("\nGenerated submit_all_experiments.sh")
    print("Run './submit_all_experiments.sh' to submit all experiments")


# ---------------------------------------------------------------------------
# Monitoring Script
# ---------------------------------------------------------------------------

def create_monitoring_script():
    """Create a script to monitor training progress"""
    
    monitor_script = '''#!/usr/bin/env python3
"""Monitor Balatro RL training progress on HPC"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
import subprocess

def get_job_status():
    """Get current SLURM job status"""
    result = subprocess.run(['squeue', '-u', os.environ['USER'], '-o', '%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R'],
                          capture_output=True, text=True)
    print("Current Jobs:")
    print(result.stdout)

def monitor_training_progress():
    """Monitor training logs and metrics"""
    log_dirs = list(Path('.').glob('run_*/logs'))
    
    for log_dir in sorted(log_dirs):
        run_id = log_dir.parent.name
        print(f"\\n{'='*50}")
        print(f"Run: {run_id}")
        
        # Check for latest checkpoint
        checkpoints = list((log_dir.parent / 'models').glob('*.zip'))
        if checkpoints:
            latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
            print(f"Latest checkpoint: {latest.name}")
            print(f"Modified: {datetime.fromtimestamp(latest.stat().st_mtime)}")
        
        # Check tensorboard logs
        tb_dirs = list(log_dir.glob('**/events.out.tfevents.*'))
        if tb_dirs:
            print(f"TensorBoard logs: {len(tb_dirs)} files")
        
        # Check eval results
        eval_log = log_dir / 'eval' / 'evaluations.npz'
        if eval_log.exists():
            import numpy as np
            data = np.load(eval_log)
            if 'results' in data:
                results = data['results']
                if len(results) > 0:
                    latest_mean = results[-1].mean()
                    print(f"Latest eval reward: {latest_mean:.2f}")

if __name__ == "__main__":
    while True:
        os.system('clear')
        print(f"Balatro RL Training Monitor - {datetime.now()}")
        print("="*60)
        
        get_job_status()
        monitor_training_progress()
        
        print(f"\\nRefreshing in 30 seconds... (Ctrl+C to exit)")
        time.sleep(30)
'''
    
    with open("monitor_training.py", "w") as f:
        f.write(monitor_script)
    
    os.chmod("monitor_training.py", 0o755)
    print("Created monitor_training.py")


if __name__ == "__main__":
    # Generate experiment scripts
    generate_experiment_scripts()
    create_monitoring_script()
    
    # Run training if called directly
    run_hpc_training()
