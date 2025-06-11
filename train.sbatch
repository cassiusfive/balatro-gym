#!/bin/bash
#SBATCH --job-name=balatro_rl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/balatro_%j.out
#SBATCH --error=logs/balatro_%j.err
#SBATCH --mail-user=your.email@oregonstate.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# Load modules
module purge
module load python/3.9
module load cuda/11.8

# Set up Python environment
if [ ! -d "$HOME/balatro_env" ]; then
    python -m venv $HOME/balatro_env
    source $HOME/balatro_env/bin/activate
    pip install --upgrade pip
    pip install torch torchvision gymnasium numpy
    pip install stable-baselines3[extra] tensorboard
    pip install -e balatro-gym/  # Install your gym
else
    source $HOME/balatro_env/bin/activate
fi

# Change to project directory
cd $HOME/balatro_rl_project

# Run training
python hpc_train.py --timesteps 10000000 --n-envs 16
