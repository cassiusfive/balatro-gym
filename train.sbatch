#!/bin/bash
#SBATCH --job-name=balatro_rl
#SBATCH --output=logs/balatro_%j.out
#SBATCH --error=logs/balatro_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --mail-user=villarec@oregonstate.edu
#SBATCH --mail-type=END,FAIL

# Print job info
echo "Starting Balatro RL Training"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Time: $(date)"

# Load required modules
module purge
module load python/3.9
module load cuda/11.8

# Activate your virtual environment
source ~/balatro_env/bin/activate

# Change to project directory
cd ~/balatro_rl_project

# Create logs directory if it doesn't exist
mkdir -p logs

# Set Python path to find your balatro_gym module
export PYTHONPATH=$PYTHONPATH:~/balatro_rl_project

# Show GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Run training
echo "Starting training..."
python simple_hpc_train.py \
    --timesteps 10000000 \
    --n-envs 16 \
    --seed 42 \
    --checkpoint-freq 100000 \
    --eval-freq 50000

echo "Training completed at $(date)"
