#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=2:00:00
#SBATCH -A r02346
#SBATCH --job-name=dp_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

mkdir -p logs
source ~/.bashrc
module load conda
conda activate dp_casa
cd ~/robocasa_diffusion_policy

CKPT=/N/u/mihparek/BigRed200/robocasa_diffusion_policy/data/outputs/2026.09.07/02.45.59_train_diffusion_transformer_hybrid_finetune_target_composite_seen_SearingMeat/checkpoints/latest.ckpt
TASK=${1:-SearingMeat}

CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl python run_diffusion_policy_robocasa.py \
    --checkpoint "$CKPT" --task_set "$TASK" --split target
