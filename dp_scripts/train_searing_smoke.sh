#!/bin/bash
#SBATCH --job-name=dp_searing_smoke
#SBATCH --output=logs/dp_searing_smoke_%j.out
#SBATCH --error=logs/dp_searing_smoke_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00 
#SBATCH --account=r02346

module load ffmpeg
module load conda
conda activate dp_casa
cd ~/robocasa_diffusion_policy

MUJOCO_GL=egl HYDRA_FULL_ERROR=1 python train.py \
    --config-name=train_diffusion_transformer_bs192 \
    task=robocasa/target_composite_seen \
    task.dataset.dataset_soup=null \
    '+task.dataset.dataset_paths=[/N/slate/mihparek/robocasa_datasets/SearingMeat/20250812/lerobot]' \
    'task.dataset.lerobot_dir_suffix=' \
    training.device=cuda:0 \
    logging.mode=online
