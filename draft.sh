#!/bin/bash
#SBATCH --job-name=scaling_crl_smoke
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/yd2247/scaling-crl/slurm_logs/smoke-%j.out

cd /scratch/yd2247/scaling-crl
mkdir -p /scratch/yd2247/.cache/uv
mkdir -p /scratch/yd2247/scaling-crl/slurm_logs

export UV_CACHE_DIR=/scratch/yd2247/.cache/uv
export UV_PROJECT_ENVIRONMENT=/scratch/yd2247/scaling-crl/.venv

uv run train.py \
  --env_id "ant_big_maze" \
  --eval_env_id "ant_big_maze_eval" \
  --num_epochs 1 \
  --total_env_steps 10000 \
  --critic_depth 2 \
  --actor_depth 2 \
  --batch_size 64 \
  --vis_length 100 \
  --save_buffer 0