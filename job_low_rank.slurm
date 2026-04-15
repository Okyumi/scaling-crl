#!/bin/bash
#SBATCH --job-name=crl_lowrank
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --output=slurm_logs/slurm-%j.out

module purge

# ---- Adjust these variables per experiment ----
# Override via environment variables when submitting:
#   ENV_ID=ant_big_maze DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 SEED=1000 sbatch job_low_rank.slurm
ENV_ID="${ENV_ID:-ant_big_maze}"
EVAL_ENV_ID="${EVAL_ENV_ID:-ant_big_maze_eval}"
DEPTH="${DEPTH:-16}"
USE_LOW_RANK="${USE_LOW_RANK:-0}"
LOW_RANK_DIM="${LOW_RANK_DIM:-64}"
LOW_RANK_ACTOR="${LOW_RANK_ACTOR:-0}"
SEED="${SEED:-1000}"
# -----------------------------------------------

if [ "$USE_LOW_RANK" -eq 1 ]; then
    LR_FLAGS="--use_low_rank 1 --low_rank_dim $LOW_RANK_DIM --low_rank_actor $LOW_RANK_ACTOR"
else
    LR_FLAGS=""
fi

echo "Running: env=$ENV_ID depth=$DEPTH seed=$SEED low_rank=$USE_LOW_RANK r=$LOW_RANK_DIM actor_lr=$LOW_RANK_ACTOR"

uv run train_low_rank.py \
    --env_id "$ENV_ID" \
    --eval_env_id "$EVAL_ENV_ID" \
    --num_epochs 100 \
    --total_env_steps 100000000 \
    --critic_depth $DEPTH \
    --actor_depth $DEPTH \
    --actor_skip_connections 4 \
    --critic_skip_connections 4 \
    --batch_size 512 \
    --num_envs 512 \
    --vis_length 1000 \
    --save_buffer 0 \
    --seed $SEED \
    $LR_FLAGS
