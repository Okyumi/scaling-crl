#!/bin/bash
#SBATCH --job-name=crl_lora
#SBATCH --account=torch_pr_301_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --time 24:00:00
#SBATCH --output=slurm_logs/slurm-%A_%a.out
#SBATCH --array=0-2

# Map SLURM array task ID to seeds
SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge

# ---- Override these via environment variables when submitting ----
# Example:
#   ENV_ID=ant_big_maze DEPTH=16 USE_RELU=1 USE_LOW_RANK=1 LOW_RANK_DIM=64 sbatch job_lora.slurm
ENV_ID="${ENV_ID:-ant_big_maze}"
EVAL_ENV_ID="${EVAL_ENV_ID:-ant_big_maze_eval}"
DEPTH="${DEPTH:-16}"
USE_RELU="${USE_RELU:-0}"
USE_LOW_RANK="${USE_LOW_RANK:-0}"
LOW_RANK_DIM="${LOW_RANK_DIM:-64}"
# ---------------------------------------------------------------

# Build conditional flags
LR_FLAGS=""
if [ "$USE_LOW_RANK" -eq 1 ]; then
    LR_FLAGS="--use_low_rank 1 --low_rank_dim $LOW_RANK_DIM"
fi

RELU_FLAG=""
if [ "$USE_RELU" -eq 1 ]; then
    RELU_FLAG="--use_relu 1"
else
    RELU_FLAG="--use_relu 0"
fi

# Build tag for logging
if [ "$USE_LOW_RANK" -eq 1 ]; then
    MODE_TAG="relu_lora${LOW_RANK_DIM}"
elif [ "$USE_RELU" -eq 1 ]; then
    MODE_TAG="relu"
else
    MODE_TAG="silu"
fi

echo "Running: env=$ENV_ID depth=$DEPTH seed=$SEED mode=$MODE_TAG"
echo "  use_relu=$USE_RELU use_low_rank=$USE_LOW_RANK low_rank_dim=$LOW_RANK_DIM"

uv run train_lora.py \
    --env_id "$ENV_ID" \
    --eval_env_id "$EVAL_ENV_ID" \
    --num_epochs 100 \
    --total_env_steps 100000000 \
    --critic_depth $DEPTH \
    --actor_depth $DEPTH \
    --critic_network_width 256 \
    --actor_network_width 256 \
    --actor_skip_connections 4 \
    --critic_skip_connections 4 \
    --batch_size 512 \
    --num_envs 512 \
    --vis_length 1000 \
    --save_buffer 0 \
    --seed $SEED \
    $RELU_FLAG \
    $LR_FLAGS
