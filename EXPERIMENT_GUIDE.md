# Low-Rank Regularization for Contrastive RL — Experiment Guide

## Overview

This guide describes experiments to investigate whether **low-rank bottleneck regularization** improves performance of deep residual networks in Contrastive RL (CRL). The technique replaces the standard wide residual MLP in the critic encoders with a bottleneck architecture where the residual blocks operate in a lower-dimensional space.

## Research Questions & Hypotheses

### RQ1: Does low-rank regularization improve CRL performance at various depths?
**Hypothesis**: Low-rank bottleneck architecture will improve or match baseline performance by preventing feature rank collapse in the critic's learned representations. The bottleneck forces the network to learn a more compressed, structured representation.

### RQ2: Does low-rank help more with deeper networks?
**Hypothesis**: Deeper networks (depth 16, 32) may suffer more from feature rank collapse. The low-rank bottleneck should provide larger relative improvements at greater depths compared to shallow networks (depth 4).

### RQ3: What is the optimal bottleneck dimension for CRL?
**Hypothesis**: There is a sweet spot for `low_rank_dim` — too small loses representational capacity, too large provides insufficient regularization. We expect `low_rank_dim` in the range [32, 128] to work well when `hidden_dim=256`.

### RQ4: Does applying low-rank to the actor help or hurt?
**Hypothesis**: The critic is the primary learning component in CRL (representations are learned via InfoNCE), so low-rank regularization should primarily benefit the critic. Applying it to the actor may hurt since the actor needs full expressiveness for the policy.

## Architecture Description

### Baseline (Full-Rank Residual MLP)
```
Input → Dense(input_dim → hidden_dim) → LayerNorm → Swish
     → [Residual Block × (depth // 4)]
     → Dense(hidden_dim → 64)
     → Output (64-dim representation)

Residual Block (4 layers):
    identity = x
    x → Dense(hidden_dim → hidden_dim) → LayerNorm → Swish  (×4)
    x = x + identity
```

### Low-Rank Bottleneck Variant
```
Input → V: Dense(input_dim → low_rank_dim) → LayerNorm → Swish
     → [Low-Rank Residual Block × (depth // 4)]  (operates at dim r)
     → U: Dense(low_rank_dim → hidden_dim) → LayerNorm → Swish
     → Dense(hidden_dim → 64)
     → Output (64-dim representation)

Low-Rank Residual Block (4 layers at dim r):
    identity = x
    x → Dense(r → r) → LayerNorm → Swish  (×4)
    x = x + identity
```

### Key Differences
- **V projection** (input → bottleneck): Uses epsilon-scaled orthogonal initialization
- **Residual blocks**: Operate at dimension `r` instead of `hidden_dim` (e.g., 64 instead of 256)
- **U expansion** (bottleneck → full-rank): Uses epsilon-scaled orthogonal initialization
- **Parameter count**: Significantly fewer parameters in the residual stack (r² per layer vs m² per layer)

## Config Flags

All new flags are in `train_low_rank.py` and default to preserving baseline behavior:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_low_rank` | int | 0 | Enable low-rank bottleneck in critic encoders (SA and Goal). 0=off (baseline), 1=on. |
| `--low_rank_dim` | int | 64 | Bottleneck dimension `r`. Only used when `use_low_rank=1`. Controls the width of the residual blocks in the low-rank MLP. |
| `--low_rank_actor` | int | 0 | Also apply low-rank to the actor network. Only used when `use_low_rank=1`. 0=standard actor, 1=low-rank actor. |
| `--low_rank_eps` | float | 0.1 | Epsilon for scaled orthogonal init of V and U projection layers. |

**Baseline guarantee**: When `--use_low_rank 0` (default), the script uses the exact same `SA_encoder`, `G_encoder`, and `Actor` classes from `train.py`. No code paths in the original `train.py` are modified.

## File Structure

```
scaling-crl-42d56645/
├── train.py                          # Original training script (UNMODIFIED)
├── train_low_rank.py                 # Extended training script with low-rank support
├── networks/
│   ├── __init__.py                   # Package exports
│   ├── low_rank_mlp.py               # LowRankMLP: bottleneck residual MLP in Flax
│   └── low_rank_encoders.py          # LowRankSAEncoder, LowRankGEncoder, LowRankActor
├── buffer.py                         # Replay buffer (UNMODIFIED)
├── evaluator.py                      # Evaluation (UNMODIFIED)
└── envs/                             # Environments (UNMODIFIED)
```

## Experiment Design

### Variables

**Controlled Variables** (fixed across all experiments):
- Environment: `ant_big_maze` (primary), `arm_push_easy` (secondary)
- `total_env_steps`: 100,000,000
- `num_epochs`: 100
- `num_envs`: 512
- `batch_size`: 512
- `actor_lr`: 3e-4
- `critic_lr`: 3e-4
- `logsumexp_penalty_coeff`: 0.1
- `critic_network_width`: 256
- `actor_network_width`: 256
- `actor_skip_connections`: 4
- `critic_skip_connections`: 4
- `vis_length`: 1000
- `save_buffer`: 0

**Independent Variables**:
- `use_low_rank`: {0, 1}
- `low_rank_dim`: {32, 64, 128} (when `use_low_rank=1`)
- `critic_depth` / `actor_depth`: {4, 16, 32}
- `low_rank_actor`: {0, 1} (secondary experiment)
- `seed`: {1000, 1001, 1002} (3 seeds per configuration)

**Dependent Variables** (measured outcomes):
- **Time at Goal** (primary metric): fraction of episode the agent spends at the goal
- **Training curves**: learning progress over env steps
- **Wall-clock time**: training time per epoch and total
- **Critic loss**: InfoNCE loss convergence
- **Steps per second (SPS)**: throughput metric

## Run Commands

### Common flags for all ant_big_maze runs:
```bash
COMMON="--env_id ant_big_maze --eval_env_id ant_big_maze_eval \
  --num_epochs 100 --total_env_steps 100000000 \
  --batch_size 512 --num_envs 512 \
  --actor_skip_connections 4 --critic_skip_connections 4 \
  --vis_length 1000 --save_buffer 0"
```

### Experiment 1: Baseline vs Low-Rank at Different Depths (ant_big_maze)

#### Depth 4 — Baseline
```bash
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --seed 1002
```

#### Depth 4 — Low-Rank (r=64)
```bash
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --use_low_rank 1 --low_rank_dim 64 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --use_low_rank 1 --low_rank_dim 64 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 4 --actor_depth 4 --use_low_rank 1 --low_rank_dim 64 --seed 1002
```

#### Depth 16 — Baseline
```bash
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --seed 1002
```

#### Depth 16 — Low-Rank (r=64)
```bash
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --seed 1002
```

#### Depth 32 — Baseline
```bash
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --seed 1002
```

#### Depth 32 — Low-Rank (r=64)
```bash
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --use_low_rank 1 --low_rank_dim 64 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --use_low_rank 1 --low_rank_dim 64 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 32 --actor_depth 32 --use_low_rank 1 --low_rank_dim 64 --seed 1002
```

### Experiment 2: Sweep over low_rank_dim (ant_big_maze, depth 16)

#### r=32
```bash
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 32 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 32 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 32 --seed 1002
```

#### r=64 (already covered in Experiment 1, depth 16)

#### r=128
```bash
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 128 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 128 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 128 --seed 1002
```

### Experiment 3: Low-Rank Actor (ant_big_maze, depth 16, r=64)

#### Low-rank critic only (already in Experiment 1)

#### Low-rank critic AND actor
```bash
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --low_rank_actor 1 --seed 1000
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --low_rank_actor 1 --seed 1001
uv run train_low_rank.py $COMMON --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --low_rank_actor 1 --seed 1002
```

### Experiment 4: Quick Validation on arm_push_easy

```bash
COMMON_ARM="--env_id arm_push_easy --eval_env_id arm_push_easy \
  --num_epochs 100 --total_env_steps 100000000 \
  --batch_size 512 --num_envs 512 \
  --actor_skip_connections 4 --critic_skip_connections 4 \
  --vis_length 1000 --save_buffer 0"

# Baseline depth 16
uv run train_low_rank.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --seed 1000
uv run train_low_rank.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --seed 1001

# Low-rank depth 16 (r=64)
uv run train_low_rank.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --seed 1000
uv run train_low_rank.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_low_rank 1 --low_rank_dim 64 --seed 1001
```

## Expected Wall-Clock Times (per run)

Based on the paper's Table 7 and SLURM job configuration on A100 GPUs:

| Environment | Depth | Approx. Time (baseline) | Approx. Time (low-rank) |
|-------------|-------|------------------------|------------------------|
| ant_big_maze | 4 | ~2 hours | ~1.5–2 hours (fewer params in residual stack) |
| ant_big_maze | 16 | ~6 hours | ~4–5 hours |
| ant_big_maze | 32 | ~12 hours | ~8–10 hours |
| arm_push_easy | 16 | ~3 hours | ~2–3 hours |

**Note**: Low-rank runs may be faster due to smaller matrix multiplications in the residual blocks (r×r instead of m×m).

## Total Experiment Budget

| Experiment | Configurations | Seeds | Total Runs |
|-----------|---------------|-------|------------|
| Exp 1: Depth scaling | 6 (3 depths × 2 modes) | 3 | 18 |
| Exp 2: low_rank_dim sweep | 2 (r=32, r=128) | 3 | 6 |
| Exp 3: Low-rank actor | 1 | 3 | 3 |
| Exp 4: arm_push_easy | 2 | 2 | 4 |
| **Total** | | | **31 runs** |

Estimated total GPU-hours: ~150–200 hours on A100.

## SLURM Job Script Template (NYU Torch HPC)

Save as `job_low_rank.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=crl_lowrank
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time 24:00:00
#SBATCH --output=slurm_logs/slurm-%j.out
#SBATCH --array=0-2

# Map SLURM array task ID to seeds
SEEDS=(1000 1001 1002)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

module purge

# ---- Adjust these variables per experiment ----
ENV_ID="ant_big_maze"
EVAL_ENV_ID="ant_big_maze_eval"
DEPTH=16
USE_LOW_RANK=1
LOW_RANK_DIM=64
LOW_RANK_ACTOR=0
# -----------------------------------------------

if [ "$USE_LOW_RANK" -eq 1 ]; then
    LR_FLAGS="--use_low_rank 1 --low_rank_dim $LOW_RANK_DIM --low_rank_actor $LOW_RANK_ACTOR"
    LR_TAG="_lowrank${LOW_RANK_DIM}"
else
    LR_FLAGS=""
    LR_TAG="_baseline"
fi

echo "Running: env=$ENV_ID depth=$DEPTH seed=$SEED low_rank=$USE_LOW_RANK r=$LOW_RANK_DIM"

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
```

### Submitting experiments:

```bash
# Create logs directory
mkdir -p slurm_logs

# Experiment 1: Baseline depth 16
ENV_ID="ant_big_maze" DEPTH=16 USE_LOW_RANK=0 sbatch job_low_rank.slurm

# Experiment 1: Low-rank depth 16 (r=64)
ENV_ID="ant_big_maze" DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 sbatch job_low_rank.slurm

# Experiment 1: Baseline depth 32
ENV_ID="ant_big_maze" DEPTH=32 USE_LOW_RANK=0 sbatch job_low_rank.slurm

# Experiment 1: Low-rank depth 32 (r=64)
ENV_ID="ant_big_maze" DEPTH=32 USE_LOW_RANK=1 LOW_RANK_DIM=64 sbatch job_low_rank.slurm

# (and so on for each configuration)
```

### Time Limits by Depth

| Depth | Recommended `--time` |
|-------|---------------------|
| 4 | `04:00:00` |
| 16 | `12:00:00` |
| 32 | `24:00:00` |

Adjust `#SBATCH --time` accordingly for each depth to avoid wasting queue allocation.

## Evaluating Results

### Primary Metric: Time at Goal
- Logged to W&B as `eval/episode_success` or similar eval metrics
- Higher is better — measures how effectively the agent reaches and stays at the goal
- Compare mean and std across 3 seeds

### How to Compare
1. **Training curves**: Plot eval metric vs. env_steps for baseline vs. low-rank at each depth
2. **Final performance**: Compare mean ± std of final eval metric across seeds
3. **Training speed**: Compare SPS (steps per second) — low-rank should be faster due to smaller matrices
4. **Depth scaling**: Plot final performance vs. depth for both baseline and low-rank

### Key Comparisons
- At each depth: does low-rank improve over baseline?
- Across depths: does the improvement grow with depth?
- Across r values: which bottleneck dimension works best?
- Low-rank critic only vs. critic+actor: does actor low-rank help or hurt?

### W&B Dashboard Setup
Create a W&B project dashboard with:
- **Panel 1**: Eval metric vs. env_steps, grouped by (depth, use_low_rank)
- **Panel 2**: Final eval metric bar chart with error bars, grouped by configuration
- **Panel 3**: Critic loss vs. env_steps, grouped by configuration
- **Panel 4**: SPS comparison across configurations

## Simplified Initialization Note

The PyTorch reference implementation (`low_rank_res.py`) uses a sophisticated SVD-based initialization procedure:
1. Train a full-rank MLP briefly
2. Compute the gradient of the loss w.r.t. the first layer
3. SVD decompose to find signal subspace
4. Initialize V and U from the top singular vectors

Our JAX implementation uses **epsilon-scaled orthogonal initialization** for the V and U projection layers instead. This is simpler and avoids the complexity of computing gradients in JAX's functional paradigm for a single initialization step. The `--low_rank_eps` flag (default 0.1) controls the scale.

If initial results are promising, a follow-up experiment could implement the full SVD-based initialization in JAX using `jax.grad` and `jnp.linalg.svd`.
