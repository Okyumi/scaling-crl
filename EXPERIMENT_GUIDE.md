# LoRA-Style Low-Rank Factorization for Contrastive RL — Experiment Guide

## Research Context

The paper "Emergent Low-Rank Training Dynamics in MLPs with Smooth Activations" (arxiv 2602.06208) shows that with smooth activations (SiLU, GELU), training dynamics naturally concentrate in low-rank subspaces. With non-smooth activations (ReLU), dynamics spread over higher-dimensional space.

**Hypothesis**: Since ReLU already learns in a wider subspace, explicitly constraining to low-rank with LoRA factorization may help focus learning and improve sample efficiency. With SiLU, the dynamics are already low-rank so the constraint may be redundant.

## Experiment: Three Baselines

1. **CRL + SiLU** (default baseline): Tests whether the existing smooth-activation CRL naturally benefits from emergent low-rank dynamics
2. **CRL + ReLU**: Tests CRL with non-smooth activation — expected to have higher-rank training dynamics
3. **CRL + ReLU + LoRA**: Tests whether explicitly imposing low-rank structure via LoRA factorization helps when using ReLU

## What the LoRA Factorization Does

### Original Full-Rank Architecture (train.py)

The SA_encoder, G_encoder, and Actor all follow this pattern:
```
input → Dense(input_dim, m) → LayerNorm → activation → [residual_block × (depth//4)] → Dense(m, output_dim)
```

Where `residual_block(x, width, normalize, activation)` does:
```
identity = x
x = Dense(width, width) → normalize → activation  (×4)
x = x + identity
```

### LoRA-Factorized Architecture

Each `Dense(m→m)` layer INSIDE the residual blocks is replaced with two factors:
```
x → Dense(m, r) → Dense(r, m)   (i.e., x @ B^T @ A^T = x @ (AB)^T)
```

Key properties:
- The network still operates at full width `m` everywhere (activations, LayerNorm, skip connections are all dimension `m`)
- Only the weight matrices are constrained to be rank-r
- The initial layer `Dense(input→m)` stays full-rank
- The final output layer `Dense(m→64)` or `Dense(m→action_size)` stays full-rank
- This is **fundamentally different** from a narrow bottleneck that reduces the hidden dimension
- Down-projection uses no bias (LoRA convention); up-projection uses bias
- Initialization: `lecun_uniform` for both factors
- Total parameter count per factorized layer: `m² + m` → `mr + rm + m = 2mr + m`

### Architecture Diagram

```
Full-rank residual block:               LoRA-factorized residual block:
  identity = x                            identity = x
  x = Dense(m→m)(x)                      x = Dense(m→r, no bias)(x)
                                          x = Dense(r→m, with bias)(x)
  x = LayerNorm(x)                       x = LayerNorm(x)
  x = activation(x)                      x = activation(x)
  ... (×4 layers)                         ... (×4 layers)
  x = x + identity                       x = x + identity
```

## Config Flags

All flags are in `train_lora.py` and default to preserving baseline behavior:

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--use_low_rank` | int | 0 | Enable LoRA factorization in critic encoder residual blocks. 0=off (baseline), 1=on. |
| `--low_rank_dim` | int | 64 | Low-rank dimension `r`. Only used when `use_low_rank=1`. Each Dense(m,m) in residual blocks becomes Dense(m,r)→Dense(r,m). |
| `--use_relu` | int | 0 | Activation function. 0=SiLU/Swish (default), 1=ReLU. |

**Baseline guarantee**: When `--use_low_rank 0` (default), the script uses the exact same `SA_encoder`, `G_encoder`, and `Actor` classes from `train.py`. No code paths in the original `train.py` are modified.

## File Structure

```
scaling-crl/
├── train.py                          # Original training script (UNMODIFIED)
├── train_lora.py                     # Training script with LoRA support
├── networks/
│   ├── __init__.py                   # Package exports
│   ├── lora_layers.py                # LoRADense: factorized Dense layer
│   └── lora_encoders.py              # LoraSAEncoder, LoraGEncoder, LoraActor
├── buffer.py                         # Replay buffer (UNMODIFIED)
├── evaluator.py                      # Evaluation (UNMODIFIED)
└── envs/                             # Environments (UNMODIFIED)
```

## Controlled Variables

Fixed across all experiments:
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

## Dependent Variables (What to Measure)

- **Time at Goal** (primary metric): fraction of episode the agent spends at the goal
- **Training curves**: episode return over time (env steps)
- **Feature rank of learned critic representations**
- **Wall-clock time**: training time per epoch and total
- **Steps per second (SPS)**: throughput metric

## Run Commands

### Common flags for ant_big_maze:
```bash
COMMON="--env_id ant_big_maze --eval_env_id ant_big_maze_eval \
  --num_epochs 100 --total_env_steps 100000000 \
  --batch_size 512 --num_envs 512 \
  --critic_network_width 256 --actor_network_width 256 \
  --actor_skip_connections 4 --critic_skip_connections 4 \
  --vis_length 1000 --save_buffer 0"
```

### Common flags for arm_push_easy:
```bash
COMMON_ARM="--env_id arm_push_easy --eval_env_id arm_push_easy \
  --num_epochs 100 --total_env_steps 100000000 \
  --batch_size 512 --num_envs 512 \
  --critic_network_width 256 --actor_network_width 256 \
  --actor_skip_connections 4 --critic_skip_connections 4 \
  --vis_length 1000 --save_buffer 0"
```

---

### ant_big_maze — Depth 4

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

### ant_big_maze — Depth 16

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

### ant_big_maze — Depth 32

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

### arm_push_easy — Depth 4

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 4 --actor_depth 4 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

### arm_push_easy — Depth 16

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 16 --actor_depth 16 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

### arm_push_easy — Depth 32

#### Baseline 1: CRL + SiLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 0 --use_low_rank 0 --seed 2
```

#### Baseline 2: CRL + ReLU
```bash
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 0 --seed 2
```

#### Baseline 3: CRL + ReLU + LoRA (r=64)
```bash
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 0
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 1
python train_lora.py $COMMON_ARM --critic_depth 32 --actor_depth 32 --use_relu 1 --use_low_rank 1 --low_rank_dim 64 --seed 2
```

---

## Total Experiment Budget

| Environment | Depths | Baselines | Seeds | Runs |
|-------------|--------|-----------|-------|------|
| ant_big_maze | 3 (4, 16, 32) | 3 (SiLU, ReLU, ReLU+LoRA) | 3 (0, 1, 2) | 27 |
| arm_push_easy | 3 (4, 16, 32) | 3 (SiLU, ReLU, ReLU+LoRA) | 3 (0, 1, 2) | 27 |
| **Total** | | | | **54 runs** |

## SLURM Job Script Template (NYU Torch HPC)

See `job_lora.slurm` for the full template. Example submissions:

```bash
# Create logs directory
mkdir -p slurm_logs

# ant_big_maze, depth 16, CRL + SiLU baseline
ENV_ID=ant_big_maze DEPTH=16 USE_RELU=0 USE_LOW_RANK=0 sbatch job_lora.slurm

# ant_big_maze, depth 16, CRL + ReLU baseline
ENV_ID=ant_big_maze DEPTH=16 USE_RELU=1 USE_LOW_RANK=0 sbatch job_lora.slurm

# ant_big_maze, depth 16, CRL + ReLU + LoRA (r=64)
ENV_ID=ant_big_maze DEPTH=16 USE_RELU=1 USE_LOW_RANK=1 LOW_RANK_DIM=64 sbatch job_lora.slurm

# arm_push_easy, depth 16, CRL + ReLU + LoRA (r=64)
ENV_ID=arm_push_easy EVAL_ENV_ID=arm_push_easy DEPTH=16 USE_RELU=1 USE_LOW_RANK=1 LOW_RANK_DIM=64 sbatch job_lora.slurm
```

Each `sbatch` call submits an array job with 3 tasks (seeds 0, 1, 2).

### Time Limits by Depth

| Depth | Recommended `--time` |
|-------|---------------------|
| 4 | `04:00:00` |
| 16 | `12:00:00` |
| 32 | `24:00:00` |

Adjust `#SBATCH --time` accordingly for each depth.

## Implementation Details

- **LoRA factorization**: Each `Dense(m→m)` in residual blocks → `Dense(m→r, no bias)` + `Dense(r→m, with bias)`
- **Down-projection uses no bias** (LoRA convention)
- **Up-projection uses bias**
- **Initialization**: `lecun_uniform = variance_scaling(1/3, "fan_in", "uniform")` for both factors
- **Parameter count**: Per factorized layer goes from `m² + m` to `mr + rm + m = 2mr + m`
  - For m=256, r=64: from 65,792 → 32,832 params per layer (50% reduction)
  - For a depth-16 network with 4 residual blocks × 4 layers = 16 factorized layers: ~527K fewer params

## Evaluating Results

### Key Comparisons

1. **CRL + SiLU vs CRL + ReLU**: Does the activation function matter for CRL training dynamics?
2. **CRL + ReLU vs CRL + ReLU + LoRA**: Does explicit low-rank constraint help when using ReLU?
3. **Depth scaling**: Do the above effects change at different network depths?

### W&B Dashboard Setup

Create a W&B project dashboard with:
- **Panel 1**: Eval metric vs. env_steps, grouped by (depth, baseline_type)
- **Panel 2**: Final eval metric bar chart with error bars, grouped by configuration
- **Panel 3**: Critic loss vs. env_steps, grouped by configuration
- **Panel 4**: SPS comparison across configurations
