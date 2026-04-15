# Submit All Low-Rank CRL Experiments (Individual Jobs)

Run from the repository root:

```bash
mkdir -p slurm_logs
```

## Experiment 1 — Depth scaling on `ant_big_maze` (18 runs)

```bash
# Depth 4 baseline
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=0 SEED=1002 sbatch job_low_rank.slurm

# Depth 4 low-rank r=64
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=4 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1002 sbatch job_low_rank.slurm

# Depth 16 baseline
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=0 SEED=1002 sbatch job_low_rank.slurm

# Depth 16 low-rank r=64
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1002 sbatch job_low_rank.slurm

# Depth 32 baseline
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=0 SEED=1002 sbatch job_low_rank.slurm

# Depth 32 low-rank r=64
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=32 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1002 sbatch job_low_rank.slurm
```

## Experiment 2 — `low_rank_dim` sweep on `ant_big_maze`, depth 16 (6 runs)

```bash
# r=32
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=32 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=32 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=32 LOW_RANK_ACTOR=0 SEED=1002 sbatch job_low_rank.slurm

# r=128
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=128 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=128 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=128 LOW_RANK_ACTOR=0 SEED=1002 sbatch job_low_rank.slurm
```

## Experiment 3 — low-rank actor on `ant_big_maze`, depth 16, r=64 (3 runs)

```bash
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=1 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=1 SEED=1001 sbatch job_low_rank.slurm
ENV_ID=ant_big_maze EVAL_ENV_ID=ant_big_maze_eval DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=1 SEED=1002 sbatch job_low_rank.slurm
```

## Experiment 4 — quick validation on `arm_push_easy` (4 runs)

```bash
# Baseline depth 16
ENV_ID=arm_push_easy EVAL_ENV_ID=arm_push_easy DEPTH=16 USE_LOW_RANK=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=arm_push_easy EVAL_ENV_ID=arm_push_easy DEPTH=16 USE_LOW_RANK=0 SEED=1001 sbatch job_low_rank.slurm

# Low-rank depth 16 r=64
ENV_ID=arm_push_easy EVAL_ENV_ID=arm_push_easy DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1000 sbatch job_low_rank.slurm
ENV_ID=arm_push_easy EVAL_ENV_ID=arm_push_easy DEPTH=16 USE_LOW_RANK=1 LOW_RANK_DIM=64 LOW_RANK_ACTOR=0 SEED=1001 sbatch job_low_rank.slurm
```
