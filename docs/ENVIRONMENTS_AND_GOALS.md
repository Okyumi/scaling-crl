# Codebase Analysis: Environments, Goals, Wrappers, and Adding New Envs

## 1. How Environments Are Initialized

Environments are created in **`train.py`** inside **`make_env(env_id)`** (lines 318–504):

1. **Dispatch by `env_id`**  
   A long `if/elif` chain matches `env_id` (e.g. `"reacher"`, `"ant"`, `"arm_reach"`, `"ant_big_maze"`) and:
   - Imports the right env class from `envs/` (or `envs/manipulation/`).
   - Instantiates it with fixed kwargs (e.g. `backend="spring"` or `"mjx"`).
   - Sets **`args.obs_dim`**, **`args.goal_start_idx`**, **`args.goal_end_idx`** for that env.

2. **Brax base**  
   All envs inherit from **`brax.envs.base.PipelineEnv`**. They load a MuJoCo system (from XML), call `super().__init__(sys=sys, backend=backend, ...)`, and implement `reset(rng)` and `step(state, action)`.

3. **Wrapping**  
   The raw env is then wrapped and JIT'd:

   ```python
   env = make_env()
   env = envs.training.wrap(env, episode_length=args.episode_length)
   env_state = jax.jit(env.reset)(env_keys)
   env.step = jax.jit(env.step)
   ```

So: **initialization = `make_env(env_id)` (local env class + args) → `brax.envs.training.wrap` → JIT `reset`/`step`.**

---

## 2. Where Environment Source and Assets Live

- **Environment source code**  
  All env logic lives **in this repo** under:
  - `envs/*.py` (e.g. `reacher.py`, `ant.py`, `ant_maze.py`, `humanoid.py`, `pusher.py`, `ant_ball.py`, `ant_push.py`, …)
  - `envs/manipulation/*.py` (e.g. `arm_reach.py`, `arm_binpick_easy.py`, `arm_push_easy.py`, `arm_grasp.py`, …)
  - Base class for manipulation: `envs/manipulation/arm_envs.py` (`ArmEnvs`).

- **Assets (XML, meshes)**  
  Stored under **`envs/assets/`** in this repo:
  - Local XMLs: `ant.xml`, `ant_maze.xml`, `ant_ball.xml`, `ant_push.xml`, `humanoid.xml`, `humanoid_maze.xml`, `half_cheetah.xml`, `simple_maze.xml`, `panda_*.xml`, etc.
  - Franka mesh assets: `envs/assets/franka_emika_panda/`.

- **Referenced from Brax (not in this repo)**  
  A few envs load XML from **Brax's package** via `epath.resource_path("brax")`:
  - **Reacher**: `brax/envs/assets/reacher.xml` (see `envs/reacher.py`).
  - **Pusher**: `brax/envs/assets/pusher.xml` (see `envs/pusher.py`).

So: **env source is local under `envs/`; most assets are under `envs/assets/`; only reacher and pusher use Brax's bundled XMLs.**

---

## 3. How the Goal Is Defined and Obtained (by Env Type)

### Reacher, Pusher, Ant, Humanoid (point / target goal)

- **Goal**  
  A fixed target (position or similar) for the episode. Stored in physics state (e.g. `q` or body position) and/or computed in `_get_obs`.

- **Where it's defined/obtained**  
  - **Reacher** (`envs/reacher.py`): `_random_target()` samples target on a circle; target is written into `q[2:]` and read as `pipeline_state.x.pos[2]`. Observation is built in `_get_obs()` as `[state..., target_pos]`; goal indices in `train.py`: `goal_start_idx=4`, `goal_end_idx=7`.
  - **Pusher** (`envs/pusher.py`): Goal position set in `reset()` into `qpos`; observation includes object and goal; `goal_indices` 10–12; `train.py`: `goal_start_idx=10`, `goal_end_idx=13`.
  - **Ant** (`envs/ant.py`): `_random_target()` gives (x,y) on a circle; target in `q[-2:]` and `pipeline_state.x.pos[-1][:2]`. `_get_obs()` returns `[qpos, qvel, target_pos]`. `train.py`: `goal_start_idx=0`, `goal_end_idx=2`.
  - **Humanoid** (`envs/humanoid.py`): Same idea: `_random_target()`; target in `qpos`; `_get_obs()` includes it; `train.py`: `goal_start_idx=0`, `goal_end_idx=3`.

So for these: **goal = target state (e.g. position) set at reset and appended to obs; indices in `train.py` tell the algorithm which slice of `obs` is the goal.**

### Ant maze

- **Goal**  
  One of a discrete set of maze cell positions (e.g. "goal" cells in the layout).

- **Where**  
  - **`envs/ant_maze.py`**: `make_maze()` builds the layout and `find_goals()` returns `possible_goals` (cell coordinates). `AntMaze._random_target()` picks one: `jax.random.randint(..., 0, len(self.possible_goals))` and sets `q[-2:]` to that goal. `_get_obs()` returns `[qpos, qvel, target_pos]` with `target_pos = pipeline_state.x.pos[-1][:2]`. Same `goal_start_idx=0`, `goal_end_idx=2` in `train.py`.

### Ant ball

- **Goal**  
  Ball (object) target position.

- **Where**  
  `_random_target()` returns object and target; both written into `q`; obs includes object and target; `train.py`: `goal_start_idx=28`, `goal_end_idx=30`.

### Manipulation tasks (Arm*)

- **Goal**  
  Defined **per task**: reach target (EEF or cube), bin goal, push goal, or grasp (finger positions + openness). It's a vector (e.g. 3D or 7D) **stored in `state.info["goal"]`** and **appended to the observation** in `_get_obs()`.

- **Where**  
  - **Base** (`envs/manipulation/arm_envs.py`):  
    - In `reset()`: `goal = self._get_initial_goal(pipeline_state, subkey1)`, then `info = {..., "goal": goal}`.  
    - In `step()`: `obs = self._get_obs(pipeline_state, state.info["goal"], timestep)`.  
    So the goal is **obtained from `state.info["goal"]`** every step; only `reset()` and optional `update_goal()` set it.
  - **Per-task** (each `envs/manipulation/arm_*.py`):
    - **`_get_initial_goal(pipeline_state, rng)`**: samples the goal (e.g. random 3D position in a box for reach/binpick/push, or finger positions + openness for grasp).
    - **`_get_obs(..., goal, timestep)`**: returns `jnp.concatenate([state_features..., goal])`.
    - **`_compute_goal_completion(obs, goal)`**: success thresholds (e.g. distance to goal).
    - **`_update_goal_visualization(pipeline_state, goal)`**: moves a visual marker in the sim.

  **Goal indices in `train.py`** (slice of full obs = state + goal):
  - **arm_reach**: goal 7:10 (EEF goal position).
  - **arm_binpick_easy/hard, arm_push_easy/hard**: goal 0:3 (cube position).
  - **arm_binpick_easy_EEF**: goal 0:3.
  - **arm_grasp**: goal 16:23 (fingertip positions + openness).

So for manipulation: **goal is in `state.info["goal"]`, appended to obs by `_get_obs`; task-specific logic lives in `_get_initial_goal` / `_get_obs` / `_compute_goal_completion` in each `envs/manipulation/arm_*.py`.**

---

## 4. What You Need to Train on a New Env (e.g. MetaWorld Sawyer)

To add something like **MetaWorld Sawyer** you'd do the following, mirroring the existing manipulation stack.

### 4.1 Important constraints

- The codebase is **JAX/Brax** and **vectorized** (batch of envs, `jax.jit(env.reset)`, `jax.jit(env.step)`). MetaWorld is **PyTorch/Gym** and typically single-env. So you either:
  - **Option A**: Implement a **Brax/JAX env** that mimics the Sawyer task (own physics in Brax/MuJoCo), or  
  - **Option B**: Add a **wrapper/adapter** that runs MetaWorld (or a reimplementation) and exposes a Brax-like, JAX-friendly API (e.g. convert to JAX arrays, vmap over multiple envs). Option B is heavier and may require stepping out of JAX for physics.

Below assumes **Option A** (new Brax-style env in this repo), which fits the current architecture.

### 4.2 New files to add

1. **Env class**  
   - **Path**: e.g. `envs/manipulation/arm_sawyer_reach.py` (or under a new `envs/metaworld/` if you prefer).  
   - **Pattern**: Same as `arm_reach.py` / `arm_binpick_easy.py`: inherit from `ArmEnvs`, implement:
     - `_get_xml_path()` → return path to a MuJoCo XML for Sawyer (you'd add this under `envs/assets/` or use a path you load).
     - `_set_environment_attributes()`: `env_name`, `episode_length`, `goal_indices`, `completion_goal_indices`, `state_dim`, noise scales.
     - `_get_initial_state(rng)`, `_get_initial_goal(pipeline_state, rng)`.
     - `_get_obs(pipeline_state, goal, timestep)` → must return **state features concatenated with goal** so that `observation_size` = state_dim + goal_dim.
     - `_compute_goal_completion(obs, goal)` → return `(success, success_easy, success_hard)`.
     - `_update_goal_visualization(pipeline_state, goal)`.
     - `_get_arm_angles(pipeline_state)` (and, if you use EEF control, action conversion like in `arm_binpick_easy_EEF`).
   - If the robot/action space differs from Franka, you can still mirror **`arm_envs.py`** but optionally subclass a slimmer base (only the parts that are shared) and override `action_size` and action conversion.

2. **Assets**  
   - **Path**: e.g. `envs/assets/sawyer_reach.xml` (and any meshes).  
   - MuJoCo XML for Sawyer (from MetaWorld or your own), compatible with Brax's `mjcf.load()` (see existing `panda_*.xml` and `arm_*.py`).

### 4.3 Integration in `train.py`

In **`make_env()`**, add a branch for your env_id, e.g. `"sawyer_reach"`:

```python
elif env_id == "sawyer_reach":
    from envs.manipulation.arm_sawyer_reach import ArmSawyerReach  # or your module path
    env = ArmSawyerReach(backend="mjx")
    args.obs_dim = <state_dim>
    args.goal_start_idx = <start>
    args.goal_end_idx = <end>
```

- **`obs_dim`**: length of the **state** part of the observation only (what the actor/critic use as "state" before concatenating goal; see `buffer.flatten_crl_fn` and critic/actor usage).
- **`goal_start_idx` / `goal_end_idx`**: slice of the **full** observation that is the goal (same convention as other arm envs: obs = state || goal).

### 4.4 Existing components to mirror / plug into

- **`envs/manipulation/arm_envs.py`**: Base `reset()` / `step()` / `update_goal()`; seed and `info["goal"]` handling; action conversion (joint or EEF). Your env should behave like `ArmReach` / `ArmBinpickEasy`: goal in `info["goal"]`, obs = `_get_obs(..., goal, timestep)`.
- **`train.py`**:  
  - `make_env()`: env creation and **args.obs_dim, goal_start_idx, goal_end_idx**.  
  - Uses **`env.observation_size`** and **`env.action_size`** (Brax/PipelineEnv convention; override `action_size` if needed like other arm envs).  
  - Buffer and CRL logic assume **observation = state || goal** and use **args.obs_dim** and **args.goal_start_idx/end** (e.g. in `buffer.flatten_crl_fn` and actor/critic).
- **`evaluator.py`**: Expects **`state.info["eval_metrics"]`** with **`episode_metrics`** containing at least one of `reward`, `success`, `success_easy`, `success_hard`, `dist`, `distance_from_origin`. Your env doesn't set these directly; they are added by **`envs.training.EvalWrapper`** (Brax). So you only need to keep using the same **metrics** in your env's `step()` (e.g. `success`, `success_easy`, `success_hard`) so that the wrapper can aggregate them.

So: **new env class + assets + one branch in `make_env()`; observation = state || goal and correct goal indices; optional `action_size` override.**

---

## 5. How `reset()` and `step()` Are Wrapped and What They Return

### 5.1 Wrapper: `brax.envs.training.wrap(env, episode_length=...)`

- **Source**: From the **Brax** library (`from brax import envs`; `envs.training.wrap`), not from this repo. So the exact behavior is that of **Brax 0.10.1**.
- **Typical Brax training wrap** (from common Brax usage and your code):
  - **reset**: Same signature `reset(rng)`; returns a **State** that the rest of the code treats like the base env's State (same fields).
  - **step**: Same signature `step(state, action)`; returns a **State**. The wrapper usually:
    - Increments an internal step counter (or uses `state.info["steps"]`).
    - Sets **truncation** when the step counter reaches **episode_length** (time limit), and may auto-reset or set `done`/truncation so that the training loop sees episode boundaries.

So: **the wrapper does not change the State type or the meaning of `obs`/`reward`/`done`; it adds time limiting and likely extra `info` fields used for truncation and evaluation.**

### 5.2 Does the wrapper change what the base env returns?

- **Observation, reward, pipeline_state**: No. The core `obs`, `reward`, and physics state come from the underlying env's `reset`/`step`. The wrapper only adds/modifies **info** (e.g. **`truncation`**, **`steps`**) and possibly **done** for timeout.
- **Your code** explicitly expects **`state.info["truncation"]`** and **`state.info["seed"]`** (e.g. `extra_fields=("truncation", "seed")` when building transitions). So the wrapper (or the base envs in this repo) are responsible for populating **`info["truncation"]`** and **`info["seed"]`**; the wrapper is what adds time-based **truncation**.

So: **the wrapper does not change obs/reward from the base env; it adds/updates info (e.g. truncation, steps) and possibly done.**

### 5.3 What `reset` and `step` return in this codebase

Both return a **Brax `State`** (or the same structure). After wrapping, the object has:

- **`state.pipeline_state`**: Brax pipeline state (positions, velocities, etc.).
- **`state.obs`**: **Observation**; in this codebase it is always **state features concatenated with goal** (e.g. `[state_dim]` or `[state_dim + goal_dim]`). So **full obs = state || goal**; `args.obs_dim` is the state part; goal is `obs[goal_start_idx:goal_end_idx]`.
- **`state.reward`**: Scalar reward for the last step.
- **`state.done`**: Done flag (e.g. 0/1).
- **`state.metrics`**: Dict of scalars (e.g. `success`, `success_easy`, `dist`, …) used for logging and evaluation.
- **`state.info`**: Dict; in this codebase it always includes:
  - **`"seed"`**: Used as trajectory/episode id for CRL (future goal sampling in the buffer).
  - **`"truncation"`**: Set by the training wrapper when the episode is truncated by **episode_length** (time limit).
  - For manipulation envs: **`"goal"`** (goal vector) and **`"timestep"`**.
  - After **EvalWrapper** (evaluator): **`"eval_metrics"`** with **`episode_metrics`** (e.g. reward, success, success_easy, success_hard, dist) and **`episode_steps`**.

So, in one line:

- **`reset(rng)`** returns a **State** with **obs = state || goal**, **reward/done/metrics** for the initial step, and **info** containing at least **seed** and (after wrap) **truncation**.
- **`step(state, action)`** returns a **State** with **obs** (state || goal), **reward**, **done**, **metrics**, and **info** (seed, truncation, and for manipulation goal/timestep); the wrapper adds time-based truncation and the evaluator adds **eval_metrics**.
