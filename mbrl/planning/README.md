# BC-CEM (`BCCEMOptimizer`)

`mbrl.planning.BCCEMOptimizer` is a multi-worker Cross-Entropy Method (CEM) optimizer for MPC-style action-sequence optimization. It runs several independent CEM “workers”, forms a **centroid** distribution across workers, and tracks an diversity index, **information radius** (`ir_norm`) to decide when returning the centroid is safe vs. when to sample from multiple modes.

## Quick start (Hydra)

Run PETS with BC-CEM (replace the env override as needed):

```bash
python -m mbrl.examples.main \
  algorithm=pets overrides=pets_cartpole \
  action_optimizer=bccem \
  overrides.cem_num_workers=4 \
  overrides.cem_population_size=360 \
  overrides.cem_population_size_per_worker=90
```

## Tunable parameters

BC-CEM inherits the usual CEM parameters and adds consensus / IR-based execution.

### CEM basics

- `num_iterations`: number of CEM generations.
- `elite_ratio`: fraction of samples per worker used as elites.
- `population_size`: samples per worker per iteration.
- `total_population_size` (optional): if set, overrides `population_size` by splitting the total budget across workers.
- `num_workers`: number of parallel CEM workers.
- `alpha`: EMA / momentum for distribution updates.
- `clipped_normal`: if `true`, samples from a normal distribution and clamps to bounds; if `false`, uses truncated normal.
- `init_jitter_scale`: adds noise to the initial mean to reduce early collapse (useful when `num_workers > 1`).

### Consensus + worker weighting

- `use_value_weights`: if `true`, weights workers by a softmax over their per-iteration value statistics (currently the per-worker median score); if `false`, uses uniform weights.
- `consensus_coef`: base strength of pulling worker distributions toward the centroid (annealed over CEM iterations).
- `consensus_anneal_power`: anneals consensus strength over iterations; effective beta scales like `((i+1)/num_iterations)^power`.

Note: the current per-worker consensus scaling is `beta_w = beta / (1 + kl_w / dim)`, i.e., workers that are already close to the centroid are pulled more strongly.

### Execution / warm-start behavior

- `return_mean_elites`: if `false`, returns the best sampled sequence (like standard CEM-family optimizers).
  If `true`, uses `ir_norm` to choose between returning the centroid mean vs. a worker mean sampled from the worker weights.
- `ir_low` / `ir_high`: thresholds (on `ir_norm`) controlling the centroid-vs-worker decision when `return_mean_elites=true`.
- `ir_high_target` (optional): if set, some training loops may anneal `ir_high` toward this target over total env steps (see below).

Warm-start: `TrajectoryOptimizer` always warm-starts from the returned/executed sequence (shifted by `replan_freq`).

### Early stopping

- `early_stop`: enables early stopping when both IR and centroid motion are small.
- `early_stop_ir`: IR threshold for early-stop.
- `early_stop_mu`: RMS centroid-mean change threshold for early-stop.
- `early_stop_patience`: number of consecutive “small-change” iterations before stopping.

## Interaction with `TrajectoryOptimizer` / `TrajectoryOptimizerAgent`

- `TrajectoryOptimizer` instantiates the underlying optimizer via Hydra and calls `optimizer.optimize(..., x0=previous_solution)`.
- `TrajectoryOptimizerAgent` plans an action sequence and caches it for `replan_freq` environment steps. The warm-start shift is handled in `TrajectoryOptimizer` when `keep_last_solution=true`.
- If `skip_replan_if_ir_low=true`, `TrajectoryOptimizerAgent` can extend the cached plan (skip replanning) when `agent.last_plan_debug["ir_norm"] <= skip_replan_ir_threshold`.
- If `particle_schedule="ir"` in the agent config, `create_trajectory_optim_agent_for_model()` adapts the model rollout particle count based on the *previous* plan’s `ir_norm` (using `ir_particles_low/high` and `particles_min_frac`).

## Config grouping (`bcem_params`)

`BCCEMOptimizer` accepts either:

- flat Hydra keys (e.g., `action_optimizer.ir_high=...`), or
- a single `bcem_params: <BCCEMParams>` object (Hydra-instantiated dataclass; `BCCEMParams` lives at `mbrl.planning.trajectory_opt.BCCEMParams`).

If `bcem_params` is provided, it takes precedence over the flat keys.

Example (nested dataclass):

```yaml
_target_: mbrl.planning.BCCEMOptimizer
bcem_params:
  _target_: mbrl.planning.trajectory_opt.BCCEMParams
  ir_low: 0.3
  ir_high: 1.0
  ir_high_target: 0.4
```
