# BCCEMOptimizer (BC-CEM)

`mbrl.planning.BCCEMOptimizer` is a multi-worker Cross-Entropy Method (CEM) optimizer for MPC-style action-sequence optimization. It runs several independent CEM “workers”, forms a moment-matched **centroid** distribution across workers, and uses an **information-radius** diagnostic (weighted KL to the centroid) to decide when returning the centroid is safe vs. when to fall back to a single best mode.

This optimizer is designed to be used through `TrajectoryOptimizer` / `TrajectoryOptimizerAgent` (Hydra configs in `mbrl/examples/conf/action_optimizer/bccem.yaml`).

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

BC-CEM-specific tuning can be done by overriding `action_optimizer.*`, e.g.:

```bash
python -m mbrl.examples.main \
  algorithm=pets overrides=pets_cartpole \
  action_optimizer=bccem \
  action_optimizer.ir_low=0.1 \
  action_optimizer.ir_high=1.0 \
  action_optimizer.consensus_coef=0.2
```

## Tunable parameters

BC-CEM inherits the usual CEM parameters and adds consensus / IR-based execution.

### Budget and CEM basics

- `num_iterations`: number of CEM generations.
- `elite_ratio`: fraction of samples per worker used as elites.
- `population_size`: samples per worker per iteration.
- `total_population_size` (optional): if set, overrides `population_size` by splitting the total budget across workers.
- `num_workers`: number of parallel CEM workers.
- `alpha`: EMA / momentum for distribution updates (higher = smoother, lower = faster adaptation).
- `clipped_normal`: if `true`, samples from a normal distribution and clamps to bounds; if `false`, uses truncated normal.
- `init_jitter_scale`: adds noise to the initial mean to reduce early collapse (useful when `num_workers > 1`).

### Consensus + worker weighting

- `use_value_weights`: if `true`, weights workers by a softmax over their per-iteration value statistics; if `false`, uses uniform weights.
- `consensus_coef`: base strength of pulling worker distributions toward the centroid (scaled per-worker by KL-to-centroid: farther => stronger).
- `consensus_anneal_power`: anneals consensus strength over iterations; effective beta scales like `((i+1)/num_iterations)^power`.

### Execution / warm-start behavior

- `return_mean_elites`: if `false`, returns the best sampled sequence (like standard CEM-family optimizers).
  If `true`, uses the information-radius `ir_norm` to choose between returning the centroid mean vs. a sampled worker mean.
- `ir_low` / `ir_high`: thresholds (on `ir_norm`) controlling the centroid-vs-sampled decision when `return_mean_elites=true`.
- Warm-start: `TrajectoryOptimizer` always warm-starts from the returned/executed sequence (shifted by `replan_freq`).

### Early stopping

- `early_stop`: enables early stopping when both IR and centroid motion are small.
- `early_stop_ir`: IR threshold for early-stop.
- `early_stop_mu`: RMS centroid-mean change threshold for early-stop.
- `early_stop_patience`: number of consecutive “small-change” iterations before stopping.

## Diagnostics (for logging and adaptive evaluation)

`BCCEMOptimizer.get_diagnostics()` returns a JSON-serializable dict that typically includes:

- `weights`: current worker weights used for centroid formation.
- `best_worker_idx`: index of best worker (by best value this call).
- `ir_norm`, `ir_avg`, `ir_max`: information-radius diagnostics (used by `TrajectoryOptimizerAgent` when `particle_schedule=ir`).
- `exec_source`: provenance of the returned solution (`best` / `centroid` / `sampled` / `best_worker_mean`).

When using `create_trajectory_optim_agent_for_model()`, the agent augments `last_plan_debug` with evaluation accounting:
`eval_calls`, `eval_sequences`, and `rollouts_est`.
