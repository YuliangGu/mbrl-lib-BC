# Scripts

Helper entry points for benchmarking and quick experiments.

## Available scripts

- `run_pets_optimizer_sweep.py`: convenience wrapper around `mbrl.examples.main` to sweep PETS across dynamics models and action optimizers. Uses `conf/override_expr/pets_optimizer_benchmark.yaml` by default. Example:
  ```bash
  python mbrl/examples/scripts/run_pets_optimizer_sweep.py \
    --overrides=pets_halfcheetah \
    --dynamics-models=mlp,gaussian_mlp_ensemble \
    --optimizers=cem,bccem,nes \
    --seeds=0,1
  ```
  Add `--dry-run` to print commands without executing, and `--fail-fast` to stop on the first error.

- `benchmark_analytic_optimizers.py`: runs CEM/DecentCEM/BCCEM/GMMCEM/CMA-ES/NES on analytic objectives (Rosenbrock or Rastrigin). Good for quick sanity checks without an environment. Example:
  ```bash
  python mbrl/examples/scripts/benchmark_analytic_optimizers.py \
    --objective=rastrigin \
    --optimizers=cem,bccem,nes \
    --plot=rastrigin.png
  ```
  Adjust `--num-iters`, `--per-iter-budget`, and other flags to tune the benchmark.

All scripts are runnable from the repo root without installing the package; they add the repo to `PYTHONPATH` internally.
