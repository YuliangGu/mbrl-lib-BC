"""Benchmark PETS trajectory optimizers across environments.

This wrapper repeatedly calls ``mbrl.examples.main`` with a shared benchmark override and
writes a small summary CSV with episode reward + planning time.

Example:
  /home/ygu/miniconda3/envs/mbrl/bin/python -m mbrl.examples.scripts.run_pets_optimizer_benchmark \
    --envs pets_cartpole pets_inv_pendulum \
    --optimizers cem decent_cem bccem gmm_cem icem nes cma_es mppi \
    --seeds 0 1 \
    --override-expr pets_optimizer_benchmark \
    --budget-mode env_round \
    --extra overrides.num_steps=2000
"""

from __future__ import annotations

import argparse
import csv
import datetime as _dt
import math
import pathlib
import subprocess
import sys
from typing import Any, Dict, Iterable, List

from omegaconf import OmegaConf

DEFAULT_OPTIMIZERS = [
    "cem",
    "decent_cem",
    "bccem",
    "gmm_cem",
    "icem",
    "nes",
    "cma_es",
    "mppi",
]
DEFAULT_ENVS = ["pets_cartpole"]

# Keep benchmarking budgets comparable for BCCEM by default.
DEFAULT_BCCEM_BENCH_OVERRIDES = [
    "action_optimizer.early_stop=true",
]
DEFAULT_OVERRIDE_EXPR = "pets_optimizer_benchmark"

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _resolve_under_repo(path: str) -> pathlib.Path:
    p = pathlib.Path(path)
    return (_REPO_ROOT / p) if not p.is_absolute() else p


def _ceil_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 1:
        return int(value)
    value = int(value)
    multiple = int(multiple)
    return ((value + multiple - 1) // multiple) * multiple


def _load_env_budget(overrides_name: str) -> Dict[str, int]:
    cfg_path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "conf"
        / "overrides"
        / f"{overrides_name}.yaml"
    )
    cfg = OmegaConf.load(cfg_path)
    pop = int(cfg.get("cem_population_size"))
    iters = int(cfg.get("cem_num_iters"))
    elite_ratio = float(cfg.get("cem_elite_ratio"))
    alpha = float(cfg.get("cem_alpha"))
    clipped_normal = bool(cfg.get("cem_clipped_normal"))
    init_jitter_scale = float(cfg.get("cem_init_jitter_scale", 0.0))
    return {
        "planning_samples_per_iter": pop,
        "planning_num_iters": iters,
        "cem_elite_ratio": elite_ratio,
        "cem_alpha": alpha,
        "cem_clipped_normal": clipped_normal,
        "cem_init_jitter_scale": init_jitter_scale,
    }


def _hydra_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _read_results_csv(run_dir: pathlib.Path) -> Dict[str, float]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        return {"episodes": 0.0}
    with results_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {"episodes": 0.0}

    def _finite_values(key: str) -> List[float]:
        out: List[float] = []
        for r in rows:
            if key not in r:
                continue
            v_str = r.get(key, "")
            if v_str is None or v_str == "":
                continue
            try:
                v = float(v_str)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                out.append(v)
        return out

    episode_rewards = _finite_values("episode_reward")
    plan_times = _finite_values("planning_time_ms")
    model_steps = _finite_values("planning_model_steps_est")
    replans = _finite_values("planning_replans_per_step")
    skips = _finite_values("planning_skips_per_step")
    centroid_frac = _finite_values("planning_centroid_frac")
    ir_mean = _finite_values("planning_ir_norm_mean")
    ir_min = _finite_values("planning_ir_norm_min")
    ir_max = _finite_values("planning_ir_norm_max")

    def _mean_or_nan(vals: List[float]) -> float:
        return float(sum(vals) / len(vals)) if vals else float("nan")
    return {
        "episodes": float(len(rows)),
        "reward_mean": _mean_or_nan(episode_rewards),
        "reward_max": float(max(episode_rewards)) if episode_rewards else float("nan"),
        "planning_time_ms_mean": _mean_or_nan(plan_times),
        "planning_model_steps_est_mean": _mean_or_nan(model_steps),
        "planning_replans_per_step_mean": _mean_or_nan(replans),
        "planning_skips_per_step_mean": _mean_or_nan(skips),
        "planning_centroid_frac_mean": _mean_or_nan(centroid_frac),
        "planning_ir_norm_mean_mean": _mean_or_nan(ir_mean),
        "planning_ir_norm_min_mean": _mean_or_nan(ir_min),
        "planning_ir_norm_max_mean": _mean_or_nan(ir_max),
    }


def _iterable_str(items: Iterable[object]) -> str:
    return " ".join(str(x) for x in items)


def run_one(
    *,
    python: str,
    override_expr: str,
    overrides_name: str,
    optimizer: str,
    seed: int,
    run_dir: pathlib.Path,
    debug_mode: bool,
    extra_overrides: List[str],
    budget_overrides: List[str],
    dry_run: bool,
) -> None:
    run_dir = run_dir.resolve()
    cmd = [
        python,
        "-m",
        "mbrl.examples.main",
        f"+override_expr={override_expr}",
        f"overrides={overrides_name}",
        f"action_optimizer={optimizer}",
        f"seed={seed}",
        f"debug_mode={'true' if debug_mode else 'false'}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    cmd += budget_overrides
    cmd += extra_overrides

    print(f"=== env={overrides_name} opt={optimizer} seed={seed} ===")
    print("Command:", _iterable_str(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark PETS trajectory optimizers.")
    parser.add_argument(
        "--envs",
        nargs="+",
        default=DEFAULT_ENVS,
        help="Environment override configs (names under conf/overrides, e.g., pets_cartpole).",
    )
    parser.add_argument(
        "-o",
        "--optimizers",
        nargs="+",
        default=DEFAULT_OPTIMIZERS,
        help="Optimizers to run (Hydra action_optimizer names).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Seeds to run.",
    )
    parser.add_argument(
        "--budget-mode",
        choices=["fixed", "env", "env_round"],
        default="fixed",
        help=(
            "How to set per-plan budget. "
            "`fixed` uses values in override_expr. "
            "`env` uses the env's cem_* baseline. "
            "`env_round` rounds samples up to be divisible by cem_num_workers."
        ),
    )
    parser.add_argument(
        "--cem-num-workers",
        type=int,
        default=4,
        help="Worker count to assume when using budget-mode=env_round.",
    )
    parser.add_argument(
        "--override-expr",
        type=str,
        default=DEFAULT_OVERRIDE_EXPR,
        help="Hydra override_expr name under conf/override_expr.",
    )
    parser.add_argument(
        "--output-root",
        default="exp/pets_optimizer_benchmark_runs",
        help=(
            "Root directory where run outputs and summary CSV are written. "
            "Relative paths are resolved under the repo root."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug_mode (prints per-step plan diagnostics; very verbose).",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional Hydra overrides to append (e.g., overrides.num_steps=2000).",
    )
    parser.add_argument(
        "--extra-bccem",
        nargs="*",
        default=[],
        help=(
            "Additional Hydra overrides applied only when optimizer=bccem "
            "(use `action_optimizer.*`; defaults disable early_stop for fair budgets)."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    args = parser.parse_args()

    python = sys.executable
    output_root = _resolve_under_repo(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_rows: List[Dict[str, object]] = []

    for env_name in args.envs:
        budget_overrides: List[str] = []
        if args.budget_mode in ("env", "env_round"):
            budget = _load_env_budget(env_name)
            samples = budget["planning_samples_per_iter"]
            if args.budget_mode == "env_round":
                samples = _ceil_to_multiple(samples, args.cem_num_workers)
            budget_overrides = [
                f"overrides.planning_samples_per_iter={samples}",
                f"overrides.planning_num_iters={budget['planning_num_iters']}",
                f"overrides.cem_elite_ratio={budget['cem_elite_ratio']}",
                f"overrides.cem_alpha={budget['cem_alpha']}",
                f"overrides.cem_clipped_normal={_hydra_literal(budget['cem_clipped_normal'])}",
                f"overrides.cem_init_jitter_scale={budget['cem_init_jitter_scale']}",
            ]

        for optimizer in args.optimizers:
            for seed in args.seeds:
                run_dir = output_root / run_id / env_name / optimizer / f"seed{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                run_one(
                    python=python,
                    override_expr=args.override_expr,
                    overrides_name=env_name,
                    optimizer=optimizer,
                    seed=seed,
                    run_dir=run_dir,
                    debug_mode=args.debug,
                    extra_overrides=args.extra
                    + (
                        DEFAULT_BCCEM_BENCH_OVERRIDES + args.extra_bccem
                        if optimizer == "bccem"
                        else []
                    ),
                    budget_overrides=budget_overrides,
                    dry_run=args.dry_run,
                )

                stats = _read_results_csv(run_dir)
                summary_rows.append(
                    {
                        "run_id": run_id,
                        "env": env_name,
                        "optimizer": optimizer,
                        "seed": seed,
                        **stats,
                    }
                )

    summary_path = output_root / run_id / "summary.csv"
    if summary_rows:
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print("Summary:", summary_path.as_posix())


if __name__ == "__main__":
    main()
