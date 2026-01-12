"""Benchmark POPLIN (policy-augmented PETS) across trajectory optimizers.

This wrapper repeatedly calls ``mbrl.examples.main`` with a shared benchmark override and
writes a small summary CSV with episode reward + planning time.

Example:
  python -m mbrl.examples.scripts.run_poplin_optimizer_benchmark \\
    --envs pets_cartpole pets_inv_pendulum \\
    --optimizers cem decent_cem bccem gmm_cem icem nes cma_es mppi \\
    --variants a p \\
    --seeds 0 1 \\
    --override-expr poplin_optimizer_benchmark \\
    --budget-mode env_round \\
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
DEFAULT_VARIANTS = ["a"]

DEFAULT_OVERRIDE_EXPR = "poplin_optimizer_benchmark"

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


def _normalize_variant(variant: str) -> str:
    v = str(variant).lower()
    if v in {"a", "action"}:
        return "a"
    if v in {"p", "param", "params", "parameter"}:
        return "p"
    raise ValueError(f"Invalid variant {variant!r}. Supported: 'a' or 'p'.")


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
    variant: str,
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
        "algorithm=poplin",
        f"overrides={overrides_name}",
        f"action_optimizer={optimizer}",
        f"algorithm.agent.poplin.variant={variant}",
        f"seed={seed}",
        f"debug_mode={'true' if debug_mode else 'false'}",
        f"hydra.run.dir={run_dir.as_posix()}",
    ]
    cmd += budget_overrides
    cmd += extra_overrides

    print(f"=== env={overrides_name} variant={variant} opt={optimizer} seed={seed} ===")
    print("Command:", _iterable_str(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark POPLIN trajectory optimizers.")
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
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help="POPLIN variants to run: 'a' (action-space) or 'p' (parameter-space).",
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
        help="Worker count to use (and override) when budget-mode is env/env_round.",
    )
    parser.add_argument(
        "--override-expr",
        type=str,
        default=DEFAULT_OVERRIDE_EXPR,
        help="Hydra override_expr name under conf/override_expr.",
    )
    parser.add_argument(
        "--output-root",
        default="exp/poplin_optimizer_benchmark_runs",
        help=(
            "Root directory where run outputs and summary CSV are written. "
            "Relative paths are resolved under the repo root."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug prints in the underlying run.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=None,
        help="Python executable to use.",
    )
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="Additional Hydra overrides to append (e.g., overrides.num_steps=2000).",
    )
    args = parser.parse_args()

    override_expr = str(args.override_expr)
    output_root = _resolve_under_repo(str(args.output_root))
    output_root.mkdir(parents=True, exist_ok=True)

    python = str(args.python) if args.python else sys.executable
    run_id = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_rows: List[Dict[str, object]] = []

    for env_name in args.envs:
        env_budget = _load_env_budget(env_name)
        budget_overrides: List[str] = []
        if args.budget_mode in {"env", "env_round"}:
            pop = int(env_budget["planning_samples_per_iter"])
            iters = int(env_budget["planning_num_iters"])
            workers = int(args.cem_num_workers)
            if args.budget_mode == "env_round":
                pop = _ceil_to_multiple(pop, workers)
            budget_overrides = [
                f"overrides.planning_samples_per_iter={pop}",
                f"overrides.planning_num_iters={iters}",
                f"overrides.cem_num_workers={workers}",
                f"overrides.cem_population_size={pop}",
                f"overrides.cem_population_size_per_worker={pop // max(1, workers)}",
                f"overrides.cem_num_iters={iters}",
                f"overrides.cem_elite_ratio={_hydra_literal(env_budget['cem_elite_ratio'])}",
                f"overrides.cem_alpha={_hydra_literal(env_budget['cem_alpha'])}",
                f"overrides.cem_clipped_normal={_hydra_literal(env_budget['cem_clipped_normal'])}",
                f"overrides.cem_init_jitter_scale={_hydra_literal(env_budget['cem_init_jitter_scale'])}",
            ]

        for variant_raw in args.variants:
            variant = _normalize_variant(variant_raw)
            for optimizer in args.optimizers:
                for seed in args.seeds:
                    run_dir = (
                        output_root
                        / run_id
                        / env_name
                        / variant
                        / optimizer
                        / f"seed{seed}"
                    )
                    run_dir.mkdir(parents=True, exist_ok=True)

                    run_one(
                        python=python,
                        override_expr=override_expr,
                        overrides_name=env_name,
                        optimizer=optimizer,
                        variant=variant,
                        seed=int(seed),
                        run_dir=run_dir,
                        debug_mode=bool(args.debug),
                        extra_overrides=list(args.extra),
                        budget_overrides=budget_overrides,
                        dry_run=bool(args.dry_run),
                    )

                    stats = _read_results_csv(run_dir)
                    summary_rows.append(
                        {
                            "run_id": run_id,
                            "env": env_name,
                            "variant": variant,
                            "optimizer": optimizer,
                            "seed": int(seed),
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
