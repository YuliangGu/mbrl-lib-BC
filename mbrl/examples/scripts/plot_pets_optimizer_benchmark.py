#!/usr/bin/env python3
"""Aggregate and plot PETS optimizer benchmark results."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

# matplotlib is only needed when actually plotting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot PETS optimizer benchmark results from exp/."
    )
    parser.add_argument(
        "--exp-root",
        type=Path,
        default=Path("exp"),
        help="Root experiments directory (default: ./exp).",
    )
    parser.add_argument(
        "--experiment",
        default="pets_optimizer_benchmark",
        help="Experiment name used in the Hydra run.",
    )
    parser.add_argument(
        "--env",
        default="cartpole_continuous",
        help="Environment name to filter (matches overrides.env).",
    )
    parser.add_argument(
        "--metric",
        choices=["episode_reward", "planning_time_ms", "training_curve"],
        default="episode_reward",
        help="Metric to aggregate/plot.",
    )
    parser.add_argument(
        "--agg",
        choices=["max", "last", "mean"],
        default="mean",
        help="Reduction over each run's timeseries (for episode_reward use max/last; for planning_time_ms use mean).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("pets_optimizer_benchmark.png"),
        help="Path to save the plot.",
    )
    parser.add_argument(
        "--show-dynamics",
        action="store_true",
        help="Include dynamics model name in the plot labels.",
    )
    return parser.parse_args()


def find_runs(exp_root: Path, experiment: str, env: str) -> List[Path]:
    root = exp_root / "pets" / experiment / env
    return list(root.rglob(".hydra/config.yaml"))


def load_config(config_path: Path) -> Dict:
    with config_path.open("r") as f:
        return yaml.safe_load(f)


def load_metric(
    run_dir: Path, metric: str, agg: str
) -> Tuple[float, int, float, List[Tuple[float, float]]]:
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing results.csv at {results_path}")

    values: List[float] = []
    curve: List[Tuple[float, float]] = []
    with results_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No columns found in {results_path}")
        if metric != "training_curve" and metric not in reader.fieldnames:
            raise KeyError(f"Metric '{metric}' not found in {results_path}")
        for row in reader:
            try:
                env_step = float(row.get("env_step", len(values)))
            except (TypeError, ValueError):
                env_step = float(len(values))
            if metric == "training_curve":
                try:
                    reward = float(row.get("episode_reward", "nan"))
                except (TypeError, ValueError):
                    continue
                curve.append((env_step, reward))
            else:
                try:
                    values.append(float(row[metric]))
                except (TypeError, ValueError):
                    continue

    if metric == "training_curve":
        if not curve:
            raise ValueError(f"No training curve values in {results_path}")
        # Use final reward as summary statistic for sorting/labels
        final_val = curve[-1][1]
        return final_val, len(curve), final_val, curve

    if not values:
        raise ValueError(f"No numeric values for {metric} in {results_path}")

    if agg == "max":
        val = max(values)
    elif agg == "last":
        val = values[-1]
    else:
        val = sum(values) / len(values)
    return val, len(values), sum(values) / len(values), curve


def summarize_runs(
    run_configs: List[Tuple[Path, Dict]], metric: str, agg: str
) -> Tuple[Dict[Tuple[str, str], List[float]], Dict[Tuple[str, str], List[List[Tuple[float, float]]]]]:
    grouped_vals: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    grouped_curves: Dict[Tuple[str, str], List[List[Tuple[float, float]]]] = defaultdict(list)
    for cfg_path, cfg in run_configs:
        run_dir = cfg_path.parent.parent  # .../<time>/
        opt_target = cfg.get("action_optimizer", {}).get("_target_", "unknown")
        dyn_target = cfg.get("dynamics_model", {}).get("_target_", "unknown")
        opt_name = opt_target.split(".")[-1]
        dyn_name = dyn_target.split(".")[-1]
        try:
            value, _, _, curve = load_metric(run_dir, metric, agg)
        except (FileNotFoundError, KeyError, ValueError):
            continue
        grouped_vals[(opt_name, dyn_name)].append(value)
        if metric == "training_curve":
            grouped_curves[(opt_name, dyn_name)].append(curve)
    return grouped_vals, grouped_curves


def plot_groups(
    grouped: Dict[Tuple[str, str], List[float]],
    include_dyn: bool,
    metric: str,
    out_path: Path,
    env: str,
    experiment: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    labels: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    for (opt, dyn), vals in sorted(
        grouped.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True
    ):
        label = opt if not include_dyn else f"{opt}\n({dyn})"
        labels.append(label)
        means.append(float(sum(vals) / len(vals)))
        stds.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, means, yerr=stds, color="#4C72B0", alpha=0.9, capsize=4)
    ax.set_ylabel(metric)
    ax.set_title(f"{experiment} on {env}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if include_dyn else 20, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    for rect, mean in zip(bars, means):
        ax.text(
            rect.get_x() + rect.get_width() / 2.0,
            rect.get_height(),
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def plot_training_curves(
    grouped_curves: Dict[Tuple[str, str], List[List[Tuple[float, float]]]],
    include_dyn: bool,
    out_path: Path,
    env: str,
    experiment: str,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(10, 6))
    for (opt, dyn), curves in grouped_curves.items():
        label = opt if not include_dyn else f"{opt} ({dyn})"
        # simple mean curve over runs
        # align by env_step using interpolation on a common grid
        all_steps = sorted({step for curve in curves for step, _ in curve})
        if not all_steps:
            continue
        grid = np.array(all_steps, dtype=float)
        interp_vals = []
        for curve in curves:
            steps = np.array([s for s, _ in curve], dtype=float)
            vals = np.array([v for _, v in curve], dtype=float)
            interp_vals.append(np.interp(grid, steps, vals))
        mean_vals = np.mean(np.stack(interp_vals, axis=0), axis=0)
        std_vals = np.std(np.stack(interp_vals, axis=0), axis=0) if len(curves) > 1 else None
        plt.plot(grid, mean_vals, label=label)
        if std_vals is not None:
            plt.fill_between(grid, mean_vals - std_vals, mean_vals + std_vals, alpha=0.2)

    plt.xlabel("env_step")
    plt.ylabel("episode_reward")
    plt.title(f"{experiment} training curves on {env}")
    plt.legend()
    plt.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved training curves to {out_path}")


def main() -> None:
    args = parse_args()
    cfg_paths = find_runs(args.exp_root, args.experiment, args.env)
    run_cfgs = [(p, load_config(p)) for p in cfg_paths]
    grouped_vals, grouped_curves = summarize_runs(run_cfgs, args.metric, args.agg)
    if not grouped_vals:
        raise SystemExit("No runs found or no metrics available.")

    # Print table
    print(f"Summary for {args.metric} (agg={args.agg}):")
    for (opt, dyn), vals in sorted(
        grouped_vals.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True
    ):
        mean = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1))
        label = f"{opt} ({dyn})" if args.show_dynamics else opt
        print(f"- {label}: mean={mean:.2f}, std={std:.2f}, n={len(vals)}")

    try:
        if args.metric == "training_curve":
            if not grouped_curves:
                raise SystemExit("No training curves available.")
            plot_training_curves(
                grouped_curves, args.show_dynamics, args.out, args.env, args.experiment
            )
        else:
            plot_groups(
                grouped_vals,
                args.show_dynamics,
                args.metric,
                args.out,
                args.env,
                args.experiment,
            )
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting; install it with `pip install matplotlib`."
        ) from exc


if __name__ == "__main__":
    main()
