#!/usr/bin/env python3
"""Quickly benchmark action optimizers on analytic objectives.

Provides a lightweight, non-environment test to sanity-check optimizer behavior.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

# Allow running directly from the repo root without installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mbrl.planning as planning
from mbrl.planning.trajectory_opt import Optimizer

Objective = Callable[[torch.Tensor], torch.Tensor]


def rosenbrock(actions: torch.Tensor, a: float = 1.0, b: float = 100.0) -> torch.Tensor:
    flat = actions.view(actions.shape[0], -1)
    x, y = flat[:, 0], flat[:, 1]
    return -((a - x) ** 2 + b * (y - x * x) ** 2)


def rastrigin(actions: torch.Tensor) -> torch.Tensor:
    freq = 1.2 * math.pi
    flat = actions.view(actions.shape[0], -1)
    dim = flat.shape[1]
    return -(10 * dim + (flat * flat - 10 * torch.cos(freq * flat)).sum(dim=1))


def run_optimizer(
    factory: Callable[[], Optimizer],
    obj_fun: Objective,
    x0: torch.Tensor,
) -> np.ndarray:
    history: List[float] = []

    def callback(pop: torch.Tensor, values: torch.Tensor, _iter: int, *args) -> None:
        history.append(float(values.max().item()))

    optimizer = factory()
    best = optimizer.optimize(obj_fun, x0=x0, callback=callback)
    best_so_far = np.maximum.accumulate(np.array(history))
    print(f"  Final best value: {best_so_far[-1]:.4f}, solution: {best.view(-1)}")
    return best_so_far


def build_factories(
    args: argparse.Namespace,
    lower: List[List[float]],
    upper: List[List[float]],
    device: torch.device,
) -> Dict[str, Callable[[], Optimizer]]:
    num_iters = args.num_iters
    per_iter_budget = args.per_iter_budget
    elite_ratio = args.elite_ratio
    num_workers = args.num_workers
    init_jitter = args.init_jitter
    lr_mean = args.lr_mean
    lr_sigma = args.lr_sigma
    sigma = args.sigma

    factories: Dict[str, Callable[[], Optimizer]] = {
        "cem": lambda: planning.CEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=elite_ratio,
            population_size=per_iter_budget,
            alpha=args.cem_alpha,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
        ),
        "decent_cem": lambda: planning.DecentCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=elite_ratio,
            population_size=max(1, per_iter_budget // 3),
            alpha=args.cem_alpha,
            lower_bound=lower,
            upper_bound=upper,
            num_workers=num_workers,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter,
        ),
        "bccem": lambda: planning.BCCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=elite_ratio,
            population_size=max(1, per_iter_budget // 3),
            alpha=args.cem_alpha,
            lower_bound=lower,
            upper_bound=upper,
            num_workers=num_workers,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter,
        ),
        "gmm_cem": lambda: planning.GMMCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.15,
            population_size=max(1, math.ceil(per_iter_budget / 3)),
            num_workers=num_workers,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter,
        ),
        "cma_es": lambda: planning.CMAESOptimizer(
            num_iterations=num_iters,
            population_size=per_iter_budget,
            elite_ratio=args.cma_elite_ratio,
            sigma=sigma,
            lower_bound=lower,
            upper_bound=upper,
            alpha=args.cma_alpha,
            device=device,
            adaptation=args.cma_adaptation,
            return_mean_elites=True,
        ),
        "nes": lambda: planning.NESOptimizer(
            num_iterations=num_iters,
            population_size=per_iter_budget,
            sigma=sigma,
            lr_mean=lr_mean,
            lr_sigma=lr_sigma,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
            min_sigma=args.nes_min_sigma,
            max_sigma=args.nes_max_sigma,
        ),
    }
    return {k: v for k, v in factories.items() if k in args.optimizers}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark optimizers on analytic objectives."
    )
    parser.add_argument(
        "--objective",
        choices=["rosenbrock", "rastrigin"],
        default="rosenbrock",
        help="Objective to optimize.",
    )
    parser.add_argument(
        "--optimizers",
        default="cem,decent_cem,bccem,gmm_cem,cma_es,nes",
        help="Comma-separated optimizers to run.",
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=4,
        help="Iterations per optimizer.",
    )
    parser.add_argument(
        "--per-iter-budget",
        type=int,
        default=500,
        help="Total samples evaluated per iteration.",
    )
    parser.add_argument(
        "--elite-ratio",
        type=float,
        default=0.1,
        help="Elite ratio for CEM variants.",
    )
    parser.add_argument(
        "--cem-alpha",
        type=float,
        default=0.1,
        help="Momentum term for CEM variants.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker populations for DecentCEM/BCCEM/GMMCEM.",
    )
    parser.add_argument(
        "--init-jitter",
        type=float,
        default=0.5,
        help="Initial jitter scale for CEM variants.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=1.0,
        help="Initial sigma for CMA-ES/NES.",
    )
    parser.add_argument(
        "--lr-mean",
        type=float,
        default=0.3,
        help="Mean learning rate for NES.",
    )
    parser.add_argument(
        "--lr-sigma",
        type=float,
        default=0.15,
        help="Sigma learning rate for NES.",
    )
    parser.add_argument(
        "--nes-min-sigma",
        type=float,
        default=1e-3,
        help="Minimum sigma for NES.",
    )
    parser.add_argument(
        "--nes-max-sigma",
        type=float,
        default=None,
        help="Maximum sigma for NES (omit for no cap).",
    )
    parser.add_argument(
        "--cma-alpha",
        type=float,
        default=0.2,
        help="EMA factor for CMA-ES updates.",
    )
    parser.add_argument(
        "--cma-elite-ratio",
        type=float,
        default=0.25,
        help="Elite ratio for CMA-ES.",
    )
    parser.add_argument(
        "--cma-adaptation",
        choices=["diagonal", "full"],
        default="diagonal",
        help="Covariance adaptation type for CMA-ES.",
    )
    parser.add_argument(
        "--planning-horizon",
        type=int,
        default=1,
        help="Planning horizon to optimize.",
    )
    parser.add_argument(
        "--action-dim",
        type=int,
        default=2,
        help="Action dimension to optimize.",
    )
    parser.add_argument(
        "--bound",
        type=float,
        default=2.0,
        help="Symmetric bound on actions ([-bound, bound]).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="If set, save a PNG of convergence curves to this path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.optimizers = [o.strip() for o in args.optimizers.split(",") if o.strip()]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cpu")

    planning_horizon = args.planning_horizon
    action_dim = args.action_dim
    lower = np.full((planning_horizon, action_dim), -args.bound).tolist()
    upper = np.full((planning_horizon, action_dim), args.bound).tolist()
    x0 = torch.zeros((planning_horizon, action_dim), device=device)

    objectives: Dict[str, Objective] = {
        "rosenbrock": rosenbrock,
        "rastrigin": rastrigin,
    }
    obj_fun = objectives[args.objective]

    factories = build_factories(args, lower, upper, device)
    if not factories:
        raise SystemExit("No optimizers selected.")

    curves: Dict[str, np.ndarray] = {}
    print(f"Running objective: {args.objective}")
    for name, factory in factories.items():
        print(f"Running {name}...")
        curves[name] = run_optimizer(factory, obj_fun, x0)

    if args.plot:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "matplotlib is required for plotting; install it with `pip install matplotlib`."
            ) from exc

        plt.figure(figsize=(8, 5))
        for name, hist in curves.items():
            plt.plot(hist, label=name)
        plt.xlabel("Iteration")
        plt.ylabel("Best value so far")
        plt.title(f"Optimizer benchmark on {args.objective}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.plot, dpi=200)
        print(f"Saved plot to {args.plot}")


if __name__ == "__main__":
    main()
