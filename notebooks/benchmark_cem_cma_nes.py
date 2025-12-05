"""Benchmark several trajectory optimizers on simple analytic objectives.

Compares CEM, DecentCEM, GMMCEM, CMA-ES, and NES on both a unimodal
Rosenbrock function and a multimodal Rastrigin function, visualizing the
best objective value per iteration.
"""

import math
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch

# Allow running the script directly from the repository root without installation.
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import mbrl.planning as planning


def rosenbrock(actions: torch.Tensor, a: float = 1.0, b: float = 100.0) -> torch.Tensor:
    flat = actions.view(actions.shape[0], -1)
    x, y = flat[:, 0], flat[:, 1]
    return -((a - x) ** 2 + b * (y - x * x) ** 2)


def rastrigin(actions: torch.Tensor) -> torch.Tensor:
    freq = 1.2 * math.pi
    flat = actions.view(actions.shape[0], -1)
    dim = flat.shape[1]
    return -(10 * dim + (flat * flat - 10 * torch.cos(freq * flat)).sum(dim=1))


Objective = Callable[[torch.Tensor], torch.Tensor]


def run_optimizer(
    name: str,
    factory: Callable[[], object],
    obj_fun: Objective,
    x0: torch.Tensor,
) -> Tuple[
    np.ndarray, torch.Tensor, List[np.ndarray], Dict[str, List[np.ndarray]]
]:
    history: List[float] = []
    populations: List[np.ndarray] = []
    extra: Dict[str, List[np.ndarray]] = {"IR": [], "mu_c": [], "sigma_c": []}

    def callback(
        pop: torch.Tensor,
        values: torch.Tensor,
        _iter: int,
        *extra_args: torch.Tensor,
    ) -> None:
        history.append(float(values.max().item()))
        populations.append(pop[..., :2].reshape(-1, 2).cpu().numpy())
        if extra_args:
            ir = extra_args[0]
            extra["IR"].append(np.array(float(ir)))
        if len(extra_args) > 1:
            mu_c = extra_args[1]
            extra["mu_c"].append(mu_c.detach().cpu().numpy().reshape(-1))
        if len(extra_args) > 2:
            sigma_c = extra_args[2]
            extra["sigma_c"].append(sigma_c.detach().cpu().numpy().reshape(-1))

    optimizer = factory()
    opt_kwargs = {"x0": x0, "callback": callback}
    if isinstance(optimizer, planning.BCCEMOptimizer):
        opt_kwargs["tau"] = 0.5
    best = optimizer.optimize(obj_fun, **opt_kwargs)
    best_so_far = np.maximum.accumulate(np.array(history))
    extra = {k: v for k, v in extra.items() if v}
    return best_so_far, best, populations, extra


def benchmark(
    obj_name: str, obj_fun: Objective, record_pop: bool = False
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, List[np.ndarray]], Dict[str, Dict[str, List[np.ndarray]]]
]:
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device("cpu")
    action_dim = 2
    planning_horizon = 1
    lower = np.full((planning_horizon, action_dim), -2.0).tolist()
    upper = np.full((planning_horizon, action_dim), 2.0).tolist()
    x0 = torch.zeros((planning_horizon, action_dim), device=device)

    init_jitter_scale = 1.0
    num_workers = 5 
    num_iters = 5
    per_iter_budget =  500 # total samples evaluated per iteration for fairness

    factories = {
        "CEM": lambda: planning.CEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.15,
            population_size=per_iter_budget,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
        ),
        "DecentCEM": lambda: planning.DecentCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.15,
            population_size=per_iter_budget // num_workers,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            num_workers=num_workers,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter_scale,
        ),
        "BCCEM": lambda: planning.BCCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.15,
            population_size=per_iter_budget // num_workers,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            num_workers=num_workers,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter_scale,
        ),
        "GMMCEM": lambda: planning.GMMCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.2,
            population_size=math.ceil(per_iter_budget / num_workers),
            num_workers=num_workers,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter_scale,
        ),
        "CMAES": lambda: planning.CMAESOptimizer(
            num_iterations=num_iters,
            population_size=per_iter_budget,
            elite_ratio=0.15,
            sigma=1.0,
            lower_bound=lower,
            upper_bound=upper,
            alpha=0.2,
            device=device,
            adaptation="full",
            return_mean_elites=True,
        ),
        "NES": lambda: planning.NESOptimizer(
            num_iterations=num_iters,
            population_size=per_iter_budget,
            sigma=0.8,
            lr_mean=0.3,
            lr_sigma=0.15,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
        ),
    }

    curves: Dict[str, np.ndarray] = {}
    populations: Dict[str, List[np.ndarray]] = {}
    extras: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for name, factory in factories.items():
        # reset RNG so all optimizers see the same randomness
        np.random.seed(0)
        torch.manual_seed(0)
        print(f"Running {name} on {obj_name}...")
        history, best, pops, extra = run_optimizer(name, factory, obj_fun, x0)
        curves[name] = history
        if record_pop:
            populations[name] = pops
        if extra:
            extras[name] = extra
        print(f"  Final best value: {history[-1]:.3f}, solution: {best.view(-1)}")
    return curves, populations, extras


def plot_curves(curves: Dict[str, np.ndarray], title: str, filename: str) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting; install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 5))
    for name, hist in curves.items():
        plt.plot(hist, label=name)
    plt.xlabel("Iteration")
    plt.ylabel("Best objective value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_bccem_diagnostics(
    obj_fun: Objective,
    extras: Dict[str, List[np.ndarray]],
    title: str,
    filename_prefix: str,
    bounds: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    if not extras:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting; install it with `pip install matplotlib`."
        ) from exc

    mu_seq = extras.get("mu_c", [])
    ir_seq = extras.get("IR", [])
    sigma_seq = extras.get("sigma_c", [])
    if not mu_seq:
        return

    mu_arr = np.stack(mu_seq)
    mu_flat = mu_arr.reshape(mu_arr.shape[0], -1)
    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200), np.linspace(bounds[0], bounds[1], 200)
    )
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    vals = obj_fun(torch.tensor(pts, dtype=torch.float32)).reshape(grid_x.shape)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    cs = axes[0].contourf(grid_x, grid_y, vals, levels=50, cmap="viridis")
    fig.colorbar(cs, ax=axes[0], label="Objective")
    axes[0].plot(mu_flat[:, 0], mu_flat[:, 1], "-o", color="crimson", markersize=4)
    axes[0].set_xlim(bounds)
    axes[0].set_ylim(bounds)
    axes[0].set_title("BCCEM mean trajectory")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    if ir_seq:
        axes[1].plot(np.arange(len(ir_seq)), ir_seq, color="navy")
        axes[1].set_title("BCCEM Information Radius")
        axes[1].set_xlabel("Iteration")
        axes[1].set_ylabel("IR")
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].axis("off")

    if sigma_seq:
        sigma_arr = np.stack(sigma_seq)
        sigma_norm = np.linalg.norm(sigma_arr.reshape(sigma_arr.shape[0], -1), axis=1)
        axes[2].plot(np.arange(len(sigma_norm)), sigma_norm, color="darkgreen")
        axes[2].set_title("BCCEM sigma_c norm")
        axes[2].set_xlabel("Iteration")
        axes[2].set_ylabel("||sigma_c||")
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis("off")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_bccem.png", dpi=150)
    plt.show()


def plot_population_snapshots(
    obj_fun: Objective,
    populations: Dict[str, List[np.ndarray]],
    title: str,
    filename: str,
    bounds: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    if not populations:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting; install it with `pip install matplotlib`."
        ) from exc

    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200), np.linspace(bounds[0], bounds[1], 200)
    )
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    vals = obj_fun(torch.tensor(pts, dtype=torch.float32)).reshape(grid_x.shape)

    plt.figure(figsize=(10, 6))
    cs = plt.contourf(grid_x, grid_y, vals, levels=50, cmap="viridis")
    plt.colorbar(cs, label="Objective")
    for name, pops in populations.items():
        last = pops[-1]
        plt.scatter(last[:, 0], last[:, 1], s=8, alpha=0.7, label=name)
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def plot_kde_subplots(
    populations: Dict[str, List[np.ndarray]],
    title: str,
    filename: str,
    bounds: Tuple[float, float] = (-2.0, 2.0),
) -> None:
    if not populations:
        return
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise SystemExit(
            "matplotlib is required for plotting; install it with `pip install matplotlib`."
        ) from exc

    try:
        from scipy import stats  # type: ignore
    except ModuleNotFoundError:
        stats = None

    num = len(populations)
    cols = min(3, num)
    rows = math.ceil(num / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 150),
        np.linspace(bounds[0], bounds[1], 150),
    )
    positions = np.vstack([grid_x.ravel(), grid_y.ravel()])

    for ax, (name, pops) in zip(axes.flat, populations.items()):
        data = pops[-1]
        if data.shape[0] < 2:
            ax.set_title(f"{name} (insufficient samples)")
            ax.axis("off")
            continue
        if stats is not None:
            kde = stats.gaussian_kde(data.T)
            z = kde(positions).reshape(grid_x.shape)
            ax.contourf(grid_x, grid_y, z, levels=40, cmap="magma")
        else:
            h, xedges, yedges = np.histogram2d(
                data[:, 0],
                data[:, 1],
                bins=50,
                range=[[bounds[0], bounds[1]], [bounds[0], bounds[1]]],
                density=True,
            )
            ax.imshow(
                h.T,
                origin="lower",
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                aspect="auto",
                cmap="magma",
            )
        ax.scatter(data[:, 0], data[:, 1], s=6, c="white", alpha=0.5, edgecolors="none")
        ax.set_title(name)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
    for ax in axes.flat[num:]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()


def main() -> None:
    output_dir = repo_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    rosen_curves, rosen_pops, rosen_extra = benchmark(
        "Rosenbrock", rosenbrock, record_pop=True
    )
    plot_curves(
        rosen_curves,
        "Rosenbrock Benchmark",
        str(output_dir / "benchmark_rosenbrock.png"),
    )
    plot_population_snapshots(
        rosenbrock,
        rosen_pops,
        "Rosenbrock Populations",
        str(output_dir / "pop_rosenbrock.png"),
    )
    plot_kde_subplots(
        rosen_pops,
        "Rosenbrock KDEs",
        str(output_dir / "kde_rosenbrock.png"),
    )
    if "BCCEM" in rosen_extra:
        plot_bccem_diagnostics(
            rosenbrock,
            rosen_extra["BCCEM"],
            "Rosenbrock BCCEM Diagnostics",
            str(output_dir / "rosenbrock"),
        )

    rastrigin_curves, rastrigin_pops, rastrigin_extra = benchmark(
        "Rastrigin", rastrigin, record_pop=True
    )
    plot_curves(
        rastrigin_curves,
        "Rastrigin Benchmark",
        str(output_dir / "benchmark_rastrigin.png"),
    )
    plot_population_snapshots(
        rastrigin,
        rastrigin_pops,
        "Rastrigin Populations",
        str(output_dir / "pop_rastrigin.png"),
    )
    plot_kde_subplots(
        rastrigin_pops,
        "Rastrigin KDEs",
        str(output_dir / "kde_rastrigin.png"),
    )
    if "BCCEM" in rastrigin_extra:
        plot_bccem_diagnostics(
            rastrigin,
            rastrigin_extra["BCCEM"],
            "Rastrigin BCCEM Diagnostics",
            str(output_dir / "rastrigin"),
        )


if __name__ == "__main__":
    main()
