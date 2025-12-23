"""Benchmark several trajectory optimizers on simple analytic objectives.

Compares CEM, DecentCEM, GMMCEM, CMA-ES, and NES on both a unimodal
Rosenbrock function and a multimodal Rastrigin function, visualizing the
best objective value per iteration.
"""

import math
import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

# Allow running the script directly from the repository root without installation.
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import mbrl.planning as planning

LOWER = -5.0
UPPER = 5.0
OFFSET = - 0.0

def bimodal_quadratic(actions: torch.Tensor) -> torch.Tensor:
    flat = actions.view(actions.shape[0], -1)
    x, y = flat[:, 0] - OFFSET, flat[:, 1] - OFFSET
    quad1 = -((x - 2) ** 2 + (y - 2) ** 2)
    quad2 = -((x + 2) ** 2 + (y + 2) ** 2)
    return torch.maximum(quad1, quad2)

def rosenbrock(actions: torch.Tensor, a: float = 1.0, b: float = 100.0) -> torch.Tensor:
    flat = actions.view(actions.shape[0], -1)
    x, y = flat[:, 0] - OFFSET, flat[:, 1] - OFFSET
    return -((a - x) ** 2 + b * (y - x * x) ** 2) 

def rastrigin(actions: torch.Tensor) -> torch.Tensor:
    freq = 2.0 * math.pi
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
    np.ndarray, torch.Tensor, List[np.ndarray], Dict[str, List]
]:
    history: List[float] = []
    populations: List[np.ndarray] = []
    extra: Dict[str, List[np.ndarray]] = {}

    def callback(
        pop: torch.Tensor,
        values: torch.Tensor,
        _iter: int,
    ) -> None:
        history.append(float(values.max().item()))
        populations.append(pop[..., :2].reshape(-1, 2).cpu().numpy())

    optimizer = factory()
    opt_kwargs = {"x0": x0, "callback": callback}
    if isinstance(optimizer, planning.BCCEMOptimizer):
        opt_kwargs["collect_trace"] = True
    best = optimizer.optimize(obj_fun, **opt_kwargs)
    best_so_far = np.maximum.accumulate(np.array(history))
    if isinstance(optimizer, planning.BCCEMOptimizer):
        trace = getattr(optimizer, "_trace", None)
        if isinstance(trace, dict) and trace:
            extra = trace
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
    lower = np.full((planning_horizon, action_dim), LOWER).tolist()
    upper = np.full((planning_horizon, action_dim), UPPER).tolist()
    x0 = torch.ones((planning_horizon, action_dim), device=device)
    # x0 = X0

    init_jitter_scale = 0.1 * (UPPER - LOWER)
    num_workers = 5 
    num_iters = 5
    per_iter_budget = 350 # total samples evaluated per iteration for fairness

    factories = {
        "CEM": lambda: planning.CEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.2,
            population_size=per_iter_budget,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            device=device,
            return_mean_elites=True,
        ),
        "DecentCEM": lambda: planning.DecentCEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.2,
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
            elite_ratio=0.2,
            population_size=per_iter_budget // num_workers,
            alpha=0.2,
            lower_bound=lower,
            upper_bound=upper,
            num_workers=num_workers,
            device=device,
            return_mean_elites=True,
            init_jitter_scale=init_jitter_scale,
            consensus_coef=0.5,
        ),
        "iCEM": lambda: planning.ICEMOptimizer(
            num_iterations=num_iters,
            elite_ratio=0.2,
            population_size=per_iter_budget,
            population_decay_factor=0.9,
            colored_noise_exponent=1.0,
            lower_bound=lower,
            upper_bound=upper,
            keep_elite_frac=0.5,
            alpha=0.2,
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
            elite_ratio=0.2,
            sigma=0.5 * (UPPER - LOWER),
            lower_bound=lower,
            upper_bound=upper,
            alpha=0.3,
            device=device,
            adaptation="full",
            return_mean_elites=True,
        ),
        # "NES": lambda: planning.NESOptimizer(
        #     num_iterations=num_iters,
        #     population_size=per_iter_budget,
        #     sigma=2.0,
        #     lr_mean=0.2,
        #     lr_sigma=0.5,
        #     lower_bound=lower,
        #     upper_bound=upper,
        #     device=device,
        #     return_mean_elites=True,
        # ),
    }

    curves: Dict[str, np.ndarray] = {}
    populations: Dict[str, List[np.ndarray]] = {}
    extras: Dict[str, Dict[str, List[np.ndarray]]] = {}
    for name, factory in factories.items():
        # reset RNG so all optimizers see the same randomness
        np.random.seed(0)
        torch.manual_seed(0)
        history, best, pops, extra = run_optimizer(name, factory, obj_fun, x0)
        curves[name] = history
        if record_pop:
            populations[name] = pops
        if extra:
            extras[name] = extra
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
    # plt.show()

def plot_bccem_diagnostics(
    obj_fun: Objective,
    extras: Dict[str, List],
    title: str,
    filename_prefix: str,
    populations: Optional[List[np.ndarray]] = None,
    bounds: Tuple[float, float] = (LOWER, UPPER),
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
    sigma_seq = extras.get("sigma_c", [])
    ir_seq = extras.get("ir_norm", extras.get("IR", []))
    kl_seq = extras.get("kl_w", [])
    w_seq = extras.get("weights", [])
    mu_w_seq = extras.get("mu_w", [])
    beta_seq = extras.get("consensus_beta", [])
    if not mu_seq:
        return

    mu_flat = np.stack([np.asarray(m).reshape(-1) for m in mu_seq])
    mu_xy = mu_flat[:, :2]
    dim = int(mu_flat.shape[1])
    inv_dim = 1.0 / float(max(1, dim))

    sigma_flat = None
    if sigma_seq:
        sigma_flat = np.stack([np.asarray(s).reshape(-1) for s in sigma_seq])

    W = None
    mu_w_xy = None
    if mu_w_seq:
        mu_w_xy = np.stack(
            [np.asarray(mw).reshape(np.asarray(mw).shape[0], -1)[:, :2] for mw in mu_w_seq]
        )
        W = int(mu_w_xy.shape[1])
    w_arr = np.stack([np.asarray(w) for w in w_seq]) if w_seq else None
    kl_arr = np.stack([np.asarray(k) for k in kl_seq]) if kl_seq else None

    grid_x, grid_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 200), np.linspace(bounds[0], bounds[1], 200)
    )
    pts = np.stack([grid_x.ravel(), grid_y.ravel()], axis=1)
    vals = obj_fun(torch.tensor(pts, dtype=torch.float32)).reshape(grid_x.shape)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax0, ax1, ax2, ax3 = axes.ravel()
    colors = None
    if W is not None:
        colors = plt.cm.tab10(np.linspace(0, 1, max(1, W)))

    cs = ax0.contourf(grid_x, grid_y, vals, levels=50, cmap="viridis")
    fig.colorbar(cs, ax=ax0, label="Objective")
    if populations:
        last = populations[-1]
        if last.size:
            n = min(800, last.shape[0])
            ax0.scatter(last[:n, 0], last[:n, 1], s=6, alpha=0.15, color="black", label="pop")
    ax0.plot(mu_xy[:, 0], mu_xy[:, 1], "-o", color="crimson", markersize=4, label="centroid")
    if mu_w_xy is not None and colors is not None:
        for w in range(W):
            ax0.plot(
                mu_w_xy[:, w, 0],
                mu_w_xy[:, w, 1],
                "--",
                color=colors[w],
                linewidth=1.0,
                alpha=0.8,
            )
    if sigma_flat is not None:
        from matplotlib.patches import Ellipse  # type: ignore

        for t in range(mu_xy.shape[0]):
            sx, sy = sigma_flat[t, 0], sigma_flat[t, 1]
            if not np.isfinite(sx) or not np.isfinite(sy):
                continue
            ell = Ellipse(
                (mu_xy[t, 0], mu_xy[t, 1]),
                width=2.0 * float(sx),
                height=2.0 * float(sy),
                angle=0.0,
                edgecolor="crimson",
                facecolor="none",
                lw=1.0,
                alpha=0.15,
            )
            ax0.add_patch(ell)
    ax0.set_xlim(bounds)
    ax0.set_ylim(bounds)
    ax0.set_title("BCCEM: centroid + workers")
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    ax0.legend(loc="best", fontsize=8)

    if ir_seq:
        ax1.plot(np.arange(len(ir_seq)), ir_seq, color="navy", label="ir_norm")
    if kl_arr is not None:
        kl_norm = kl_arr * inv_dim
        W_kl = int(kl_norm.shape[1])
        for w in range(W_kl):
            ax1.plot(
                np.arange(kl_norm.shape[0]),
                kl_norm[:, w],
                color="gray",
                linewidth=1.0,
                alpha=0.25,
            )
    ax1.set_title("Disagreement (IR / KL)")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Normalized KL")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best", fontsize=8)

    if w_arr is not None and colors is not None:
        for w in range(int(w_arr.shape[1])):
            ax2.plot(
                np.arange(w_arr.shape[0]),
                w_arr[:, w],
                color=colors[w],
                linewidth=1.5,
            )
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title("Worker weights")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("weight")
        ax2.grid(True, alpha=0.3)
    else:
        ax2.axis("off")

    if sigma_flat is not None:
        sigma_norm = np.linalg.norm(sigma_flat, axis=1)
        ax3.plot(np.arange(len(sigma_norm)), sigma_norm, color="darkgreen", label="||sigma_c||")
    ax3.set_title("Centroid spread / consensus")
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("spread")
    ax3.grid(True, alpha=0.3)
    if beta_seq:
        ax3b = ax3.twinx()
        ax3b.plot(
            np.arange(len(beta_seq)),
            beta_seq,
            color="purple",
            linewidth=1.5,
            alpha=0.8,
            label="beta_mean",
        )
        ax3b.set_ylabel("consensus beta")

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_bccem.png", dpi=150)

    if populations:
        T = min(len(populations), int(mu_xy.shape[0]))
        if T > 0:
            fig2, axes2 = plt.subplots(1, T, figsize=(3.2 * T, 3.2))
            if T == 1:
                axes2 = [axes2]
            for t in range(T):
                ax = axes2[t]
                ax.contourf(grid_x, grid_y, vals, levels=30, cmap="viridis", alpha=0.9)
                pts_t = populations[t]
                if pts_t.size:
                    n = min(800, pts_t.shape[0])
                    ax.scatter(pts_t[:n, 0], pts_t[:n, 1], s=6, alpha=0.25, color="black")
                ax.scatter(mu_xy[t, 0], mu_xy[t, 1], s=30, color="crimson", marker="o")
                if mu_w_xy is not None and colors is not None:
                    for w in range(W):
                        ax.scatter(
                            mu_w_xy[t, w, 0],
                            mu_w_xy[t, w, 1],
                            s=25,
                            color=colors[w],
                            marker="x",
                        )
                if sigma_flat is not None:
                    from matplotlib.patches import Ellipse  # type: ignore

                    sx, sy = sigma_flat[t, 0], sigma_flat[t, 1]
                    if np.isfinite(sx) and np.isfinite(sy):
                        ax.add_patch(
                            Ellipse(
                                (mu_xy[t, 0], mu_xy[t, 1]),
                                width=2.0 * float(sx),
                                height=2.0 * float(sy),
                                angle=0.0,
                                edgecolor="crimson",
                                facecolor="none",
                                lw=1.0,
                                alpha=0.3,
                            )
                        )
                ax.set_xlim(bounds)
                ax.set_ylim(bounds)
                ax.set_title(f"iter {t}")
                ax.set_xticks([])
                ax.set_yticks([])
            fig2.suptitle(f"{title} (pop evolution)")
            plt.tight_layout()
            plt.savefig(f"{filename_prefix}_bccem_evolution.png", dpi=150)


def plot_population_snapshots(
    obj_fun: Objective,
    populations: Dict[str, List[np.ndarray]],
    title: str,
    filename: str,
    bounds: Tuple[float, float] = (LOWER, UPPER),
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
    vals = (
        obj_fun(torch.tensor(pts, dtype=torch.float32))
        .reshape(grid_x.shape)
        .detach()
        .cpu()
        .numpy()
    )
    vals_to_plot = vals
    colorbar_label = "Objective"
    if obj_fun is rosenbrock:
        vals_to_plot = np.log10(np.maximum(-vals, 1e-12))
        colorbar_label = "log10(-Objective)"

    plt.figure(figsize=(10, 6))
    cs = plt.contourf(grid_x, grid_y, vals_to_plot, levels=30, cmap="coolwarm")
    plt.colorbar(cs, label=colorbar_label)
    for name, pops in populations.items():
        # plot samples
        last = pops[-1]
        num_samples = min(100, last.shape[0])
        plt.scatter(last[:num_samples, 0], last[:num_samples, 1], s=5, alpha=0.3, label=name)
        # plot solution
        mean_sol = last.mean(axis=0)
        plt.plot(
            mean_sol[0], mean_sol[1], "o", markersize=10, label=f"{name} mean solution"
        )   
    plt.legend()
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    # plt.show()

def plot_kde_subplots(
    populations: Dict[str, List[np.ndarray]],
    title: str,
    filename: str,
    bounds: Tuple[float, float] = (LOWER, UPPER),
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
        N, _ = data.shape
        n_samples = N
        ax.scatter(data[:n_samples, 0], data[:n_samples, 1], s=5, c="white", alpha=0.3, edgecolors="none")
        ax.set_title(name)
        ax.set_xlim(bounds)
        ax.set_ylim(bounds)
    for ax in axes.flat[num:]:
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    # plt.show()


def main() -> None:
    # Seed everything
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = repo_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    quad_curves, quad_pops, quad_extra = benchmark(
        "Quadratic", bimodal_quadratic, record_pop=True
    )
    plot_curves(
        quad_curves,
        "Quadratic Benchmark",
        str(output_dir / "benchmark_quadratic.png"),
    )
    plot_population_snapshots(
        bimodal_quadratic,
        quad_pops,
        "Quadratic Populations",
        str(output_dir / "pop_quadratic.png"),
    )
    plot_kde_subplots(
        quad_pops,
        "Quadratic KDEs",
        str(output_dir / "kde_quadratic.png"),
    )
    if "BCCEM" in quad_extra:
        plot_bccem_diagnostics(
            bimodal_quadratic,
            quad_extra["BCCEM"],
            "Quadratic BCCEM Diagnostics",
            str(output_dir / "quadratic"),
            populations=quad_pops.get("BCCEM"),
        )

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
            populations=rosen_pops.get("BCCEM"),
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
            populations=rastrigin_pops.get("BCCEM"),
        )


if __name__ == "__main__":
    main()
