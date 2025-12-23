"""Plot PETS optimizer benchmark training stats.

This reads the output directory created by
``mbrl.examples.scripts.run_pets_optimizer_benchmark`` and plots learning curves
(returns) and compute metrics (planning time/model steps), including planning time
both in absolute terms and normalized to CEM. If BCCEM is present, it also plots
BCCEM-specific diagnostics (centroid action fraction + IR norm).

Example:
  python -m mbrl.examples.scripts.plot_pets_optimizer_benchmark --run-id 20251217_181336
"""

from __future__ import annotations

import argparse
import csv
import difflib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Allow running directly from the repo root without installation.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_RUNS_ROOT = REPO_ROOT / "exp" / "pets_optimizer_benchmark_runs"
DEFAULT_OPTIMIZERS = ["cem", "decent_cem", "gmm_cem", "icem", "bccem"]

_OPT_LABELS: Dict[str, str] = {
    "cem": "CEM",
    "decent_cem": "DecentCEM",
    "gmm_cem": "GMMCEM",
    "icem": "iCEM",
    "bccem": "BCCEM",
    "cma_es": "CMA-ES",
    "nes": "NES",
    "mppi": "MPPI",
}

_OPT_COLOR_INDEX: Dict[str, int] = {
    "cem": 0,
    "decent_cem": 1,
    "gmm_cem": 2,
    "bccem": 3,
    "icem": 4,
    "nes": 5,
    "cma_es": 6,
    "mppi": 7,
}

_RESULT_FIELDS = (
    "env_step",
    "step",
    "episode_reward",
    "planning_time_ms",
    "planning_model_steps_est",
    "planning_replans_per_step",
    "planning_skips_per_step",
    "planning_centroid_frac",
    "planning_ir_norm_mean",
    "planning_ir_norm_min",
    "planning_ir_norm_max",
)


@dataclass(frozen=True)
class ResultSpec:
    env: str
    optimizer: str
    seed: int
    path: Path


def _opt_label(opt: str) -> str:
    return _OPT_LABELS.get(opt, opt)


def _parse_int(value: object) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))  # some CSVs end up with "0.0"
        except (TypeError, ValueError):
            return None


def _parse_float(value: object) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _smooth_nanmean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(int(window), dtype=float)
    mask = np.isfinite(values).astype(float)
    weighted = np.convolve(np.nan_to_num(values, nan=0.0), kernel, mode="same")
    counts = np.convolve(mask, kernel, mode="same")
    out = weighted / np.where(counts > 0.0, counts, np.nan)
    return out


def _human_format(value: float) -> str:
    if not np.isfinite(value):
        return ""
    abs_v = abs(float(value))
    if abs_v >= 1_000_000:
        return f"{value/1_000_000:.1f}M"
    if abs_v >= 1_000:
        return f"{value/1_000:.0f}k"
    if abs_v >= 10:
        return f"{value:.0f}"
    return f"{value:.1f}"


def _interp_to_common_x(x: np.ndarray, y: np.ndarray, x_common: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return np.full_like(x_common, np.nan, dtype=float)
    x = x[mask]
    y = y[mask]
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    x_u, idx = np.unique(x, return_index=True)
    y_u = y[idx]
    return np.interp(x_common, x_u, y_u, left=np.nan, right=np.nan)


def _aggregate_xy(xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    if not xs:
        return np.array([]), np.array([[]])
    if _aligned_x(xs):
        min_len = min(y.size for y in ys)
        x_common = np.asarray(xs[0], dtype=float)[:min_len]
        stack = np.vstack([np.asarray(y, dtype=float)[:min_len] for y in ys])
        return x_common, stack
    x_values = [np.asarray(x, dtype=float) for x in xs]
    x_cat = np.concatenate([x[np.isfinite(x)] for x in x_values]) if x_values else np.array([])
    x_common = np.unique(x_cat)
    if x_common.size == 0:
        return np.array([]), np.array([[]])
    stack = np.vstack(
        [_interp_to_common_x(x, y, x_common) for x, y in zip(x_values, ys)]
    )
    return x_common, stack


def _apply_plot_style(plt) -> None:
    plt.rcParams.update(
        {
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 13,
        }
    )


def _aligned_x(xs: Sequence[np.ndarray]) -> bool:
    if not xs:
        return False
    first = np.asarray(xs[0])
    for x in xs[1:]:
        x = np.asarray(x)
        if x.shape != first.shape:
            return False
        if not np.allclose(x, first, rtol=0.0, atol=1e-6, equal_nan=True):
            return False
    return True


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _find_latest_run_dir(runs_root: Path) -> Path:
    candidates = sorted([p for p in runs_root.iterdir() if p.is_dir()])
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {runs_root}")
    return candidates[-1]


def _list_run_dirs(runs_root: Path) -> List[Path]:
    if not runs_root.exists():
        return []
    return sorted([p for p in runs_root.iterdir() if p.is_dir()])


def _resolve_run_dir(runs_root: Path, run_id: str) -> Path:
    direct = (runs_root / run_id).resolve()
    if direct.exists():
        return direct

    candidates = _list_run_dirs(runs_root)
    names = [p.name for p in candidates]
    prefix_matches = [p for p in candidates if p.name.startswith(run_id)]
    if prefix_matches:
        prefix_matches = sorted(prefix_matches)
        chosen = prefix_matches[-1]
        if len(prefix_matches) > 1:
            print(
                f"[info] run-id prefix matched {len(prefix_matches)} runs; using {chosen.name}"
            )
        return chosen.resolve()

    suggestion = difflib.get_close_matches(run_id, names, n=1)
    tail = ", ".join(names[-10:]) if names else "(none)"
    msg = [f"Run id not found: {run_id}", f"runs_root: {runs_root}"]
    if suggestion:
        msg.append(f"Did you mean: {suggestion[0]} ?")
    if prefix_matches:
        msg.append("Prefix matches: " + ", ".join(p.name for p in prefix_matches))
    msg.append(f"Available runs (last 10): {tail}")
    raise FileNotFoundError("\n".join(msg))


def _load_summary_specs(run_dir: Path) -> List[ResultSpec]:
    summary_path = run_dir / "summary.csv"
    if not summary_path.exists():
        return []
    rows = _read_csv(summary_path)
    specs: List[ResultSpec] = []
    for row in rows:
        env = row.get("env")
        optimizer = row.get("optimizer")
        seed = _parse_int(row.get("seed"))
        if not env or not optimizer or seed is None:
            continue
        path = run_dir / env / optimizer / f"seed{seed}" / "results.csv"
        specs.append(ResultSpec(env=env, optimizer=optimizer, seed=seed, path=path))
    return specs


def _scan_results_specs(run_dir: Path) -> List[ResultSpec]:
    specs: List[ResultSpec] = []
    for path in run_dir.rglob("results.csv"):
        try:
            rel = path.relative_to(run_dir)
        except ValueError:
            continue
        parts = rel.parts
        if len(parts) != 4:
            continue
        env, optimizer, seed_part, _ = parts
        if not seed_part.startswith("seed"):
            continue
        seed = _parse_int(seed_part[len("seed") :])
        if seed is None:
            continue
        specs.append(ResultSpec(env=env, optimizer=optimizer, seed=seed, path=path))
    return specs


def _filter_specs(
    specs: Iterable[ResultSpec],
    *,
    env: Optional[str],
    optimizers: Optional[Sequence[str]],
    seeds: Optional[Sequence[int]],
) -> List[ResultSpec]:
    out: List[ResultSpec] = []
    for spec in specs:
        if env is not None and spec.env != env:
            continue
        if optimizers is not None and spec.optimizer not in optimizers:
            continue
        if seeds is not None and spec.seed not in seeds:
            continue
        out.append(spec)
    return out


def _load_results_series(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    rows = _read_csv(path)
    if not rows:
        return {}
    series: Dict[str, List[float]] = {k: [] for k in _RESULT_FIELDS}
    for row in rows:
        for k in _RESULT_FIELDS:
            series[k].append(_parse_float(row.get(k)))
    return {k: np.asarray(v, dtype=float) for k, v in series.items()}


def _group_by_env_and_optimizer(
    specs: Sequence[ResultSpec], x_key: str
) -> Dict[Tuple[str, str], List[Dict[str, np.ndarray]]]:
    grouped: Dict[Tuple[str, str], List[Dict[str, np.ndarray]]] = {}
    for spec in specs:
        data = _load_results_series(spec.path)
        if not data or x_key not in data:
            continue
        grouped.setdefault((spec.env, spec.optimizer), []).append(data)
    return grouped


def _plot_with_optional_mean(
    ax,
    *,
    series_list: Sequence[Dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    label: Optional[str],
    color,
    smooth: int,
    show_seeds: bool,
    linestyle: str = "-",
) -> None:
    line_label = label if label is not None else "_nolegend_"
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in series_list:
        x = s.get(x_key)
        y = s.get(y_key)
        if x is None or y is None:
            continue
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size == 0 or y.size == 0:
            continue
        pairs.append((x, y))
    if not pairs:
        return
    xs, ys = zip(*pairs)

    if show_seeds:
        for x, y in zip(xs, ys):
            ax.plot(
                x,
                _smooth_nanmean(y, smooth),
                color=color,
                alpha=0.20,
                linewidth=1.0,
                linestyle=linestyle,
            )

    if len(xs) == 1:
        ax.plot(
            xs[0],
            _smooth_nanmean(ys[0], smooth),
            color=color,
            linewidth=2.5,
            linestyle=linestyle,
            label=line_label,
        )
        return

    x_common, stack = _aggregate_xy(xs, ys)
    if x_common.size == 0 or stack.size == 0:
        return
    keep = np.any(np.isfinite(stack), axis=0)
    if not np.any(keep):
        return
    x_common = x_common[keep]
    stack = stack[:, keep]
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    mean_s = _smooth_nanmean(mean, smooth)
    ax.plot(
        x_common,
        mean_s,
        color=color,
        linewidth=2.8,
        linestyle=linestyle,
        label=line_label,
    )
    ax.fill_between(
        x_common, mean_s - std, mean_s + std, color=color, alpha=0.15, linewidth=0.0
    )


def _fill_between_mean_range(
    ax,
    *,
    series_list: Sequence[Dict[str, np.ndarray]],
    x_key: str,
    y_min_key: str,
    y_max_key: str,
    color,
    smooth: int,
    alpha: float = 0.12,
) -> None:
    xs: List[np.ndarray] = []
    ymins: List[np.ndarray] = []
    ymaxs: List[np.ndarray] = []
    for s in series_list:
        x = s.get(x_key)
        y_min = s.get(y_min_key)
        y_max = s.get(y_max_key)
        if x is None or y_min is None or y_max is None:
            continue
        x = np.asarray(x)
        y_min = np.asarray(y_min)
        y_max = np.asarray(y_max)
        if x.size == 0 or y_min.size == 0 or y_max.size == 0:
            continue
        xs.append(x)
        ymins.append(y_min)
        ymaxs.append(y_max)
    if not xs:
        return
    x_common, stack_min = _aggregate_xy(xs, ymins)
    x_common2, stack_max = _aggregate_xy(xs, ymaxs)
    if x_common.size == 0 or x_common2.size == 0 or x_common.shape != x_common2.shape:
        return
    keep = np.any(np.isfinite(stack_min), axis=0) & np.any(np.isfinite(stack_max), axis=0)
    if not np.any(keep):
        return
    x_common = x_common[keep]
    stack_min = stack_min[:, keep]
    stack_max = stack_max[:, keep]
    min_mean = np.nanmean(stack_min, axis=0)
    max_mean = np.nanmean(stack_max, axis=0)
    min_s = _smooth_nanmean(min_mean, smooth)
    max_s = _smooth_nanmean(max_mean, smooth)
    low = np.minimum(min_s, max_s)
    high = np.maximum(min_s, max_s)
    ax.fill_between(x_common, low, high, color=color, alpha=alpha, linewidth=0.0)


def _mean_curve(
    series_list: Sequence[Dict[str, np.ndarray]], x_key: str, y_key: str
) -> Tuple[np.ndarray, np.ndarray]:
    pairs: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in series_list:
        x = s.get(x_key)
        y = s.get(y_key)
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        pairs.append((x, y))
    if not pairs:
        return np.array([]), np.array([])
    xs, ys = zip(*pairs)
    if len(xs) == 1:
        return np.asarray(xs[0], dtype=float), np.asarray(ys[0], dtype=float)
    x_common, stack = _aggregate_xy(xs, ys)
    if x_common.size == 0 or stack.size == 0:
        return np.array([]), np.array([])
    return x_common, np.nanmean(stack, axis=0)


def _normalize_series_list_to_baseline_mean(
    *,
    series_list: Sequence[Dict[str, np.ndarray]],
    x_key: str,
    y_key: str,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    out_y_key: str,
) -> List[Dict[str, np.ndarray]]:
    out: List[Dict[str, np.ndarray]] = []
    for s in series_list:
        x = s.get(x_key)
        y = s.get(y_key)
        if x is None or y is None:
            continue
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        if x.size == 0 or y.size == 0:
            continue
        baseline_at_x = _interp_to_common_x(baseline_x, baseline_y, x)
        ratio = np.full_like(y, np.nan, dtype=float)
        denom_ok = np.isfinite(baseline_at_x) & (baseline_at_x != 0.0)
        num_ok = np.isfinite(y)
        ok = denom_ok & num_ok
        ratio[ok] = y[ok] / baseline_at_x[ok]
        out.append({x_key: x, out_y_key: ratio})
    return out


def _plot_env(
    *,
    run_dir: Path,
    env: str,
    optimizers: Sequence[str],
    grouped: Dict[Tuple[str, str], List[Dict[str, np.ndarray]]],
    x_key: str,
    smooth: int,
    show_seeds: bool,
    out_dir: Path,
    show: bool,
    plt,
) -> None:
    import matplotlib.ticker as mticker

    present_opts = [opt for opt in optimizers if grouped.get((env, opt))]
    if not present_opts:
        return

    cmap = plt.get_cmap("tab10")
    used_idx: set[int] = set()
    color_map: Dict[str, object] = {}
    for opt in present_opts:
        idx = _OPT_COLOR_INDEX.get(opt)
        if idx is not None:
            idx = int(idx) % 10
            color_map[opt] = cmap(idx)
            used_idx.add(idx)
    next_candidates = [i for i in range(10) if i not in used_idx]
    for opt in present_opts:
        if opt in color_map:
            continue
        idx = next_candidates.pop(0) if next_candidates else len(color_map) % 10
        color_map[opt] = cmap(idx)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 4, height_ratios=(2.2, 1.0), hspace=0.32, wspace=0.26)
    ax_reward = fig.add_subplot(gs[0, :])
    ax_time = fig.add_subplot(gs[1, 0], sharex=ax_reward)
    ax_time_norm = fig.add_subplot(gs[1, 1], sharex=ax_reward)
    ax_steps = fig.add_subplot(gs[1, 2], sharex=ax_reward)
    ax_replans = fig.add_subplot(gs[1, 3], sharex=ax_reward)
    ax_reward.tick_params(labelbottom=False)

    cem_baseline = grouped.get((env, "cem"), [])
    baseline_x, baseline_y = _mean_curve(cem_baseline, x_key, "planning_time_ms")
    baseline_y_s = _smooth_nanmean(baseline_y, smooth)
    have_baseline = baseline_x.size > 0 and baseline_y_s.size > 0
    if have_baseline:
        ax_time_norm.axhline(1.0, color="black", linewidth=1.0, alpha=0.18, zorder=0)

    norm_for_ylim: List[np.ndarray] = []
    for opt in present_opts:
        series_list = grouped[(env, opt)]
        color = color_map[opt]
        label = _opt_label(opt)

        _plot_with_optional_mean(
            ax_reward,
            series_list=series_list,
            x_key=x_key,
            y_key="episode_reward",
            label=label,
            color=color,
            smooth=smooth,
            show_seeds=show_seeds,
        )
        _plot_with_optional_mean(
            ax_time,
            series_list=series_list,
            x_key=x_key,
            y_key="planning_time_ms",
            label=None,
            color=color,
            smooth=smooth,
            show_seeds=show_seeds,
        )
        if have_baseline:
            norm_series_list = _normalize_series_list_to_baseline_mean(
                series_list=series_list,
                x_key=x_key,
                y_key="planning_time_ms",
                baseline_x=baseline_x,
                baseline_y=baseline_y_s,
                out_y_key="planning_time_ms_norm_to_cem",
            )
            if opt != "bccem":
                for s in norm_series_list:
                    y = s.get("planning_time_ms_norm_to_cem")
                    if y is not None:
                        norm_for_ylim.append(np.asarray(y, dtype=float))
            _plot_with_optional_mean(
                ax_time_norm,
                series_list=norm_series_list,
                x_key=x_key,
                y_key="planning_time_ms_norm_to_cem",
                label=None,
                color=color,
                smooth=smooth,
                show_seeds=show_seeds,
            )
        _plot_with_optional_mean(
            ax_steps,
            series_list=series_list,
            x_key=x_key,
            y_key="planning_model_steps_est",
            label=None,
            color=color,
            smooth=smooth,
            show_seeds=show_seeds,
        )
        _plot_with_optional_mean(
            ax_replans,
            series_list=series_list,
            x_key=x_key,
            y_key="planning_replans_per_step",
            label=None,
            color=color,
            smooth=smooth,
            show_seeds=show_seeds,
        )

    x_label = "Env steps" if x_key == "env_step" else "Episode"
    if x_key == "env_step":
        formatter = mticker.FuncFormatter(lambda x, _pos: _human_format(x))
        for ax in (ax_time, ax_time_norm, ax_steps, ax_replans):
            ax.xaxis.set_major_formatter(formatter)
    else:
        for ax in (ax_time, ax_time_norm, ax_steps, ax_replans):
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_reward.set_title("Episode Return")
    ax_reward.set_ylabel("return")
    ax_time.set_title("Planning Time")
    ax_time.set_ylabel("ms / env step")
    ax_time_norm.set_title("Planning Time (norm. to CEM)")
    ax_time_norm.set_ylabel("\u00d7 CEM")
    if have_baseline and norm_for_ylim:
        finite = [v[np.isfinite(v)] for v in norm_for_ylim]
        finite = [v for v in finite if v.size]
        if finite:
            vals = np.concatenate(finite, axis=0)
            dev = float(np.nanpercentile(np.abs(vals - 1.0), 95))
            dev = max(dev, 0.05)
            pad = 1.25
            ax_time_norm.set_ylim(0.2, 1.1)
    if not have_baseline:
        ax_time_norm.text(
            0.5,
            0.5,
            "CEM baseline missing",
            transform=ax_time_norm.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            color="gray",
        )
    ax_steps.set_title("Model Steps (est.)")
    ax_steps.set_ylabel("model steps / env step")
    ax_steps.set_yscale("log")
    ax_steps.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _pos: _human_format(y)))
    ax_replans.set_title("Replanning")
    ax_replans.set_ylabel("replans / env step")
    ax_replans.set_ylim(-0.05, 1.05)

    for ax in (ax_reward, ax_time, ax_time_norm, ax_steps, ax_replans):
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=0.25)

    ax_steps.grid(True, which="minor", alpha=0.12)
    for ax in (ax_time, ax_time_norm, ax_steps, ax_replans):
        ax.set_xlabel(x_label)

    handles, labels = ax_reward.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(6, len(labels)),
            frameon=False,
        )
    fig.suptitle(f"{env} \u2014 {run_dir.name}")
    fig.subplots_adjust(top=0.86)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_dir.name}_{env}_training.png"
    fig.savefig(out_path, dpi=180)
    print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)

    if "bccem" in optimizers and grouped.get((env, "bccem")):
        _plot_bccem_diagnostics(
            run_dir=run_dir,
            env=env,
            series_list=grouped[(env, "bccem")],
            x_key=x_key,
            smooth=smooth,
            show_seeds=show_seeds,
            out_dir=out_dir,
            show=show,
            plt=plt,
        )


def _plot_bccem_diagnostics(
    *,
    run_dir: Path,
    env: str,
    series_list: Sequence[Dict[str, np.ndarray]],
    x_key: str,
    smooth: int,
    show_seeds: bool,
    out_dir: Path,
    show: bool,
    plt,
) -> None:
    import matplotlib.ticker as mticker

    cmap = plt.get_cmap("tab10")
    fig, (ax_mode, ax_ir) = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    _plot_with_optional_mean(
        ax_mode,
        series_list=series_list,
        x_key=x_key,
        y_key="planning_centroid_frac",
        label="centroid exec frac",
        color=cmap(0),
        smooth=smooth,
        show_seeds=show_seeds,
    )
    _plot_with_optional_mean(
        ax_mode,
        series_list=series_list,
        x_key=x_key,
        y_key="planning_replans_per_step",
        label="replan frac",
        color=cmap(1),
        smooth=smooth,
        show_seeds=show_seeds,
        linestyle="--",
    )
    _plot_with_optional_mean(
        ax_mode,
        series_list=series_list,
        x_key=x_key,
        y_key="planning_skips_per_step",
        label="skip frac",
        color=cmap(2),
        smooth=smooth,
        show_seeds=show_seeds,
        linestyle=":",
    )

    _plot_with_optional_mean(
        ax_ir,
        series_list=series_list,
        x_key=x_key,
        y_key="planning_ir_norm_mean",
        label="IR norm (mean)",
        color=cmap(3),
        smooth=smooth,
        show_seeds=show_seeds,
    )
    _fill_between_mean_range(
        ax_ir,
        series_list=series_list,
        x_key=x_key,
        y_min_key="planning_ir_norm_min",
        y_max_key="planning_ir_norm_max",
        color=cmap(3),
        smooth=smooth,
        alpha=0.10,
    )

    x_label = "Env steps" if x_key == "env_step" else "Episode"
    if x_key == "env_step":
        formatter = mticker.FuncFormatter(lambda x, _pos: _human_format(x))
        for ax in (ax_mode, ax_ir):
            ax.xaxis.set_major_formatter(formatter)
    else:
        for ax in (ax_mode, ax_ir):
            ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    ax_mode.set_title("BCCEM: Execution Strategy")
    ax_mode.set_ylabel("fraction")
    ax_mode.set_xlabel(x_label)
    ax_mode.set_ylim(-0.05, 1.05)
    ax_mode.legend(loc="best", frameon=False)

    ax_ir.set_title("BCCEM: IR Norm")
    ax_ir.set_ylabel("ir_norm")
    ax_ir.set_xlabel(x_label)
    ax_ir.legend(loc="best", frameon=False)

    for ax in (ax_mode, ax_ir):
        ax.set_axisbelow(True)
        ax.grid(True, which="major", alpha=0.25)

    fig.suptitle(f"{env} \u2014 {run_dir.name} \u2014 BCCEM diagnostics")
    fig.subplots_adjust(top=0.82, wspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{run_dir.name}_{env}_bccem.png"
    fig.savefig(out_path, dpi=160)
    print(f"[saved] {out_path}")
    if show:
        plt.show()
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PETS optimizer benchmark stats.")
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to a benchmark run directory (containing summary.csv).",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=str(DEFAULT_RUNS_ROOT),
        help="Root directory containing multiple benchmark runs.",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Benchmark run id under --runs-root (e.g., 20251217_181336).",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Filter to a single env folder (e.g., pets_cartpole). Default: all envs found.",
    )
    parser.add_argument(
        "-o",
        "--optimizers",
        nargs="+",
        default=DEFAULT_OPTIMIZERS,
        help="Optimizers to plot (default: CEM family).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Seeds to plot (default: all).",
    )
    parser.add_argument(
        "--x",
        choices=["env_step", "step"],
        default="env_step",
        help="X-axis for training curves.",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=2,
        help="Moving-average window (in episodes). Use 1 to disable.",
    )
    parser.add_argument(
        "--show-seeds",
        action="store_true",
        help="Overlay individual seed curves (otherwise show mean \u00b1 std).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output directory for plots (default: <run-dir>/plots).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show interactive windows instead of only saving PNGs.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        import matplotlib
        if not args.show:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib.\n"
            "Install requirements (e.g., `pip install -r requirements/main.txt`) and retry."
        ) from exc
    _apply_plot_style(plt)

    runs_root = Path(args.runs_root).expanduser().resolve()

    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
    elif args.run_id:
        run_dir = _resolve_run_dir(runs_root, args.run_id)
    else:
        run_dir = _find_latest_run_dir(runs_root)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    specs_all = _load_summary_specs(run_dir)
    if not specs_all:
        specs_all = _scan_results_specs(run_dir)
    specs = _filter_specs(
        specs_all, env=args.env, optimizers=args.optimizers, seeds=args.seeds
    )
    if not specs:
        envs_avail = sorted({s.env for s in specs_all})
        opts_avail = sorted({s.optimizer for s in specs_all})
        seeds_avail = sorted({s.seed for s in specs_all})
        avail_str = (
            f"Available envs: {envs_avail}\n"
            f"Available optimizers: {opts_avail}\n"
            f"Available seeds: {seeds_avail}"
            if specs_all
            else "No results.csv found under run directory."
        )
        raise FileNotFoundError(
            f"No results.csv found after filtering under {run_dir} "
            f"(env={args.env}, optimizers={args.optimizers}, seeds={args.seeds}).\n"
            f"{avail_str}"
        )

    out_dir = Path(args.out).expanduser().resolve() if args.out else (run_dir / "plots")
    grouped = _group_by_env_and_optimizer(specs, args.x)
    envs = sorted({spec.env for spec in specs})
    for env in envs:
        _plot_env(
            run_dir=run_dir,
            env=env,
            optimizers=args.optimizers,
            grouped=grouped,
            x_key=args.x,
            smooth=max(1, int(args.smooth)),
            show_seeds=bool(args.show_seeds),
            out_dir=out_dir,
            show=bool(args.show),
            plt=plt,
        )
    if not args.show:
        print(f"[done] plots in {out_dir}")


if __name__ == "__main__":
    main()
