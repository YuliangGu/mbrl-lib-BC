# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import time
import inspect
from typing import Callable, List, Optional, Sequence, Tuple, cast

import hydra
import numpy as np
import omegaconf
import torch
import torch.distributions

import mbrl.models
import mbrl.types
import mbrl.util.math

from .core import Agent, complete_agent_cfg

DISPERSION_EPS = 1e-4

ObjectiveFn = Callable[[torch.Tensor], torch.Tensor]
OptimizerCallback = Optional[Callable[[torch.Tensor, torch.Tensor, int], None]]

class Optimizer:
    """Base class for trajectory optimizers.

    For benchmarking and MPC integration, we provide a few
    optional hooks (diagnostics, persistent state).
    """

    def __init__(self):
        self._buffers: dict[str, torch.Tensor] = {}

    def _get_buffer(
        self,
        key: str,
        shape: Tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # Retrieve or create a persistent buffer tensor.
        buf = self._buffers.get(key, None)
        if (
            buf is None
            or tuple(buf.shape) != tuple(shape)
            or buf.device != device
            or buf.dtype != dtype
        ):
            buf = torch.empty(shape, device=device, dtype=dtype)
            self._buffers[key] = buf
        return buf

    @staticmethod
    def _sanitize_values_(values: torch.Tensor) -> torch.Tensor:
        bad = -torch.finfo(values.dtype).max
        try:
            torch.nan_to_num_(values, nan=bad, posinf=bad, neginf=bad)
        except TypeError:
            values[~torch.isfinite(values)] = bad
        return values

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution / warm start, if necessary.

        Returns:
            (torch.Tensor): the best solution found.
        """
        raise NotImplementedError

    # ---- Optional hooks for MPC integration / benchmarking ----
    def reset(self) -> None:
        """Clears any persistent optimizer state (called at episode reset)."""
        return None

    def get_diagnostics(self) -> dict:
        """Returns a JSON-serializable dict of diagnostics for logging."""
        return {}

class RandomOptimizer(Optimizer):
    """Implements a random search optimization algorithm.

    Args:
        population_size (int): the size of the population.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        device (torch.device): device where computations will be performed.
    """

    def __init__(
        self,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        device: torch.device,
        **kwargs,   # for compatibility with other optimizers
    ):
        super().__init__()
        self.population_size = population_size
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.device = device
        self._bound_range = self.upper_bound - self.lower_bound

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        population = self._get_buffer(
            "population",
            (self.population_size,) + tuple(self.lower_bound.shape),
            device=self.device,
        )
        population.uniform_(0.0, 1.0)
        population.mul_(self._bound_range).add_(self.lower_bound)
        values = obj_fun(population)

        if callback is not None:
            callback(population, values, 0)

        values = self._sanitize_values_(values).reshape(-1)
        best_idx = int(torch.argmax(values).detach().cpu())
        best_solution = population[best_idx].clone()

        return best_solution

class CEMOptimizer(Optimizer):
    """Implements the Cross-Entropy Method optimization algorithm.

    A good description of CEM [1] can be found at https://arxiv.org/pdf/2008.06389.pdf. This
    code implements the version described in Section 2.1, labeled CEM_PETS
    (but note that the shift-initialization between planning time steps is handled outside of
    this class by TrajectoryOptimizer).

    This implementation also returns the best solution found as opposed
    to the mean of the last generation.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.
        clipped_normal (bool); if ``True`` samples are drawn from a normal distribution
            and clipped to the bounds. If ``False``, sampling uses a truncated normal
            distribution up to the bounds. Defaults to ``False``.

    [1] R. Rubinstein and W. Davidson. "The cross-entropy method for combinatorial and continuous
    optimization". Methodology and Computing in Applied Probability, 1999.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = max(
            1, np.ceil(self.population_size * self.elite_ratio).astype(np.int32)
        )
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device

        self._clipped_normal = clipped_normal

    def _init_population_params(
        self, x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x0.clone()
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            dispersion = ((self.upper_bound - self.lower_bound) ** 2) / 16
        return mean, dispersion

    def _sample_population(
        self, mean: torch.Tensor, dispersion: torch.Tensor, population: torch.Tensor
    ) -> torch.Tensor:
        # fills population with random samples
        # for truncated normal, dispersion should be the variance
        # for clipped normal, dispersion should be the standard deviation
        if self._clipped_normal:
            population.normal_()
            population.mul_(dispersion).add_(mean)
            torch.maximum(population, self.lower_bound, out=population)
            torch.minimum(population, self.upper_bound, out=population)
            return population
        else:
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, dispersion)

            mbrl.util.math.truncated_normal_(population)
            population.mul_(torch.sqrt(constrained_var)).add_(mean)
            return population

    def _update_population_params(
        self, elite: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_mu = torch.mean(elite, dim=0)
        if self._clipped_normal:
            new_dispersion = torch.std(elite, dim=0, unbiased=False)
        else:
            new_dispersion = torch.var(elite, dim=0, unbiased=False)
        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
        dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)
        return mu, dispersion

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs the optimization using CEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.

        Returns:
            (torch.Tensor): the best solution found.
        """
        mu, dispersion = self._init_population_params(x0)
        best_solution = torch.empty_like(mu)
        best_value = -float("inf")
        population = self._get_buffer(
            "population",
            (self.population_size,) + tuple(x0.shape),
            device=self.device,
        )
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            values = obj_fun(population)

            if callback is not None:
                callback(population, values, i)

            self._sanitize_values_(values)
            best_values, elite_idx = values.topk(self.elite_num)
            elite = population[elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        return mu if self.return_mean_elites else best_solution

class DecentCEMOptimizer(Optimizer):
    """Runs multiple independent CEM optimizers in parallel and picks the best result."""

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        total_population_size: Optional[int] = None,
        num_workers: int = 4,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
        init_jitter_scale: float = 0.0,
        min_component_weight: float = 0.1,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.num_workers = num_workers
        if total_population_size is not None:
            population_size = int(math.ceil(float(total_population_size) / float(max(1, num_workers)))) # rounded up.
        self.population_size = int(population_size)
        self.total_population_size = self.population_size * self.num_workers
        self.elite_num = max(1, int(np.ceil(self.population_size * self.elite_ratio)))
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        self._clipped_normal = clipped_normal
        self._eps = 1e-8
        self.init_jitter_scale = init_jitter_scale

    def _init_population_params(
        self, x0: torch.Tensor, jitter_scale: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers):
            mean = x0.clone()
        else:
            mean = x0.expand((self.num_workers,) + x0.shape)
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            base_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
            dispersion = base_var.expand_as(mean)
        if jitter_scale > 0:
            mean = torch.clamp(
                mean
                + jitter_scale
                * torch.randn_like(mean)
                * torch.sqrt(dispersion + self._eps),
                min=self.lower_bound,
                max=self.upper_bound,
            )
        return mean, dispersion

    def _sample_population(
        self, mean: torch.Tensor, dispersion: torch.Tensor, population: torch.Tensor
    ) -> torch.Tensor:
        if self._clipped_normal:
            population.normal_()
            population.mul_(dispersion.unsqueeze(1)).add_(mean.unsqueeze(1))
            torch.maximum(population, self.lower_bound, out=population)
            torch.minimum(population, self.upper_bound, out=population)
            return population

        lb_dist = mean - self.lower_bound
        ub_dist = self.upper_bound - mean
        mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
        constrained_var = torch.min(mv, dispersion)

        mbrl.util.math.truncated_normal_(population)
        population.mul_(torch.sqrt(constrained_var).unsqueeze(1)).add_(mean.unsqueeze(1))
        return population

    def _update_population_params(
        self, elite: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor, alpha: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if alpha is None:
            alpha = self.alpha
        new_mu = torch.mean(elite, dim=1)
        if self._clipped_normal:
            new_dispersion = torch.std(elite, dim=1, unbiased=False)
        else:
            new_dispersion = torch.var(elite, dim=1, unbiased=False)
        mu = alpha * mu + (1 - alpha) * new_mu
        dispersion = alpha * dispersion + (1 - alpha) * new_dispersion
        dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)
        return mu, dispersion

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        if x0 is None:
            raise ValueError("DecentCEMOptimizer requires an initial solution x0.")
        action_shape = tuple(x0.shape)
        x0_like = x0
        if x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers):
            action_shape = tuple(x0.shape[1:])
            x0_like = x0[0]

        mu, dispersion = self._init_population_params(x0, jitter_scale=self.init_jitter_scale)
        best_solution = torch.empty_like(x0_like)
        best_value = -float("inf")
        best_worker_idx = 0
        population = self._get_buffer(
            "population",
            (self.num_workers, self.population_size) + tuple(action_shape),
            device=self.device,
        )
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            flat_population = population.reshape(-1, *action_shape)
            values = obj_fun(flat_population).reshape(
                self.num_workers, self.population_size
            )
            self._sanitize_values_(values)

            if callback is not None:
                callback(flat_population, values.reshape(-1), i)

            best_values, elite_idx = values.topk(self.elite_num, dim=1)
            worker_indices = torch.arange(self.num_workers, device=self.device).view(
                -1, 1
            )
            elite = population[worker_indices, elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)

            worker_best_values = best_values[:, 0]
            candidate_value, candidate_worker = torch.max(worker_best_values, dim=0)
            candidate_value_item = float(candidate_value)
            if candidate_value_item > best_value:
                best_value = candidate_value_item
                best_worker_idx = int(candidate_worker)
                best_solution = population[
                    best_worker_idx, elite_idx[best_worker_idx, 0]
                ].clone()

        # Expose final worker distributions (useful for POPLIN-style distillation).
        try:
            self._last_mu_w = mu.detach().clone()
            self._last_disp_w = dispersion.detach().clone()
            scores = worker_best_values.detach()  # type: ignore[name-defined]
            scores = scores - torch.max(scores)
            weights = torch.softmax(scores, dim=0)
            self._last_worker_weights = weights.detach().clone()
            self._last_best_worker_idx = int(best_worker_idx)
        except Exception:
            pass

        if self.return_mean_elites:
            return mu[best_worker_idx]
        return best_solution

class BCCEMOptimizer(DecentCEMOptimizer):
    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        total_population_size: Optional[int] = None,
        num_workers: int = 4,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
        init_jitter_scale: float = 0.0,
        # ---- BC-CEM specific knobs ----
        consensus_coef: float = 0.2,
        consensus_anneal_power: float = 0.9,
        use_value_weights: bool = True,
        ir_low: float = 0.3,
        ir_high: float = 1.0,
        early_stop: bool = True,
        early_stop_mu: float = 1e-3,
        early_stop_patience: int = 1,
    ):
        super().__init__(
            num_iterations=num_iterations,
            elite_ratio=elite_ratio,
            population_size=population_size,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            alpha=alpha,
            device=device,
            total_population_size=total_population_size,
            num_workers=num_workers,
            return_mean_elites=return_mean_elites,
            clipped_normal=clipped_normal,
            init_jitter_scale=init_jitter_scale,
        )

        self.consensus_coef = float(consensus_coef)
        self.consensus_anneal_power = float(consensus_anneal_power)
        self.use_value_weights = bool(use_value_weights)
        self.ir_low = float(ir_low)
        self.ir_high = float(ir_high)
        self.early_stop = bool(early_stop)
        self.early_stop_mu = float(early_stop_mu)
        self.early_stop_patience = int(early_stop_patience)

        # BC-CEM internal state used by MPC 
        self._centroid_type = "moment"   # "moment" or "geometric"
        self._diagnostics: dict = {}
        self._centroid_mean: Optional[torch.Tensor] = None
        self._centroid_dispersion: Optional[torch.Tensor] = None

        # Persistent MPC state (kept across calls to optimize; clear on reset
        self._exec_mode = "centroid"
        self._sticky_worker_idx: Optional[int] = None
        self._state_mu_w: Optional[torch.Tensor] = None
        self._state_disp_w: Optional[torch.Tensor] = None
        self._last_ir_norm: Optional[float] = None
        self._last_jitter_scale = self.init_jitter_scale

        self.use_sticky_worker = True
        self.sticky_patience_max = 5
        self.sticky_patience = self.sticky_patience_max
        self.var_eps = DISPERSION_EPS

        # ---- IR adaptation ----
        self.ir_adapt = True
        self.ir_adapt_alpha = 0.97         # slow update; 0.95–0.995 are reasonable
        self.ir_adapt_warmup = 10          # MPC calls before using adapted thresholds
        self.ir_adapt_k_low = 0.95          # band width below mean in log-space
        self.ir_adapt_k_high = 0.95         # band width above mean in log-space
        self.ir_min_low = 0.01
        self.ir_max_high = 10.0
        self.ir_min_gap = 0.05            

        self._ir_log_ema = None
        self._ir_log_var_ema = None
        self._ir_adapt_steps = 0

    def _init_population_params(
        self, x0: torch.Tensor, jitter_scale: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        W = int(self.num_workers)
        is_worker_init = bool(x0.dim() == 3 and int(x0.shape[0]) == W)

        # Start from provided means (per-worker if available).
        mean = x0.clone() if is_worker_init else x0.expand((W,) + x0.shape)
        action_range = (self.upper_bound - self.lower_bound).unsqueeze(0)

        # If the initial solution isn't per-worker, add noise to break symmetry across workers.
        if not is_worker_init:
            mean = torch.clamp(
                mean + (0.01 * action_range) * torch.randn_like(mean),
                min=self.lower_bound,
                max=self.upper_bound,
            )

        # Initialize dispersion around the default CEM scale.
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
            if not is_worker_init:
                dispersion = dispersion + (0.05 * action_range) * torch.randn_like(mean).abs()
        else:
            base_var = (((self.upper_bound - self.lower_bound) ** 2) / 16).unsqueeze(0)
            dispersion = base_var.expand_as(mean)
            if not is_worker_init:
                # Multiplicative perturbation keeps units consistent (variance).
                dispersion = dispersion * (1.0 + 0.5 * torch.randn_like(mean).abs())

        dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)

        # Optional extra jitter (typically annealed across MPC steps).
        if jitter_scale > 0:
            mean = torch.clamp(
                mean
                + jitter_scale
                * torch.randn_like(mean)
                * torch.sqrt(dispersion + self._eps),
                min=self.lower_bound,
                max=self.upper_bound,
            )
        return mean, dispersion

    def reset(self) -> None:
        self._diagnostics = {}
        self._centroid_mean = None
        self._centroid_dispersion = None
        self._sticky_worker_idx = None
        self.sticky_patience = self.sticky_patience_max
        self._state_mu_w = None
        self._state_disp_w = None
        self._last_jitter_scale = self.init_jitter_scale
        # self._last_ir_norm = None
    
    def get_diagnostics(self) -> dict:
        out = {}
        for k, v in getattr(self, "_diagnostics", {}).items():
            if isinstance(v, (np.floating, np.integer)):
                out[k] = v.item()
            elif torch.is_tensor(v):
                out[k] = float(v.detach().cpu().item()) if v.numel() == 1 else None
            else:
                out[k] = v
        return out

    def _as_var(self, dispersion: torch.Tensor) -> torch.Tensor:
        if self._clipped_normal:
            return torch.square(dispersion)
        return dispersion

    def _from_var(self, var: torch.Tensor) -> torch.Tensor:
        if self._clipped_normal:
            return torch.sqrt(var)
        return var
    
    @staticmethod
    def _diag_gaussian_kl(
        mu_p: torch.Tensor, var_p: torch.Tensor, mu_q: torch.Tensor, var_q: torch.Tensor
    ) -> torch.Tensor:
        var_p = torch.clamp(var_p, min=DISPERSION_EPS)
        var_q = torch.clamp(var_q, min=DISPERSION_EPS)
        log_term = torch.log(var_q / var_p)
        quad = (var_p + torch.square(mu_p - mu_q)) / var_q
        elem = log_term + quad - 1.0
        return 0.5 * elem.sum(dim=tuple(range(1, elem.dim())))

    def _compute_worker_weights(
        self, worker_scores: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        W = int(worker_scores.shape[0])
        if (not self.use_value_weights) or (W <= 1):
            return torch.full((W,), 1.0 / max(1, W), device=worker_scores.device)

        # EXPERIMENTAL: softmax weighting
        logits = worker_scores / temperature
        logits = logits - torch.max(logits)
        
        # EXPERIMENTAL: standard score weighting     
        # logits = (worker_scores - worker_scores.mean()) / (worker_scores.std(unbiased=False) + 1e-8)
        # w = torch.softmax(logits, dim=0)

        # # EXPERIMENTAL: rank weighting (better stability) 
        # ranks = torch.argsort(torch.argsort(-worker_scores))
        # logits = -ranks.to(dtype=worker_scores.dtype)  

        w = torch.softmax(logits, dim=0)
        return w

    def _centroid_from_workers(
        self, mu_w: torch.Tensor, disp_w: torch.Tensor, w: torch.Tensor, centroid_type: str = "moment"
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Computes the weighted centroid of worker distributions.
        
        Centroid types:
          - "moment": moment-matching centroid | KL direction: centroid -> workers
          - "geometric": geometric centroid | KL direction: workers -> centroid
        """
        w_view = w.view(-1, 1, 1)
        var_w = self._as_var(disp_w)

        if centroid_type == "geometric":
            inv_var_w = 1.0 / torch.clamp(var_w, min=self.var_eps)
            prec_c = torch.sum(w_view * inv_var_w, dim=0)
            mu_c = torch.sum(w_view * mu_w * inv_var_w, dim=0) / prec_c
            var_c = 1.0 / prec_c
            disp_c = self._from_var(var_c)
            return mu_c, disp_c, var_c
        
        mu_c = torch.sum(w_view * mu_w, dim=0)
        second_w = var_w + torch.square(mu_w)
        second_c = torch.sum(w_view * second_w, dim=0)
        var_c = second_c - torch.square(mu_c)
        var_c = torch.clamp(var_c, min=self.var_eps)
        disp_c = self._from_var(var_c)
        return mu_c, disp_c, var_c

    def _update_ir_stats(self, ir_norm: Optional[float]) -> None:
        if ir_norm is None:
            return
        try:
            x = float(ir_norm)
        except (TypeError, ValueError):
            return
        if (not math.isfinite(x)) or (x <= 0.0):
            return
        lx = math.log(x + 1e-12)

        if self._ir_log_ema is None or self._ir_log_var_ema is None:
            self._ir_log_ema = lx
            self._ir_log_var_ema = 0.0
            self._ir_adapt_steps = 1
            return

        a = float(self.ir_adapt_alpha)
        m_old = float(self._ir_log_ema)                 # old mean for variance update
        self._ir_log_ema = a * m_old + (1.0 - a) * lx
        # EWMA variance on residuals (stable enough for thresholding)
        r = lx - m_old
        v_old = float(self._ir_log_var_ema)
        self._ir_log_var_ema = a * v_old + (1.0 - a) * (r * r)

        self._ir_adapt_steps += 1


    def _adaptive_ir_thresholds(self) -> Tuple[float, float]:
        # fallback to config defaults
        low0, high0 = float(self.ir_low), float(self.ir_high)

        if (not getattr(self, "ir_adapt", False)) or self._ir_log_ema is None or self._ir_log_var_ema is None:
            return low0, high0
        if int(getattr(self, "_ir_adapt_steps", 0)) < int(getattr(self, "ir_adapt_warmup", 0)):
            return low0, high0

        mu = float(self._ir_log_ema)
        std = math.sqrt(max(1e-12, float(self._ir_log_var_ema)))

        low = math.exp(mu - float(self.ir_adapt_k_low) * std)
        high = math.exp(mu + float(self.ir_adapt_k_high) * std)

        low = max(float(self.ir_min_low), low)
        high = min(float(self.ir_max_high), high)
        if high <= low + float(self.ir_min_gap):
            high = low + float(self.ir_min_gap)

        return low, high

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        self._diagnostics = {}
        curr_low, curr_high = self._adaptive_ir_thresholds()

        if x0 is None:
            raise ValueError("BCCEMOptimizer requires an initial solution x0.")
        x0 = x0.to(device=self.device)
        action_shape = tuple(x0.shape)
        x0_like = x0
        if x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers):
            action_shape = tuple(x0.shape[1:])
            x0_like = x0[0]

        # anneal jitter scale
        if self.init_jitter_scale > 0:
            self._last_jitter_scale *= 0.9
            self._last_jitter_scale = max(self._last_jitter_scale, 0.0)

        # warm start
        mu_w, disp_w = self._init_population_params(x0, jitter_scale=self._last_jitter_scale)
        if (
            (self._state_mu_w is not None)
            and (self._state_disp_w is not None)
            and not (x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers))
        ):
            mu_prev, disp_prev = self._state_mu_w, self._state_disp_w
            H = int(mu_w.shape[1])
            if H > 1:
                frac = torch.arange(H, device=mu_w.device, dtype=mu_w.dtype).mul_(1.0 / float(H - 1))
                w_state = ((1.0 - frac) * (1.0 - frac)).view(1, H, 1)
            else:
                w_state = torch.zeros((1, H, 1), device=mu_w.device, dtype=mu_w.dtype)

            mu_w = torch.lerp(mu_w, mu_prev, w_state)
            disp_w = torch.lerp(disp_w, disp_prev, w_state)
            mu_w = torch.clamp(mu_w, min=self.lower_bound, max=self.upper_bound)
            disp_w = torch.clamp(disp_w, min=DISPERSION_EPS)
        else:
            self._state_mu_w = None
            self._state_disp_w = None

        W = int(self.num_workers)
        weights = torch.full((W,), 1.0 / max(1, W), device=self.device)
        dim = int(x0_like.numel())
        population = self._get_buffer(
            "population",
            (W, self.population_size) + tuple(action_shape),
            device=self.device,
        )

        best_solution = torch.empty_like(x0_like)
        best_value = -float("inf")
        best_worker_idx = 0
        worker_best_solution = self._get_buffer(
            "worker_best_solution",
            (W,) + tuple(action_shape),
            device=self.device,
        )
        worker_best_value = torch.full((W,), -float("inf"), device=self.device)

        prev_mu_c: Optional[torch.Tensor] = None
        stall_count = 0

        ir_norm = None
        ir_max = None

        for i in range(self.num_iterations):
            population = self._sample_population(mu_w, disp_w, population)
            flat_population = population.reshape(-1, *action_shape)
            values = obj_fun(flat_population).reshape(W, self.population_size)
            self._sanitize_values_(values)
                
            if callback is not None:
                callback(flat_population, values.reshape(-1), i)

            best_values, elite_idx = values.topk(self.elite_num, dim=1)
            worker_indices = torch.arange(W, device=self.device).view(-1, 1)
            elite = population[worker_indices, elite_idx] 
            mu_w, disp_w = self._update_population_params(elite, mu_w, disp_w)

            iter_best_vals, iter_best_solutions = best_values[:, 0], elite[:, 0]
            improved = iter_best_vals > worker_best_value
            if improved.any():
                idx = improved.nonzero(as_tuple=False).view(-1)
                worker_best_value[idx] = iter_best_vals[idx]
                worker_best_solution[idx].copy_(iter_best_solutions[idx])

            cand_val, cand_worker = torch.max(iter_best_vals, dim=0)
            cand_val_item = float(cand_val)
            if cand_val_item > best_value:
                best_value = cand_val_item
                best_worker_idx = int(cand_worker)
                best_solution = iter_best_solutions[best_worker_idx].clone()

            # value_quantiles = torch.quantile(values, 0.5, dim=1)  
            value_quantiles = torch.mean(best_values, dim=1)  
            weights = self._compute_worker_weights(value_quantiles)
            mu_c, disp_c, var_c = self._centroid_from_workers(
                mu_w, disp_w, weights, centroid_type=self._centroid_type
            )
            var_w = self._as_var(disp_w)

            # compute inter-worker KL divergence
            if self._centroid_type == "moment":
                kl_w = self._diag_gaussian_kl(mu_w, var_w, mu_c, var_c)
            else:
                kl_w = self._diag_gaussian_kl(mu_c, var_c, mu_w, var_w)

            ir_avg, ir_max = torch.sum(weights * kl_w), torch.max(kl_w)
            ir_norm = ir_avg / float(max(1, dim))
            ir_max = float(ir_max.detach().cpu())
            ir_norm = float(ir_norm.detach().cpu())


            # gated consensus coefficient
            beta_base = self.consensus_coef
            beta = beta_base * float(((i + 1) / max(1, self.num_iterations)) ** self.consensus_anneal_power)

            if beta > 0.0 and W > 1:
                # Scale consensus per worker by its normalized KL to the centroid.
                scale = (kl_w / float(max(1, dim))).clamp(min=0.0)  # (W,)
                beta_w = (beta / (1.0 + scale)).clamp(0.0, 1.0)
                beta_w = beta_w.view(-1, 1, 1)

                # Unimodal: consensus update
                if ir_norm < curr_low:
                    mu_w = (1.0 - beta_w) * mu_w + beta_w * mu_c.unsqueeze(0)
                    mu_w = torch.clamp(mu_w, min=self.lower_bound, max=self.upper_bound)

                    var_w = self._as_var(disp_w)
                    var_w = (1.0 - beta_w) * var_w + beta_w * var_c.unsqueeze(0)
                    var_w = torch.clamp(var_w, min=self.var_eps)
                    disp_w = self._from_var(var_w)

                else:
                    """
                    Decomposes the scalar Information Radius into Principal Components of Fisher-weighted disagreement.
                    """ 
                    # FAST path
                    mu_diff = mu_w - mu_c.unsqueeze(0)                               # (W,H,A)
                    sigma_noise_diag = torch.clamp(var_w.mean(dim=0), min=self.var_eps)  # (H,A)
                    inv_sqrt = torch.rsqrt(sigma_noise_diag)                         # (H,A)
                    sqrt_w = torch.sqrt(weights).view(W, 1, 1)                       # (W,1,1)

                    # Zt: (H,A,W) where columns are z_w = sqrt(pi_w)*D^{-1/2}*(mu_w - mu_c)
                    Zt = (mu_diff * sqrt_w * inv_sqrt.unsqueeze(0)).permute(1, 2, 0).contiguous()  # (H,A,W)
                    
                    # Gram matrix: G = Z^T Z  (H,W,W)
                    G = torch.matmul(Zt.transpose(-2, -1), Zt)                       # (H,W,W)
                    lam, U = torch.linalg.eigh(G)                                    # lam asc, U columns  (H,W), (H,W,W)
                    tau = 1.0
                    lam_safe = torch.clamp(lam, min=1e-12)

                    inv_lam_disagree = (lam > tau).to(mu_w.dtype) / lam_safe         # (H,W)
                    delta = (mu_c.unsqueeze(0) - mu_w).permute(1, 0, 2).contiguous() # (H,W,A)
                    ZtT = Zt.transpose(-2, -1)       
                    y = torch.matmul(ZtT.unsqueeze(1), delta.unsqueeze(-1)).squeeze(-1)  # (H,W,W)

                    # coordinates in eigenbasis of G
                    c = torch.matmul(y, U)                                           # (H,W,W)

                    # apply Λ^{-1} only to disagree coords (others 0)
                    c = c * inv_lam_disagree.unsqueeze(1)                             # (H,W,W)

                    # back to W-space: y_back = U c
                    y_back = torch.matmul(c, U.transpose(-2, -1))                     # (H,W,W)

                    # project back to A: proj_disagree = (Z y_back)  (row-form)
                    proj_disagree = torch.matmul(y_back, ZtT)                         # (H,W,A)
                    delta_shared = (delta - proj_disagree).permute(1, 0, 2).contiguous() # (W,H,A)

                    # Apply the same per-worker gated consensus strength to the shared component.
                    mu_w = mu_w + beta_w * delta_shared
                    mu_w = torch.clamp(mu_w, min=self.lower_bound, max=self.upper_bound)

                    # Keep variance unchanged

                    # mu_diff = mu_w - mu_c.unsqueeze(0)              # (W,H,A)
                    # w_view = weights.view(-1, 1, 1, 1)              # (W,1,1,1)
                    # sigma_signal = torch.sum(w_view * (mu_diff.unsqueeze(-1) * mu_diff.unsqueeze(-2)), dim=0)  # (H,A,A)
                    # sigma_noise_diag = torch.clamp(var_w.mean(dim=0), min=self.var_eps)  # (H,A)
                    # inv_sqrt = 1.0 / torch.sqrt(sigma_noise_diag)                          # (H,A)

                    # # symmetric whitened disagreement matrix
                    # F = sigma_signal * inv_sqrt.unsqueeze(-1) * inv_sqrt.unsqueeze(-2)     # (H,A,A)
                    # evals, evecs = torch.linalg.eigh(F)                                    # evals asc, evecs columns

                    # tau = 1.0  # threshold for shared subspace
                    # shared_mask = (evals <= tau).to(dtype=mu_w.dtype)                      # (H,A)

                    # # project (mu_c - mu_w) onto shared subspace
                    # delta = (mu_c.unsqueeze(0) - mu_w)                                     # (W,H,A)
                    # Vt = evecs.transpose(-2, -1)                                           # (H,A,A)

                    # # coeff = V^T delta
                    # coeff = torch.matmul(Vt.unsqueeze(0), delta.unsqueeze(-1)).squeeze(-1) # (W,H,A)
                    # coeff = coeff * shared_mask.unsqueeze(0)                               # zero mode directions
                    # delta_shared = torch.matmul(evecs.unsqueeze(0), coeff.unsqueeze(-1)).squeeze(-1)  # (W,H,A)

                    # beta_shared = self.consensus_coef
                    # mu_w = mu_w + (beta_shared * beta_w) * delta_shared
                    # mu_w = torch.clamp(mu_w, min=self.lower_bound, max=self.upper_bound)

                    # # keep variance unchanged
                    # # optional: share variance in shared subspace (diagonal approximation)
                    # V2 = evecs * evecs                                                    # (H,A,A)
                    # s_w = torch.einsum("hak,wha->whk", V2, var_w)                          # (W,H,A)
                    # s_c = torch.einsum("hak,ha->hk", V2, var_c)                            # (H,A)

                    # mask_w = shared_mask.unsqueeze(0)                                      # (1,H,A)
                    # s_w_new = (1.0 - (beta_shared * beta_w)) * s_w + (beta_shared * beta_w) * s_c.unsqueeze(0)
                    # s_w = mask_w * s_w_new + (1.0 - mask_w) * s_w                          # only shared directions
                    # var_w = torch.einsum("hak,whk->wha", V2, s_w)                           # back to diag
                    # var_w = torch.clamp(var_w, min=self.var_eps)
                    # disp_w = self._from_var(var_w)
                
            if self.early_stop:
                mu_shift = None
                var_c_flag = False
                if prev_mu_c is not None:
                    mu_shift = float(
                        torch.sqrt(torch.mean(torch.square(mu_c - prev_mu_c))).detach().cpu()
                    )
                    var_c_flag = bool(torch.mean(var_c) <= self.var_eps * 10.0)
                prev_mu_c = mu_c.detach()
                if mu_shift is not None:
                    if (mu_shift <= self.early_stop_mu) and var_c_flag:
                        stall_count += 1
                    else:
                        stall_count = 0
                    if stall_count >= self.early_stop_patience:
                        break

        mu_c, disp_c, var_c = self._centroid_from_workers(mu_w, disp_w, weights)
        self._centroid_mean = mu_c.detach()
        self._centroid_dispersion = disp_c.detach()

        exec_solution = best_solution
        exec_source = "best"
        mode = getattr(self, "_exec_mode", "centroid")
        
        if self.return_mean_elites:
            if mode not in {"centroid", "worker"}:
                mode = "centroid"
            if ir_norm is not None:
                # Hysteresis on IR to avoid rapid switching.
                if mode == "worker":
                    if ir_norm < curr_low:
                        mode = "centroid"
                else:
                    if ir_norm > curr_high:
                        mode = "worker"
            
            self._exec_mode = mode

            if mode == "centroid":
                exec_solution = mu_c
                exec_source = "centroid"
            else:
                chosen, chosen_ok = self._sticky_worker_idx, False
                if self.use_sticky_worker and chosen is not None and self.sticky_patience > 0:
                    chosen_ok = True
                    self.sticky_patience -= 1
                if not chosen_ok:
                    if (not torch.isfinite(weights).all()) or float(weights.sum()) <= 0.0:
                        weights = torch.full_like(weights, 1.0 / max(1, int(weights.numel())))
                    chosen = int(
                        torch.distributions.Categorical(weights).sample().detach().cpu().item()
                    )
                    self.sticky_patience = self.sticky_patience_max 
                self._sticky_worker_idx = chosen
                exec_solution = mu_w[chosen]
                exec_source = f"worker_{chosen}"

        self._last_ir_norm = ir_norm
        self._update_ir_stats(ir_norm)

        # Expose final worker distributions (useful for POPLIN).
        self._last_mu_w = mu_w.detach().clone()
        self._last_disp_w = disp_w.detach().clone()
        self._last_worker_weights = weights.detach().clone()

        self._state_mu_w = mu_w.detach().clone()
        self._state_disp_w = disp_w.detach().clone()

        self._diagnostics = {
            "sticky_worker_idx": int(self._sticky_worker_idx) if self._sticky_worker_idx is not None else None,
            "ir_norm": float(ir_norm) if ir_norm is not None else None,
            "ir_low": float(curr_low),
            "ir_high": float(curr_high),
            "ir_max": float(ir_max) if ir_max is not None else None,
            "exec_source": exec_source,
        }

        return exec_solution
    
    def advance(self, steps: int = 1) -> None:
        """Advances the internal state by the given number of steps and fills in new steps with default distributions."""
        # Sanity checks
        if self._state_mu_w is None or self._state_disp_w is None:
            return
        if steps <= 0:
            return

        mu = self._state_mu_w
        disp = self._state_disp_w

        if (not torch.is_tensor(mu)) or (not torch.is_tensor(disp)) or mu.dim() != 3 or disp.dim() != 3:
            self._state_mu_w = None
            self._state_disp_w = None
            self._sticky_worker_idx = None
            return

        W, H, A = int(mu.shape[0]), int(mu.shape[1]), int(mu.shape[2])
        mid = (self.lower_bound + self.upper_bound) * 0.5  # (H,A)

        # If we are advancing past the horizon, reset to mid-range
        if steps >= H:
            mu_new = mid.unsqueeze(0).expand(W, H, A).clone()
            if self._clipped_normal:
                base_disp = torch.ones_like(mid)            # (H,A) std
            else:
                base_disp = ((self.upper_bound - self.lower_bound) ** 2) / 16  # (H,A) var
            disp_new = base_disp.unsqueeze(0).expand(W, H, A).clone()
            self._state_mu_w = mu_new
            self._state_disp_w = torch.clamp(disp_new, min=DISPERSION_EPS)
            self._sticky_worker_idx = None
            return

        # Roll the distributions and fill in the last `steps` with mid-range values
        mu = mu.roll(shifts=-steps, dims=1)
        disp = disp.roll(shifts=-steps, dims=1)

        mu[:, -steps:, :] = mid[-steps:, :].unsqueeze(0)

        if self._clipped_normal:
            base_disp = torch.ones_like(mid)                # (H,A)
        else:
            base_disp = ((self.upper_bound - self.lower_bound) ** 2) / 16      # (H,A)

        disp[:, -steps:, :] = base_disp[-steps:, :].unsqueeze(0)

        self._state_mu_w = torch.clamp(mu, min=self.lower_bound, max=self.upper_bound)
        self._state_disp_w = torch.clamp(disp, min=DISPERSION_EPS)


class GMMCEMOptimizer(Optimizer):
    """Runs a Gaussian Mixture Model variant of CEM with optional initial mean jitter."""

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        total_population_size: Optional[int] = None,
        num_workers: int = 4,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
        min_component_weight: float = 0.0,
        init_jitter_scale: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.num_workers = num_workers
        if total_population_size is not None:
            population_size = int(
                math.ceil(float(total_population_size) / float(max(1, num_workers)))
            )
        self.population_size = int(population_size)
        self.total_population_size = self.population_size * self.num_workers
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        self._clipped_normal = clipped_normal
        self._eps = 1e-8
        self.min_component_weight = float(min_component_weight)
        self.init_jitter_scale = init_jitter_scale

    @staticmethod
    def _project_onto_simplex(v: torch.Tensor, s: float) -> torch.Tensor:
        """Projects ``v`` onto the simplex ``{w >= 0, sum(w) = s}``."""
        if s <= 0:
            return torch.zeros_like(v)
        v_sorted, _ = torch.sort(v, descending=True)
        cssv = torch.cumsum(v_sorted, dim=0) - s
        ind = torch.arange(1, v.numel() + 1, device=v.device, dtype=v.dtype)
        cond = v_sorted - cssv / ind > 0
        if not bool(cond.any()):
            return torch.full_like(v, s / float(v.numel()))
        rho = int(torch.nonzero(cond, as_tuple=False)[-1].item())
        theta = cssv[rho] / float(rho + 1)
        return torch.clamp(v - theta, min=0.0)

    def _enforce_min_component_weight(self, mix: torch.Tensor) -> torch.Tensor:
        mix = torch.clamp(mix, min=0.0)
        mix = mix / (mix.sum() + self._eps)
        min_w = self.min_component_weight
        if min_w <= 0:
            return mix
        k = int(mix.numel())
        if min_w * k >= 1.0:
            return torch.full_like(mix, 1.0 / float(k))
        v = mix - min_w
        w = self._project_onto_simplex(v, 1.0 - float(k) * min_w)
        mix = w + min_w
        return mix / (mix.sum() + self._eps)

    def _init_population_params(
        self, x0: torch.Tensor, jitter_scale: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers):
            mean = x0.clone()
        else:
            mean = x0.expand((self.num_workers,) + x0.shape).clone()
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            base_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
            dispersion = base_var.expand_as(mean)
        if jitter_scale > 0:
            mean = torch.clamp(
                mean
                + jitter_scale
                * torch.randn_like(mean)
                * torch.sqrt(dispersion + self._eps),
                min=self.lower_bound,
                max=self.upper_bound,
            )
        mix = torch.full(
            (self.num_workers,), 1.0 / self.num_workers, device=self.device
        )
        return mean, dispersion, self._enforce_min_component_weight(mix)

    def _sample_population(
        self,
        mean: torch.Tensor,
        dispersion: torch.Tensor,
        mix: torch.Tensor,
        population: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draws samples from the current GMM.

        Returns population of shape ``(N, H, A)`` and the component ids for each sample.
        """
        total_samples = self.population_size * self.num_workers
        if population is None or population.shape[0] != total_samples:
            population = self._get_buffer(
                "population", (total_samples,) + tuple(mean.shape[1:]), device=self.device
            )

        comp_ids = torch.distributions.Categorical(mix).sample((total_samples,))
        comp_mean = mean[comp_ids]
        comp_dispersion = dispersion[comp_ids]

        if self._clipped_normal:
            population.normal_()
            population.mul_(comp_dispersion).add_(comp_mean)
            torch.maximum(population, self.lower_bound, out=population)
            torch.minimum(population, self.upper_bound, out=population)
            return population, comp_ids

        lb_dist = comp_mean - self.lower_bound
        ub_dist = self.upper_bound - comp_mean
        mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
        constrained_var = torch.min(mv, comp_dispersion)
        constrained_var = torch.clamp(constrained_var, min=DISPERSION_EPS)

        mbrl.util.math.truncated_normal_(population)
        population.mul_(torch.sqrt(constrained_var)).add_(comp_mean)
        torch.maximum(population, self.lower_bound, out=population)
        torch.minimum(population, self.upper_bound, out=population)
        return population, comp_ids

    def _log_prob(
        self, flat_population: torch.Tensor, mean: torch.Tensor, dispersion: torch.Tensor
    ) -> torch.Tensor:
        # flat_population: (S, H, A); mean/dispersion: (K, H, A)
        diff = flat_population.unsqueeze(0) - mean.unsqueeze(1)
        var = (dispersion ** 2 if self._clipped_normal else dispersion).unsqueeze(1)
        log_var = torch.log(var + self._eps)
        quad_term = (diff * diff) / (var + self._eps)
        reduce_dims = tuple(range(2, diff.dim()))
        dim = flat_population[0].numel()
        return -0.5 * (
            log_var.sum(dim=reduce_dims)
            + quad_term.sum(dim=reduce_dims)
            + dim * math.log(2 * math.pi)
        )

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        if x0 is None:
            raise ValueError("GMMCEMOptimizer requires an initial solution x0.")
        action_shape = tuple(x0.shape)
        x0_like = x0
        if x0.dim() == 3 and int(x0.shape[0]) == int(self.num_workers):
            action_shape = tuple(x0.shape[1:])
            x0_like = x0[0]

        mu, dispersion, mix = self._init_population_params(
            x0, jitter_scale=self.init_jitter_scale
        )
        best_solution = torch.empty_like(x0_like)
        best_value = -float("inf")
        best_worker = 0

        total_samples = self.population_size * self.num_workers
        population = self._get_buffer(
            "population", (total_samples,) + tuple(action_shape), device=self.device
        )

        for i in range(self.num_iterations):
            population, comp_ids = self._sample_population(
                mu, dispersion, mix, population
            )
            values = obj_fun(population)
            self._sanitize_values_(values)

            if callback is not None:
                callback(population, values, i)

            candidate_idx = int(torch.argmax(values))
            candidate_value = float(values[candidate_idx])
            if candidate_value > best_value:
                best_value = candidate_value
                best_solution = population[candidate_idx].clone()
                best_worker = int(comp_ids[candidate_idx])

            value_weights = torch.softmax((values - torch.max(values)) / 10.0, dim=0)

            log_prob = self._log_prob(population, mu, dispersion)
            log_mix = torch.log(mix + self._eps).unsqueeze(1)
            log_joint = log_mix + log_prob
            log_norm = torch.logsumexp(log_joint, dim=0, keepdim=True)
            responsibilities = torch.exp(log_joint - log_norm)

            weighted_resp = responsibilities * value_weights.unsqueeze(0)
            comp_weight = weighted_resp.sum(dim=1) + self._eps

            weighted_resp_exp = weighted_resp.unsqueeze(-1).unsqueeze(-1)
            flat_expanded = population.unsqueeze(0)
            comp_weight_exp = comp_weight.view(self.num_workers, 1, 1)
            new_mu = (weighted_resp_exp * flat_expanded).sum(dim=1) / comp_weight_exp
            diff = flat_expanded - new_mu.unsqueeze(1)
            new_dispersion = (
                weighted_resp_exp * diff * diff
            ).sum(dim=1) / comp_weight_exp

            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
            dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)

            mix_update = comp_weight / (comp_weight.sum() + self._eps)
            mix = self.alpha * mix + (1 - self.alpha) * mix_update
            mix = self._enforce_min_component_weight(mix)

            mu = torch.clamp(mu, min=self.lower_bound, max=self.upper_bound)

        # Expose final mixture components (useful for POPLIN-style distillation).
        try:
            self._last_mu_w = mu.detach().clone()
            self._last_disp_w = dispersion.detach().clone()
            self._last_worker_weights = mix.detach().clone()
            self._last_best_worker_idx = int(best_worker)
        except Exception:
            pass

        if self.return_mean_elites:
            return mu[best_worker]
        return best_solution

class NESOptimizer(Optimizer):
    """Implements a Natural Evolution Strategy (NES) optimizer.

    This implementation uses a diagonal Gaussian search distribution and natural-gradient
    style updates for both the mean and the (log) standard deviation of the distribution.

    Args:
        num_iterations (int): number of optimization iterations.
        population_size (int): number of sampled candidates per iteration.
        sigma (float): initial standard deviation for sampling.
        lr_mean (float): learning rate for mean updates.
        lr_sigma (float): learning rate for standard deviation updates.
        lower_bound (sequence of floats): lower bounds for the variables.
        upper_bound (sequence of floats): upper bounds for the variables.
        device (torch.device): computation device.
        return_mean_elites (bool): if ``True`` returns the final mean instead of best sample.
        min_sigma (float): minimum standard deviation allowed during optimization.
        max_sigma (float, optional): maximum standard deviation allowed. Defaults to ``None``.
    """

    def __init__(
        self,
        num_iterations: int,
        population_size: int,
        sigma: float,
        lr_mean: float,
        lr_sigma: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        device: torch.device,
        return_mean_elites: bool = False,
        min_sigma: float = 1e-3,
        max_sigma: Optional[float] = None,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.sigma = sigma
        self.lr_mean = lr_mean
        self.lr_sigma = lr_sigma
        self.return_mean_elites = return_mean_elites
        self.device = device
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self._eps = 1e-8

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs a diagonal NES optimization loop."""
        if x0 is None:
            raise ValueError("NESOptimizer requires an initial solution x0.")

        mu = x0.reshape(-1).to(self.device)
        dim = mu.shape[0]
        log_sigma = torch.full_like(mu, math.log(self.sigma))

        flat_lb = self.lower_bound.reshape(-1)
        flat_ub = self.upper_bound.reshape(-1)

        best_solution = x0.clone()
        best_value = -float("inf")

        for i in range(self.num_iterations):
            sigma_vec = torch.exp(log_sigma)
            half = self.population_size // 2
            noise_half = torch.randn((half, dim), device=self.device)
            noise = (
                torch.cat([noise_half, -noise_half], dim=0)
                if self.population_size % 2 == 0
                else torch.cat(
                    [noise_half, -noise_half, torch.randn((1, dim), device=self.device)],
                    dim=0,
                )
            )
            candidates = mu.unsqueeze(0) + sigma_vec.unsqueeze(0) * noise
            candidates = torch.clamp(candidates, min=flat_lb, max=flat_ub)
            population = candidates.view(self.population_size, *x0.shape)

            values = obj_fun(population)
            self._sanitize_values_(values)

            if callback is not None:
                callback(population, values, i)

            values_flat = values.reshape(-1)
            candidate_idx = int(torch.argmax(values_flat))
            candidate_value = float(values_flat[candidate_idx])
            if candidate_value > best_value:
                best_value = candidate_value
                best_solution = population.view(-1, *x0.shape)[candidate_idx].clone()
            best_candidate = candidates[candidate_idx]

            n = values_flat.numel()
            if n <= 1:
                continue
            weights = torch.softmax(values_flat - values_flat.max(), dim=0)
            centered = weights - weights.mean()
            std = centered.std(unbiased=False)
            if std < self._eps:
                continue
            weights = centered / (std + self._eps)

            perturbation = candidates - mu.unsqueeze(0)
            scaled = perturbation / (sigma_vec.unsqueeze(0) + self._eps)

            grad_mu = (weights.unsqueeze(1) * scaled).mean(dim=0)
            grad_log_sigma = (
                weights.unsqueeze(1) * (scaled * scaled - 1.0)
            ).mean(dim=0)

            mu = mu + self.lr_mean * sigma_vec * grad_mu
            log_sigma = log_sigma + 0.5 * self.lr_sigma * grad_log_sigma
            mu = mu + 0.2 * (best_candidate - mu)

            mu = torch.clamp(mu, min=flat_lb, max=flat_ub)
            if self.max_sigma is not None:
                log_sigma = torch.clamp(
                    log_sigma,
                    min=math.log(self.min_sigma),
                    max=math.log(self.max_sigma),
                )
            else:
                log_sigma = torch.maximum(
                    log_sigma, torch.full_like(log_sigma, math.log(self.min_sigma))
                )

        if self.return_mean_elites:
            return mu.view_as(x0)
        return best_solution


class CMAESOptimizer(Optimizer):
    """Implements a CMA-ES style optimizer with configurable covariance adaptation.

    Args:
        num_iterations (int): number of optimization iterations.
        population_size (int): number of sampled candidates per iteration.
        elite_ratio (float): fraction of top samples to use for updates.
        sigma (float): initial standard deviation for sampling.
        lower_bound (sequence of floats): lower bounds for the variables.
        upper_bound (sequence of floats): upper bounds for the variables.
        alpha (float): smoothing factor for updates.
        adaptation (str): either ``diagonal`` or ``full`` covariance adaptation.
        device (torch.device): computation device.
        return_mean_elites (bool): if ``True`` returns the final mean instead of best sample.
    """

    def __init__(
        self,
        num_iterations: int,
        population_size: int,
        elite_ratio: float,
        sigma: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        alpha: float,
        device: torch.device,
        adaptation: str = "diagonal",
        return_mean_elites: bool = False,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.population_size = population_size
        self.elite_num = max(1, int(math.ceil(population_size * elite_ratio)))
        self.sigma = sigma
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        self.adaptation = adaptation.lower()
        if self.adaptation not in ("diagonal", "full"):
            raise ValueError("adaptation must be either 'diagonal' or 'full'.")
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self._eps = 1e-8

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs a diagonal CMA-ES loop."""
        if x0 is None:
            raise ValueError("CMAESOptimizer requires an initial solution x0.")

        flat_lb = self.lower_bound.reshape(-1)
        flat_ub = self.upper_bound.reshape(-1)
        mean = x0.reshape(-1).to(self.device)
        dim = mean.shape[0]
        eye = torch.eye(dim, device=self.device, dtype=torch.float32)
        if self.adaptation == "full":
            cov = eye * (self.sigma**2)
        else:
            var = torch.full(
                (dim,), self.sigma**2, device=self.device, dtype=torch.float32
            )

        best_solution = x0.clone()
        best_value = -float("inf")

        for i in range(self.num_iterations):
            noise = torch.randn((self.population_size, dim), device=self.device)
            if self.adaptation == "full":
                chol = torch.linalg.cholesky(cov + self._eps * eye)
                candidates = mean.unsqueeze(0) + noise @ chol.T
            else:
                candidates = mean.unsqueeze(0) + noise * torch.sqrt(var).unsqueeze(0)
            candidates = torch.clamp(candidates, min=flat_lb, max=flat_ub)
            population = candidates.view(self.population_size, *x0.shape)

            values = obj_fun(population)
            self._sanitize_values_(values)

            if callback is not None:
                callback(population, values, i)

            top_values, elite_idx = values.topk(self.elite_num)
            elite = candidates[elite_idx]

            raw_weights = torch.log(
                torch.arange(1, self.elite_num + 1, device=self.device).float() + 0.5
            )
            weights = raw_weights / (raw_weights.sum() + self._eps)

            new_mean = (weights.unsqueeze(1) * elite).sum(dim=0)
            diff = elite - new_mean.unsqueeze(0)
            if self.adaptation == "full":
                weighted = weights.unsqueeze(1) * diff
                new_cov = weighted.T @ diff
                cov = self.alpha * cov + (1 - self.alpha) * new_cov
            else:
                new_var = (weights.unsqueeze(1) * (diff * diff)).sum(dim=0)
                var = self.alpha * var + (1 - self.alpha) * new_var

            mean = self.alpha * mean + (1 - self.alpha) * new_mean

            mean = torch.clamp(mean, min=flat_lb, max=flat_ub)
            if self.adaptation == "full":
                cov = 0.5 * (cov + cov.T) + self._eps * eye
                cov = torch.clamp(cov, min=DISPERSION_EPS)
            else:
                var = torch.clamp(var, min=DISPERSION_EPS)
                

            if top_values[0] > best_value:
                best_value = float(top_values[0])
                best_solution = population[elite_idx[0]].clone()

        if self.return_mean_elites:
            return mean.view_as(x0)
        return best_solution


class MPPIOptimizer(Optimizer):
    """Implements the Model Predictive Path Integral optimization algorithm.

    A derivation of MPPI can be found at https://arxiv.org/abs/2102.09027
    This version is closely related to the original TF implementation used in PDDM with
    some noise sampling modifications and the addition of refinement steps.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        population_size (int): the size of the population.
        gamma (float): reward scaling term.
        sigma (float): noise scaling term used in action sampling.
        beta (float): correlation term between time steps.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        device (torch.device): device where computations will be performed.
    """

    def __init__(
        self,
        num_iterations: int,
        population_size: int,
        gamma: float,
        sigma: float,
        beta: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        device: torch.device,
    ):
        super().__init__()
        self.planning_horizon = len(lower_bound)
        self.population_size = population_size
        self.action_dimension = len(lower_bound[0])
        self.mean = torch.zeros(
            (self.planning_horizon, self.action_dimension),
            device=device,
            dtype=torch.float32,
        )

        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.var = sigma**2 * torch.ones_like(self.lower_bound)
        self.beta = beta
        self.gamma = gamma
        self.refinements = num_iterations
        self.device = device

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        """Implementation of MPPI planner.
        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): Not required
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        past_action = self.mean[0]
        self.mean[:-1] = self.mean[1:].clone()

        for k in range(self.refinements):
            # sample noise and update constrained variances
            noise = self._get_buffer(
                "noise",
                (self.population_size, self.planning_horizon, self.action_dimension),
                device=self.device,
            )
            mbrl.util.math.truncated_normal_(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.minimum(mv, self.var)
            population = self._get_buffer(
                "population",
                (self.population_size, self.planning_horizon, self.action_dimension),
                device=self.device,
            )
            population.copy_(noise)
            population.mul_(torch.sqrt(constrained_var))

            # smoothed actions with noise
            population[:, 0, :] = (
                self.beta * (self.mean[0, :] + noise[:, 0, :])
                + (1 - self.beta) * past_action
            )
            for i in range(max(self.planning_horizon - 1, 0)):
                population[:, i + 1, :] = (
                    self.beta * (self.mean[i + 1] + noise[:, i + 1, :])
                    + (1 - self.beta) * population[:, i, :]
                )
            # clipping actions
            # This should still work if the bounds between dimensions are different.
            torch.minimum(population, self.upper_bound, out=population)
            torch.maximum(population, self.lower_bound, out=population)
            values = obj_fun(population)
            self._sanitize_values_(values)

            if callback is not None:
                callback(population, values, k)

            # weight actions
            weights = torch.reshape(
                torch.exp(self.gamma * (values - values.max())),
                (self.population_size, 1, 1),
            )
            norm = torch.sum(weights) + 1e-10
            weighted_actions = population * weights
            self.mean = torch.sum(weighted_actions, dim=0) / norm

        return self.mean.clone()


class ICEMOptimizer(Optimizer):
    """Implements the Improved Cross-Entropy Method (iCEM) optimization algorithm.

    iCEM improves the sample efficiency over standard CEM and was introduced by
    [2] for real-time planning.

    Args:
        num_iterations (int): the number of iterations (generations) to perform.
        elite_ratio (float): the proportion of the population that will be kept as
            elite (rounds up).
        population_size (int): the size of the population.
        population_decay_factor (float): fixed factor for exponential decrease in population size
        colored_noise_exponent (float): colored-noise scaling exponent for generating correlated
            action sequences.
        lower_bound (sequence of floats): the lower bound for the optimization variables.
        upper_bound (sequence of floats): the upper bound for the optimization variables.
        keep_elite_frac (float): the fraction of elites to keep (or shift) during CEM iterations
        alpha (float): momentum term.
        device (torch.device): device where computations will be performed.
        return_mean_elites (bool): if ``True`` returns the mean of the elites of the last
            iteration. Otherwise, it returns the max solution found over all iterations.
        population_size_module (int, optional): if specified, the population is rounded to be
            a multiple of this number. Defaults to ``None``.
        init_jitter_scale (float): optional scale for adding initial noise to the mean to prevent
            early collapse. Defaults to ``0.0``.

    [2] C. Pinneri, S. Sawant, S. Blaes, J. Achterhold, J. Stueckler, M. Rolinek and
    G, Martius, Georg. "Sample-efficient Cross-Entropy Method for Real-time Planning".
    Conference on Robot Learning, 2020.
    """

    def __init__(
        self,
        num_iterations: int,
        elite_ratio: float,
        population_size: int,
        population_decay_factor: float,
        colored_noise_exponent: float,
        lower_bound: Sequence[Sequence[float]],
        upper_bound: Sequence[Sequence[float]],
        keep_elite_frac: float,
        alpha: float,
        device: torch.device,
        return_mean_elites: bool = False,
        population_size_module: Optional[int] = None,
        init_jitter_scale: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.population_decay_factor = population_decay_factor
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.colored_noise_exponent = colored_noise_exponent
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.initial_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
        self.keep_elite_frac = keep_elite_frac
        self.keep_elite_size = np.ceil(keep_elite_frac * self.elite_num).astype(
            np.int32
        )
        self.elite = None
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.population_size_module = population_size_module
        self.device = device
        self.init_jitter_scale = init_jitter_scale
        self._eps = 1e-8

        if self.population_size_module:
            self.keep_elite_size = self._round_up_to_module(
                self.keep_elite_size, self.population_size_module
            )

    @staticmethod
    def _round_up_to_module(value: int, module: int) -> int:
        if value % module == 0:
            return value
        return value + (module - value % module)

    def optimize(
        self,
        obj_fun: ObjectiveFn,
        x0: Optional[torch.Tensor] = None,
        callback: OptimizerCallback = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs the optimization using iCEM.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial mean for the population. Must
                be consistent with lower/upper bounds.
            callback (callable(tensor, tensor, int) -> any, optional): if given, this
                function will be called after every iteration, passing it as input the full
                population tensor, its corresponding objective function values, and
                the index of the current iteration. This can be used for logging and plotting
                purposes.
        Returns:
            (torch.Tensor): the best solution found.
        """
        mu = x0.clone()
        var = self.initial_var.clone()

        best_solution = torch.empty_like(mu)
        best_value = -float("inf")

        for i in range(self.num_iterations):
            decay_population_size = np.ceil(
                np.max(
                    (
                        self.population_size * self.population_decay_factor**-i,
                        2 * self.elite_num,
                    )
                )
            ).astype(np.int32)

            if self.population_size_module:
                decay_population_size = self._round_up_to_module(
                    decay_population_size, self.population_size_module
                )
            # the last dimension is used for temporal correlations
            population = mbrl.util.math.powerlaw_psd_gaussian(
                self.colored_noise_exponent,
                size=(decay_population_size, x0.shape[1], x0.shape[0]),
                device=self.device,
            ).transpose(1, 2)
            population = torch.minimum(
                population * torch.sqrt(var) + mu, self.upper_bound
            )
            population = torch.maximum(population, self.lower_bound)
            if self.elite is not None:
                kept_elites = torch.index_select(
                    self.elite,
                    dim=0,
                    index=torch.randperm(self.elite_num, device=self.device)[
                        : self.keep_elite_size
                    ],
                )
                if i == 0:
                    end_action = (
                        torch.normal(
                            mu[-1, :].repeat(kept_elites.shape[0], 1),
                            torch.sqrt(var[-1, :]).repeat(kept_elites.shape[0], 1),
                        )
                        .unsqueeze(1)
                        .to(self.device)
                    )
                    kept_elites_shifted = torch.cat(
                        (kept_elites[:, 1:, :], end_action), dim=1
                    )
                    population = torch.cat((population, kept_elites_shifted), dim=0)
                elif i == self.num_iterations - 1:
                    population = torch.cat((population, mu.unsqueeze(dim=0)), dim=0)
                else:
                    population = torch.cat((population, kept_elites), dim=0)

            values = obj_fun(population)

            if callback is not None:
                callback(population, values, i)

            self._sanitize_values_(values)
            best_values, elite_idx = values.topk(self.elite_num)
            self.elite = population[elite_idx]

            new_mu = torch.mean(self.elite, dim=0)
            new_var = torch.var(self.elite, unbiased=False, dim=0)
            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            var = self.alpha * var + (1 - self.alpha) * new_var

            if best_values[0] > best_value:
                best_value = best_values[0]
                best_solution = population[elite_idx[0]].clone()

        return mu if self.return_mean_elites else best_solution


class TrajectoryOptimizer:
    """Class for using generic optimizers on trajectory optimization problems.

    This is a convenience class that sets up optimization problem for trajectories, given only
    action bounds and the length of the horizon. Using this class, the concern of handling
    appropriate tensor shapes for the optimization problem is hidden from the users, which only
    need to provide a function that is capable of evaluating trajectories of actions. It also
    takes care of shifting previous solution for the next optimization call, if the user desires.

    The optimization variables for the problem will have shape ``H x A``, where ``H`` and ``A``
    represent planning horizon and action dimension, respectively. The initial solution for the
    optimizer will be computed as (action_ub + action_lb) / 2 (the midpoint), for each time step.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the optimizer to use.
        action_lb (np.ndarray): the lower bound for actions.
        action_ub (np.ndarray): the upper bound for actions.
        planning_horizon (int): the length of the trajectories that will be optimized.
        replan_freq (int): the frequency of re-planning. This is used for shifting the previous
        solution for the next time step, when ``keep_last_solution == True``. Defaults to 1.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: np.ndarray,
        action_ub: np.ndarray,
        planning_horizon: int,
        replan_freq: int = 1,
        keep_last_solution: bool = True,
    ):
        optimizer_cfg.lower_bound = np.tile(action_lb, (planning_horizon, 1)).tolist()
        optimizer_cfg.upper_bound = np.tile(action_ub, (planning_horizon, 1)).tolist()
        self.optimizer: Optimizer = hydra.utils.instantiate(optimizer_cfg)
        device = optimizer_cfg.device
        if isinstance(device, str):
            device = torch.device(device)
        lb_t = torch.as_tensor(action_lb, device=device, dtype=torch.float32)
        ub_t = torch.as_tensor(action_ub, device=device, dtype=torch.float32)
        self._initial_action = (lb_t + ub_t) * 0.5
        self.previous_solution = self._initial_action.expand(planning_horizon, -1).clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon
        self.last_plan_debug: Optional[dict] = None

    def optimize(
        self,
        trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: OptimizerCallback = None,
        x0: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Runs the trajectory optimization.

        Args:
            trajectory_eval_fn (callable(tensor) -> tensor): A function that receives a batch
                of action sequences and returns a batch of objective function values (e.g.,
                accumulated reward for each sequence). The shape of the action sequence tensor
                will be ``B x H x A``, where ``B``, ``H``, and ``A`` represent batch size,
                planning horizon, and action dimension, respectively.
            callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (np.ndarray): the best action sequence.
        """
        self.last_plan_debug = None
        x0_in = self.previous_solution if x0 is None else x0
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=x0_in,
            callback=callback,
        )
        self.last_plan_debug = self.optimizer.get_diagnostics()
        if self.keep_last_solution:
            warmstart_solution = best_solution
            rf = int(self.replan_freq)
            if rf >= self.horizon:
                self.previous_solution[:] = self._initial_action
                # keep any optimizer persistent state
                opt_adv = getattr(self.optimizer, "advance", None)
                if callable(opt_adv):
                    try:
                        opt_adv(rf)
                    except TypeError:
                        opt_adv(steps=rf)
            else:
                if warmstart_solution is self.previous_solution:
                    self.previous_solution.copy_(warmstart_solution.roll(-rf, dims=0))
                    self.previous_solution[-rf:].copy_(self._initial_action)
                else:
                    self.previous_solution[:-rf].copy_(warmstart_solution[rf:])
                    self.previous_solution[-rf:].copy_(self._initial_action)
                opt_adv = getattr(self.optimizer, "advance", None)
                if callable(opt_adv):
                    try:
                        opt_adv(rf)
                    except TypeError:
                        opt_adv(steps=rf)
        return best_solution.detach().cpu().numpy()

    def reset(self):
        """Resets the previous solution cache (and any optimizer internal state)."""
        self.previous_solution[:] = self._initial_action
        self.last_plan_debug = None
        self.optimizer.reset()

    def advance(self, steps: int = 1) -> None:
        """Advances the warm-start solution by ``steps`` without running optimization."""
        if not self.keep_last_solution:
            return
        steps = int(steps)
        if steps <= 0:
            return
        if steps >= self.horizon:
            self.previous_solution[:] = self._initial_action
            opt_adv = getattr(self.optimizer, "advance", None)
            if callable(opt_adv):
                try:
                    opt_adv(steps)
                except TypeError:
                    opt_adv(steps=steps)
            return
        self.previous_solution.copy_(self.previous_solution.roll(-steps, dims=0))
        self.previous_solution[-steps:].copy_(self._initial_action)
        opt_adv = getattr(self.optimizer, "advance", None)
        if callable(opt_adv):
            try:
                opt_adv(steps)
            except TypeError:
                opt_adv(steps=steps)

class TrajectoryOptimizerAgent(Agent):
    """Agent that performs trajectory optimization on a given objective function for each action.

    This class uses an internal :class:`TrajectoryOptimizer` object to generate
    sequence of actions, given a user-defined trajectory optimization function.

    Args:
        optimizer_cfg (omegaconf.DictConfig): the configuration of the base optimizer to pass to
            the trajectory optimizer.
        action_lb (sequence of floats): the lower bound of the action space.
        action_ub (sequence of floats): the upper bound of the action space.
        planning_horizon (int): the length of action sequences to evaluate. Defaults to 1.
        replan_freq (int): the frequency of re-planning. The agent will keep a cache of the
            generated sequences an use it for ``replan_freq`` number of :meth:`act` calls.
            Defaults to 1.
        verbose (bool): if ``True``, prints the planning time on the console.
        keep_last_solution (bool): if ``True``, the last solution found by a call to
            :meth:`optimize` is kept as the initial solution for the next step. This solution is
            shifted ``replan_freq`` time steps, and the new entries are filled using the initial
            solution. Defaults to ``True``.

    Note:
        After constructing an agent of this type, the user must call
        :meth:`set_trajectory_eval_fn`. This is not passed to the constructor so that the agent can
        be automatically instantiated with Hydra (which in turn makes it easy to replace this
        agent with an agent of another type via config-only changes).
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True,
        # NEW: two-stage evaluation
        two_stage_eval: bool = True,
        two_stage_particles_frac: float = 0.5,
        two_stage_topk_frac: float = 0.2,
        two_stage_max_topk: int = 64,
        risk_spread_coef: float = 0.0,
        particle_schedule: str = "fixed",       # "fixed" or "ir" (used by BC-CEM)
        cv_low_threshold: float = 0.1,
        particles_min_frac: float = 0.25,
        skip_replan_if_ir_low: bool = False,
        skip_replan_ir_threshold: float = 0.1,
        skip_replan_max_frac: float = 0.5,
    ):
        self.optimizer = TrajectoryOptimizer(
            optimizer_cfg,
            np.array(action_lb),
            np.array(action_ub),
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            keep_last_solution=keep_last_solution,
        )
        self.optimizer_args = {
            "optimizer_cfg": optimizer_cfg,
            "action_lb": np.array(action_lb),
            "action_ub": np.array(action_ub),
        }
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.verbose = verbose
        self.planning_horizon = planning_horizon
        self._action_cache: Optional[np.ndarray] = None
        self._action_cache_idx: int = 0
        self._replan_cache_len: int = 0
        self.replan_skips: int = 0
        self.last_plan_model_steps_est: int = 0
        self.last_plan_time: float = 0.0
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None

        # ---- objective evaluation knobs (used by create_trajectory_optim_agent_for_model) ----
        self.two_stage_eval = two_stage_eval
        self.two_stage_particles_frac = two_stage_particles_frac
        self.two_stage_topk_frac = two_stage_topk_frac
        self.two_stage_max_topk = two_stage_max_topk
        self.risk_spread_coef = risk_spread_coef
        schedule = str(particle_schedule).lower()
        if schedule not in {"fixed", "ir"}:
            raise ValueError(
                f"Invalid particle_schedule={particle_schedule!r}. Supported values: 'fixed', 'ir'."
            )
        self.particle_schedule = schedule
        self.cv_low_threshold = float(cv_low_threshold)
        self.particles_min_frac = particles_min_frac
        self.skip_replan_if_ir_low = skip_replan_if_ir_low
        self.skip_replan_ir_threshold = float(skip_replan_ir_threshold)
        self.skip_replan_max_frac = float(skip_replan_max_frac)

        # last plan diagnostics (JSON-serializable dict from optimizer)
        self.last_plan_debug: Optional[dict] = None

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def _get_warm_start(self, _obs: np.ndarray) -> Optional[torch.Tensor]:
        """Optional hook for providing a warm-start solution to the underlying optimizer."""
        return None

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
                keep_last_solution=self.keep_last_solution,
            )
            self.planning_horizon = int(planning_horizon)

        self.optimizer.reset()
        self._action_cache = None
        self._action_cache_idx = 0
        self._replan_cache_len = 0
        self.replan_skips = 0
        self.last_plan_model_steps_est = 0
        self.last_plan_time = 0.0
        self.last_plan_debug = None

    def act(
        self, obs: np.ndarray, optimizer_callback: OptimizerCallback = None, **_kwargs
    ) -> np.ndarray:
        """Issues an action given an observation.

        This method optimizes a full sequence of length ``self.planning_horizon`` and returns
        the first action in the sequence. If ``self.replan_freq > 1``, future calls will use
        subsequent actions in the sequence, for ``self.replan_freq`` number of steps.
        After that, the method will plan again, and repeat this process.

        Args:
            obs (np.ndarray): the observation for which the action is needed.
            optimizer_callback (callable, optional): a callback function
                to pass to the optimizer.

        Returns:
            (np.ndarray): the action.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError("Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent")

        self.last_plan_model_steps_est = 0

        def _ir_low_skip_ok() -> bool:
            if not self.skip_replan_if_ir_low:
                return False
            dbg = self.last_plan_debug
            if not isinstance(dbg, dict):
                return False
            ir_norm = dbg.get("ir_norm", None)
            ir_low = dbg.get("ir_low", None)
            if ir_norm is None:
                return False
            try:
                if ir_low is not None:
                    return float(ir_norm) <= float(ir_low)
                else:
                    return float(ir_norm) <= self.skip_replan_ir_threshold
            except (TypeError, ValueError):
                return False

        did_replan = False
        plan_time = 0.0
        if self._action_cache is not None and self._action_cache_idx >= self._replan_cache_len:
            # At the nominal replan boundary: optionally skip replanning if IR indicates a
            # stable/unimodal regime (e.g., BC-CEM).
            if _ir_low_skip_ok() and self._action_cache_idx < self._action_cache.shape[0]:
                max_len = int(self._action_cache.shape[0])
                frac = float(getattr(self, "skip_replan_max_frac", 1.0))
                if frac < 1.0:
                    max_len = max(1, int(math.ceil(max_len * frac)))
                rf = int(max(1, self.replan_freq))
                max_len = max(max_len, rf)

                new_len = min(
                    int(self._action_cache.shape[0]),
                    int(self._replan_cache_len) + rf,
                    max_len,
                )
                if new_len > int(self._replan_cache_len):
                    self._replan_cache_len = int(new_len)
                    self.replan_skips += 1

        need_replan = (
            self._action_cache is None
            or self._action_cache_idx >= self._action_cache.shape[0]
            or self._action_cache_idx >= self._replan_cache_len
        )
        if need_replan:
            did_replan = True
            x0_override = None
            try:
                x0_override = self._get_warm_start(obs)
            except Exception:
                x0_override = None

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences)

            start_time = time.perf_counter()
            plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback, x0=x0_override
            )
            plan_time = time.perf_counter() - start_time
            plan_debug = getattr(self.optimizer, "last_plan_debug", None)
            self.last_plan_debug = plan_debug if isinstance(plan_debug, dict) else None

            # Optional: add model-predicted return uncertainty for the selected plan.
            eval_fn = getattr(self, "_model_env_eval_action_sequences", None)
            dbg = self.last_plan_debug
            if (
                callable(eval_fn)
                and isinstance(dbg, dict)
                and (dbg.get("ir_norm", None) is not None)
                and bool(getattr(self, "_model_env_supports_return_variance", False))
            ):
                try:
                    device = getattr(self, "_model_env_device", torch.device("cpu"))
                    particles = int(
                        getattr(
                            self,
                            "_last_eval_particles",
                            getattr(self, "_model_env_base_num_particles", 1),
                        )
                    )
                    particles = max(1, particles)
                    plan_t = torch.from_numpy(np.asarray(plan)).to(
                        device=device, dtype=torch.float32
                    )
                    plan_t = plan_t.unsqueeze(0)
                    rng = getattr(self, "_model_env_rng", None)
                    rng_state = None
                    if rng is not None and hasattr(rng, "get_state") and hasattr(rng, "set_state"):
                        try:
                            rng_state = rng.get_state()
                        except Exception:
                            rng_state = None
                    try:
                        out = eval_fn(
                            plan_t,
                            initial_state=obs,
                            num_particles=particles,
                            return_variance=True,
                        )
                    finally:
                        if rng_state is not None:
                            try:
                                rng.set_state(rng_state)
                            except Exception:
                                pass
                    mean = None
                    var = None
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        mean, var = out[0], out[1]
                    if mean is not None and var is not None:
                        m = float(mean.reshape(-1)[0].detach().cpu().item())
                        v = float(var.reshape(-1)[0].detach().cpu().item())
                        cv = math.sqrt(max(0.0, v)) / (abs(m) + 1e-8)
                        if self.last_plan_debug is None:
                            self.last_plan_debug = {}
                        self.last_plan_debug["pred_return_mean"] = m
                        self.last_plan_debug["pred_return_var"] = v
                        self.last_plan_debug["pred_return_cv"] = cv
                        self.last_plan_debug["pred_return_var_particles"] = int(particles)
                except Exception:
                    pass

            self._action_cache = plan
            self._action_cache_idx = 0
            self._replan_cache_len = int(min(int(self.replan_freq), int(plan.shape[0])))

        # ``last_plan_time`` is the time spent on *this* step's planning (0.0 if no re-plan).
        self.last_plan_time = plan_time
        idx = int(self._action_cache_idx)
        action = self._action_cache[idx]
        self._action_cache_idx = idx + 1

        # If we executed beyond the nominal warm-start shift window, advance the warm start.
        rf = int(max(1, self.replan_freq))
        if (not did_replan) and idx >= rf:
            self.optimizer.advance(1)

        if self._action_cache_idx >= self._action_cache.shape[0]:
            self._action_cache = None
            self._action_cache_idx = 0
            self._replan_cache_len = 0

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns a sequence of length ``self.planning_horizon``.

        Args:
            obs (np.ndarray): the observation for which the sequence is needed.

        Returns:
            (np.ndarray): a sequence of actions.
        """
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )

        def trajectory_eval_fn(action_sequences):
            return self.trajectory_eval_fn(obs, action_sequences)

        plan = self.optimizer.optimize(trajectory_eval_fn)
        plan_debug = getattr(self.optimizer, "last_plan_debug", None)
        self.last_plan_debug = plan_debug if isinstance(plan_debug, dict) else None
        return plan


def create_trajectory_optim_agent_for_model(
    model_env: mbrl.models.ModelEnv,
    agent_cfg: omegaconf.DictConfig,
    num_particles: int = 1,
) -> TrajectoryOptimizerAgent:
    """Utility function for creating a trajectory optimizer agent for a model environment.

    This is a convenience function for creating a :class:`TrajectoryOptimizerAgent`,
    using :meth:`mbrl.models.ModelEnv.evaluate_action_sequences` as its objective function.


    Args:
        model_env (mbrl.models.ModelEnv): the model environment.
        agent_cfg (omegaconf.DictConfig): the agent's configuration.
        num_particles (int): the number of particles for taking averages of action sequences'
            total rewards.

    Returns:
        (:class:`TrajectoryOptimizerAgent`): the agent.

    """
    complete_agent_cfg(model_env, agent_cfg)
    try:
        agent = hydra.utils.instantiate(agent_cfg, _recursive_=False)
    except Exception as e:
        if "_recursive_" not in str(e):
            raise
        agent = hydra.utils.instantiate(agent_cfg)

    eval_action_sequences = model_env.evaluate_action_sequences

    # Precompute whether the model env can return per-sequence variance.
    supports_return_variance = False
    try:
        sig = inspect.signature(eval_action_sequences)
        supports_return_variance = "return_variance" in sig.parameters
    except (TypeError, ValueError):
        supports_return_variance = False

    def _infer_model_batch_divisor() -> Optional[int]:
        dyn = getattr(model_env, "dynamics_model", None)
        base_model = getattr(dyn, "model", None) if dyn is not None else None
        if base_model is None:
            return None
        elite = getattr(base_model, "elite_models", None)
        if elite is not None:
            try:
                n = int(len(elite))
                return n if n > 1 else None
            except TypeError:
                return None
        try:
            n = int(len(base_model))
            return n if n > 1 else None
        except TypeError:
            return None

    base_num_particles = int(num_particles)

    # Expose model eval helper for optional diagnostics (e.g., return variance).
    agent._model_env_eval_action_sequences = eval_action_sequences
    agent._model_env_device = model_env.device
    agent._model_env_supports_return_variance = bool(supports_return_variance)
    agent._model_env_base_num_particles = int(base_num_particles)
    agent._last_eval_particles = int(base_num_particles)
    agent._model_env_rng = getattr(model_env, "_rng", None)

    def _pad_to_model_multiple(
        seqs: torch.Tensor, n_particles: int
    ) -> Tuple[torch.Tensor, int]:
        """Pads the batch so (B * n_particles) is divisible by #models (if required)."""
        divisor = _infer_model_batch_divisor()
        if not divisor:
            return seqs, 0
        B = int(seqs.shape[0])
        if B <= 0:
            return seqs, 0
        step = int(divisor) // max(1, math.gcd(int(divisor), int(n_particles)))
        if step <= 1:
            return seqs, 0
        r = B % step
        if r == 0:
            return seqs, 0
        pad_k = step - r
        pad = (
            seqs[:pad_k]
            if B >= pad_k
            else seqs.repeat((pad_k + B - 1) // B, 1, 1)[:pad_k]
        )
        return torch.cat([seqs, pad], dim=0), pad_k

    def _eval_once(
        initial_state: torch.Tensor,
        seqs: torch.Tensor,
        n_particles: int,
        want_var: bool,
    ):
        seqs_eval, pad_k = _pad_to_model_multiple(seqs, n_particles)
        try:
            agent.last_plan_model_steps_est += int(seqs_eval.shape[0]) * int(n_particles) * int(
                seqs_eval.shape[1]
            )
        except Exception:
            pass

        if want_var and supports_return_variance:
            out = eval_action_sequences(seqs_eval, initial_state=initial_state, num_particles=n_particles, return_variance=True)
            if isinstance(out, (tuple, list)) and len(out) >= 2:
                mean, var = out[0], out[1]
                if pad_k:
                    mean = mean[:-pad_k]
                    var = var[:-pad_k]
                return mean, var
            mean = out
            if pad_k:
                mean = mean[:-pad_k]
            return mean, None

        mean = eval_action_sequences(seqs_eval, initial_state=initial_state, num_particles=n_particles)
        if pad_k:
            mean = mean[:-pad_k]
        return mean, None

    def trajectory_eval_fn(initial_state, action_sequences):
        """Evaluates a batch of candidate action sequences.

        This wrapper optionally applies:
          - IR-adaptive particle scheduling (based on the *previous* plan's diagnostics),
          - a cheap multi-fidelity (two-stage) evaluation,
          - risk-sensitive scoring via a variance penalty (if supported by the model env).

        It always returns a 1-D tensor of scores to *maximize*.
        """
        # ---- dynamic particle schedule (uses diagnostics from previous MPC step) ----
        particles = base_num_particles
        stable_unimodal = False
        if agent.particle_schedule == "ir":
            dbg = agent.last_plan_debug
            if isinstance(dbg, dict):
                try:
                    pred_cv_v = float(dbg.get("pred_return_cv", None))
                except (TypeError, ValueError):
                    pred_cv_v = None
                cv_ok = (
                    pred_cv_v is not None
                    and math.isfinite(pred_cv_v)
                    and pred_cv_v <= float(agent.cv_low_threshold)
                )
                if cv_ok:
                    try:
                        ir_norm_v = float(dbg.get("ir_norm", None))
                        ir_low_v = float(dbg.get("ir_low", None))
                        ir_high_v = float(dbg.get("ir_high", None))
                    except (TypeError, ValueError):
                        ir_norm_v = None
                        ir_low_v = None
                        ir_high_v = None
                    if (
                        ir_norm_v is not None
                        and ir_low_v is not None
                        and ir_high_v is not None
                        and math.isfinite(ir_norm_v)
                        and math.isfinite(ir_low_v)
                        and math.isfinite(ir_high_v)
                        and (ir_high_v > ir_low_v)
                        and (ir_norm_v <= ir_low_v)
                    ):
                        p_min = max(
                            1,
                            int(
                                round(
                                    base_num_particles
                                    * float(agent.particles_min_frac)
                                )
                            ),
                        )
                        particles = min(base_num_particles, p_min)
                        stable_unimodal = True
        agent._last_eval_particles = int(particles)

        # ---- risk + multi-fidelity knobs ----
        risk_coef = float(agent.risk_spread_coef)
        two_stage = bool(agent.two_stage_eval)
        stage1_frac = float(agent.two_stage_particles_frac)
        topk_frac = float(agent.two_stage_topk_frac)
        max_topk = int(agent.two_stage_max_topk)

        if stable_unimodal:
            stage1_frac = max(0.1, min(1.0, stage1_frac * 0.5))
            topk_frac = max(0.05, min(1.0, topk_frac * 0.5))

        need_var = (risk_coef > 0.0) and supports_return_variance

        with torch.no_grad():
            B = int(action_sequences.shape[0])

            # single-stage fallback
            if (not two_stage) or (B <= 1):
                mean, var = _eval_once(initial_state, action_sequences, particles, need_var)
                return mean if var is None else (mean - risk_coef * var)

            # stage-1 coarse eval
            stage1_particles = max(1, int(round(particles * stage1_frac)))
            stage1_particles = min(stage1_particles, particles)
            if stage1_particles >= particles:
                mean, var = _eval_once(initial_state, action_sequences, particles, need_var)
                return mean if var is None else (mean - risk_coef * var)

            coarse_mean, _ = _eval_once(
                initial_state, action_sequences, stage1_particles, want_var=False
            )

            # pick top-k to refine
            topk = max(1, int(round(topk_frac * B)))
            topk = min(topk, B)
            if max_topk > 0:
                topk = min(topk, max_topk)

            # To avoid wasting compute on padding duplicates (for ensemble batch divisibility),
            # snap the refine batch size up to the next valid multiple when possible.
            divisor = _infer_model_batch_divisor()
            if divisor:
                step = int(divisor) // max(1, math.gcd(int(divisor), int(particles)))
                if step > 1:
                    topk = topk if (topk % step == 0) else (topk + (step - topk % step))
                    if topk > B:
                        topk = B
                    if max_topk > 0 and topk > max_topk:
                        topk = max_topk

            if topk >= B:
                mean, var = _eval_once(initial_state, action_sequences, particles, need_var)
                return mean if var is None else (mean - risk_coef * var)

            _, idx = torch.topk(coarse_mean, topk, dim=0)

            refined_mean, refined_var = _eval_once(
                initial_state, action_sequences[idx], particles, need_var
            )
            out = coarse_mean.clone()
            if refined_var is None:
                out[idx] = refined_mean
            else:
                out[idx] = refined_mean - risk_coef * refined_var
            return out

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent
