# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import time
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

DISPERSION_EPS = 1e-3

class Optimizer:
    def __init__(self):
        pass

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Runs optimization.

        Args:
            obj_fun (callable(tensor) -> tensor): objective function to maximize.
            x0 (tensor, optional): initial solution, if necessary.

        Returns:
            (torch.Tensor): the best solution found.
        """
        pass


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
            pop = mean + dispersion * torch.randn_like(population)
            pop = torch.where(pop > self.lower_bound, pop, self.lower_bound)
            population = torch.where(pop < self.upper_bound, pop, self.upper_bound)
            return population
        else:
            lb_dist = mean - self.lower_bound
            ub_dist = self.upper_bound - mean
            mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.min(mv, dispersion)

            population = mbrl.util.math.truncated_normal_(population)
            return population * torch.sqrt(constrained_var) + mean

    def _update_population_params(
        self, elite: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_mu = torch.mean(elite, dim=0)
        if self._clipped_normal:
            new_dispersion = torch.std(elite, dim=0)
        else:
            new_dispersion = torch.var(elite, dim=0)
        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
        dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)
        return mu, dispersion

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
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
        best_value = -np.inf
        population = torch.zeros((self.population_size,) + x0.shape).to(
            device=self.device
        )
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            values = obj_fun(population)

            if callback is not None:
                callback(population, values, i)

            # filter out NaN values
            values[values.isnan()] = -1e-10
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
        num_workers: int = 4,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
        init_jitter_scale: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        self.num_workers = num_workers
        self._clipped_normal = clipped_normal
        self._eps = 1e-8
        self.init_jitter_scale = init_jitter_scale

    def _init_population_params(
        self, x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = x0.expand((self.num_workers,) + x0.shape)
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            base_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
            dispersion = base_var.expand_as(mean)
        if self.init_jitter_scale > 0:
            mean = torch.clamp(
                mean
                + self.init_jitter_scale
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
            pop = mean.unsqueeze(1) + dispersion.unsqueeze(1) * torch.randn_like(
                population
            )
            return torch.max(torch.min(pop, self.upper_bound), self.lower_bound)

        lb_dist = mean - self.lower_bound
        ub_dist = self.upper_bound - mean
        mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
        constrained_var = torch.min(mv, dispersion)

        population = mbrl.util.math.truncated_normal_(population)
        scaled_noise = population * torch.sqrt(constrained_var).unsqueeze(1)
        return scaled_noise + mean.unsqueeze(1)

    def _update_population_params(
        self, elite: torch.Tensor, mu: torch.Tensor, dispersion: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        new_mu = torch.mean(elite, dim=1)
        if self._clipped_normal:
            new_dispersion = torch.std(elite, dim=1, unbiased=False)
        else:
            new_dispersion = torch.var(elite, dim=1, unbiased=False)
        mu = self.alpha * mu + (1 - self.alpha) * new_mu
        dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
        dispersion = torch.clamp(dispersion, min=DISPERSION_EPS)
        return mu, dispersion

    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        mu, dispersion = self._init_population_params(x0)
        best_solution = torch.empty_like(x0)
        best_value = -np.inf
        best_worker_idx = 0
        population = torch.zeros(
            (self.num_workers, self.population_size) + x0.shape, device=self.device
        )
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            flat_population = population.reshape(-1, *x0.shape)
            values = obj_fun(flat_population).reshape(
                self.num_workers, self.population_size
            )
            values[values.isnan()] = -1e-10

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

        if self.return_mean_elites:
            return mu[best_worker_idx]
        return best_solution

class BCCEMOptimizer(DecentCEMOptimizer):
    """ TESTING: Bregman(KL)-Centroid guided ensemble CEM method. """
    
    def optimize(
        self,
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[
            Callable[
                [
                    torch.Tensor,
                    torch.Tensor,
                    int,
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                ],
                None,
            ]
        ] = None,
        **kwargs,
    ) -> torch.Tensor:
        tau = kwargs.pop("tau", 1.0)    # higher tau -> more uniform weights
        mu, dispersion = self._init_population_params(x0)
        best_solution = torch.empty_like(x0)
        best_value = -np.inf
        best_worker_idx = 0
        population = torch.zeros(
            (self.num_workers, self.population_size) + x0.shape, device=self.device
        )
        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            flat_population = population.reshape(-1, *x0.shape)
            values = obj_fun(flat_population).reshape(
                self.num_workers, self.population_size
            )
            values[values.isnan()] = -1e-10

            best_values, elite_idx = values.topk(self.elite_num, dim=1)
            worker_indices = torch.arange(self.num_workers, device=self.device).view(
                -1, 1
            )
            elite = population[worker_indices, elite_idx]

            mu, dispersion = self._update_population_params(elite, mu, dispersion)

            """ TESTING adds-on: no algorithmic change to original CEM """
            # per-iteration worker quality: best value[:,0]
            # rewards = best_values[:, 0] # shape: (num_workers, )
            rewards = torch.mean(best_values, dim=1)  # shape: (num_workers, ) try average
            logits = (rewards - rewards.max()) / tau  # tau: temperature parameter for softmax
            w = torch.softmax(logits, dim=0)  # worker weights

            mu_c = (w.view(-1, 1, 1) * mu).sum(dim=0)  # (H, A)

            # simple variance proxy: use average dispersion as sigma^2
            sigma2_c = (w.view(-1, 1, 1) * dispersion).sum(dim=0)  # (H, A)
            sigma2_c = torch.clamp(sigma2_c, min=1e-3)  # avoid IR explosion

            # gamma and IR
            diff = (mu - mu_c) / torch.sqrt(sigma2_c) # boardcast to (K, H, A)
            gamma = 0.5 * (diff * diff).sum(dim=(1,2))  # (K, ) | Bregman divergence
            IR = (w * gamma).sum()  # Information Radius

            if callback is not None:
                sigma_c = torch.sqrt(sigma2_c)
                callback(flat_population, values.reshape(-1), i, IR, mu_c, sigma_c)

            worker_best_values = best_values[:, 0]
            candidate_value, candidate_worker = torch.max(worker_best_values, dim=0)
            candidate_value_item = float(candidate_value)
            if candidate_value_item > best_value:
                best_value = candidate_value_item
                best_worker_idx = int(candidate_worker)
                best_solution = population[
                    best_worker_idx, elite_idx[best_worker_idx, 0]
                ].clone()

        if self.return_mean_elites:
            return mu[best_worker_idx]
        return best_solution   # Override optimize method

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
        num_workers: int = 4,
        return_mean_elites: bool = False,
        clipped_normal: bool = False,
        init_jitter_scale: float = 0.0,
    ):
        super().__init__()
        self.num_iterations = num_iterations
        self.elite_ratio = elite_ratio
        self.population_size = population_size
        self.elite_num = np.ceil(self.population_size * self.elite_ratio).astype(
            np.int32
        )
        self.lower_bound = torch.tensor(lower_bound, device=device, dtype=torch.float32)
        self.upper_bound = torch.tensor(upper_bound, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.return_mean_elites = return_mean_elites
        self.device = device
        self.num_workers = num_workers
        self._clipped_normal = clipped_normal
        self._eps = 1e-8
        self.init_jitter_scale = init_jitter_scale

    def _init_population_params(
        self, x0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x0.expand((self.num_workers,) + x0.shape).clone()
        if self._clipped_normal:
            dispersion = torch.ones_like(mean)
        else:
            base_var = ((self.upper_bound - self.lower_bound) ** 2) / 16
            dispersion = base_var.expand_as(mean)
        if self.init_jitter_scale > 0:
            mean = torch.clamp(
                mean
                + self.init_jitter_scale
                * torch.randn((self.num_workers,) + x0.shape, device=self.device)
                * torch.sqrt(dispersion + self._eps),
                min=self.lower_bound,
                max=self.upper_bound,
            )
        mix = torch.full(
            (self.num_workers,), 1.0 / self.num_workers, device=self.device
        )
        return mean, dispersion, mix

    def _sample_population(
        self,
        mean: torch.Tensor,
        dispersion: torch.Tensor,
        population: torch.Tensor,
    ) -> torch.Tensor:
        if self._clipped_normal:
            pop = mean.unsqueeze(1) + dispersion.unsqueeze(1) * torch.randn_like(
                population
            )
            return torch.max(torch.min(pop, self.upper_bound), self.lower_bound)

        lb_dist = mean - self.lower_bound
        ub_dist = self.upper_bound - mean
        mv = torch.min(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
        constrained_var = torch.min(mv, dispersion)
        constrained_var = torch.clamp(constrained_var, min=DISPERSION_EPS)

        population = mbrl.util.math.truncated_normal_(population)
        scaled_noise = population * torch.sqrt(constrained_var).unsqueeze(1)
        return scaled_noise + mean.unsqueeze(1)

    def _log_prob(
        self, flat_population: torch.Tensor, mean: torch.Tensor, dispersion: torch.Tensor
    ) -> torch.Tensor:
        # flat_population: (S, H, A); mean/dispersion: (K, H, A)
        diff = flat_population.unsqueeze(0) - mean.unsqueeze(1)
        var = dispersion.unsqueeze(1)
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
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
        **kwargs,
    ) -> torch.Tensor:
        mu, dispersion, mix = self._init_population_params(x0)
        best_solution = torch.empty_like(x0)
        best_value = -np.inf
        best_worker = 0
        population = torch.zeros(
            (self.num_workers, self.population_size) + x0.shape, device=self.device
        )

        for i in range(self.num_iterations):
            population = self._sample_population(mu, dispersion, population)
            flat_population = population.reshape(-1, *x0.shape)
            values = obj_fun(flat_population).reshape(
                self.num_workers, self.population_size
            )
            values[values.isnan()] = -1e-10

            if callback is not None:
                callback(flat_population, values.reshape(-1), i)

            values_flat = values.reshape(-1)
            candidate_value, candidate_idx = torch.max(values_flat, dim=0)
            candidate_value_item = float(candidate_value)
            if candidate_value_item > best_value:
                best_value = candidate_value_item
                best_solution = flat_population[candidate_idx].clone()
                best_worker = int(candidate_idx) // self.population_size

            value_weights = torch.softmax(values, dim=1).reshape(-1)
            log_prob = self._log_prob(flat_population, mu, dispersion)
            log_mix = torch.log(mix + self._eps).unsqueeze(1)
            log_joint = log_mix + log_prob
            log_norm = torch.logsumexp(log_joint, dim=0, keepdim=True)
            responsibilities = torch.exp(log_joint - log_norm)

            weighted_resp = responsibilities * value_weights.unsqueeze(0)
            comp_weight = weighted_resp.sum(dim=1) + self._eps

            sample_dims = flat_population.dim() - 1
            weight_shape = (self.num_workers, flat_population.shape[0]) + (
                (1,) * sample_dims
            )
            weighted_resp_exp = weighted_resp.reshape(weight_shape)
            flat_expanded = flat_population.unsqueeze(0)
            comp_weight_exp = comp_weight.reshape(
                (self.num_workers,) + (1,) * sample_dims
            )
            new_mu = (weighted_resp_exp * flat_expanded).sum(dim=1) / comp_weight_exp
            diff = flat_expanded - new_mu.unsqueeze(1)
            new_dispersion = (
                weighted_resp_exp * diff * diff
            ).sum(dim=1) / comp_weight_exp

            mu = self.alpha * mu + (1 - self.alpha) * new_mu
            dispersion = self.alpha * dispersion + (1 - self.alpha) * new_dispersion
            mix_update = comp_weight / (comp_weight.sum() + self._eps)
            mix = self.alpha * mix + (1 - self.alpha) * mix_update
            mix = mix / (mix.sum() + self._eps)

            mu = torch.clamp(mu, min=self.lower_bound, max=self.upper_bound)
            dispersion = torch.clamp(dispersion, min=self._eps)

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
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
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
        best_value = -np.inf

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
            values[values.isnan()] = -1e-10

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
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
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
        best_value = -np.inf

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
            values[values.isnan()] = -1e-10

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
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
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
            noise = torch.empty(
                size=(
                    self.population_size,
                    self.planning_horizon,
                    self.action_dimension,
                ),
                device=self.device,
            )
            noise = mbrl.util.math.truncated_normal_(noise)

            lb_dist = self.mean - self.lower_bound
            ub_dist = self.upper_bound - self.mean
            mv = torch.minimum(torch.square(lb_dist / 2), torch.square(ub_dist / 2))
            constrained_var = torch.minimum(mv, self.var)
            population = noise.clone() * torch.sqrt(constrained_var)

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
            population = torch.where(
                population > self.upper_bound, self.upper_bound, population
            )
            population = torch.where(
                population < self.lower_bound, self.lower_bound, population
            )
            values = obj_fun(population)
            values[values.isnan()] = -1e-10

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
        obj_fun: Callable[[torch.Tensor], torch.Tensor],
        x0: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[torch.Tensor, torch.Tensor, int], None]] = None,
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
        if self.init_jitter_scale > 0:
            mu = torch.clamp(
                mu
                + self.init_jitter_scale
                * torch.randn_like(mu)
                * torch.sqrt(var).clamp_min(self._eps),
                min=self.lower_bound,
                max=self.upper_bound,
            )

        best_solution = torch.empty_like(mu)
        best_value = -np.inf

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

            # filter out NaN values
            values[values.isnan()] = -1e-10
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
    optimizer will be computed as (action_ub - action_lb) / 2, for each time step.

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
        self.initial_solution = (
            ((torch.tensor(action_lb) + torch.tensor(action_ub)) / 2)
            .float()
            .to(optimizer_cfg.device)
        )
        self.initial_solution = self.initial_solution.repeat((planning_horizon, 1))
        self.previous_solution = self.initial_solution.clone()
        self.replan_freq = replan_freq
        self.keep_last_solution = keep_last_solution
        self.horizon = planning_horizon

    def optimize(
        self,
        trajectory_eval_fn: Callable[[torch.Tensor], torch.Tensor],
        callback: Optional[Callable] = None,
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
            (tuple of np.ndarray and float): the best action sequence.
        """
        best_solution = self.optimizer.optimize(
            trajectory_eval_fn,
            x0=self.previous_solution,
            callback=callback,
        )
        if self.keep_last_solution:
            self.previous_solution = best_solution.roll(-self.replan_freq, dims=0)
            # Note that initial_solution[i] is the same for all values of [i],
            # so just pick i = 0
            self.previous_solution[-self.replan_freq :] = self.initial_solution[0]
        return best_solution.cpu().numpy()

    def reset(self):
        """Resets the previous solution cache to the initial solution."""
        self.previous_solution = self.initial_solution.clone()


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
        self.trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType = None
        self.actions_to_use: List[np.ndarray] = []
        self.replan_freq = replan_freq
        self.verbose = verbose

    def set_trajectory_eval_fn(
        self, trajectory_eval_fn: mbrl.types.TrajectoryEvalFnType
    ):
        """Sets the trajectory evaluation function.

        Args:
            trajectory_eval_fn (callable): a trajectory evaluation function, as described in
                :class:`TrajectoryOptimizer`.
        """
        self.trajectory_eval_fn = trajectory_eval_fn

    def reset(self, planning_horizon: Optional[int] = None):
        """Resets the underlying trajectory optimizer."""
        if planning_horizon:
            self.optimizer = TrajectoryOptimizer(
                cast(omegaconf.DictConfig, self.optimizer_args["optimizer_cfg"]),
                cast(np.ndarray, self.optimizer_args["action_lb"]),
                cast(np.ndarray, self.optimizer_args["action_ub"]),
                planning_horizon=planning_horizon,
                replan_freq=self.replan_freq,
            )

        self.optimizer.reset()

    def act(
        self, obs: np.ndarray, optimizer_callback: Optional[Callable] = None, **_kwargs
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
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using TrajectoryOptimizerAgent"
            )
        plan_time = 0.0
        if not self.actions_to_use:  # re-plan is necessary

            def trajectory_eval_fn(action_sequences):
                return self.trajectory_eval_fn(obs, action_sequences)

            start_time = time.time()
            plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback
            )
            plan_time = time.time() - start_time

            self.actions_to_use.extend([a for a in plan[: self.replan_freq]])
        action = self.actions_to_use.pop(0)

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def plan(self, obs: np.ndarray, **_kwargs) -> np.ndarray:
        """Issues a sequence of actions given an observation.

        Returns s sequence of length self.planning_horizon.

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
    agent = hydra.utils.instantiate(agent_cfg)

    def trajectory_eval_fn(initial_state, action_sequences):
        return model_env.evaluate_action_sequences(
            action_sequences, initial_state=initial_state, num_particles=num_particles
        )

    agent.set_trajectory_eval_fn(trajectory_eval_fn)
    return agent
