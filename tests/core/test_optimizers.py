# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pytest
import torch

import mbrl.planning as planning

DEVICE = torch.device("cpu")


@pytest.fixture
def quadratic_problem():
    target = torch.tensor(
        [[1.2, -0.8], [0.4, 0.6]], dtype=torch.float32, device=DEVICE
    )
    lower = torch.full_like(target, -2.0)
    upper = torch.full_like(target, 2.0)

    def objective(actions: torch.Tensor) -> torch.Tensor:
        diff = actions - target
        return -(diff * diff).sum(dim=(1, 2))

    return target, lower.tolist(), upper.tolist(), objective


@pytest.mark.parametrize(
    ("optimizer_cls", "kwargs", "tolerance"),
    [
        (
            planning.CEMOptimizer,
            {
                "num_iterations": 7,
                "elite_ratio": 0.2,
                "population_size": 256,
                "alpha": 0.15,
            },
            0.1,
        ),
        (
            planning.DecentCEMOptimizer,
            {
                "num_iterations": 7,
                "elite_ratio": 0.2,
                "population_size": 128,
                "alpha": 0.15,
                "num_workers": 4,
            },
            0.1,
        ),
        (
            planning.GMMCEMOptimizer,
            {
                "num_iterations": 6,
                "elite_ratio": 0.2,
                "population_size": 96,
                "alpha": 0.2,
                "num_workers": 3,
            },
            0.15,
        ),
    ],
)
def test_cem_family_optimize_quadratic(quadratic_problem, optimizer_cls, kwargs, tolerance):
    target, lower, upper, objective = quadratic_problem
    torch.manual_seed(0)
    optimizer = optimizer_cls(lower_bound=lower, upper_bound=upper, device=DEVICE, **kwargs)

    solution = optimizer.optimize(objective, x0=torch.zeros_like(target))

    assert solution.shape == target.shape
    assert torch.norm(solution - target) < tolerance


@pytest.mark.parametrize("adaptation", ["diagonal", "full"])
def test_cmaes_optimizer_converges(quadratic_problem, adaptation):
    target, lower, upper, objective = quadratic_problem
    torch.manual_seed(0)
    optimizer = planning.CMAESOptimizer(
        num_iterations=8,
        population_size=96,
        elite_ratio=0.25,
        sigma=1.5,
        lower_bound=lower,
        upper_bound=upper,
        alpha=0.2,
        device=DEVICE,
        adaptation=adaptation,
    )

    solution = optimizer.optimize(objective, x0=torch.zeros_like(target))

    assert solution.shape == target.shape
    assert torch.norm(solution - target) < 0.2


def test_nes_optimizer_converges(quadratic_problem):
    target, lower, upper, objective = quadratic_problem
    torch.manual_seed(0)
    optimizer = planning.NESOptimizer(
        num_iterations=12,
        population_size=128,
        sigma=1.0,
        lr_mean=0.2,
        lr_sigma=0.15,
        lower_bound=lower,
        upper_bound=upper,
        device=DEVICE,
        return_mean_elites=True,
    )

    solution = optimizer.optimize(objective, x0=torch.zeros_like(target))

    assert solution.shape == target.shape
    assert torch.norm(solution - target) < 0.25


def test_mppi_optimizer_moves_toward_optimum(quadratic_problem):
    target, lower, upper, objective = quadratic_problem
    torch.manual_seed(0)
    optimizer = planning.MPPIOptimizer(
        num_iterations=7,
        population_size=600,
        gamma=6.0,
        sigma=0.9,
        beta=0.65,
        lower_bound=lower,
        upper_bound=upper,
        device=DEVICE,
    )

    solution = optimizer.optimize(objective)

    assert solution.shape == target.shape
    assert torch.norm(solution - target) < 0.25


def test_icem_optimizer_tracks_elites(quadratic_problem):
    target, lower, upper, objective = quadratic_problem
    torch.manual_seed(0)
    optimizer = planning.ICEMOptimizer(
        num_iterations=6,
        elite_ratio=0.1,
        population_size=256,
        population_decay_factor=1.25,
        colored_noise_exponent=2.0,
        lower_bound=lower,
        upper_bound=upper,
        keep_elite_frac=0.2,
        alpha=0.1,
        device=DEVICE,
    )

    solution = optimizer.optimize(objective, x0=torch.zeros_like(target))

    assert solution.shape == target.shape
    assert torch.norm(solution - target) < 0.15
