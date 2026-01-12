# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import omegaconf
import pytest
import torch

import mbrl.planning as planning
from mbrl.planning.trajectory_opt import Optimizer


class DummyOptimizer(Optimizer):
    """A tiny optimizer used to test warm-start shifting and state advancement.

    Notes:
      - Must accept `lower_bound` / `upper_bound` because `TrajectoryOptimizer` injects them.
      - Keeps a log of `advance()` calls so tests can assert correct advancement behavior.
    """

    def __init__(
        self,
        lower_bound: Optional[Sequence[Sequence[float]]] = None,
        upper_bound: Optional[Sequence[Sequence[float]]] = None,
        device: Union[str, torch.device] = "cpu",
        **_kwargs,
    ):
        super().__init__()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.device = torch.device(device) if isinstance(device, str) else device

        self.optimize_calls = 0
        self.advance_calls: list[int] = []
        self.last_x0: Optional[torch.Tensor] = None

    def optimize(self, obj_fun, x0: Optional[torch.Tensor] = None, **_kwargs) -> torch.Tensor:
        self.optimize_calls += 1
        if x0 is None:
            raise ValueError("DummyOptimizer requires x0.")
        self.last_x0 = x0.detach().clone()
        return torch.arange(x0.numel(), device=x0.device, dtype=x0.dtype).view_as(x0)

    def advance(self, steps: int = 1) -> None:
        self.advance_calls.append(int(steps))

    def reset(self) -> None:
        self.optimize_calls = 0
        self.advance_calls = []
        self.last_x0 = None

    def get_diagnostics(self) -> dict:
        # Used by TrajectoryOptimizerAgent's IR-based skip logic.
        return {"ir_norm": 0.0, "ir_low": 0.1}


def _dummy_optimizer_cfg() -> omegaconf.DictConfig:
    return omegaconf.OmegaConf.create(
        {
            "_target_": "tests.core.test_warm_start.DummyOptimizer",
            "device": "cpu",
        }
    )


def test_trajectory_optimizer_warm_start_and_optimizer_advance():
    cfg = _dummy_optimizer_cfg()
    trajopt = planning.TrajectoryOptimizer(
        cfg,
        action_lb=np.array([-10.0, -10.0], dtype=np.float32),
        action_ub=np.array([10.0, 10.0], dtype=np.float32),
        planning_horizon=5,
        replan_freq=2,
        keep_last_solution=True,
    )

    def obj_fun(population: torch.Tensor) -> torch.Tensor:
        return torch.zeros((population.shape[0],), device=population.device)

    plan = trajopt.optimize(obj_fun)
    dummy = trajopt.optimizer
    assert isinstance(dummy, DummyOptimizer)
    assert dummy.optimize_calls == 1
    assert dummy.advance_calls == [2]

    expected_plan = np.arange(10, dtype=np.float32).reshape(5, 2)
    assert np.allclose(plan, expected_plan)

    expected_prev = np.vstack([expected_plan[2:], np.zeros((2, 2), dtype=np.float32)])
    assert np.allclose(trajopt.previous_solution.detach().cpu().numpy(), expected_prev)

    trajopt.advance(1)
    assert dummy.advance_calls == [2, 1]
    expected_prev_2 = np.vstack([expected_plan[3:], np.zeros((3, 2), dtype=np.float32)])
    assert np.allclose(trajopt.previous_solution.detach().cpu().numpy(), expected_prev_2)


def test_agent_skip_replan_advances_warm_start_state():
    cfg = _dummy_optimizer_cfg()
    agent = planning.TrajectoryOptimizerAgent(
        optimizer_cfg=cfg,
        action_lb=[-10.0, -10.0],
        action_ub=[10.0, 10.0],
        planning_horizon=5,
        replan_freq=2,
        keep_last_solution=True,
        skip_replan_if_ir_low=True,
        skip_replan_max_frac=1.0,
    )

    def traj_eval_fn(_obs: np.ndarray, action_sequences: torch.Tensor) -> torch.Tensor:
        return torch.zeros((action_sequences.shape[0],), device=action_sequences.device)

    agent.set_trajectory_eval_fn(traj_eval_fn)
    agent.reset()

    # Execute the full plan length; IR-based skipping should avoid replanning.
    obs = np.zeros((1,), dtype=np.float32)
    for _ in range(5):
        agent.act(obs)

    dummy = agent.optimizer.optimizer
    assert isinstance(dummy, DummyOptimizer)
    assert dummy.optimize_calls == 1
    assert dummy.advance_calls == [2, 1, 1, 1]
    assert np.allclose(agent.optimizer.previous_solution.detach().cpu().numpy(), 0.0)


def test_bccem_optimizer_advance_shifts_and_fills():
    device = torch.device("cpu")
    horizon = 4
    lower = np.full((horizon, 1), -20.0, dtype=np.float32).tolist()
    upper = np.full((horizon, 1), 20.0, dtype=np.float32).tolist()

    opt = planning.BCCEMOptimizer(
        num_iterations=1,
        elite_ratio=0.5,
        population_size=4,
        lower_bound=lower,
        upper_bound=upper,
        alpha=0.0,
        device=device,
        num_workers=2,
        return_mean_elites=True,
        init_jitter_scale=0.0,
    )

    mu = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]],
        device=device,
    )
    disp = torch.full_like(mu, 0.25)
    opt._state_mu_w = mu.clone()
    opt._state_disp_w = disp.clone()

    opt.advance(steps=2)
    assert opt._state_mu_w is not None and opt._state_disp_w is not None

    expected_mu = torch.tensor(
        [[[2.0], [3.0], [0.0], [0.0]], [[12.0], [13.0], [0.0], [0.0]]],
        device=device,
    )
    expected_disp = torch.tensor(
        [[[0.25], [0.25], [100.0], [100.0]], [[0.25], [0.25], [100.0], [100.0]]],
        device=device,
    )
    assert torch.allclose(opt._state_mu_w, expected_mu)
    assert torch.allclose(opt._state_disp_w, expected_disp)
