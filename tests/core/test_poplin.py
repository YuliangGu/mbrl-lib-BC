# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import omegaconf
import pytest
import torch

from mbrl.planning.poplin import PoplinDataset, PoplinPlanPolicy, PoplinTrainer
from mbrl.planning.poplin_agent import PoplinTrajectoryOptimizerAgent


def _make_zero_output_policy(*, obs_dim: int, horizon: int, action_dim: int, num_heads: int):
    policy = PoplinPlanPolicy(
        obs_dim=obs_dim,
        action_lb=np.full((action_dim,), -1.0, dtype=np.float32),
        action_ub=np.full((action_dim,), 1.0, dtype=np.float32),
        planning_horizon=horizon,
        action_dim=action_dim,
        num_heads=num_heads,
        hidden_sizes=(),
        activation="relu",
    )
    with torch.no_grad():
        policy.plan_head.weight.zero_()
        policy.plan_head.bias.zero_()
        if policy.weight_head is not None:
            policy.weight_head.weight.zero_()
            policy.weight_head.bias.zero_()
    return policy


def test_poplin_trainer_wta_vs_weighted():
    torch.manual_seed(0)
    obs_dim, horizon, action_dim, num_heads = 1, 1, 1, 2

    dataset = PoplinDataset(
        capacity=32,
        obs_dim=obs_dim,
        num_heads=num_heads,
        planning_horizon=horizon,
        action_dim=action_dim,
        device="cpu",
    )

    obs = np.array([0.0], dtype=np.float32)
    target_plans = torch.tensor([[[0.0]], [[1.0]]], dtype=torch.float32)  # (K,H,A)
    target_weights = torch.tensor([0.7, 0.3], dtype=torch.float32)  # winner is head 0
    for _ in range(16):
        dataset.add(obs, target_plans, target_weights)

    policy_weighted = _make_zero_output_policy(
        obs_dim=obs_dim, horizon=horizon, action_dim=action_dim, num_heads=num_heads
    )
    trainer_weighted = PoplinTrainer(
        policy_weighted,
        lr=0.0,
        weight_decay=0.0,
        action_loss_coef=1.0,
        weight_loss_coef=0.0,
        action_loss_mode="weighted",
    )
    losses_weighted = trainer_weighted.update(dataset, batch_size=8, num_updates=1, device="cpu")
    assert losses_weighted is not None
    assert losses_weighted.action == pytest.approx(0.3, abs=1e-6)

    policy_wta = _make_zero_output_policy(
        obs_dim=obs_dim, horizon=horizon, action_dim=action_dim, num_heads=num_heads
    )
    trainer_wta = PoplinTrainer(
        policy_wta,
        lr=0.0,
        weight_decay=0.0,
        action_loss_coef=1.0,
        weight_loss_coef=0.0,
        action_loss_mode="wta",
        wta_mode="targets",
    )
    losses_wta = trainer_wta.update(dataset, batch_size=8, num_updates=1, device="cpu")
    assert losses_wta is not None
    assert losses_wta.action == pytest.approx(0.0, abs=1e-6)


def test_poplin_p_parameter_planning_runs():
    torch.manual_seed(0)
    optimizer_cfg = omegaconf.OmegaConf.create(
        {
            "_target_": "mbrl.planning.CEMOptimizer",
            "num_iterations": 2,
            "elite_ratio": 0.2,
            "population_size": 64,
            "alpha": 0.1,
            "device": "cpu",
            "return_mean_elites": True,
            "clipped_normal": True,
        }
    )
    poplin_cfg = omegaconf.OmegaConf.create(
        {
            "enabled": True,
            "variant": "p",
            "num_heads": "auto",
            "hidden_sizes": [],
            "activation": "relu",
            "param_lb": -1.0,
            "param_ub": 1.0,
            "param_avg_coef": 1.0,
            "param_return_mean_elites": True,
        }
    )
    agent = PoplinTrajectoryOptimizerAgent(
        optimizer_cfg=optimizer_cfg,
        action_lb=[-1.0],
        action_ub=[1.0],
        obs_dim=2,
        planning_horizon=3,
        replan_freq=1,
        verbose=False,
        keep_last_solution=True,
        poplin=poplin_cfg,
    )

    target = torch.zeros((3, 1), dtype=torch.float32)

    def trajectory_eval_fn(_obs, action_sequences):
        diff = action_sequences - target.to(device=action_sequences.device)
        return -(diff * diff).sum(dim=(1, 2))

    agent.set_trajectory_eval_fn(trajectory_eval_fn)

    obs = np.array([0.5, -0.25], dtype=np.float32)
    action = agent.act(obs)

    assert action.shape == (1,)
    assert action[0] <= 1.0 + 1e-6
    assert action[0] >= -1.0 - 1e-6
