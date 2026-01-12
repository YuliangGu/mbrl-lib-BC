# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""POPLIN: PETS-style training with a policy-augmented planner.

This implementation currently reuses the PETS training loop and relies on
`mbrl.planning.PoplinTrajectoryOptimizerAgent` to implement both:

- POPLIN-A (action-space): warm-start CEM-family optimizers from a learned policy and train the
  policy online by distillation from optimizer targets.
- POPLIN-P (parameter-space): plan in the policy-parameter space, initializing the sampling
  distribution from the current policy parameters and updating the policy via parameter averaging.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
import omegaconf

from . import pets
import mbrl.types


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    return pets.train(
        env,
        termination_fn,
        reward_fn,
        cfg,
        silent=silent,
        work_dir=work_dir,
    )
