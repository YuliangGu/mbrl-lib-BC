# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
from typing import Optional

import gymnasium as gym
import numpy as np
import omegaconf
import torch

import mbrl.constants
import mbrl.models
import mbrl.planning
import mbrl.types
import mbrl.util
import mbrl.util.common
import mbrl.util.math

EVAL_LOG_FORMAT = mbrl.constants.EVAL_LOG_FORMAT


def train(
    env: gym.Env,
    termination_fn: mbrl.types.TermFnType,
    reward_fn: mbrl.types.RewardFnType,
    cfg: omegaconf.DictConfig,
    silent: bool = False,
    work_dir: Optional[str] = None,
) -> np.float32:
    # ------------------- Initialization -------------------
    debug_mode = cfg.get("debug_mode", False)
    use_color = (not os.environ.get("NO_COLOR")) and bool(getattr(sys.stderr, "isatty", lambda: False)())
    warning_tag = "\033[33m[BCCEM]\033[0m" if use_color else "[BCCEM]"

    obs_shape = env.observation_space.shape
    act_shape = env.action_space.shape

    rng = np.random.default_rng(seed=cfg.seed)
    torch_generator = torch.Generator(device=cfg.device)
    if cfg.seed is not None:
        torch_generator.manual_seed(cfg.seed)

    work_dir = work_dir or os.getcwd()
    print(f"Results will be saved at {work_dir}.")

    if silent:
        logger = None
    else:
        logger = mbrl.util.Logger(work_dir)
        logger.register_group(
            mbrl.constants.RESULTS_LOG_NAME, EVAL_LOG_FORMAT, color="green"
        )

    # -------- Create and populate initial env dataset --------
    dynamics_model = mbrl.util.common.create_one_dim_tr_model(cfg, obs_shape, act_shape)
    use_double_dtype = cfg.algorithm.get("normalize_double_precision", False)
    dtype = np.double if use_double_dtype else np.float32
    replay_buffer = mbrl.util.common.create_replay_buffer(
        cfg,
        obs_shape,
        act_shape,
        rng=rng,
        obs_type=dtype,
        action_type=dtype,
        reward_type=dtype,
    )
    mbrl.util.common.rollout_agent_trajectories(
        env,
        cfg.algorithm.initial_exploration_steps,
        mbrl.planning.RandomAgent(env),
        {},
        replay_buffer=replay_buffer,
    )
    replay_buffer.save(work_dir)

    # ---------------------------------------------------------
    # ---------- Create model environment and agent -----------
    model_env = mbrl.models.ModelEnv(
        env, dynamics_model, termination_fn, reward_fn, generator=torch_generator
    )
    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    max_total_reward = -np.inf
    while env_steps < cfg.overrides.num_steps:
        obs, _ = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        total_reward = 0.0
        plan_time_sum = 0.0
        model_steps_sum = 0
        replan_calls = 0
        centroid_action_steps = 0
        bccem_ir_norm_sum = 0.0
        bccem_ir_norm_count = 0
        bccem_ir_norm_min = None
        bccem_ir_norm_max = None
        steps_trial = 0
        while not terminated and not truncated:
            # --------------- Model Training -----------------
            if env_steps % cfg.algorithm.freq_train_model == 0:
                mbrl.util.common.train_model_and_save_model_and_data(
                    dynamics_model,
                    model_trainer,
                    cfg.overrides,
                    replay_buffer,
                    work_dir=work_dir,
                    silent=bool(cfg.algorithm.get("silent_model_train", False)),
                )

            # --- Optional BC-CEM schedule: anneal ir_high toward target over training steps ---
            opt = getattr(getattr(agent, "optimizer", None), "optimizer", None)
            bcem_params = getattr(opt, "bcem_params", None)
            if bcem_params is not None:
                target = getattr(bcem_params, "ir_high_target", None)
                if target is not None:
                    try:
                        total_steps = int(getattr(cfg.overrides, "num_steps", 0))
                    except (TypeError, ValueError):
                        total_steps = 0
                    if total_steps > 0:
                        denom = max(1, total_steps - 1)
                        t = float(env_steps) / float(denom)
                        t = max(0.0, min(1.0, t))
                        start = float(getattr(opt, "_ir_high_start", bcem_params.ir_high))
                        # TODO: try cosine annealing?
                        bcem_params.ir_high = start + t * (float(target) - start)

            # --- Doing env step using the agent and adding to model dataset ---
            (
                next_obs,
                reward,
                terminated,
                truncated,
                _,
            ) = mbrl.util.common.step_env_and_add_to_buffer(
                env, obs, agent, {}, replay_buffer
            )

            obs = next_obs
            total_reward += reward
            steps_trial += 1
            env_steps += 1
            last_plan_time = float(getattr(agent, "last_plan_time", 0.0))
            plan_time_sum += last_plan_time
            model_steps_sum += int(getattr(agent, "last_plan_model_steps_est", 0))
            dbg = getattr(agent, "last_plan_debug", None)
            if isinstance(dbg, dict):
                if dbg.get("exec_source", None) == "centroid":
                    centroid_action_steps += 1
                if last_plan_time > 0.0:
                    replan_calls += 1
                    ir_norm = dbg.get("ir_norm", None)
                    if ir_norm is not None:
                        # ---- BC-CEM specific logging and reset ----
                        try:
                            ir_v = float(ir_norm)
                            bccem_ir_norm_sum += ir_v
                            bccem_ir_norm_count += 1
                            bccem_ir_norm_min = (
                                ir_v
                                if bccem_ir_norm_min is None
                                else min(bccem_ir_norm_min, ir_v)
                            )
                            bccem_ir_norm_max = (
                                ir_v
                                if bccem_ir_norm_max is None
                                else max(bccem_ir_norm_max, ir_v)
                            )
                            if debug_mode:
                                print(f"{warning_tag} step {env_steps} | return {total_reward:.3f} | ir_norm {ir_v:.3f} | source {dbg.get('exec_source', 'N/A')}")

                        except (TypeError, ValueError):
                            pass

        skips = int(getattr(agent, "replan_skips", 0))
        if logger is not None:
            centroid_frac = float(centroid_action_steps) / float(max(1, steps_trial))
            ir_norm_mean = (
                float(bccem_ir_norm_sum) / float(bccem_ir_norm_count)
                if bccem_ir_norm_count
                else float("nan")
            )
            logger.log_data(
                mbrl.constants.RESULTS_LOG_NAME,
                {
                    "env_step": env_steps,
                    "episode_reward": total_reward,
                    "planning_time_ms": 1000.0
                    * plan_time_sum / max(1, steps_trial),
                    "planning_model_steps_est": float(model_steps_sum) / max(1, steps_trial),
                    "planning_replans": float(replan_calls),
                    "planning_replans_per_step": float(replan_calls)
                    / float(max(1, steps_trial)),
                    "planning_skips": float(skips),
                    "planning_skips_per_step": float(skips) / float(max(1, steps_trial)),
                    "planning_centroid_frac": centroid_frac,
                    "planning_ir_norm_mean": ir_norm_mean,
                    "planning_ir_norm_min": float(bccem_ir_norm_min)
                    if bccem_ir_norm_min is not None
                    else float("nan"),
                    "planning_ir_norm_max": float(bccem_ir_norm_max)
                    if bccem_ir_norm_max is not None
                    else float("nan"),
                },
            )
        if cfg.algorithm.get("print_planning_info", False):
            opt = getattr(getattr(agent, "optimizer", None), "optimizer", None)
            opt_name = type(opt).__name__ if opt is not None else type(agent).__name__
            msg = f"[planning] opt={opt_name} replans={replan_calls} skips={skips}"
            if bccem_ir_norm_count:
                msg += f" ir_mean={bccem_ir_norm_sum / float(bccem_ir_norm_count):.3f}"
            if bccem_ir_norm_min is not None:
                msg += f" ir_min={float(bccem_ir_norm_min):.3f}"
            if bccem_ir_norm_max is not None:
                msg += f" ir_max={float(bccem_ir_norm_max):.3f}"
            if steps_trial:
                msg += f" centroid_frac={centroid_action_steps / float(steps_trial):.3f}"
            print(msg)
        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
