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
IR_MEAN_EMA_ALPHA = 0.9

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

    agent = mbrl.planning.create_trajectory_optim_agent_for_model(
        model_env, cfg.algorithm.agent, num_particles=cfg.algorithm.num_particles
    )

    opt = getattr(getattr(agent, "optimizer", None), "optimizer", None)
    is_bccem = type(opt).__name__ == "BCCEMOptimizer"

    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    max_total_reward = -np.inf
    bccem_ir_norm_mean_ema = None
    episode_idx = 0
    while env_steps < cfg.overrides.num_steps:
        episode_idx += 1
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
        bccem_ir_low_sum = 0.0
        bccem_ir_low_count = 0
        bccem_ir_high_sum = 0.0
        bccem_ir_high_count = 0
        pred_cv_sum = 0.0
        pred_cv_count = 0
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
                    pred_cv = dbg.get("pred_return_cv", None)
                    pred_cv_v = None
                    if pred_cv is not None:
                        try:
                            pred_cv_v = float(pred_cv)
                            if np.isfinite(pred_cv_v):
                                pred_cv_sum += pred_cv_v
                                pred_cv_count += 1
                        except (TypeError, ValueError):
                            pred_cv_v = None

                    ir_low = dbg.get("ir_low", None)
                    if ir_low is not None:
                        try:
                            v = float(ir_low)
                            if np.isfinite(v):
                                bccem_ir_low_sum += v
                                bccem_ir_low_count += 1
                        except (TypeError, ValueError):
                            pass

                    ir_high = dbg.get("ir_high", None)
                    if ir_high is not None:
                        try:
                            v = float(ir_high)
                            if np.isfinite(v):
                                bccem_ir_high_sum += v
                                bccem_ir_high_count += 1
                        except (TypeError, ValueError):
                            pass

                    ir_norm = dbg.get("ir_norm", None)
                    if ir_norm is not None:
                        # ---- BC-CEM specific logging and reset ----
                        try:
                            ir_v = float(ir_norm)
                            bccem_ir_norm_sum += ir_v
                            bccem_ir_norm_count += 1
                            if debug_mode:
                                cv_msg = (
                                    f" | pred_cv {pred_cv_v:.3g}"
                                    if pred_cv_v is not None
                                    else ""
                                )
                                print(
                                    f"{warning_tag} step {env_steps} | return {total_reward:.2f} | ir_norm {ir_v:.3f}{cv_msg} | source {dbg.get('exec_source', 'N/A')}"
                                )

                        except (TypeError, ValueError):
                            pass

        skips = int(getattr(agent, "replan_skips", 0))
        centroid_frac = float(centroid_action_steps) / float(max(1, steps_trial))
        ir_norm_mean = (
            float(bccem_ir_norm_sum) / float(bccem_ir_norm_count)
            if bccem_ir_norm_count
            else float("nan")
        )
        if bccem_ir_norm_count:
            if bccem_ir_norm_mean_ema is None:
                bccem_ir_norm_mean_ema = ir_norm_mean
            else:
                bccem_ir_norm_mean_ema = (
                    IR_MEAN_EMA_ALPHA * bccem_ir_norm_mean_ema
                    + (1.0 - IR_MEAN_EMA_ALPHA) * ir_norm_mean
                )

        pred_cv_mean = (
            float(pred_cv_sum) / float(pred_cv_count)
            if pred_cv_count
            else float("nan")
        )
        ir_low_mean = (
            float(bccem_ir_low_sum) / float(bccem_ir_low_count)
            if bccem_ir_low_count
            else float("nan")
        )
        ir_high_mean = (
            float(bccem_ir_high_sum) / float(bccem_ir_high_count)
            if bccem_ir_high_count
            else float("nan")
        )

        if logger is not None:
            poplin_total = float("nan")
            poplin_action = float("nan")
            poplin_weights = float("nan")
            poplin_losses = getattr(agent, "poplin_last_losses", None)
            if poplin_losses is not None:
                try:
                    poplin_total = float(getattr(poplin_losses, "total", float("nan")))
                    poplin_action = float(getattr(poplin_losses, "action", float("nan")))
                    poplin_weights = float(getattr(poplin_losses, "weights", float("nan")))
                except (TypeError, ValueError):
                    pass
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
                    "planning_ir_norm_mean_ema": float(bccem_ir_norm_mean_ema)
                    if bccem_ir_norm_mean_ema is not None
                    else float("nan"),
                    "planning_pred_return_cv_mean": pred_cv_mean,
                    "planning_ir_low_mean": ir_low_mean,
                    "planning_ir_high_mean": ir_high_mean,
                    "poplin_policy_loss_total": poplin_total,
                    "poplin_policy_loss_action": poplin_action,
                    "poplin_policy_loss_weights": poplin_weights,
                },
            )
        if cfg.algorithm.get("print_planning_info", False):
            opt = getattr(getattr(agent, "optimizer", None), "optimizer", None)
            opt_name = type(opt).__name__ if opt is not None else type(agent).__name__
            msg = f"[planning] opt={opt_name} replans={replan_calls} skips={skips}"
            if bccem_ir_norm_count:
                msg += f" ir_mean={bccem_ir_norm_sum / float(bccem_ir_norm_count):.3f}"
                if bccem_ir_norm_mean_ema is not None:
                    msg += f" ir_ema={bccem_ir_norm_mean_ema:.3f}"
            if pred_cv_count:
                msg += f" pred_cv={pred_cv_mean:.3g}"
            if steps_trial:
                msg += f" centroid_frac={centroid_action_steps / float(steps_trial):.3f}"
            print(msg)
        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
