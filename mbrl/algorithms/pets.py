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
META_COPY_EMA_ALPHA = 0.99
DISTILL_BATCH_SIZE = 256

# TODO: for the ensemble growth, only keeps meta-copy and random init strategies. remove others and related.

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

    # ---------------------------------------------------------
    # ---- Optional BC-CEM ensemble growth (speed-up early) ----
    ensemble_grow_cfg = cfg.algorithm.get("ensemble_grow", None)
    ensemble_grow_enabled_cfg = (
        None if ensemble_grow_cfg is None else ensemble_grow_cfg.get("enabled", None)
    )
    ensemble_grow_enabled = (
        bool(is_bccem)
        if ensemble_grow_enabled_cfg is None
        else bool(ensemble_grow_enabled_cfg)
    )

    ensemble_size_max = int(len(dynamics_model))
    if ensemble_grow_enabled and is_bccem:
        if ensemble_grow_cfg is None:
            init_size = min(2, ensemble_size_max)
        else:
            init_size = int(ensemble_grow_cfg.get("init_size", min(2, ensemble_size_max)))
            max_size_cfg = ensemble_grow_cfg.get("max_size", None)
            if max_size_cfg is not None:
                ensemble_size_max = int(max_size_cfg)
        init_size = max(1, min(init_size, ensemble_size_max))

        if init_size < len(dynamics_model):
            base_model = getattr(dynamics_model, "model", None)
            resize_fn = getattr(base_model, "resize_ensemble", None)
            if callable(resize_fn):
                resize_fn(init_size)
                if debug_mode:
                    print(
                        f"{warning_tag} ensemble_grow init: {len(dynamics_model)}/{ensemble_size_max}"
                    )
            else:
                ensemble_grow_enabled = False
                if debug_mode:
                    print(
                        f"{warning_tag} ensemble_grow disabled (model has no resize_ensemble)."
                    )

    # Cache growth init settings (also used for meta_copy EMA candidate prep).
    ensemble_grow_init_strategy = None
    ensemble_grow_meta_member_idx = None
    ensemble_grow_init_noise_std = None
    ensemble_grow_meta_copy_ema_alpha = None
    ensemble_grow_meta_copy_candidate = False
    if ensemble_grow_enabled and is_bccem:
        ensemble_grow_init_strategy = (
            "meta_copy"
            if ensemble_grow_cfg is None
            else str(ensemble_grow_cfg.get("init_strategy", "meta_copy"))
        )
        init_strategy_norm = str(ensemble_grow_init_strategy).lower()
        ensemble_grow_meta_copy_candidate = init_strategy_norm in {
            "meta_copy",
            "copy",
            "meta",
            "inherit",
        }
        ensemble_grow_meta_member_idx = (
            0
            if ensemble_grow_cfg is None
            else int(ensemble_grow_cfg.get("meta_member_idx", 0))
        )
        ensemble_grow_init_noise_std = (
            0.0
            if ensemble_grow_cfg is None
            else float(ensemble_grow_cfg.get("init_noise_std", 0.0))
        )
        ensemble_grow_meta_copy_ema_alpha = (
            float(META_COPY_EMA_ALPHA)
            if ensemble_grow_cfg is None
            else float(ensemble_grow_cfg.get("meta_copy_ema_alpha", META_COPY_EMA_ALPHA))
        )

    if is_bccem:
        if ensemble_grow_meta_member_idx is None:
            ensemble_grow_meta_member_idx = 0
        if ensemble_grow_meta_copy_ema_alpha is None:
            ensemble_grow_meta_copy_ema_alpha = float(META_COPY_EMA_ALPHA)

    model_trainer = mbrl.models.ModelTrainer(
        dynamics_model,
        optim_lr=cfg.overrides.model_lr,
        weight_decay=cfg.overrides.model_wd,
        logger=logger,
    )

    # Prepare on-hold candidate for meta_copy growth (kept updated after each model training).
    if is_bccem:
        base_model = getattr(dynamics_model, "model", None)
        update_cand_fn = getattr(base_model, "update_meta_copy_candidate", None)
        if callable(update_cand_fn):
            update_cand_fn(
                meta_member_idx=int(ensemble_grow_meta_member_idx),
                ema_alpha=float(ensemble_grow_meta_copy_ema_alpha),
            )

    # ---------------------------------------------------------
    # --------------------- Training Loop ---------------------
    env_steps = 0
    max_total_reward = -np.inf
    bccem_ir_norm_mean_ema = None
    bccem_ir_norm_mean_ema_prev = None
    bccem_ir_ema_stable = 0
    episode_reward_ema = None
    episode_reward_ema_prev = None
    episode_reward_ema_stable = 0
    last_grow_episode = -10**9
    dynamics_distill_mse = float("nan")
    episode_idx = 0
    while env_steps < cfg.overrides.num_steps:
        episode_idx += 1
        obs, _ = env.reset()
        agent.reset()
        terminated = False
        truncated = False
        grew_this_episode = 0.0
        total_reward = 0.0
        plan_time_sum = 0.0
        model_steps_sum = 0
        replan_calls = 0
        centroid_action_steps = 0
        bccem_ir_norm_sum = 0.0
        bccem_ir_norm_count = 0
        bccem_ir_norm_min = None
        bccem_ir_norm_max = None
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
                if is_bccem:
                    base_model = getattr(dynamics_model, "model", None)
                    update_cand_fn = getattr(base_model, "update_meta_copy_candidate", None)
                    if callable(update_cand_fn):
                        update_cand_fn(
                            meta_member_idx=int(ensemble_grow_meta_member_idx),
                            ema_alpha=float(ensemble_grow_meta_copy_ema_alpha),
                        )
                    distill_fn = getattr(base_model, "meta_copy_distillation_score", None)
                    if callable(distill_fn) and hasattr(dynamics_model, "_get_model_input"):
                        try:
                            distill_bs = int(
                                getattr(
                                    cfg.overrides,
                                    "model_batch_size",
                                    int(DISTILL_BATCH_SIZE),
                                )
                            )
                            distill_bs = max(1, min(int(DISTILL_BATCH_SIZE), distill_bs))
                            batch = replay_buffer.sample(distill_bs)
                            model_in = dynamics_model._get_model_input(batch.obs, batch.act)
                            dynamics_distill_mse = float(distill_fn(model_in))
                            if debug_mode:
                                print(
                                    f"{warning_tag} distill_mse {dynamics_distill_mse:.3g}"
                                )
                        except Exception:
                            dynamics_distill_mse = float("nan")

            # --- Optional BC-CEM schedule: anneal ir_high toward target over training steps ---
            opt = getattr(getattr(agent, "optimizer", None), "optimizer", None)
            target = getattr(opt, "ir_high_target", None)
            if target is not None:
                try:
                    total_steps = int(getattr(cfg.overrides, "num_steps", 0))
                except (TypeError, ValueError):
                    total_steps = 0
                if total_steps > 0:
                    denom = max(1, total_steps - 1)
                    t = float(env_steps) / float(denom)
                    t = max(0.0, min(1.0, t))
                    ir_high = getattr(opt, "ir_high", None)
                    if ir_high is not None:
                        start = float(getattr(opt, "_ir_high_start", ir_high))
                        opt.ir_high = start + t * (float(target) - start)

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
                                cv_msg = (
                                    f" | pred_cv {pred_cv_v:.3g}"
                                    if pred_cv_v is not None
                                    else ""
                                )
                                print(
                                    f"{warning_tag} step {env_steps} | return {total_reward:.3f} | ir_norm {ir_v:.3f}{cv_msg} | source {dbg.get('exec_source', 'N/A')}"
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

        # Track episode return EMA for ensemble growth (stagnation signal).
        if episode_reward_ema is None:
            episode_reward_ema = float(total_reward)
        else:
            return_ema_alpha = (
                0.9
                if ensemble_grow_cfg is None
                else float(ensemble_grow_cfg.get("return_ema_alpha", 0.9))
            )
            return_ema_alpha = max(0.0, min(1.0, return_ema_alpha))
            episode_reward_ema = (
                return_ema_alpha * float(episode_reward_ema)
                + (1.0 - return_ema_alpha) * float(total_reward)
            )

        # ---- ensemble grow trigger (BC-CEM) ----
        if ensemble_grow_enabled and is_bccem and (bccem_ir_norm_count > 0):
            ema_abs_tol = 0.01 if ensemble_grow_cfg is None else float(
                ensemble_grow_cfg.get("ema_abs_tol", 0.01)
            )
            ema_rel_tol = 0.02 if ensemble_grow_cfg is None else float(
                ensemble_grow_cfg.get("ema_rel_tol", 0.02)
            )
            patience = 3 if ensemble_grow_cfg is None else int(
                ensemble_grow_cfg.get("patience", 3)
            )
            warmup_episodes = 1 if ensemble_grow_cfg is None else int(
                ensemble_grow_cfg.get("warmup_episodes", 1)
            )
            cooldown_episodes = 0 if ensemble_grow_cfg is None else int(
                ensemble_grow_cfg.get("cooldown_episodes", 0)
            )

            if bccem_ir_norm_mean_ema_prev is not None and bccem_ir_norm_mean_ema is not None:
                delta = abs(float(bccem_ir_norm_mean_ema) - float(bccem_ir_norm_mean_ema_prev))
                rel = delta / (abs(float(bccem_ir_norm_mean_ema_prev)) + 1e-8)
                if (delta <= ema_abs_tol) or (rel <= ema_rel_tol):
                    bccem_ir_ema_stable += 1
                else:
                    bccem_ir_ema_stable = 0
            bccem_ir_norm_mean_ema_prev = bccem_ir_norm_mean_ema

            return_abs_tol = 0.5 if ensemble_grow_cfg is None else float(
                ensemble_grow_cfg.get("return_ema_abs_tol", 0.5)
            )
            return_rel_tol = 0.01 if ensemble_grow_cfg is None else float(
                ensemble_grow_cfg.get("return_ema_rel_tol", 0.01)
            )
            return_patience = patience if ensemble_grow_cfg is None else int(
                ensemble_grow_cfg.get("return_patience", patience)
            )
            if episode_reward_ema_prev is not None and episode_reward_ema is not None:
                delta_r = abs(float(episode_reward_ema) - float(episode_reward_ema_prev))
                rel_r = delta_r / (abs(float(episode_reward_ema_prev)) + 1e-8)
                if (delta_r <= return_abs_tol) or (rel_r <= return_rel_tol):
                    episode_reward_ema_stable += 1
                else:
                    episode_reward_ema_stable = 0
            episode_reward_ema_prev = episode_reward_ema

            if (
                episode_idx >= warmup_episodes
                and bccem_ir_ema_stable >= max(1, patience)
                and episode_reward_ema_stable >= max(1, return_patience)
                and (episode_idx - last_grow_episode) >= max(0, cooldown_episodes)
                and len(dynamics_model) < ensemble_size_max
            ):
                grow_step = 1 if ensemble_grow_cfg is None else int(
                    ensemble_grow_cfg.get("grow_step", 1)
                )
                new_size = min(ensemble_size_max, int(len(dynamics_model)) + max(1, grow_step))

                init_strategy = str(ensemble_grow_init_strategy)
                meta_member_idx = int(ensemble_grow_meta_member_idx)
                noise_std = float(ensemble_grow_init_noise_std)

                base_model = getattr(dynamics_model, "model", None)
                grow_fn = getattr(base_model, "grow_ensemble", None)
                resize_fn = getattr(base_model, "resize_ensemble", None)
                if callable(grow_fn):
                    grow_fn(
                        new_size,
                        init_strategy=init_strategy,
                        meta_member_idx=meta_member_idx,
                        noise_std=noise_std,
                    )
                elif callable(resize_fn):
                    resize_fn(new_size)
                else:
                    ensemble_grow_enabled = False

                if ensemble_grow_enabled:
                    model_trainer = mbrl.models.ModelTrainer(
                        dynamics_model,
                        optim_lr=cfg.overrides.model_lr,
                        weight_decay=cfg.overrides.model_wd,
                        logger=logger,
                    )
                    grew_this_episode = 1.0
                    last_grow_episode = episode_idx
                    bccem_ir_ema_stable = 0
                    episode_reward_ema_stable = 0
                    if debug_mode or cfg.algorithm.get("print_planning_info", False):
                        print(
                            f"{warning_tag} ensemble_grow: {len(dynamics_model)}/{ensemble_size_max}"
                        )

        if logger is not None:
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
                    "episode_reward_ema": float(episode_reward_ema)
                    if episode_reward_ema is not None
                    else float("nan"),
                    "dynamics_ensemble_size": float(len(dynamics_model)),
                    "dynamics_ensemble_grew": float(grew_this_episode),
                    "dynamics_distill_mse": float(dynamics_distill_mse),
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
                if bccem_ir_norm_mean_ema is not None:
                    msg += f" ir_ema={bccem_ir_norm_mean_ema:.3f}"
            if pred_cv_count:
                msg += f" pred_cv={pred_cv_mean:.3g}"
            if ensemble_grow_enabled and is_bccem:
                msg += f" ens={len(dynamics_model)}/{ensemble_size_max}"
            if np.isfinite(float(dynamics_distill_mse)):
                msg += f" distill_mse={float(dynamics_distill_mse):.3g}"
            if bccem_ir_norm_min is not None:
                msg += f" ir_min={float(bccem_ir_norm_min):.3f}"
            if bccem_ir_norm_max is not None:
                msg += f" ir_max={float(bccem_ir_norm_max):.3f}"
            if steps_trial:
                msg += f" centroid_frac={centroid_action_steps / float(steps_trial):.3f}"
            if ensemble_grow_enabled and is_bccem:
                patience = 3 if ensemble_grow_cfg is None else int(ensemble_grow_cfg.get("patience", 3))
                return_patience = (
                    patience
                    if ensemble_grow_cfg is None
                    else int(ensemble_grow_cfg.get("return_patience", patience))
                )
                msg += (
                    f" grow_ir={int(bccem_ir_ema_stable)}/{max(1, int(patience))}"
                    f" grow_ret={int(episode_reward_ema_stable)}/{max(1, int(return_patience))}"
                )
            print(msg)
        max_total_reward = max(max_total_reward, total_reward)

    return np.float32(max_total_reward)
