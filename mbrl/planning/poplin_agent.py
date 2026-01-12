from __future__ import annotations

import math
import time
from typing import Optional, Sequence, Tuple

import hydra
import numpy as np
import omegaconf
import torch
import torch.nn as nn
from torch.func import functional_call, vmap
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .poplin import PoplinDataset, PoplinLosses, PoplinPlanPolicy, PoplinTrainer
from .trajectory_opt import OptimizerCallback, TrajectoryOptimizerAgent


class PoplinTrajectoryOptimizerAgent(TrajectoryOptimizerAgent):
    """Trajectory optimizer agent augmented with a POPLIN-style distillation policy.

    Supports two variants:

    - POPLIN-A (action space, ``poplin.variant=a``): the policy predicts open-loop action plans
      (single head, or multi-head aligned with multi-worker optimizers) and can optionally
      predict head weights (gating). The policy is trained online by distillation (BC) from
      optimizer targets, with optional winner-take-all (WTA) action loss.
    - POPLIN-P (parameter space, ``poplin.variant=p``): planning is performed in the *policy
      parameter space*. The current policy parameters initialize the mean(s) of the sampling
      distribution, and planning updates parameters directly. During interaction, the policy
      parameters are updated via an AVG-style parameter averaging scheme from the planner.
    """

    def __init__(
        self,
        optimizer_cfg: omegaconf.DictConfig,
        action_lb: Sequence[float],
        action_ub: Sequence[float],
        obs_dim: int,
        planning_horizon: int = 1,
        replan_freq: int = 1,
        verbose: bool = False,
        keep_last_solution: bool = True,
        # Evaluation knobs (same as TrajectoryOptimizerAgent)
        two_stage_eval: bool = True,
        two_stage_particles_frac: float = 0.5,
        two_stage_topk_frac: float = 0.2,
        two_stage_max_topk: int = 64,
        risk_spread_coef: float = 0.0,
        particle_schedule: str = "fixed",
        cv_low_threshold: float = 0.1,
        particles_min_frac: float = 0.25,
        skip_replan_if_ir_low: bool = False,
        skip_replan_ir_threshold: float = 0.1,
        skip_replan_max_frac: float = 0.5,
        # POPLIN config
        poplin: Optional[omegaconf.DictConfig] = None,
    ):
        super().__init__(
            optimizer_cfg=optimizer_cfg,
            action_lb=action_lb,
            action_ub=action_ub,
            planning_horizon=planning_horizon,
            replan_freq=replan_freq,
            verbose=verbose,
            keep_last_solution=keep_last_solution,
            two_stage_eval=two_stage_eval,
            two_stage_particles_frac=two_stage_particles_frac,
            two_stage_topk_frac=two_stage_topk_frac,
            two_stage_max_topk=two_stage_max_topk,
            risk_spread_coef=risk_spread_coef,
            particle_schedule=particle_schedule,
            cv_low_threshold=cv_low_threshold,
            particles_min_frac=particles_min_frac,
            skip_replan_if_ir_low=skip_replan_if_ir_low,
            skip_replan_ir_threshold=skip_replan_ir_threshold,
            skip_replan_max_frac=skip_replan_max_frac,
        )

        self._poplin_cfg = poplin if poplin is not None else omegaconf.OmegaConf.create()
        self.poplin_enabled = bool(self._poplin_cfg.get("enabled", True))
        variant = str(self._poplin_cfg.get("variant", "a")).lower()
        if variant in {"a", "action"}:
            self.poplin_variant = "a"
        elif variant in {"p", "param", "params", "parameter"}:
            self.poplin_variant = "p"
        else:
            raise ValueError(
                f"Invalid poplin.variant={variant!r}. Supported: 'a' (action), 'p' (parameter)."
            )

        # Action-space POPLIN knobs (variant="a").
        self.poplin_use_actions = bool(self._poplin_cfg.get("use_policy_actions", True))
        self.poplin_use_weights = bool(self._poplin_cfg.get("use_policy_weights", True))
        self.poplin_warm_start_mode = str(self._poplin_cfg.get("warm_start_mode", "auto")).lower()
        self.poplin_warm_start_mix = float(self._poplin_cfg.get("warm_start_mix", 1.0))
        self.poplin_warm_start_mix = max(0.0, min(1.0, self.poplin_warm_start_mix))

        # Training controls.
        self.poplin_store_every = int(self._poplin_cfg.get("store_every", 1))
        self.poplin_train_every = int(self._poplin_cfg.get("train_every", 1))
        self.poplin_batch_size = int(self._poplin_cfg.get("batch_size", 256))
        self.poplin_updates_per_train = int(self._poplin_cfg.get("updates_per_train", 1))

        # Expose last losses for logging/diagnostics.
        self.poplin_last_losses: Optional[PoplinLosses] = None

        self._poplin_step = 0

        if not self.poplin_enabled:
            self.poplin_policy = None
            self.poplin_dataset = None
            self.poplin_trainer = None
            self.poplin_param_policies = None
            self.poplin_param_optimizer = None
            return

        device = optimizer_cfg.get("device", "cpu")
        device = torch.device(device) if isinstance(device, str) else device

        base_opt = getattr(self.optimizer, "optimizer", None)
        num_workers = int(getattr(base_opt, "num_workers", 1)) if base_opt is not None else 1
        opt_name = type(base_opt).__name__.lower() if base_opt is not None else ""
        self._poplin_is_bccem = "bccem" in opt_name

        num_heads_cfg = self._poplin_cfg.get("num_heads", "auto")
        if str(num_heads_cfg).lower() == "auto":
            num_heads = max(1, num_workers)
        else:
            num_heads = int(num_heads_cfg)
        num_heads = max(1, num_heads)

        hidden_sizes = tuple(self._poplin_cfg.get("hidden_sizes", (256, 256)))
        activation = str(self._poplin_cfg.get("activation", "relu"))

        self._poplin_obs_dim = int(obs_dim)
        self._poplin_last_policy_plan: Optional[torch.Tensor] = None
        self._poplin_last_plan_is_residual: bool = False

        # ---- Variant A: action-space distillation ----
        if self.poplin_variant == "a":
            self.poplin_param_policies = None
            self.poplin_param_optimizer = None

            self.poplin_policy = PoplinPlanPolicy(
                obs_dim=int(obs_dim),
                action_lb=np.asarray(action_lb, dtype=np.float32),
                action_ub=np.asarray(action_ub, dtype=np.float32),
                planning_horizon=int(planning_horizon),
                action_dim=int(len(action_lb)),
                num_heads=int(num_heads),
                hidden_sizes=hidden_sizes,
                activation=activation,
            ).to(device=device)

            cap = int(self._poplin_cfg.get("dataset_capacity", 100_000))
            self.poplin_dataset = PoplinDataset(
                capacity=cap,
                obs_dim=int(obs_dim),
                num_heads=int(num_heads),
                planning_horizon=int(planning_horizon),
                action_dim=int(len(action_lb)),
                device="cpu",
            )

            self.poplin_trainer = PoplinTrainer(
                self.poplin_policy,
                lr=float(self._poplin_cfg.get("lr", 1e-3)),
                weight_decay=float(self._poplin_cfg.get("weight_decay", 0.0)),
                action_loss_coef=float(self._poplin_cfg.get("action_loss_coef", 1.0)),
                weight_loss_coef=float(self._poplin_cfg.get("weight_loss_coef", 1.0)),
                weight_temperature=float(self._poplin_cfg.get("weight_temperature", 1.0)),
                weight_targets=str(self._poplin_cfg.get("weight_targets", "soft")),
                action_loss_mode=self._poplin_cfg.get("action_loss_mode", None),
                action_loss_weighted_by_targets=bool(
                    self._poplin_cfg.get("action_loss_weighted_by_targets", True)
                ),
                use_mdn=bool(self._poplin_is_bccem),
            )
            return

        # ---- Variant P: parameter-space planning (POPLIN-P) ----
        # In this variant, the policy parameters define the mean(s) of the sampling distribution.
        # We maintain one policy "head" per worker so each worker is warm-started from its own
        # parameter vector. Planning optimizes parameters; training uses parameter averaging (AVG).
        self.poplin_policy = None
        self.poplin_dataset = None
        self.poplin_trainer = None

        # One policy module per head (per optimizer worker).
        self.poplin_param_policies = nn.ModuleList()
        template = PoplinPlanPolicy(
            obs_dim=int(obs_dim),
            action_lb=np.asarray(action_lb, dtype=np.float32),
            action_ub=np.asarray(action_ub, dtype=np.float32),
            planning_horizon=int(planning_horizon),
            action_dim=int(len(action_lb)),
            num_heads=1,
            hidden_sizes=hidden_sizes,
            activation=activation,
        ).to(device=device)
        self.poplin_param_policies.append(template)
        for _ in range(int(num_heads) - 1):
            head = PoplinPlanPolicy(
                obs_dim=int(obs_dim),
                action_lb=np.asarray(action_lb, dtype=np.float32),
                action_ub=np.asarray(action_ub, dtype=np.float32),
                planning_horizon=int(planning_horizon),
                action_dim=int(len(action_lb)),
                num_heads=1,
                hidden_sizes=hidden_sizes,
                activation=activation,
            ).to(device=device)
            head.load_state_dict(template.state_dict())
            self.poplin_param_policies.append(head)

        # Build a flat-parameter specification for stateless evaluation.
        self._poplin_param_total = int(parameters_to_vector(template.parameters()).numel())
        spec = []
        offset = 0
        for name, p in template.named_parameters():
            n = int(p.numel())
            spec.append((name, offset, offset + n, tuple(p.shape)))
            offset += n
        self._poplin_param_spec = spec
        self._poplin_param_buffers = dict(template.named_buffers())

        # Parameter-space optimizer (uses the same optimizer class/config, but bounds are over
        # the policy parameter vector, represented as a (1,P) "trajectory").
        param_lb = float(self._poplin_cfg.get("param_lb", -1.0))
        param_ub = float(self._poplin_cfg.get("param_ub", 1.0))
        if not math.isfinite(param_lb) or not math.isfinite(param_ub) or param_ub <= param_lb:
            raise ValueError(
                f"Invalid parameter bounds: poplin.param_lb={param_lb}, poplin.param_ub={param_ub}."
            )
        self.poplin_param_avg_coef = float(self._poplin_cfg.get("param_avg_coef", 1.0))
        self.poplin_param_avg_coef = max(0.0, min(1.0, self.poplin_param_avg_coef))

        opt_cfg_container = omegaconf.OmegaConf.to_container(optimizer_cfg, resolve=True)
        param_opt_cfg = omegaconf.OmegaConf.create(opt_cfg_container)
        lb = np.full((1, self._poplin_param_total), param_lb, dtype=np.float32)
        ub = np.full((1, self._poplin_param_total), param_ub, dtype=np.float32)
        param_opt_cfg.lower_bound = lb.tolist()
        param_opt_cfg.upper_bound = ub.tolist()
        if "return_mean_elites" in param_opt_cfg:
            param_opt_cfg.return_mean_elites = bool(
                self._poplin_cfg.get("param_return_mean_elites", True)
            )
        self.poplin_param_optimizer = hydra.utils.instantiate(param_opt_cfg)

    def _policy_device(self) -> torch.device:
        if self.poplin_policy is not None:
            return next(self.poplin_policy.parameters()).device
        if self.poplin_param_policies is not None and len(self.poplin_param_policies) > 0:
            return next(self.poplin_param_policies[0].parameters()).device
        return torch.device("cpu")

    @torch.no_grad()
    def _get_warm_start(self, obs: np.ndarray) -> Optional[torch.Tensor]:
        if not self.poplin_enabled or not self.poplin_use_actions or self.poplin_policy is None:
            return None
        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_flat = obs_arr.reshape(-1)
        if int(obs_flat.size) != int(getattr(self, "_poplin_obs_dim", obs_flat.size)):
            return None

        device = self._policy_device()
        obs_t = torch.from_numpy(obs_flat).to(device=device, dtype=torch.float32).view(1, -1)
        plans, logits = self.poplin_policy(obs_t)
        plans = plans[0]  # (K,H,A)

        base_opt = getattr(self.optimizer, "optimizer", None)
        num_workers = int(getattr(base_opt, "num_workers", 1)) if base_opt is not None else 1

        mode = self.poplin_warm_start_mode
        if bool(getattr(self, "_poplin_is_bccem", False)):
            mode = "wta"
        if mode == "auto":
            mode = "multihead" if int(plans.shape[0]) == int(num_workers) and num_workers > 1 else "centroid"

        if mode == "multihead" and int(plans.shape[0]) == int(num_workers) and num_workers > 1:
            x0 = plans
        else:
            if plans.shape[0] <= 1 or not self.poplin_use_weights:
                x0 = plans[0]
            else:
                w = torch.softmax(logits[0], dim=0)
                if mode == "wta":
                    x0 = plans[int(torch.argmax(w).detach().cpu())]
                else:
                    x0 = torch.sum(w.view(-1, 1, 1) * plans, dim=0)

        if self.poplin_warm_start_mix < 1.0 and (not bool(getattr(self, "_poplin_is_bccem", False))):
            try:
                prev = self.optimizer.previous_solution  # (H,A)
                prev = prev.to(device=device, dtype=torch.float32)
                if x0.dim() == 3:
                    prev = prev.unsqueeze(0).expand_as(x0)
                x0 = (1.0 - self.poplin_warm_start_mix) * prev + self.poplin_warm_start_mix * x0
            except Exception:
                pass
        return x0

    def _extract_poplin_targets(self) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        if self.poplin_policy is None:
            return None
        num_heads = int(getattr(self.poplin_policy, "num_heads", 1))
        horizon = int(getattr(self.poplin_policy, "planning_horizon", self.planning_horizon))
        action_dim = int(getattr(self.poplin_policy, "action_dim", len(self.optimizer_args["action_lb"])))

        base_opt = getattr(self.optimizer, "optimizer", None)
        mu_w = getattr(base_opt, "_last_mu_w", None)
        w = getattr(base_opt, "_last_worker_weights", None)

        plans = None
        weights = None
        if torch.is_tensor(mu_w) and mu_w.dim() == 3 and int(mu_w.shape[0]) == num_heads:
            plans = mu_w.detach().cpu()
        if torch.is_tensor(w) and w.numel() == num_heads:
            weights = w.detach().cpu().view(num_heads)

        if plans is None:
            cache = getattr(self, "_action_cache", None)
            if cache is None:
                return None
            plan_np = np.asarray(cache, dtype=np.float32).reshape(horizon, action_dim)
            plans = torch.from_numpy(plan_np).unsqueeze(0).expand(num_heads, horizon, action_dim)

        # If we optimized in residual space (delta), convert optimizer means to action space by
        # adding the policy baseline plan used for this optimization.
        if bool(getattr(self, "_poplin_last_plan_is_residual", False)):
            base = getattr(self, "_poplin_last_policy_plan", None)
            if torch.is_tensor(base):
                try:
                    base = base.to(dtype=plans.dtype).view(1, horizon, action_dim)
                    plans = plans + base
                    lb = torch.as_tensor(self.optimizer_args["action_lb"], dtype=plans.dtype).view(
                        1, 1, action_dim
                    )
                    ub = torch.as_tensor(self.optimizer_args["action_ub"], dtype=plans.dtype).view(
                        1, 1, action_dim
                    )
                    plans = torch.clamp(plans, min=lb, max=ub)
                except Exception:
                    pass

        if weights is None:
            weights = torch.full((num_heads,), 1.0 / float(num_heads), dtype=torch.float32)
        else:
            weights = torch.clamp(weights.to(dtype=torch.float32), min=0.0)
            weights = weights / (weights.sum() + 1e-8)

        # Temper + gate worker weights using BCCEM diagnostics (IR + exec_source), if available.
        # When IR is low / centroid executes, we want *peaky* (confident) targets; otherwise we
        # temper (soften) targets to avoid premature mode collapse.
        if num_heads > 1:
            gate = 1.0
            dbg = getattr(self, "last_plan_debug", None)
            if isinstance(dbg, dict):
                try:
                    ir_norm = dbg.get("ir_norm", None)
                    ir_low = dbg.get("ir_low", None)
                    ir_high = dbg.get("ir_high", None)
                    if ir_norm is not None and ir_low is not None and ir_high is not None:
                        ir_norm_f = float(ir_norm)
                        ir_low_f = float(ir_low)
                        ir_high_f = float(ir_high)
                        if (
                            math.isfinite(ir_norm_f)
                            and math.isfinite(ir_low_f)
                            and math.isfinite(ir_high_f)
                            and ir_high_f > ir_low_f
                        ):
                            gate = (ir_norm_f - ir_low_f) / (ir_high_f - ir_low_f)
                            gate = max(0.0, min(1.0, gate))
                except (TypeError, ValueError):
                    pass
                src = dbg.get("exec_source", None)
                if isinstance(src, str) and src.startswith("centroid"):
                    gate = 0.0
                elif isinstance(src, str) and src.startswith("worker"):
                    gate = 1.0

            # Confident / peaky weights (one-hot) for centroid / low-IR regime.
            hard = torch.zeros_like(weights)
            hard[int(torch.argmax(weights).detach().cpu())] = 1.0

            soft = weights
            if gate > 0.0:
                temp = float(self._poplin_cfg.get("weight_temperature", 1.0))
                if math.isfinite(temp) and temp > 0.0 and temp != 1.0:
                    logw = torch.log(torch.clamp(soft, min=1e-8)) / temp
                    logw = logw - torch.max(logw)
                    soft = torch.softmax(logw, dim=0)

            weights = (1.0 - gate) * hard + gate * soft
            weights = weights / (weights.sum() + 1e-8)

        return plans, weights

    def _act_bccem_residual(
        self, obs: np.ndarray, optimizer_callback: OptimizerCallback = None, **kwargs
    ) -> np.ndarray:
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using PoplinTrajectoryOptimizerAgent"
            )

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
            base_plan = None
            try:
                base_plan = self._get_warm_start(obs)
            except Exception:
                base_plan = None
            if base_plan is None:
                return super().act(obs, optimizer_callback=optimizer_callback, **kwargs)

            device = getattr(self.optimizer, "previous_solution", torch.zeros((), device="cpu")).device
            base_t = base_plan.to(device=device, dtype=torch.float32)
            action_lb = np.asarray(self.optimizer_args["action_lb"], dtype=np.float32)
            action_ub = np.asarray(self.optimizer_args["action_ub"], dtype=np.float32)
            lb_t = torch.as_tensor(action_lb, device=device, dtype=torch.float32).view(1, 1, -1)
            ub_t = torch.as_tensor(action_ub, device=device, dtype=torch.float32).view(1, 1, -1)

            # ---- Residual MPC: delta bounds + recentered warm-start ----
            # We optimize deltas with bounds: a_delta in [lb - a_policy, ub - a_policy].
            base_np = base_t.detach().cpu().numpy()  # (H,A)
            action_mid = (action_lb + action_ub) * 0.5
            delta_lb_np = action_lb.reshape(1, -1) - base_np
            delta_ub_np = action_ub.reshape(1, -1) - base_np

            base_opt = getattr(self.optimizer, "optimizer", None)
            delta_lb_opt = None
            delta_ub_opt = None
            if base_opt is not None:
                try:
                    opt_dev = getattr(base_opt, "device", device)
                    delta_lb_opt = torch.as_tensor(delta_lb_np, device=opt_dev, dtype=torch.float32)
                    delta_ub_opt = torch.as_tensor(delta_ub_np, device=opt_dev, dtype=torch.float32)
                    if torch.is_tensor(getattr(base_opt, "lower_bound", None)) and base_opt.lower_bound.shape == delta_lb_opt.shape:
                        base_opt.lower_bound.copy_(delta_lb_opt)
                    else:
                        base_opt.lower_bound = delta_lb_opt
                    if torch.is_tensor(getattr(base_opt, "upper_bound", None)) and base_opt.upper_bound.shape == delta_ub_opt.shape:
                        base_opt.upper_bound.copy_(delta_ub_opt)
                    else:
                        base_opt.upper_bound = delta_ub_opt
                except Exception:
                    delta_lb_opt = None
                    delta_ub_opt = None

            # Recenter the optimizer's persistent delta-state when the policy baseline changes.
            prev_base = getattr(self, "_poplin_last_policy_plan", None)
            if (
                base_opt is not None
                and torch.is_tensor(prev_base)
                and torch.is_tensor(getattr(base_opt, "_state_mu_w", None))
                and base_opt._state_mu_w.dim() == 3
                and (delta_lb_opt is not None)
                and (delta_ub_opt is not None)
            ):
                try:
                    mu_state = base_opt._state_mu_w
                    W, H, A = int(mu_state.shape[0]), int(mu_state.shape[1]), int(mu_state.shape[2])
                    prev_np = prev_base.detach().cpu().numpy().reshape(H, A)
                    if base_np.shape == (H, A):
                        shift = int(max(0, min(int(getattr(self, "_action_cache_idx", 0)), H)))
                        prev_shift = np.empty((H, A), dtype=np.float32)
                        if shift >= H:
                            prev_shift[:] = action_mid.reshape(1, A)
                        else:
                            prev_shift[: H - shift] = prev_np[shift:]
                            prev_shift[H - shift :] = action_mid.reshape(1, A)
                        offset_np = prev_shift - base_np
                        offset_t = torch.as_tensor(
                            offset_np, device=mu_state.device, dtype=mu_state.dtype
                        ).view(1, H, A)
                        mu_state.add_(offset_t.expand(W, -1, -1))
                        mu_state = torch.minimum(mu_state, delta_ub_opt.to(mu_state.dtype).unsqueeze(0))
                        mu_state = torch.maximum(mu_state, delta_lb_opt.to(mu_state.dtype).unsqueeze(0))
                        base_opt._state_mu_w = mu_state
                except Exception:
                    pass

            # Warm-start deltas from the previous *action* plan (shifted), re-centered to the new
            # baseline. This avoids stale delta warm-starts when the policy plan changes.
            x0_action = None
            if self._action_cache is not None:
                try:
                    cache = np.asarray(self._action_cache, dtype=np.float32).reshape(base_np.shape)
                    idx0 = int(max(0, min(int(getattr(self, "_action_cache_idx", 0)), int(cache.shape[0]))))
                    rem = cache[idx0:]
                    if rem.shape[0] > 0:
                        fill = cache.shape[0] - rem.shape[0]
                        if fill > 0:
                            tail = np.tile(action_mid.reshape(1, -1), (fill, 1))
                            x0_action = np.concatenate([rem, tail], axis=0)
                        else:
                            x0_action = rem
                except Exception:
                    x0_action = None
            if x0_action is None:
                x0_action = base_np

            x0_delta_np = x0_action - base_np
            x0_delta_np = np.clip(x0_delta_np, delta_lb_np, delta_ub_np)
            x0_delta_t = torch.as_tensor(x0_delta_np, device=device, dtype=torch.float32)

            self._poplin_last_policy_plan = base_t.detach().cpu().clone()
            self._poplin_last_plan_is_residual = True

            def trajectory_eval_fn(delta_sequences: torch.Tensor) -> torch.Tensor:
                actions = base_t.unsqueeze(0) + delta_sequences
                actions = torch.clamp(actions, min=lb_t, max=ub_t)
                return self.trajectory_eval_fn(obs, actions)

            start_time = time.perf_counter()
            delta_plan = self.optimizer.optimize(
                trajectory_eval_fn, callback=optimizer_callback, x0=x0_delta_t
            )
            plan_time = time.perf_counter() - start_time
            plan_debug = getattr(self.optimizer, "last_plan_debug", None)
            self.last_plan_debug = plan_debug if isinstance(plan_debug, dict) else None

            action_plan = base_np + np.asarray(delta_plan, dtype=np.float32)
            action_plan = np.clip(action_plan, action_lb, action_ub)

            # Optional: add model-predicted return uncertainty diagnostics for this plan.
            eval_fn = getattr(self, "_model_env_eval_action_sequences", None)
            dbg = self.last_plan_debug
            if (
                callable(eval_fn)
                and isinstance(dbg, dict)
                and (dbg.get("ir_norm", None) is not None)
                and bool(getattr(self, "_model_env_supports_return_variance", False))
            ):
                try:
                    device_eval = getattr(self, "_model_env_device", torch.device("cpu"))
                    particles = int(
                        getattr(
                            self,
                            "_last_eval_particles",
                            getattr(self, "_model_env_base_num_particles", 1),
                        )
                    )
                    particles = max(1, particles)
                    plan_t = torch.from_numpy(np.asarray(action_plan)).to(
                        device=device_eval, dtype=torch.float32
                    )
                    plan_t = plan_t.unsqueeze(0)
                    out = eval_fn(
                        plan_t,
                        initial_state=obs,
                        num_particles=particles,
                        return_variance=True,
                    )
                    mean = None
                    var = None
                    if isinstance(out, (tuple, list)) and len(out) >= 2:
                        mean, var = out[0], out[1]
                    if mean is not None and var is not None:
                        m = float(mean.reshape(-1)[0].detach().cpu().item())
                        v = float(var.reshape(-1)[0].detach().cpu().item())
                        cv = math.sqrt(max(0.0, v)) / (abs(m) + 1e-8)
                        dbg["pred_return_mean"] = m
                        dbg["pred_return_var"] = v
                        dbg["pred_return_cv"] = cv
                        dbg["pred_return_var_particles"] = int(particles)
                except Exception:
                    pass
            self._action_cache = action_plan
            self._action_cache_idx = 0
            self._replan_cache_len = int(
                min(int(self.replan_freq), int(self._action_cache.shape[0]))
            )

        self.last_plan_time = plan_time
        idx = int(self._action_cache_idx)
        action = self._action_cache[idx]
        self._action_cache_idx = idx + 1

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

    @torch.no_grad()
    def _poplin_param_x0(self) -> Optional[torch.Tensor]:
        if self.poplin_param_policies is None or self.poplin_param_optimizer is None:
            return None
        if len(self.poplin_param_policies) <= 0:
            return None
        device = self._policy_device()
        P = int(getattr(self, "_poplin_param_total", 0))
        if P <= 0:
            return None

        vecs = []
        for head in self.poplin_param_policies:
            v = parameters_to_vector(head.parameters()).detach()
            vecs.append(v.to(device=device, dtype=torch.float32))
        x0 = torch.stack(vecs, dim=0).view(len(vecs), 1, P)

        W = int(getattr(self.poplin_param_optimizer, "num_workers", 1))
        if W <= 1:
            return x0[0]
        if x0.shape[0] == W:
            return x0
        if x0.shape[0] == 1:
            return x0.expand(W, 1, P).clone()
        reps = int(math.ceil(float(W) / float(int(x0.shape[0]))))
        tiled = x0.repeat((reps, 1, 1))
        return tiled[:W].clone()

    def _poplin_param_unflatten_batched(self, flat_params: torch.Tensor) -> dict:
        if flat_params.dim() != 2:
            raise ValueError(f"Expected flat_params with shape (B,P), got {tuple(flat_params.shape)}.")
        B = int(flat_params.shape[0])
        out = {}
        for name, start, end, shape in getattr(self, "_poplin_param_spec", []):
            out[name] = flat_params[:, int(start) : int(end)].view((B,) + tuple(shape))
        return out

    @torch.no_grad()
    def _poplin_param_plans_from_flat(self, flat_params: torch.Tensor, obs: np.ndarray) -> torch.Tensor:
        if self.poplin_param_policies is None or len(self.poplin_param_policies) <= 0:
            raise RuntimeError("POPLIN-P is enabled but no parameter-space policies are initialized.")

        obs_arr = np.asarray(obs, dtype=np.float32)
        obs_flat = obs_arr.reshape(-1)
        if int(obs_flat.size) != int(getattr(self, "_poplin_obs_dim", obs_flat.size)):
            raise ValueError(
                f"POPLIN-P requires observations with {getattr(self, '_poplin_obs_dim', None)} elements."
            )

        device = self._policy_device()
        obs_t = torch.from_numpy(obs_flat).to(device=device, dtype=torch.float32)
        obs_b = obs_t.expand(int(flat_params.shape[0]), -1)

        params_batched = self._poplin_param_unflatten_batched(flat_params)
        buffers = getattr(self, "_poplin_param_buffers", {})
        template = self.poplin_param_policies[0]

        def _single(params, obs_one):
            plans, _ = functional_call(template, (params, buffers), (obs_one.view(1, -1),))
            return plans[0, 0]  # (H,A)

        return vmap(_single, in_dims=(0, 0))(params_batched, obs_b)

    @torch.no_grad()
    def _poplin_param_plan_from_solution(self, solution: torch.Tensor, obs: np.ndarray) -> torch.Tensor:
        flat = solution.view(1, -1)
        plans = self._poplin_param_plans_from_flat(flat, obs)
        return plans[0]

    @torch.no_grad()
    def _poplin_param_avg_update(self, fallback_solution: Optional[torch.Tensor] = None) -> None:
        if self.poplin_param_policies is None or self.poplin_param_optimizer is None:
            return
        if len(self.poplin_param_policies) <= 0:
            return
        coef = float(getattr(self, "poplin_param_avg_coef", 1.0))
        if coef <= 0.0:
            return

        mu_w = getattr(self.poplin_param_optimizer, "_last_mu_w", None)
        if torch.is_tensor(mu_w) and mu_w.dim() == 3:
            W = int(mu_w.shape[0])
            for k in range(min(W, len(self.poplin_param_policies))):
                target = mu_w[k].reshape(-1).detach()
                head = self.poplin_param_policies[k]
                if coef < 1.0:
                    curr = parameters_to_vector(head.parameters()).detach()
                    target = (1.0 - coef) * curr + coef * target
                vector_to_parameters(target, head.parameters())
            return

        mu = getattr(self.poplin_param_optimizer, "_last_mu", None)
        if torch.is_tensor(mu):
            target = mu.reshape(-1).detach()
        elif torch.is_tensor(fallback_solution):
            target = fallback_solution.reshape(-1).detach()
        else:
            return

        head0 = self.poplin_param_policies[0]
        if coef < 1.0:
            curr = parameters_to_vector(head0.parameters()).detach()
            target = (1.0 - coef) * curr + coef * target
        vector_to_parameters(target, head0.parameters())

    def _act_parameter_space(
        self, obs: np.ndarray, optimizer_callback: OptimizerCallback = None
    ) -> np.ndarray:
        if self.trajectory_eval_fn is None:
            raise RuntimeError(
                "Please call `set_trajectory_eval_fn()` before using PoplinTrajectoryOptimizerAgent"
            )
        if self.poplin_param_optimizer is None:
            # Fallback to action-space planning if POPLIN-P isn't fully initialized.
            return super().act(obs, optimizer_callback=optimizer_callback)

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
            x0 = self._poplin_param_x0()
            if x0 is None:
                return super().act(obs, optimizer_callback=optimizer_callback)

            def obj_fun(param_population: torch.Tensor) -> torch.Tensor:
                flat = param_population.view(int(param_population.shape[0]), -1)
                plans = self._poplin_param_plans_from_flat(flat, obs)
                return self.trajectory_eval_fn(obs, plans)

            start = time.perf_counter()
            sol = self.poplin_param_optimizer.optimize(obj_fun, x0=x0, callback=optimizer_callback)
            plan_time = time.perf_counter() - start
            dbg = self.poplin_param_optimizer.get_diagnostics()
            self.last_plan_debug = dbg if isinstance(dbg, dict) else None

            # AVG training update (set policy parameters to final mean(s)).
            try:
                self._poplin_param_avg_update(fallback_solution=sol)
            except Exception:
                pass

            plan_t = self._poplin_param_plan_from_solution(sol, obs)
            self._action_cache = plan_t.detach().cpu().numpy()
            self._action_cache_idx = 0
            self._replan_cache_len = int(min(int(self.replan_freq), int(self._action_cache.shape[0])))

        self.last_plan_time = plan_time
        idx = int(self._action_cache_idx)
        action = self._action_cache[idx]
        self._action_cache_idx = idx + 1

        if self._action_cache_idx >= self._action_cache.shape[0]:
            self._action_cache = None
            self._action_cache_idx = 0
            self._replan_cache_len = 0

        if self.verbose:
            print(f"Planning time: {plan_time:.3f}")
        return action

    def act(
        self, obs: np.ndarray, optimizer_callback: OptimizerCallback = None, **kwargs
    ) -> np.ndarray:
        if not self.poplin_enabled or self.poplin_variant != "p":
            if (
                self.poplin_enabled
                and self.poplin_variant == "a"
                and bool(getattr(self, "_poplin_is_bccem", False))
            ):
                action = self._act_bccem_residual(
                    obs, optimizer_callback=optimizer_callback, **kwargs
                )
            else:
                self._poplin_last_plan_is_residual = False
                action = super().act(obs, optimizer_callback=optimizer_callback, **kwargs)

            if self.poplin_enabled and self.poplin_policy is not None and self.poplin_dataset is not None:
                self._poplin_step += 1
                did_replan = bool(getattr(self, "last_plan_time", 0.0) > 0.0)
                if (
                    did_replan
                    and (self.poplin_store_every > 0)
                    and (self._poplin_step % self.poplin_store_every == 0)
                ):
                    targets = self._extract_poplin_targets()
                    if targets is not None:
                        target_plans, target_weights = targets
                        try:
                            self.poplin_dataset.add(obs, target_plans, target_weights)
                        except Exception:
                            pass

                if (
                    (self.poplin_train_every > 0)
                    and (self._poplin_step % self.poplin_train_every == 0)
                    and self.poplin_trainer is not None
                ):
                    losses = self.poplin_trainer.update(
                        self.poplin_dataset,
                        batch_size=self.poplin_batch_size,
                        num_updates=self.poplin_updates_per_train,
                        device=self._policy_device(),
                    )
                    if losses is not None:
                        self.poplin_last_losses = losses

            return action

        self._poplin_step += 1
        return self._act_parameter_space(obs, optimizer_callback=optimizer_callback)

    def reset(self, planning_horizon: Optional[int] = None):
        super().reset(planning_horizon=planning_horizon)
        self._poplin_step = 0
        self.poplin_last_losses = None
        self._poplin_last_policy_plan = None
        self._poplin_last_plan_is_residual = False
        if self.poplin_param_optimizer is not None:
            try:
                self.poplin_param_optimizer.reset()
            except Exception:
                pass
