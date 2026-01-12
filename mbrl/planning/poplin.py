from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoplinLosses:
    total: float
    action: float
    weights: float


class PoplinPlanPolicy(nn.Module):
    """Simple MLP policy that outputs open-loop action sequences and (optionally) head weights.

    Output shapes:
      - plans: (B, K, H, A)
      - logvar: (B, K, H, A) (available via ``forward_dist``)
      - weight_logits: (B, K)
    """

    def __init__(
        self,
        *,
        obs_dim: int,
        action_lb: Union[np.ndarray, torch.Tensor],
        action_ub: Union[np.ndarray, torch.Tensor],
        planning_horizon: int,
        action_dim: int,
        num_heads: int,
        hidden_sizes: Sequence[int] = (256, 256),
        activation: str = "relu",
    ):
        super().__init__()
        if obs_dim <= 0:
            raise ValueError(f"obs_dim must be > 0, got {obs_dim}.")
        if planning_horizon <= 0:
            raise ValueError(f"planning_horizon must be > 0, got {planning_horizon}.")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be > 0, got {action_dim}.")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}.")

        self.obs_dim = int(obs_dim)
        self.planning_horizon = int(planning_horizon)
        self.action_dim = int(action_dim)
        self.num_heads = int(num_heads)

        act_fn: nn.Module
        act_name = str(activation).lower()
        if act_name == "relu":
            act_fn = nn.ReLU()
        elif act_name == "tanh":
            act_fn = nn.Tanh()
        elif act_name == "gelu":
            act_fn = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation={activation!r}.")

        layers = []
        in_dim = self.obs_dim
        for h in hidden_sizes:
            h = int(h)
            if h <= 0:
                raise ValueError(f"Hidden size must be > 0, got {h}.")
            layers.append(nn.Linear(in_dim, h))
            layers.append(act_fn)
            in_dim = h
        self.trunk = nn.Sequential(*layers) if layers else nn.Identity()

        out_dim = self.num_heads * self.planning_horizon * self.action_dim
        # Predict both mean and log-variance for a diagonal Gaussian in action space.
        self.plan_head = nn.Linear(in_dim, 2 * out_dim)
        self.weight_head = nn.Linear(in_dim, self.num_heads) if self.num_heads > 1 else None

        lb = torch.as_tensor(action_lb, dtype=torch.float32).view(1, 1, -1)
        ub = torch.as_tensor(action_ub, dtype=torch.float32).view(1, 1, -1)
        if lb.numel() != self.action_dim or ub.numel() != self.action_dim:
            raise ValueError(
                "action_lb/action_ub must have shape (A,) consistent with action_dim."
            )
        lb = lb.expand(1, self.planning_horizon, self.action_dim)
        ub = ub.expand(1, self.planning_horizon, self.action_dim)
        self.register_buffer("_lb", lb)
        self.register_buffer("_ub", ub)

    def forward_dist(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(
                f"Expected obs with shape (B,{self.obs_dim}), got {tuple(obs.shape)}."
            )
        x = self.trunk(obs)
        raw = self.plan_head(x).view(
            obs.shape[0], self.num_heads, self.planning_horizon, self.action_dim, 2
        )
        mu = raw[..., 0]
        logvar = raw[..., 1].clamp(-10.0, 2.0)
        # Squash to bounds via tanh.
        plans = torch.tanh(mu)
        mid = (self._lb + self._ub) * 0.5
        half_range = (self._ub - self._lb) * 0.5
        plans = mid + half_range * plans

        if self.weight_head is None:
            logits = torch.zeros(
                (obs.shape[0], 1), device=obs.device, dtype=plans.dtype
            )
        else:
            logits = self.weight_head(x)
        return plans, logvar, logits

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        plans, _logvar, logits = self.forward_dist(obs)
        return plans, logits


class PoplinDataset:
    """A simple ring buffer for supervised POPLIN targets."""

    def __init__(
        self,
        *,
        capacity: int,
        obs_dim: int,
        num_heads: int,
        planning_horizon: int,
        action_dim: int,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}.")
        self.capacity = capacity
        self.obs_dim = int(obs_dim)
        self.num_heads = int(num_heads)
        self.planning_horizon = int(planning_horizon)
        self.action_dim = int(action_dim)
        self.device = torch.device(device)
        self.dtype = dtype

        self._obs = torch.empty((capacity, self.obs_dim), device=self.device, dtype=self.dtype)
        self._plans = torch.empty(
            (capacity, self.num_heads, self.planning_horizon, self.action_dim),
            device=self.device,
            dtype=self.dtype,
        )
        self._weights = torch.empty((capacity, self.num_heads), device=self.device, dtype=self.dtype)

        self._size = 0
        self._ptr = 0

    def __len__(self) -> int:
        return int(self._size)

    @torch.no_grad()
    def add(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        target_plans: Union[np.ndarray, torch.Tensor],
        target_weights: Union[np.ndarray, torch.Tensor],
    ) -> None:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=self.dtype).view(-1)
        if obs_t.numel() != self.obs_dim:
            raise ValueError(f"obs must have {self.obs_dim} elements, got {obs_t.numel()}.")

        plans_t = torch.as_tensor(target_plans, device=self.device, dtype=self.dtype)
        plans_t = plans_t.view(self.num_heads, self.planning_horizon, self.action_dim)

        weights_t = torch.as_tensor(target_weights, device=self.device, dtype=self.dtype).view(
            self.num_heads
        )
        s = float(weights_t.sum().detach().cpu())
        if not np.isfinite(s) or s <= 0.0:
            weights_t = torch.full_like(weights_t, 1.0 / float(self.num_heads))
        else:
            weights_t = torch.clamp(weights_t, min=0.0)
            weights_t = weights_t / (weights_t.sum() + 1e-8)

        idx = int(self._ptr)
        self._obs[idx].copy_(obs_t)
        self._plans[idx].copy_(plans_t)
        self._weights[idx].copy_(weights_t)

        self._ptr = (idx + 1) % self.capacity
        self._size = min(self.capacity, self._size + 1)

    def sample(self, batch_size: int, *, device: Optional[Union[str, torch.device]] = None):
        if self._size <= 0:
            raise RuntimeError("Cannot sample from an empty PoplinDataset.")
        batch_size = int(batch_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}.")
        device_t = self.device if device is None else torch.device(device)
        idx = torch.randint(0, int(self._size), (batch_size,), device=self.device)
        obs = self._obs[idx].to(device=device_t, non_blocking=True)
        plans = self._plans[idx].to(device=device_t, non_blocking=True)
        weights = self._weights[idx].to(device=device_t, non_blocking=True)
        return obs, plans, weights


class PoplinTrainer:
    def __init__(
        self,
        policy: PoplinPlanPolicy,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        action_loss_coef: float = 1.0,
        weight_loss_coef: float = 1.0,
        weight_temperature: float = 1.0,
        weight_targets: str = "soft",  # {"soft","hard"}
        action_loss_mode: Optional[str] = None,  # {"uniform","weighted"} (None -> legacy)
        action_loss_weighted_by_targets: bool = True,
        use_mdn: bool = False,
    ):
        self.policy = policy
        self.optim = torch.optim.Adam(
            self.policy.parameters(), lr=float(lr), weight_decay=float(weight_decay)
        )
        self.action_loss_coef = float(action_loss_coef)
        self.weight_loss_coef = float(weight_loss_coef)
        self.weight_temperature = float(weight_temperature)
        self.weight_targets = str(weight_targets).lower()
        if self.weight_targets not in {"soft", "hard"}:
            raise ValueError(
                f"Invalid weight_targets={weight_targets!r}. Supported: 'soft', 'hard'."
            )
        self.action_loss_mode = None if action_loss_mode is None else str(action_loss_mode).lower()
        if self.action_loss_mode is not None and self.action_loss_mode not in {"uniform", "weighted"}:
            raise ValueError(
                "Invalid action_loss_mode. Supported: None, 'uniform', 'weighted'."
            )
        self.action_loss_weighted_by_targets = bool(action_loss_weighted_by_targets)
        self.use_mdn = bool(use_mdn)

    def update(
        self,
        dataset: PoplinDataset,
        *,
        batch_size: int,
        num_updates: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ) -> Optional[PoplinLosses]:
        if len(dataset) < int(batch_size):
            return None

        device_t = torch.device(device) if device is not None else next(self.policy.parameters()).device
        total_loss_v = 0.0
        action_loss_v = 0.0
        weight_loss_v = 0.0

        self.policy.train(True)
        for _ in range(int(num_updates)):
            obs, target_plans, target_weights = dataset.sample(batch_size, device=device_t)
            pred_mu, pred_logvar, pred_logits = self.policy.forward_dist(obs)

            mode = self.action_loss_mode
            if mode is None:
                mode = "weighted" if self.action_loss_weighted_by_targets else "uniform"

            if self.use_mdn and pred_mu.shape[1] > 1 and target_plans.shape[1] > 1:
                # MDN-style loss: each target plan is explained by the mixture over heads.
                temp = max(1e-8, float(self.weight_temperature))
                log_pi = F.log_softmax(pred_logits / temp, dim=1)  # (B,K)
                diff_pair = target_plans[:, None, :, :, :] - pred_mu[:, :, None, :, :]  # (B,K,K,H,A)
                logvar = pred_logvar[:, :, None, :, :]  # (B,K,1,H,A)
                nll_pair = 0.5 * (logvar + torch.square(diff_pair) * torch.exp(-logvar))
                nll_pair = torch.sum(nll_pair, dim=(3, 4))  # (B,K,K)
                log_prob = -nll_pair  # ignore constant
                log_mix = log_pi[:, :, None] + log_prob  # (B,K,K)
                log_p_target = torch.logsumexp(log_mix, dim=1)  # (B,K)
                nll_targets = -log_p_target  # (B,K targets)
                if mode == "weighted":
                    action_loss = torch.mean(torch.sum(nll_targets * target_weights, dim=1))
                else:
                    action_loss = torch.mean(nll_targets)
            else:
                diff = target_plans - pred_mu
                nll = 0.5 * (pred_logvar + torch.square(diff) * torch.exp(-pred_logvar))
                nll = torch.mean(nll, dim=(2, 3))  # (B,K)
                if mode == "weighted":
                    action_loss = torch.mean(torch.sum(nll * target_weights, dim=1))
                else:
                    action_loss = torch.mean(nll)

            if pred_logits.shape[1] <= 1:
                weights_loss = torch.zeros((), device=device_t, dtype=pred_mu.dtype)
            else:
                logits = pred_logits / max(1e-8, float(self.weight_temperature))
                if self.weight_targets == "hard":
                    labels = torch.argmax(target_weights, dim=1)
                    weights_loss = F.cross_entropy(logits, labels)
                else:
                    log_probs = F.log_softmax(logits, dim=1)
                    weights_loss = -torch.mean(torch.sum(target_weights * log_probs, dim=1))

            loss = self.action_loss_coef * action_loss + self.weight_loss_coef * weights_loss

            self.optim.zero_grad(set_to_none=True)
            loss.backward()
            self.optim.step()

            total_loss_v += float(loss.detach().cpu())
            action_loss_v += float(action_loss.detach().cpu())
            weight_loss_v += float(weights_loss.detach().cpu())

        n = float(max(1, int(num_updates)))
        return PoplinLosses(
            total=total_loss_v / n, action=action_loss_v / n, weights=weight_loss_v / n
        )
