# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import pathlib
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import torch
from torch import nn as nn
from torch.nn import functional as F

import mbrl.util.math

from .model import Ensemble
from .util import EnsembleLinearLayer, truncated_normal_init


class GaussianMLP(Ensemble):
    """Implements an ensemble of multi-layer perceptrons each modeling a Gaussian distribution.

    This model corresponds to a Probabilistic Ensemble in the Chua et al.,
    NeurIPS 2018 paper (PETS) https://arxiv.org/pdf/1805.12114.pdf

    It predicts per output mean and log variance, and its weights are updated using a Gaussian
    negative log likelihood loss. The log variance is bounded between learned ``min_log_var``
    and ``max_log_var`` parameters, trained as explained in Appendix A.1 of the paper.

    This class can also be used to build an ensemble of GaussianMLP models, by setting
    ``ensemble_size > 1`` in the constructor. Then, a single forward pass can be used to evaluate
    multiple independent MLPs at the same time. When this mode is active, the constructor will
    set ``self.num_members = ensemble_size``.

    For the ensemble variant, uncertainty propagation methods are available that can be used
    to aggregate the outputs of the different models in the ensemble.
    Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

    The default value of ``None`` indicates that no uncertainty propagation, and the forward
    method returns all outputs of all models.

    Args:
        in_size (int): size of model input.
        out_size (int): size of model output.
        device (str or torch.device): the device to use for the model.
        num_layers (int): the number of layers in the model
                          (e.g., if ``num_layers == 3``, then model graph looks like
                          input -h1-> -h2-> -l3-> output).
        ensemble_size (int): the number of members in the ensemble. Defaults to 1.
        hid_size (int): the size of the hidden layers (e.g., size of h1 and h2 in the graph above).
        deterministic (bool): if ``True``, the model will be trained using MSE loss and no
            logvar prediction will be done. Defaults to ``False``.
        propagation_method (str, optional): the uncertainty propagation method to use (see
            above). Defaults to ``None``.
        learn_logvar_bounds (bool): if ``True``, the logvar bounds will be learned, otherwise
            they will be constant. Defaults to ``False``.
        activation_fn_cfg (dict or omegaconf.DictConfig, optional): configuration of the
            desired activation function. Defaults to torch.nn.ReLU when ``None``.
        weight_init_method (str): weight initialization method. Valid options are
            ``"truncated_normal"`` (default), ``"xavier_uniform"``, ``"xavier_normal"``,
            ``"kaiming_uniform"``, and ``"kaiming_normal"``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        device: Union[str, torch.device],
        num_layers: int = 4,
        ensemble_size: int = 1,
        hid_size: int = 200,
        deterministic: bool = False,
        propagation_method: Optional[str] = None,
        learn_logvar_bounds: bool = False,
        activation_fn_cfg: Optional[Union[Dict, omegaconf.DictConfig]] = None,
        weight_init_method: str = "truncated_normal",
    ):
        super().__init__(
            ensemble_size, device, propagation_method, deterministic=deterministic
        )

        self.in_size = in_size
        self.out_size = out_size
        self.weight_init_method = weight_init_method

        if weight_init_method not in {
            "truncated_normal",
            "xavier_uniform",
            "xavier_normal",
            "kaiming_uniform",
            "kaiming_normal",
        }:
            raise ValueError(f"Invalid weight_init_method={weight_init_method}.")

        def create_activation():
            if activation_fn_cfg is None:
                activation_func = nn.ReLU()
            else:
                # Handle the case where activation_fn_cfg is a dict
                cfg = omegaconf.OmegaConf.create(activation_fn_cfg)
                activation_func = hydra.utils.instantiate(cfg)
            return activation_func

        def create_linear_layer(l_in, l_out):
            return EnsembleLinearLayer(ensemble_size, l_in, l_out)

        hidden_layers = [
            nn.Sequential(create_linear_layer(in_size, hid_size), create_activation())
        ]
        for i in range(num_layers - 1):
            hidden_layers.append(
                nn.Sequential(
                    create_linear_layer(hid_size, hid_size),
                    create_activation(),
                )
            )
        self.hidden_layers = nn.Sequential(*hidden_layers)

        if deterministic:
            self.mean_and_logvar = create_linear_layer(hid_size, out_size)
        else:
            self.mean_and_logvar = create_linear_layer(hid_size, 2 * out_size)
            self.min_logvar = nn.Parameter(
                -10 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )
            self.max_logvar = nn.Parameter(
                0.5 * torch.ones(1, out_size), requires_grad=learn_logvar_bounds
            )

        self._reset_parameters()
        self.to(self.device)

        self.elite_models: List[int] = None

    @staticmethod
    def _infer_init_nonlinearity(activation: nn.Module) -> Tuple[str, float]:
        if isinstance(activation, (nn.ReLU, nn.SiLU, nn.ELU, nn.GELU)):
            return "relu", 0.0
        if isinstance(activation, nn.LeakyReLU):
            return "leaky_relu", float(activation.negative_slope)
        if isinstance(activation, nn.Tanh):
            return "tanh", 0.0
        if isinstance(activation, nn.Sigmoid):
            return "sigmoid", 0.0
        return "linear", 0.0

    @staticmethod
    @torch.no_grad()
    def _init_ensemble_linear_layer_(
        layer: EnsembleLinearLayer, method: str, nonlinearity: str, a: float
    ) -> None:
        fan_in = layer.in_size
        fan_out = layer.out_size

        if method == "truncated_normal":
            stddev = 1.0 / (2.0 * math.sqrt(fan_in))
            trunc_normal = getattr(torch.nn.init, "trunc_normal_", None)
            if trunc_normal is not None:
                trunc_normal(
                    layer.weight,
                    mean=0.0,
                    std=float(stddev),
                    a=float(-2.0 * stddev),
                    b=float(2.0 * stddev),
                )
            else:
                for i in range(layer.weight.shape[0]):
                    mbrl.util.math.truncated_normal_(layer.weight[i], std=stddev)
        elif method == "xavier_uniform":
            gain = nn.init.calculate_gain(
                nonlinearity, a if nonlinearity == "leaky_relu" else None
            )
            bound = gain * math.sqrt(6.0 / float(fan_in + fan_out))
            layer.weight.uniform_(-bound, bound)
        elif method == "xavier_normal":
            gain = nn.init.calculate_gain(
                nonlinearity, a if nonlinearity == "leaky_relu" else None
            )
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            layer.weight.normal_(0.0, std)
        elif method == "kaiming_uniform":
            gain = (
                nn.init.calculate_gain("leaky_relu", a)
                if nonlinearity == "leaky_relu"
                else nn.init.calculate_gain("relu")
                if nonlinearity == "relu"
                else 1.0
            )
            bound = math.sqrt(3.0) * gain / math.sqrt(float(fan_in))
            layer.weight.uniform_(-bound, bound)
        elif method == "kaiming_normal":
            gain = (
                nn.init.calculate_gain("leaky_relu", a)
                if nonlinearity == "leaky_relu"
                else nn.init.calculate_gain("relu")
                if nonlinearity == "relu"
                else 1.0
            )
            std = gain / math.sqrt(float(fan_in))
            layer.weight.normal_(0.0, std)
        else:
            raise ValueError(f"Invalid init method {method}.")

        if layer.use_bias:
            layer.bias.zero_()

    def _reset_parameters(self) -> None:
        if self.weight_init_method == "truncated_normal":
            self.apply(truncated_normal_init)
            return
        first_act = self.hidden_layers[0][1]
        hidden_nonlin, hidden_a = self._infer_init_nonlinearity(first_act)
        for layer in self.hidden_layers:
            self._init_ensemble_linear_layer_(
                layer[0], self.weight_init_method, hidden_nonlin, hidden_a
            )
        self._init_ensemble_linear_layer_(
            self.mean_and_logvar, self.weight_init_method, "linear", 0.0
        )

    def _maybe_toggle_layers_use_only_elite(self, only_elite: bool):
        if self.elite_models is None:
            return
        if self.num_members > 1 and only_elite:
            for layer in self.hidden_layers:
                # each layer is (linear layer, activation_func)
                layer[0].set_elite(self.elite_models)
                layer[0].toggle_use_only_elite()
            self.mean_and_logvar.set_elite(self.elite_models)
            self.mean_and_logvar.toggle_use_only_elite()

    def _default_forward(
        self, x: torch.Tensor, only_elite: bool = False, **_kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._maybe_toggle_layers_use_only_elite(only_elite)
        x = self.hidden_layers(x)
        mean_and_logvar = self.mean_and_logvar(x)
        self._maybe_toggle_layers_use_only_elite(only_elite)
        if self.deterministic:
            return mean_and_logvar, None
        else:
            mean = mean_and_logvar[..., : self.out_size]
            logvar = mean_and_logvar[..., self.out_size :]
            logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
            logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
            return mean, logvar

    def _forward_from_indices(
        self, x: torch.Tensor, model_shuffle_indices: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, batch_size, _ = x.shape

        num_models = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        shuffled_x = x[:, model_shuffle_indices, ...].view(
            num_models, batch_size // num_models, -1
        )

        mean, logvar = self._default_forward(shuffled_x, only_elite=True)
        # note that mean and logvar are shuffled
        mean = mean.view(batch_size, -1)
        mean[model_shuffle_indices] = mean.clone()  # invert the shuffle

        if logvar is not None:
            logvar = logvar.view(batch_size, -1)
            logvar[model_shuffle_indices] = logvar.clone()  # invert the shuffle

        return mean, logvar

    def _forward_ensemble(
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.propagation_method is None:
            mean, logvar = self._default_forward(x, only_elite=False)
            if self.num_members == 1:
                mean = mean[0]
                logvar = logvar[0] if logvar is not None else None
            return mean, logvar
        assert x.ndim == 2
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if x.shape[0] % model_len != 0:
            raise ValueError(
                f"GaussianMLP ensemble requires batch size to be a multiple of the "
                f"number of models. Current batch size is {x.shape[0]} for "
                f"{model_len} models."
            )
        x = x.unsqueeze(0)
        if self.propagation_method == "random_model":
            # passing generator causes segmentation fault
            # see https://github.com/pytorch/pytorch/issues/44714
            model_indices = torch.randperm(x.shape[1], device=self.device)
            return self._forward_from_indices(x, model_indices)
        if self.propagation_method == "fixed_model":
            if propagation_indices is None:
                raise ValueError(
                    "When using propagation='fixed_model', `propagation_indices` must be provided."
                )
            return self._forward_from_indices(x, propagation_indices)
        if self.propagation_method == "expectation":
            mean, logvar = self._default_forward(x, only_elite=True)
            return mean.mean(dim=0), logvar.mean(dim=0)
        raise ValueError(f"Invalid propagation method {self.propagation_method}.")

    def forward(  # type: ignore
        self,
        x: torch.Tensor,
        rng: Optional[torch.Generator] = None,
        propagation_indices: Optional[torch.Tensor] = None,
        use_propagation: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes mean and logvar predictions for the given input.

        When ``self.num_members > 1``, the model supports uncertainty propagation options
        that can be used to aggregate the outputs of the different models in the ensemble.
        Valid propagation options are:

            - "random_model": for each output in the batch a model will be chosen at random.
              This corresponds to TS1 propagation in the PETS paper.
            - "fixed_model": for output j-th in the batch, the model will be chosen according to
              the model index in `propagation_indices[j]`. This can be used to implement TSinf
              propagation, described in the PETS paper.
            - "expectation": the output for each element in the batch will be the mean across
              models.

        If a set of elite models has been indicated (via :meth:`set_elite()`), then all
        propagation methods will operate with only on the elite set. This has no effect when
        ``propagation is None``, in which case the forward pass will return one output for
        each model.

        Args:
            x (tensor): the input to the model. When ``self.propagation is None``,
                the shape must be ``E x B x Id`` or ``B x Id``, where ``E``, ``B``
                and ``Id`` represent ensemble size, batch size, and input dimension,
                respectively. In this case, each model in the ensemble will get one slice
                from the first dimension (e.g., the i-th ensemble member gets ``x[i]``).

                For other values of ``self.propagation`` (and ``use_propagation=True``),
                the shape must be ``B x Id``.
            rng (torch.Generator, optional): random number generator to use for "random_model"
                propagation.
            propagation_indices (tensor, optional): propagation indices to use,
                as generated by :meth:`sample_propagation_indices`. Ignore if
                `use_propagation == False` or `self.propagation_method != "fixed_model".
            use_propagation (bool): if ``False``, the propagation method will be ignored
                and the method will return outputs for all models. Defaults to ``True``.

        Returns:
            (tuple of two tensors): the predicted mean and log variance of the output. If
            ``propagation is not None``, the output will be 2-D (batch size, and output dimension).
            Otherwise, the outputs will have shape ``E x B x Od``, where ``Od`` represents
            output dimension.

        Note:
            For efficiency considerations, the propagation method used by this class is an
            approximate version of that described by Chua et al. In particular, instead of
            sampling models independently for each input in the batch, we ensure that each
            model gets exactly the same number of samples (which are assigned randomly
            with equal probability), resulting in a smaller batch size which we use for the forward
            pass. If this is a concern, consider using ``propagation=None``, and passing
            the output to :func:`mbrl.util.math.propagate`.

        """
        if use_propagation:
            return self._forward_ensemble(
                x, rng=rng, propagation_indices=propagation_indices
            )
        return self._default_forward(x)

    def _mse_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add model dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        return F.mse_loss(pred_mean, target, reduction="none").sum((1, 2)).sum()

    def _nll_loss(self, model_in: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert model_in.ndim == target.ndim
        if model_in.ndim == 2:  # add ensemble dimension
            model_in = model_in.unsqueeze(0)
            target = target.unsqueeze(0)
        pred_mean, pred_logvar = self.forward(model_in, use_propagation=False)
        if target.shape[0] != self.num_members:
            target = target.repeat(self.num_members, 1, 1)
        nll = (
            mbrl.util.math.gaussian_nll(pred_mean, pred_logvar, target, reduce=False)
            .mean((1, 2))  # average over batch and target dimension
            .sum()
        )  # sum over ensemble dimension
        nll += 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        return nll

    def loss(
        self,
        model_in: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes Gaussian NLL loss.

        It also includes terms for ``max_logvar`` and ``min_logvar`` with small weights,
        with positive and negative signs, respectively.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``E x B x Id``, or ``B x Id``
                where ``E``, ``B`` and ``Id`` represent ensemble size, batch size, and input
                dimension, respectively.
            target (tensor): target tensor. The shape must be ``E x B x Id``, or ``B x Od``
                where ``E``, ``B`` and ``Od`` represent ensemble size, batch size, and output
                dimension, respectively.

        Returns:
            (tensor): a loss tensor representing the Gaussian negative log-likelihood of
            the model over the given input/target. If the model is an ensemble, returns
            the average over all models.
        """
        if self.deterministic:
            return self._mse_loss(model_in, target), {}
        else:
            return self._nll_loss(model_in, target), {}

    def eval_score(  # type: ignore
        self, model_in: torch.Tensor, target: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Computes the squared error for the model over the given input/target.

        When model is not an ensemble, this is equivalent to
        `F.mse_loss(model(model_in, target), reduction="none")`. If the model is ensemble,
        then return is batched over the model dimension.

        This function returns no metadata, so the second output is set to an empty dict.

        Args:
            model_in (tensor): input tensor. The shape must be ``B x Id``, where `B`` and ``Id``
                batch size, and input dimension, respectively.
            target (tensor): target tensor. The shape must be ``B x Od``, where ``B`` and ``Od``
                represent batch size, and output dimension, respectively.

        Returns:
            (tensor): a tensor with the squared error per output dimension, batched over model.
        """
        assert model_in.ndim == 2 and target.ndim == 2
        with torch.no_grad():
            pred_mean, _ = self.forward(model_in, use_propagation=False)
            target = target.repeat((self.num_members, 1, 1))
            return F.mse_loss(pred_mean, target, reduction="none"), {}

    def sample_propagation_indices(
        self, batch_size: int, _rng: torch.Generator
    ) -> torch.Tensor:
        model_len = (
            len(self.elite_models) if self.elite_models is not None else len(self)
        )
        if batch_size % model_len != 0:
            raise ValueError(
                "To use GaussianMLP's ensemble propagation, the batch size must "
                "be a multiple of the number of models in the ensemble."
            )
        # rng causes segmentation fault, see https://github.com/pytorch/pytorch/issues/44714
        return torch.randperm(batch_size, device=self.device)

    def set_elite(self, elite_indices: Sequence[int]):
        if len(elite_indices) != self.num_members:
            self.elite_models = list(elite_indices)

    def save(self, save_dir: Union[str, pathlib.Path]):
        """Saves the model to the given directory."""
        model_dict = {
            "state_dict": self.state_dict(),
            "elite_models": self.elite_models,
        }
        torch.save(model_dict, pathlib.Path(save_dir) / self._MODEL_FNAME)

    def load(self, load_dir: Union[str, pathlib.Path]):
        """Loads the model from the given path."""
        model_dict = torch.load(pathlib.Path(load_dir) / self._MODEL_FNAME)
        self.load_state_dict(model_dict["state_dict"])
        self.elite_models = model_dict["elite_models"]


class GaussianMLPAda(GaussianMLP):
    """GaussianMLP variant that supports changing ensemble size after construction."""

    @torch.no_grad()
    def update_meta_copy_candidate(
        self, *, meta_member_idx: int = 0, ema_alpha: float = 0.99
    ) -> None:
        """Updates an EMA "candidate" member for later `init_strategy="meta_copy"` growth."""
        num_members = int(self.num_members)
        if num_members <= 0:
            return

        meta_member_idx = int(meta_member_idx)
        meta_member_idx = max(0, min(num_members - 1, meta_member_idx))
        ema_alpha = float(ema_alpha)
        ema_alpha = max(0.0, min(1.0, ema_alpha))

        candidate = getattr(self, "_meta_copy_candidate", None)
        candidate_meta_idx = getattr(self, "_meta_copy_candidate_meta_idx", None)
        expected_len = len(self.hidden_layers) + 1  # + mean_and_logvar

        def _clone_member(
            layer: EnsembleLinearLayer,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            w = layer.weight.data[meta_member_idx].clone()
            b = layer.bias.data[meta_member_idx].clone() if layer.use_bias else None
            return w, b

        if (
            not isinstance(candidate, list)
            or candidate_meta_idx != meta_member_idx
            or len(candidate) != expected_len
        ):
            new_candidate: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for seq in self.hidden_layers:
                new_candidate.append(_clone_member(seq[0]))
            new_candidate.append(_clone_member(self.mean_and_logvar))
            self._meta_copy_candidate = new_candidate
            self._meta_copy_candidate_meta_idx = meta_member_idx
            return

        # If device/dtype changed since candidate creation, re-init.
        try:
            cand_w0, _ = candidate[0]
            src_w0 = self.hidden_layers[0][0].weight.data[meta_member_idx]
            if (
                cand_w0.device != src_w0.device
                or cand_w0.dtype != src_w0.dtype
                or cand_w0.shape != src_w0.shape
            ):
                raise RuntimeError("meta_copy candidate is stale")
        except Exception:
            new_candidate: List[Tuple[torch.Tensor, Optional[torch.Tensor]]] = []
            for seq in self.hidden_layers:
                new_candidate.append(_clone_member(seq[0]))
            new_candidate.append(_clone_member(self.mean_and_logvar))
            self._meta_copy_candidate = new_candidate
            self._meta_copy_candidate_meta_idx = meta_member_idx
            return

        one_minus = 1.0 - ema_alpha
        for i, seq in enumerate(self.hidden_layers):
            layer = seq[0]
            cand_w, cand_b = candidate[i]
            cand_w.mul_(ema_alpha).add_(layer.weight.data[meta_member_idx], alpha=one_minus)
            if layer.use_bias:
                if cand_b is None:
                    cand_b = layer.bias.data[meta_member_idx].clone()
                    candidate[i] = (cand_w, cand_b)
                cand_b.mul_(ema_alpha).add_(layer.bias.data[meta_member_idx], alpha=one_minus)

        layer = self.mean_and_logvar
        cand_w, cand_b = candidate[-1]
        cand_w.mul_(ema_alpha).add_(layer.weight.data[meta_member_idx], alpha=one_minus)
        if layer.use_bias:
            if cand_b is None:
                cand_b = layer.bias.data[meta_member_idx].clone()
                candidate[-1] = (cand_w, cand_b)
            cand_b.mul_(ema_alpha).add_(layer.bias.data[meta_member_idx], alpha=one_minus)

    @torch.no_grad()
    def _forward_meta_copy_candidate(
        self, x: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        candidate = getattr(self, "_meta_copy_candidate", None)
        if not (isinstance(candidate, list) and len(candidate) == (len(self.hidden_layers) + 1)):
            return None

        h = x
        for i, seq in enumerate(self.hidden_layers):
            layer = seq[0]
            act = seq[1]
            w, b = candidate[i]
            h = h.matmul(w)
            if layer.use_bias:
                if b is None:
                    return None
                h = h + b
            h = act(h)

        layer = self.mean_and_logvar
        w, b = candidate[-1]
        out = h.matmul(w)
        if layer.use_bias:
            if b is None:
                return None
            out = out + b
        if self.deterministic:
            return out, None

        mean = out[..., : self.out_size]
        logvar = out[..., self.out_size :]
        logvar = self.max_logvar - F.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        return mean, logvar

    @torch.no_grad()
    def meta_copy_distillation_score(self, model_in: torch.Tensor) -> float:
        """Mean squared prediction error between current ensemble and the EMA candidate."""
        cand = self._forward_meta_copy_candidate(model_in)
        if cand is None:
            return float("nan")
        cand_mean, _ = cand
        pred_mean, _ = self.forward(model_in, use_propagation=False)
        if pred_mean.ndim == 2:
            pred_mean = pred_mean.unsqueeze(0)
        mse = torch.mean((pred_mean - cand_mean.unsqueeze(0)) ** 2)
        return float(mse.item())

    @torch.no_grad()
    def resize_ensemble(
        self, new_ensemble_size: int, member_indices: Optional[Sequence[int]] = None
    ) -> None:
        """Resizes the ensemble in-place.

        Args:
            new_ensemble_size: New ensemble size.
            member_indices: Optional indices of current members to keep/copy into the
                front of the resized ensemble. If ``None``, keeps the first
                ``min(old_size, new_ensemble_size)`` members (or all current members when
                growing).
        """
        if new_ensemble_size < 1:
            raise ValueError("new_ensemble_size must be >= 1.")
        old_size = self.num_members
        if new_ensemble_size == old_size:
            return

        if member_indices is None:
            keep_count = min(old_size, new_ensemble_size)
            member_indices = tuple(range(keep_count))
        else:
            if len(member_indices) > new_ensemble_size:
                raise ValueError(
                    "member_indices can't be longer than new_ensemble_size."
                )
            member_indices = tuple(member_indices)

        if not all(isinstance(i, int) for i in member_indices):
            raise ValueError("member_indices must be a sequence of ints.")
        if not all(0 <= i < old_size for i in member_indices):
            raise ValueError("member_indices must be valid indices into the old ensemble.")

        idx = torch.as_tensor(member_indices, device=self.device)

        first_act = self.hidden_layers[0][1]
        hidden_nonlin, hidden_a = self._infer_init_nonlinearity(first_act)

        def _init_weights_(w: torch.Tensor, nonlinearity: str, a: float) -> None:
            if w.numel() == 0:
                return
            fan_in = w.shape[1]
            fan_out = w.shape[2]
            method = self.weight_init_method
            if method == "truncated_normal":
                stddev = 1.0 / (2.0 * math.sqrt(fan_in))
                trunc_normal = getattr(torch.nn.init, "trunc_normal_", None)
                if trunc_normal is not None:
                    trunc_normal(
                        w,
                        mean=0.0,
                        std=float(stddev),
                        a=float(-2.0 * stddev),
                        b=float(2.0 * stddev),
                    )
                else:
                    for i in range(w.shape[0]):
                        mbrl.util.math.truncated_normal_(w[i], std=stddev)
            elif method == "xavier_uniform":
                gain = nn.init.calculate_gain(
                    nonlinearity, a if nonlinearity == "leaky_relu" else None
                )
                bound = gain * math.sqrt(6.0 / float(fan_in + fan_out))
                w.uniform_(-bound, bound)
            elif method == "xavier_normal":
                gain = nn.init.calculate_gain(
                    nonlinearity, a if nonlinearity == "leaky_relu" else None
                )
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                w.normal_(0.0, std)
            elif method == "kaiming_uniform":
                gain = (
                    nn.init.calculate_gain("leaky_relu", a)
                    if nonlinearity == "leaky_relu"
                    else nn.init.calculate_gain("relu")
                    if nonlinearity == "relu"
                    else 1.0
                )
                bound = math.sqrt(3.0) * gain / math.sqrt(float(fan_in))
                w.uniform_(-bound, bound)
            elif method == "kaiming_normal":
                gain = (
                    nn.init.calculate_gain("leaky_relu", a)
                    if nonlinearity == "leaky_relu"
                    else nn.init.calculate_gain("relu")
                    if nonlinearity == "relu"
                    else 1.0
                )
                std = gain / math.sqrt(float(fan_in))
                w.normal_(0.0, std)
            else:
                raise ValueError(f"Invalid init method {method}.")

        def _resize_layer(layer: EnsembleLinearLayer, nonlinearity: str, a: float) -> None:
            src_w = layer.weight.detach()
            dst_w = src_w.new_empty((new_ensemble_size, layer.in_size, layer.out_size))

            if idx.numel() > 0:
                dst_w[: idx.numel()].copy_(src_w.index_select(0, idx))
            _init_weights_(dst_w[idx.numel() :], nonlinearity, a)

            layer.weight = nn.Parameter(dst_w)
            layer.num_members = new_ensemble_size

            if layer.use_bias:
                src_b = layer.bias.detach()
                dst_b = src_b.new_empty((new_ensemble_size, 1, layer.out_size))
                if idx.numel() > 0:
                    dst_b[: idx.numel()].copy_(src_b.index_select(0, idx))
                dst_b[idx.numel() :].zero_()
                layer.bias = nn.Parameter(dst_b)

            layer.elite_models = None
            layer.use_only_elite = False

        for layer in self.hidden_layers:
            _resize_layer(layer[0], hidden_nonlin, hidden_a)
        _resize_layer(self.mean_and_logvar, "linear", 0.0)

        self.num_members = new_ensemble_size

        if self.elite_models is not None:
            old_to_new = {old_i: new_i for new_i, old_i in enumerate(member_indices)}
            self.elite_models = [
                old_to_new[i] for i in self.elite_models if i in old_to_new
            ] or None

    @torch.no_grad()
    def grow_ensemble(
        self,
        new_ensemble_size: int,
        *,
        init_strategy: str = "meta_copy",
        meta_member_idx: int = 0,
        noise_std: float = 0.0,
        member_indices: Optional[Sequence[int]] = None,
    ) -> None:
        """Resizes the ensemble, optionally initializing new members by inheritance.

        Args:
            new_ensemble_size: desired ensemble size.
            init_strategy: initialization strategy for *new* members when growing.
                Supported values:
                  - "random": use the model's configured weight init.
                  - "meta_copy": copy weights/biases from ``meta_member_idx``.
                  - "mean": copy the mean weights/biases of the existing members.
            meta_member_idx: which existing member to inherit from for "meta_copy".
            noise_std: optional Gaussian noise (stddev) to add to newly initialized members
                (applied after copy/mean init; ignored for "random").
            member_indices: optional indices of current members to keep/copy into the resized
                ensemble (see :meth:`resize_ensemble`).
        """
        old_size = int(self.num_members)
        new_ensemble_size = int(new_ensemble_size)
        if new_ensemble_size == old_size:
            return
        if new_ensemble_size < 1:
            raise ValueError("new_ensemble_size must be >= 1.")

        # Shrinking: no special init required.
        if new_ensemble_size < old_size:
            self.resize_ensemble(new_ensemble_size, member_indices=member_indices)
            return

        # Growing: first allocate and init new slots, then optionally overwrite.
        self.resize_ensemble(new_ensemble_size, member_indices=member_indices)

        init_strategy = str(init_strategy).lower()
        init_strategy = "meta_copy" if init_strategy in {"copy", "meta", "inherit"} else init_strategy
        if init_strategy == "random":
            return

        if old_size <= 0:
            return
        start = old_size
        end = new_ensemble_size
        if start >= end:
            return

        def _copy_from_tensor_mean(dst: torch.Tensor, src: torch.Tensor) -> None:
            if src.ndim < 1 or src.shape[0] != old_size:
                raise ValueError("Unexpected parameter shape for ensemble initialization.")
            mean = src.mean(dim=0, keepdim=True)
            dst[start:end].copy_(mean.expand(end - start, *mean.shape[1:]))

        def _copy_from_tensor_member(dst: torch.Tensor, src: torch.Tensor, idx: int) -> None:
            if src.ndim < 1 or src.shape[0] != old_size:
                raise ValueError("Unexpected parameter shape for ensemble initialization.")
            dst[start:end].copy_(src[idx : idx + 1].expand(end - start, *src.shape[1:]))

        def _copy_from_candidate(dst: torch.Tensor, cand: torch.Tensor) -> None:
            dst[start:end].copy_(cand.unsqueeze(0).expand(end - start, *cand.shape))

        def _init_layer(
            layer: EnsembleLinearLayer,
            idx: int,
            candidate_params: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None,
        ) -> None:
            if init_strategy == "mean":
                _copy_from_tensor_mean(layer.weight, layer.weight[:old_size])
                if layer.use_bias:
                    _copy_from_tensor_mean(layer.bias, layer.bias[:old_size])
            elif init_strategy == "meta_copy":
                if candidate_params is not None:
                    cand_w, cand_b = candidate_params
                    _copy_from_candidate(layer.weight, cand_w)
                    if layer.use_bias:
                        if cand_b is None:
                            raise ValueError("meta_copy candidate is missing bias parameters.")
                        _copy_from_candidate(layer.bias, cand_b)
                else:
                    _copy_from_tensor_member(layer.weight, layer.weight[:old_size], idx)
                    if layer.use_bias:
                        _copy_from_tensor_member(layer.bias, layer.bias[:old_size], idx)
            else:
                raise ValueError(f"Invalid init_strategy={init_strategy}.")

            if noise_std > 0.0:
                layer.weight.data[start:end].add_(float(noise_std) * torch.randn_like(layer.weight.data[start:end]))
                if layer.use_bias:
                    layer.bias.data[start:end].add_(float(noise_std) * torch.randn_like(layer.bias.data[start:end]))

        meta_member_idx = int(meta_member_idx)
        meta_member_idx = max(0, min(old_size - 1, meta_member_idx))

        candidate = None
        if init_strategy == "meta_copy":
            candidate = getattr(self, "_meta_copy_candidate", None)
            if not (
                isinstance(candidate, list)
                and getattr(self, "_meta_copy_candidate_meta_idx", None) == meta_member_idx
                and len(candidate) == (len(self.hidden_layers) + 1)
            ):
                candidate = None

        if candidate is None:
            for layer in self.hidden_layers:
                _init_layer(layer[0], meta_member_idx)
            _init_layer(self.mean_and_logvar, meta_member_idx)
        else:
            for i, layer in enumerate(self.hidden_layers):
                _init_layer(layer[0], meta_member_idx, candidate_params=candidate[i])
            _init_layer(
                self.mean_and_logvar, meta_member_idx, candidate_params=candidate[-1]
            )
