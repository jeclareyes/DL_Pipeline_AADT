# src/train/diagnostics/training_audits.py

"""
Training Audits
===============

This module contains optional model-agnostic training diagnostics.

Project context
---------------
The GeneralTrainer is responsible for executing the training loop. However,
some diagnostic information is useful during debugging:

- gradient health by top-level module;
- gradient contribution by loss component;
- physics-related values returned by the model forward pass;
- solver convergence metadata returned by the model.

These diagnostics are optional and configurable. They should not define the core
training behavior.

Responsibilities
----------------
- Read diagnostic configuration defensively.
- Decide whether an audit should be collected at a given epoch.
- Collect forward-output diagnostics.
- Collect gradient diagnostics after backward().
- Collect loss-component gradient diagnostics before backward().

This module does not:
- train the model;
- step the optimizer;
- clip gradients;
- decide whether optimizer updates should be skipped;
- save checkpoints;
- evaluate validation or hold-out performance.

Design principles
-----------------
- Keep diagnostics model-agnostic.
- Never mutate model parameters.
- Fail softly whenever possible.
- Keep expensive audits limited to early epochs or configured intervals.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from omegaconf import DictConfig


# =============================================================================
# Public decision helpers
# =============================================================================


def should_collect_forward_physics_audit(
    cfg: Any,
    epoch_idx: int,
) -> bool:
    """
    Decide whether forward physics diagnostics should be collected.

    The configuration is intentionally read from training.gradient_audit to
    preserve the existing YAML structure.

    Expected YAML
    -------------
    training:
      gradient_audit:
        enabled: true
        first_n_epochs: 10
        every_n_epochs: 0
        epochs: []
        include_forward_physics: true
    """

    audit_cfg = _cfg_get(
        cfg,
        "training.gradient_audit",
        default={},
    )

    if not bool(_dict_get(audit_cfg, "enabled", default=False)):
        return False

    if not bool(_dict_get(audit_cfg, "include_forward_physics", default=True)):
        return False

    return _epoch_matches_audit_schedule(
        cfg=audit_cfg,
        epoch_idx=epoch_idx,
    )


def should_collect_gradient_audit(
    cfg: Any,
    epoch_idx: int,
) -> bool:
    """
    Decide whether grouped gradient diagnostics should be collected.

    The configuration is read from training.gradient_audit to keep all optional
    training audits under one YAML block.
    """

    audit_cfg = _cfg_get(
        cfg,
        "training.gradient_audit",
        default={},
    )

    if not bool(_dict_get(audit_cfg, "enabled", default=False)):
        return False

    if not bool(_dict_get(audit_cfg, "include_gradient_groups", default=True)):
        return False

    return _epoch_matches_audit_schedule(
        cfg=audit_cfg,
        epoch_idx=epoch_idx,
    )


def should_collect_loss_gradient_audit(
    cfg: Any,
    epoch_idx: int,
) -> bool:
    """
    Decide whether per-loss-component gradient diagnostics should be collected.

    This audit is more expensive than grouped gradient diagnostics because it
    calls torch.autograd.grad for each differentiable scalar loss component.

    For that reason, the recommended default is:

        include_loss_components: false
    """

    audit_cfg = _cfg_get(
        cfg,
        "training.gradient_audit",
        default={},
    )

    if not bool(_dict_get(audit_cfg, "enabled", default=False)):
        return False

    if not bool(_dict_get(audit_cfg, "include_loss_components", default=False)):
        return False

    return _epoch_matches_audit_schedule(
        cfg=audit_cfg,
        epoch_idx=epoch_idx,
    )


# =============================================================================
# Public collection helpers
# =============================================================================


def collect_tensor_stats(
    value: Any,
    prefix: str,
) -> Dict[str, float]:
    """
    Collect scalar statistics from a tensor-like value.

    Parameters
    ----------
    value : Any
        Candidate tensor.

    prefix : str
        Metric name prefix.

    Returns
    -------
    Dict[str, float]
        Tensor statistics. Empty if value is not a usable tensor.
    """

    if value is None or not torch.is_tensor(value):
        return {}

    tensor = value.detach().float().reshape(-1)

    if tensor.numel() == 0:
        return {}

    finite_mask = torch.isfinite(tensor)

    if not bool(finite_mask.any().item()):
        return {
            f"{prefix}_numel": float(tensor.numel()),
            f"{prefix}_finite_count": 0.0,
            f"{prefix}_non_finite_count": float(tensor.numel()),
        }

    safe_tensor = tensor[finite_mask]

    return {
        f"{prefix}_numel": float(tensor.numel()),
        f"{prefix}_finite_count": float(finite_mask.sum().item()),
        f"{prefix}_non_finite_count": float((~finite_mask).sum().item()),
        f"{prefix}_mean": float(safe_tensor.mean().cpu().item()),
        f"{prefix}_min": float(safe_tensor.min().cpu().item()),
        f"{prefix}_max": float(safe_tensor.max().cpu().item()),
        f"{prefix}_std": float(safe_tensor.std(unbiased=False).cpu().item()),
    }


def collect_forward_physics_audit(
    outputs: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Collect physics and solver diagnostics from model outputs.

    This function is intentionally defensive. It only records values that are
    present in outputs and safely convertible to floats.

    Parameters
    ----------
    outputs : Dict[str, Any]
        Model forward output dictionary.

    Returns
    -------
    Dict[str, Any]
        Forward-pass diagnostic payload.
    """

    if not isinstance(outputs, dict):
        return {}

    audit: Dict[str, Any] = {}

    audit.update(
        collect_tensor_stats(
            outputs.get("learned_alpha"),
            "alpha",
        )
    )

    audit.update(
        collect_tensor_stats(
            outputs.get("learned_beta"),
            "beta",
        )
    )

    audit.update(
        collect_tensor_stats(
            outputs.get("learned_capacity_multiplier"),
            "capacity_multiplier",
        )
    )

    audit.update(
        collect_tensor_stats(
            outputs.get("learned_theta"),
            "theta",
        )
    )

    convergence_info = outputs.get("convergence_info", {})

    if isinstance(convergence_info, dict):
        audit.update(
            _collect_solver_convergence_info(
                convergence_info=convergence_info,
            )
        )

    return audit


def collect_loss_component_gradient_audit(
    model: torch.nn.Module,
    loss_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Estimate gradient contribution of each differentiable scalar loss component.

    Important
    ---------
    This function should be called before total_loss.backward().

    It uses torch.autograd.grad(..., retain_graph=True), which makes it more
    expensive than the standard gradient audit. For this reason, it should only
    run in early epochs or at configured intervals.

    Parameters
    ----------
    model : torch.nn.Module
        Model being trained.

    loss_dict : Dict[str, Any]
        Loss dictionary returned by the model.

    Returns
    -------
    Dict[str, Any]
        Per-loss-component gradient diagnostics.
    """

    parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    audit: Dict[str, Any] = {}

    for loss_name, loss_value in loss_dict.items():
        audit[loss_name] = _audit_single_loss_component(
            loss_name=loss_name,
            loss_value=loss_value,
            parameters=parameters,
        )

    return audit


def collect_gradient_audit(
    model: torch.nn.Module,
) -> Dict[str, Any]:
    """
    Collect gradient diagnostics grouped by top-level module name.

    This function should be called after total_loss.backward() and before
    optimizer.step().

    Parameters
    ----------
    model : torch.nn.Module
        Model being trained.

    Returns
    -------
    Dict[str, Any]
        Gradient diagnostics grouped by top-level parameter prefix.
    """

    audit: Dict[str, Dict[str, float]] = {}

    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue

        group_name = name.split(".")[0]
        grad = parameter.grad.detach()

        finite_mask = torch.isfinite(grad)
        non_finite_entries = int((~finite_mask).sum().item())

        safe_grad = torch.nan_to_num(
            grad,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        grad_norm = float(torch.norm(safe_grad, p=2).detach().cpu().item())

        if safe_grad.numel() > 0:
            grad_abs_max = float(
                safe_grad.abs().max().detach().cpu().item()
            )
        else:
            grad_abs_max = 0.0

        if group_name not in audit:
            audit[group_name] = {
                "num_tensors": 0.0,
                "total_norm_sq": 0.0,
                "max_abs_grad": 0.0,
                "non_finite_entries": 0.0,
            }

        audit[group_name]["num_tensors"] += 1.0
        audit[group_name]["total_norm_sq"] += grad_norm**2
        audit[group_name]["max_abs_grad"] = max(
            audit[group_name]["max_abs_grad"],
            grad_abs_max,
        )
        audit[group_name]["non_finite_entries"] += float(non_finite_entries)

    for group_stats in audit.values():
        group_stats["total_grad_norm"] = float(
            group_stats["total_norm_sq"] ** 0.5
        )
        del group_stats["total_norm_sq"]

    return audit


# =============================================================================
# Internal helpers
# =============================================================================


def _audit_single_loss_component(
    loss_name: str,
    loss_value: Any,
    parameters: list[torch.nn.Parameter],
) -> Dict[str, Any]:
    """
    Audit one scalar loss component.

    Parameters
    ----------
    loss_name : str
        Loss component name.

    loss_value : Any
        Loss value.

    parameters : list[torch.nn.Parameter]
        Trainable model parameters.

    Returns
    -------
    Dict[str, Any]
        Diagnostic entry for one loss component.
    """

    if not torch.is_tensor(loss_value):
        return {
            "status": "skipped_non_tensor",
            "value": float(loss_value) if isinstance(loss_value, (int, float)) else None,
        }

    if loss_value.numel() != 1:
        return {
            "status": "skipped_non_scalar_tensor",
            "shape": tuple(int(dim) for dim in loss_value.shape),
        }

    loss_scalar = loss_value.reshape(())

    audit_entry: Dict[str, Any] = {
        "value": float(loss_scalar.detach().cpu().item()),
        "requires_grad": bool(loss_scalar.requires_grad),
        "has_grad_fn": loss_scalar.grad_fn is not None,
    }

    if not loss_scalar.requires_grad or loss_scalar.grad_fn is None:
        audit_entry.update(
            {
                "status": "skipped_detached_loss_component",
                "num_tensors": 0,
                "total_grad_norm": 0.0,
                "max_abs_grad": 0.0,
                "non_finite_entries": 0,
            }
        )
        return audit_entry

    try:
        grads = torch.autograd.grad(
            outputs=loss_scalar,
            inputs=parameters,
            retain_graph=True,
            allow_unused=True,
        )

    except RuntimeError as exc:
        audit_entry.update(
            {
                "status": "autograd_failed",
                "error": str(exc),
                "num_tensors": 0,
                "total_grad_norm": 0.0,
                "max_abs_grad": 0.0,
                "non_finite_entries": 0,
            }
        )
        return audit_entry

    total_norm_sq = 0.0
    max_abs_grad = 0.0
    num_tensors = 0
    non_finite_entries = 0

    for grad in grads:
        if grad is None:
            continue

        finite_mask = torch.isfinite(grad)
        non_finite_entries += int((~finite_mask).sum().item())

        safe_grad = torch.nan_to_num(
            grad.detach(),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        grad_norm = float(torch.norm(safe_grad, p=2).detach().cpu().item())
        total_norm_sq += grad_norm**2

        if safe_grad.numel() > 0:
            max_abs_grad = max(
                max_abs_grad,
                float(safe_grad.abs().max().detach().cpu().item()),
            )

        num_tensors += 1

    audit_entry.update(
        {
            "status": "ok",
            "num_tensors": int(num_tensors),
            "total_grad_norm": float(total_norm_sq**0.5),
            "max_abs_grad": float(max_abs_grad),
            "non_finite_entries": int(non_finite_entries),
        }
    )

    return audit_entry


def _collect_solver_convergence_info(
    convergence_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Extract safe scalar values from solver convergence metadata.

    Parameters
    ----------
    convergence_info : Dict[str, Any]
        Solver convergence dictionary returned by the model.

    Returns
    -------
    Dict[str, Any]
        Safe solver diagnostics.
    """

    audit: Dict[str, Any] = {}

    candidate_keys = [
        "iterations",
        "converged",
        "wardrop_gap",
        "relative_flow_change",
        "feasibility_abs_error",
        "feasibility_rel_error",
        "implicit_grad",
    ]

    for key in candidate_keys:
        if key not in convergence_info:
            continue

        value = convergence_info[key]

        try:
            if torch.is_tensor(value):
                if value.numel() == 1:
                    audit[f"solver_{key}"] = float(value.detach().cpu().item())
                else:
                    audit[f"solver_{key}_mean"] = float(
                        value.detach().float().mean().cpu().item()
                    )
            elif isinstance(value, bool):
                audit[f"solver_{key}"] = bool(value)
            elif isinstance(value, (int, float)):
                audit[f"solver_{key}"] = float(value)
            else:
                audit[f"solver_{key}"] = str(value)

        except Exception:
            audit[f"solver_{key}"] = "unavailable"

    return audit


def _epoch_matches_audit_schedule(
    cfg: Any,
    epoch_idx: int,
) -> bool:
    """
    Check whether an epoch matches a diagnostic schedule.

    Supported config keys
    ---------------------
    first_n_epochs : int
        Run for the first N epochs.

    every_n_epochs : int
        Run every N epochs.

    epochs : list[int]
        Explicit list of epochs.

    Parameters
    ----------
    cfg : Any
        Audit config.

    epoch_idx : int
        One-based epoch index.

    Returns
    -------
    bool
        True if the epoch should be audited.
    """

    epoch_idx = int(epoch_idx)

    first_n_epochs = int(
        _dict_get(
            cfg,
            "first_n_epochs",
            default=0,
        )
        or 0
    )

    if first_n_epochs > 0 and epoch_idx <= first_n_epochs:
        return True

    every_n_epochs = int(
        _dict_get(
            cfg,
            "every_n_epochs",
            default=0,
        )
        or 0
    )

    if every_n_epochs > 0 and epoch_idx % every_n_epochs == 0:
        return True

    explicit_epochs = _dict_get(
        cfg,
        "epochs",
        default=[],
    )

    try:
        explicit_epochs = {
            int(epoch)
            for epoch in explicit_epochs
        }
    except Exception:
        explicit_epochs = set()

    return epoch_idx in explicit_epochs


def _cfg_get(
    cfg: Any,
    dotted_key: str,
    default: Any = None,
) -> Any:
    """
    Safely read a nested value from DictConfig, dict or object attributes.

    Parameters
    ----------
    cfg : Any
        Configuration object.

    dotted_key : str
        Dotted key path.

    default : Any, default=None
        Fallback value.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    current = cfg

    for part in dotted_key.split("."):
        if current is None:
            return default

        if isinstance(current, DictConfig):
            if part not in current:
                return default
            current = current[part]
            continue

        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        return default

    return current


def _dict_get(
    obj: Any,
    key: str,
    default: Any = None,
) -> Any:
    """
    Safely read one key from DictConfig, dict or object attributes.

    Parameters
    ----------
    obj : Any
        Source object.

    key : str
        Key or attribute name.

    default : Any, default=None
        Fallback value.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    if obj is None:
        return default

    if isinstance(obj, DictConfig):
        if key not in obj:
            return default
        return obj[key]

    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key)

    return default