# src/train/oracle/oracle_adapter.py

"""
Oracle Adapter
==============

This module contains model pre-instantiation hooks used for oracle-based
training or diagnostic runs.

Project context
---------------
The training pipeline prepares model_params from the unified training artifact.
Before Hydra instantiates the model, optional hooks may inject additional
parameters into model_params.

This module is responsible only for oracle-specific hook logic.

Responsibilities
----------------
- Detect whether the model is configured in oracle supply mode.
- Extract true/oracle BPR parameters from model-ready network_params.
- Validate that oracle parameters are finite and aligned with the model link
  dimension.
- Return updates to be merged into model_params before model instantiation.
- Optionally combine OD initialization with oracle supply injection.

This module does not:
- train models;
- run evaluation;
- run diagnostics;
- save files;
- access run_context;
- access task-level evaluation outputs;
- execute code at import time.

Design principles
-----------------
- Keep oracle hooks side-effect free.
- Fail early when oracle parameters are missing or malformed.
- Keep all returned values explicit.
- Do not mutate global state.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from src.contracts.runtime_contracts import ConfigurationContractError
from src.train._pipeline_utils import vi_od_initialization_hook


# =============================================================================
# Public hooks
# =============================================================================


def vi_oracle_supply_hook(
    cfg: DictConfig,
    model_params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Inject oracle BPR parameters into VI model parameters.

    This hook is intended for diagnostic or oracle runs where the supply-side
    BPR parameters are known and should not be learned.

    The hook reads the true parameters from the model-ready network_params
    produced by the data-processing artifact. It does not build parameters from
    raw files.

    Expected source keys
    --------------------
    The network_params dictionary must provide one of these alpha alternatives:

    - oracle_alpha
    - b
    - alpha
    - bpr_alpha

    and one of these beta alternatives:

    - oracle_beta
    - power
    - beta
    - bpr_beta

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    model_params : Dict[str, Any]
        Model parameter dictionary prepared before Hydra instantiation.

    context : Optional[Dict[str, Any]], default=None
        Optional context passed by TrainingInputPreparer. Expected keys include:

        - network_params

    Returns
    -------
    Dict[str, Any]
        Updates to merge into model_params before model instantiation.

    Raises
    ------
    ConfigurationContractError
        If oracle mode is enabled but required BPR parameters are missing,
        non-finite or dimensionally incompatible.
    """

    context = context or {}

    supply_mode = str(
        _cfg_get(
            cfg,
            "model.supply_mode",
            default="learned",
        )
    ).lower()

    if supply_mode != "oracle":
        return {}

    network_params = context.get(
        "network_params",
        model_params,
    )

    if not isinstance(network_params, dict):
        raise ConfigurationContractError(
            "vi_oracle_supply_hook expected network_params to be a dictionary. "
            f"Got {type(network_params)}."
        )

    num_links = _read_num_links(model_params)

    alpha_source = _first_existing_key(
        network_params,
        candidate_keys=[
            "oracle_alpha",
            "b",
            "alpha",
            "bpr_alpha",
        ],
    )

    beta_source = _first_existing_key(
        network_params,
        candidate_keys=[
            "oracle_beta",
            "power",
            "beta",
            "bpr_beta",
        ],
    )

    if alpha_source is None or beta_source is None:
        available_keys = sorted(str(key) for key in network_params.keys())

        raise ConfigurationContractError(
            "model.supply_mode='oracle' requires oracle BPR parameters in "
            "artifact['model_ready']['network_params']. Expected one alpha key "
            "from ['oracle_alpha', 'b', 'alpha', 'bpr_alpha'] and one beta key "
            "from ['oracle_beta', 'power', 'beta', 'bpr_beta']. "
            f"Available network_params keys: {available_keys}"
        )

    oracle_alpha = _to_1d_float_tensor(
        alpha_source,
        name="oracle_alpha",
    )

    oracle_beta = _to_1d_float_tensor(
        beta_source,
        name="oracle_beta",
    )

    _validate_link_parameter_length(
        tensor=oracle_alpha,
        expected_num_links=num_links,
        name="oracle_alpha",
    )

    _validate_link_parameter_length(
        tensor=oracle_beta,
        expected_num_links=num_links,
        name="oracle_beta",
    )

    updates: Dict[str, Any] = {
        "oracle_alpha": oracle_alpha,
        "oracle_beta": oracle_beta,
    }

    # These semantic aliases are useful when the model, diagnostics or logging
    # code expects physical BPR names instead of oracle-specific names.
    updates["b"] = oracle_alpha
    updates["power"] = oracle_beta

    return updates


def vi_oracle_supply_and_od_initialization_hook(
    cfg: DictConfig,
    model_params: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Composite hook for OD initialization and oracle supply injection.

    This hook first applies the VI OD initialization hook and then injects oracle
    supply parameters. The order matters because oracle supply injection may use
    dimensions already defined in model_params, while OD initialization may add
    OD-related priors or initialization tensors.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    model_params : Dict[str, Any]
        Model parameters before Hydra instantiation.

    context : Optional[Dict[str, Any]], default=None
        Optional context passed by TrainingInputPreparer.

    Returns
    -------
    Dict[str, Any]
        Combined model parameter updates.
    """

    context = context or {}

    updates: Dict[str, Any] = {}

    od_updates = vi_od_initialization_hook(
        cfg=cfg,
        model_params=model_params,
        context=context,
    )

    if od_updates:
        updates.update(od_updates)

    oracle_updates = vi_oracle_supply_hook(
        cfg=cfg,
        model_params={
            **model_params,
            **updates,
        },
        context=context,
    )

    if oracle_updates:
        updates.update(oracle_updates)

    return updates


# =============================================================================
# Internal helpers
# =============================================================================


def _read_num_links(
    model_params: Dict[str, Any],
) -> int:
    """
    Read and validate the number of model links from model_params.

    Parameters
    ----------
    model_params : Dict[str, Any]
        Model parameter dictionary.

    Returns
    -------
    int
        Number of links.

    Raises
    ------
    ConfigurationContractError
        If num_links is missing or invalid.
    """

    if "num_links" not in model_params:
        raise ConfigurationContractError(
            "model_params must contain 'num_links' before oracle supply "
            "parameters can be injected."
        )

    num_links = int(model_params["num_links"])

    if num_links <= 0:
        raise ConfigurationContractError(
            f"model_params['num_links'] must be positive. Got {num_links}."
        )

    return num_links


def _first_existing_key(
    mapping: Dict[str, Any],
    candidate_keys: list[str],
) -> Any:
    """
    Return the first available value from a dictionary.

    Parameters
    ----------
    mapping : Dict[str, Any]
        Source dictionary.

    candidate_keys : list[str]
        Candidate keys in priority order.

    Returns
    -------
    Any
        First found value or None.
    """

    for key in candidate_keys:
        if key in mapping:
            return mapping[key]

    return None


def _to_1d_float_tensor(
    value: Any,
    name: str,
) -> torch.Tensor:
    """
    Convert an array-like value to a one-dimensional float tensor.

    The tensor is kept device-agnostic here. TrainingInputPreparer or the model
    will move it to the configured device later.

    Parameters
    ----------
    value : Any
        Tensor, NumPy array, list or scalar-like object.

    name : str
        Human-readable name used in error messages.

    Returns
    -------
    torch.Tensor
        One-dimensional float32 tensor.

    Raises
    ------
    ConfigurationContractError
        If the tensor contains NaN or infinite values.
    """

    if torch.is_tensor(value):
        tensor = value.detach().float().reshape(-1)
    else:
        tensor = torch.as_tensor(
            np.asarray(
                value,
                dtype=np.float32,
            ),
            dtype=torch.float32,
        ).reshape(-1)

    if tensor.numel() == 0:
        raise ConfigurationContractError(
            f"{name} cannot be empty."
        )

    if not torch.isfinite(tensor).all():
        raise ConfigurationContractError(
            f"{name} contains NaN or infinite values."
        )

    return tensor


def _validate_link_parameter_length(
    tensor: torch.Tensor,
    expected_num_links: int,
    name: str,
) -> None:
    """
    Validate that a link-level parameter contains exactly one value per link.

    Parameters
    ----------
    tensor : torch.Tensor
        Link-level parameter tensor.

    expected_num_links : int
        Expected number of links.

    name : str
        Parameter name used in error messages.
    """

    actual_num_links = int(tensor.numel())

    if actual_num_links != int(expected_num_links):
        raise ConfigurationContractError(
            f"{name} must contain one value per model link. "
            f"Expected {expected_num_links}, got {actual_num_links}."
        )


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
        Dotted key path, for example 'model.supply_mode'.

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