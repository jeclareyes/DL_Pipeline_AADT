# src/contracts/runtime_contracts.py

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

#%%%

class ContractError(RuntimeError):
    """Base exception for runtime contract violations."""


class ConfigurationContractError(ContractError):
    """Raised when runtime configuration violates strict policy."""


class ModelInputContractError(ContractError):
    """Raised when model input tensors do not match expected contract."""


class ModelOutputContractError(ContractError):
    """Raised when model outputs do not match expected contract."""


class ArtifactSchemaError(ContractError):
    """Raised when model artifacts payload is malformed."""


class EvalBundleContractError(ContractError):
    """Raised when eval bundle structure is invalid."""


class TaskDispatchContractError(ContractError):
    """Raised when evaluation dispatch configuration is invalid."""


def _ensure_mapping(value: Any, *, context: str, exc_type: type[ContractError]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        if hasattr(value, "items"):
            try:
                return dict(value.items())
            except Exception as exc:
                raise exc_type(f"{context} must be a mapping/dict, got {type(value)}") from exc
        raise exc_type(f"{context} must be a mapping/dict, got {type(value)}")
    return value


def require_keys(
    container: Any,
    required_keys: Sequence[str],
    *,
    context: str,
    exc_type: type[ContractError],
) -> Mapping[str, Any]:
    mapping = _ensure_mapping(container, context=context, exc_type=exc_type)
    missing = [k for k in required_keys if k not in mapping]
    if missing:
        raise exc_type(f"{context} missing required keys: {missing}")
    return mapping


def validate_model_input_contract(train_tensors: Any, val_tensors: Any) -> None:
    require_keys(
        train_tensors,
        ["flows", "mask", "od", "od_mask"],
        context="train_tensors",
        exc_type=ModelInputContractError,
    )
    require_keys(
        val_tensors,
        ["mask"],
        context="val_tensors",
        exc_type=ModelInputContractError,
    )


def validate_model_output_contract(outputs: Any) -> Mapping[str, Any]:
    """
    Validate the model output contract for training.
    """

    outputs_map = require_keys(
        outputs,
        ["reconstructed_flows", "estimated_demand", "loss"],
        context="model outputs",
        exc_type=ModelOutputContractError,
    )
    loss_map = require_keys(
        outputs_map["loss"],
        ["total_loss"],
        context="model outputs.loss",
        exc_type=ModelOutputContractError,
    )

    total_loss = loss_map["total_loss"]
    if not torch.is_tensor(total_loss):
        raise ModelOutputContractError(
            f"model outputs.loss.total_loss must be a torch.Tensor, got {type(total_loss)}"
        )
    return outputs_map


def validate_model_inference_output_contract(outputs: Any) -> Mapping[str, Any]:
    """
    Validate the minimal output contract required for model inference (testing).

    This contract is intentionally lighter than the training output contract.
    Evaluation utilities only need reconstructed link flows, and should not
    require a loss dictionary because loss computation depends on supervision
    targets that may not be available during hold-out or reconstruction
    evaluation.

    Required keys
    -------------
    - reconstructed_flows

    Parameters
    ----------
    outputs : Any
        Model output object returned by model.forward().

    Returns
    -------
    Mapping[str, Any]
        Validated model output mapping.

    Raises
    ------
    ModelOutputContractError
        If outputs is not mapping-like, if required keys are missing, or if
        reconstructed_flows is not a torch.Tensor.
    """

    outputs_map = require_keys(
        outputs,
        ["reconstructed_flows"],
        context="model inference outputs",
        exc_type=ModelOutputContractError,
    )

    reconstructed_flows = outputs_map["reconstructed_flows"]

    if not torch.is_tensor(reconstructed_flows):
        raise ModelOutputContractError(
            "model inference outputs.reconstructed_flows must be a "
            f"torch.Tensor, got {type(reconstructed_flows)}"
        )

    return outputs_map


def validate_eval_bundle_contract(bundle: Any) -> Mapping[str, Any]:
    bundle_map = require_keys(
        bundle,
        ["schema_version", "static_data", "epochs_history"],
        context="evaluation bundle",
        exc_type=EvalBundleContractError,
    )

    schema_version = bundle_map["schema_version"]
    if not isinstance(schema_version, int):
        raise EvalBundleContractError(
            f"evaluation bundle.schema_version must be int, got {type(schema_version)}"
        )
    if schema_version < 1:
        raise EvalBundleContractError("evaluation bundle.schema_version must be >= 1")

    _ensure_mapping(
        bundle_map["static_data"],
        context="evaluation bundle.static_data",
        exc_type=EvalBundleContractError,
    )

    epochs_history = _ensure_mapping(
        bundle_map["epochs_history"],
        context="evaluation bundle.epochs_history",
        exc_type=EvalBundleContractError,
    )
    if len(epochs_history) == 0:
        raise EvalBundleContractError("evaluation bundle.epochs_history is empty")

    return bundle_map


def validate_testing_dispatch_contract(testing_cfg: Any) -> list[str]:
    plan = resolve_testing_dispatch_plan(testing_cfg)
    return list(plan["tasks_callable"])


def resolve_testing_dispatch_plan(
    testing_cfg: Any,
    *,
    available_task_names: Sequence[str] | None = None,
) -> Mapping[str, Any]:
    """Resolve capability-based dispatch and return a preflight plan.

    Returns a mapping with keys:
    - model_key: selected model in testing config.
    - capabilities_selected: capabilities requested by that model.
    - capabilities_unused: capabilities defined in capability_dispatch but not used by model.
    - tasks_declared: tasks resolved from capabilities before task-name filtering.
    - tasks_callable: resolved tasks that exist in available_task_names (or tasks_declared if not provided).
    - tasks_unknown: resolved tasks not present in available_task_names.
    """
    if not hasattr(testing_cfg, "model_to_test"):
        raise TaskDispatchContractError("testing config missing 'model_to_test'")
    if not hasattr(testing_cfg, "capabilities"):
        raise TaskDispatchContractError("testing config missing 'capabilities'")
    if not hasattr(testing_cfg, "capability_dispatch"):
        raise TaskDispatchContractError("testing config missing 'capability_dispatch'")

    model_key = str(testing_cfg.model_to_test)
    capabilities_map = _ensure_mapping(
        testing_cfg.capabilities,
        context="testing.capabilities",
        exc_type=TaskDispatchContractError,
    )
    dispatch_map = _ensure_mapping(
        testing_cfg.capability_dispatch,
        context="testing.capability_dispatch",
        exc_type=TaskDispatchContractError,
    )

    if model_key not in capabilities_map:
        available = sorted(str(k) for k in capabilities_map.keys())
        raise TaskDispatchContractError(
            f"testing.model_to_test='{model_key}' has no entry in capabilities. Available: {available}"
        )

    capability_names = capabilities_map[model_key]
    if isinstance(capability_names, str):
        capability_names = [capability_names]
    elif isinstance(capability_names, Mapping):
        # Support boolean flags to turn capabilities on/off
        capability_names = [k for k, v in capability_names.items() if v is True]
    elif not isinstance(capability_names, Sequence):
        if hasattr(capability_names, "__iter__"):
            capability_names = list(capability_names)
        else:
            raise TaskDispatchContractError(
                f"capabilities['{model_key}'] must be a list of capability names or a boolean mapping"
            )

    capability_list = [str(c).strip() for c in capability_names if str(c).strip()]
    if len(capability_list) == 0:
        raise TaskDispatchContractError(
            f"capabilities['{model_key}'] has no valid capability names"
        )

    tasks_declared: list[str] = []
    for capability in capability_list:
        if capability not in dispatch_map:
            available = sorted(str(k) for k in dispatch_map.keys())
            raise TaskDispatchContractError(
                f"capability '{capability}' is not defined in capability_dispatch. Available: {available}"
            )

        mapped_tasks = dispatch_map[capability]
        if isinstance(mapped_tasks, str):
            mapped_tasks = [mapped_tasks]
        elif not isinstance(mapped_tasks, Sequence):
            if hasattr(mapped_tasks, "__iter__"):
                mapped_tasks = list(mapped_tasks)
            else:
                raise TaskDispatchContractError(
                    f"capability_dispatch['{capability}'] must be a task or list of task names"
                )

        for task_name in mapped_tasks:
            task_name_str = str(task_name).strip()
            if task_name_str:
                tasks_declared.append(task_name_str)

    tasks_declared = list(dict.fromkeys(tasks_declared))
    if len(tasks_declared) == 0:
        raise TaskDispatchContractError(
            f"Resolved task list for model '{model_key}' is empty after capability dispatch"
        )

    if available_task_names is None:
        tasks_callable = list(tasks_declared)
        tasks_unknown: list[str] = []
    else:
        available_set = {str(t).strip() for t in available_task_names if str(t).strip()}
        tasks_callable = [task for task in tasks_declared if task in available_set]
        tasks_unknown = [task for task in tasks_declared if task not in available_set]

    if len(tasks_callable) == 0:
        raise TaskDispatchContractError(
            f"No callable evaluation tasks resolved for model '{model_key}'"
        )

    capabilities_unused = [
        str(cap)
        for cap in dispatch_map.keys()
        if str(cap) not in set(capability_list)
    ]

    return {
        "model_key": model_key,
        "capabilities_selected": capability_list,
        "capabilities_unused": capabilities_unused,
        "tasks_declared": tasks_declared,
        "tasks_callable": tasks_callable,
        "tasks_unknown": tasks_unknown,
    }


def validate_artifacts_contract(artifacts: Any) -> Mapping[str, Any]:
    return require_keys(
        artifacts,
        ["pred_flows", "pred_od"],
        context="evaluation artifacts",
        exc_type=ArtifactSchemaError,
    )
