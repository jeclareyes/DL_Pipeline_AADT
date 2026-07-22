# src/train/simple_evaluation/evaluator.py

"""
Simple Training Evaluator
=========================

This module contains the standard post-training evaluation logic used by the
training pipeline.

Project context
---------------
The training pipeline trains one or more tasks. After each task finishes, this
module evaluates the trained model on explicit flow masks:

- validation links;
- global hold-out links;
- all observed links.

This evaluator is intentionally simple and model-agnostic. It only assumes that
the trained model can run inference with the same public forward contract used
during training and that it returns a dictionary containing:

    reconstructed_flows

This module does not:

- train models;
- instantiate models;
- load training artifacts;
- build network tensors;
- run oracle diagnostics;
- run VI-specific debugging;
- inspect gradients.

Those responsibilities belong to:

- GeneralTrainer;
- training_pipeline.py;
- TrainingInputPreparer;
- diagnostics modules;
- oracle modules.

Design principles
-----------------
- Evaluate only explicit masks.
- Keep all tensor and array alignment checks local and defensive.
- Return structured EvaluationResult objects.
- Save metrics and prediction tables in a reproducible format.
- Avoid model-specific branching as much as possible.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch

from src.contracts.runtime_contracts import (
    ModelInputContractError,
    validate_model_inference_output_contract,
)
from src.train._pipeline_utils import to_json_serializable


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvaluationResult:
    """
    Structured result returned by simple evaluation functions.

    Attributes
    ----------
    metrics : Dict[str, Any]
        Regression metrics and aggregate diagnostic values.

    predictions_df : Optional[pd.DataFrame]
        Optional row-level prediction table. This is useful for downstream
        inspection, maps, plots or manual debugging.

    metadata : Dict[str, Any]
        Evaluation metadata, such as the evaluated split, device and mask sizes.
    """

    metrics: Dict[str, Any]
    predictions_df: Optional[pd.DataFrame]
    metadata: Dict[str, Any]


# =============================================================================
# Public evaluation API
# =============================================================================


def evaluate_validation_metrics(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    train_mask_np: np.ndarray,
    val_mask_np: np.ndarray,
    device: str,
    return_predictions: bool = False,
    link_metadata: Optional[pd.DataFrame] = None,
    true_od_t: Optional[torch.Tensor] = None,
    od_mask_t: Optional[torch.Tensor] = None,
) -> EvaluationResult:
    """
    Evaluate model reconstruction on validation flow links.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    true_flows_t : torch.Tensor
        Complete link-flow target tensor.

    train_mask_np : np.ndarray
        Binary mask used as model input during reconstruction.

    val_mask_np : np.ndarray
        Binary validation mask selecting the evaluated links.

    device : str
        Torch device used for inference.

    return_predictions : bool, default=False
        If True, return a prediction DataFrame.

    link_metadata : Optional[pd.DataFrame], default=None
        Optional link metadata aligned with the flow vector.

    true_od_t : Optional[torch.Tensor], default=None
        Optional OD target tensor. Some models need OD information during
        evaluation to reproduce the same assignment contract used in training.

    od_mask_t : Optional[torch.Tensor], default=None
        Optional OD supervision mask.

    Returns
    -------
    EvaluationResult
        Validation metrics, optional prediction table and metadata.
    """

    return evaluate_masked_reconstruction(
        model=model,
        true_flows_t=true_flows_t,
        input_mask_np=train_mask_np,
        evaluation_mask_np=val_mask_np,
        device=device,
        split_name="validation",
        metric_prefix="validation",
        return_predictions=return_predictions,
        link_metadata=link_metadata,
        true_od_t=true_od_t,
        od_mask_t=od_mask_t,
        extra_metadata={
            "num_train_input_links": int(np.asarray(train_mask_np).sum()),
            "num_validation_links": int(np.asarray(val_mask_np).sum()),
            "has_validation": bool(np.asarray(val_mask_np).sum() > 0),
        },
    )


def evaluate_holdout_metrics(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    train_mask_np: np.ndarray,
    holdout_mask_np: np.ndarray,
    device: str,
    return_predictions: bool = False,
    link_metadata: Optional[pd.DataFrame] = None,
    true_od_t: Optional[torch.Tensor] = None,
    od_mask_t: Optional[torch.Tensor] = None,
) -> EvaluationResult:
    """
    Evaluate model reconstruction on global hold-out flow links.

    Hold-out links are observed links that were not used as training inputs.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    true_flows_t : torch.Tensor
        Complete link-flow target tensor.

    train_mask_np : np.ndarray
        Binary mask used as model input during reconstruction.

    holdout_mask_np : np.ndarray
        Binary hold-out mask selecting evaluated links.

    device : str
        Torch device used for inference.

    return_predictions : bool, default=False
        If True, return a prediction DataFrame.

    link_metadata : Optional[pd.DataFrame], default=None
        Optional link metadata aligned with the flow vector.

    true_od_t : Optional[torch.Tensor], default=None
        Optional OD target tensor.

    od_mask_t : Optional[torch.Tensor], default=None
        Optional OD supervision mask.

    Returns
    -------
    EvaluationResult
        Hold-out metrics, optional prediction table and metadata.
    """

    return evaluate_masked_reconstruction(
        model=model,
        true_flows_t=true_flows_t,
        input_mask_np=train_mask_np,
        evaluation_mask_np=holdout_mask_np,
        device=device,
        split_name="holdout",
        metric_prefix="holdout",
        return_predictions=return_predictions,
        link_metadata=link_metadata,
        true_od_t=true_od_t,
        od_mask_t=od_mask_t,
        extra_metadata={
            "num_train_input_links": int(np.asarray(train_mask_np).sum()),
            "num_holdout_links": int(np.asarray(holdout_mask_np).sum()),
            "has_holdout": bool(np.asarray(holdout_mask_np).sum() > 0),
        },
    )


def evaluate_full_reconstruction(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    input_mask_np: np.ndarray,
    observed_mask_np: np.ndarray,
    device: str,
    return_predictions: bool = True,
    link_metadata: Optional[pd.DataFrame] = None,
    true_od_t: Optional[torch.Tensor] = None,
    od_mask_t: Optional[torch.Tensor] = None,
) -> EvaluationResult:
    """
    Evaluate model reconstruction over all observed flow links.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    true_flows_t : torch.Tensor
        Complete link-flow target tensor.

    input_mask_np : np.ndarray
        Binary mask used as model input during reconstruction.

    observed_mask_np : np.ndarray
        Binary observed-flow mask selecting all observed links.

    device : str
        Torch device used for inference.

    return_predictions : bool, default=True
        If True, return a prediction DataFrame.

    link_metadata : Optional[pd.DataFrame], default=None
        Optional link metadata aligned with the flow vector.

    true_od_t : Optional[torch.Tensor], default=None
        Optional OD target tensor.

    od_mask_t : Optional[torch.Tensor], default=None
        Optional OD supervision mask.

    Returns
    -------
    EvaluationResult
        Full observed-link reconstruction metrics.
    """

    return evaluate_masked_reconstruction(
        model=model,
        true_flows_t=true_flows_t,
        input_mask_np=input_mask_np,
        evaluation_mask_np=observed_mask_np,
        device=device,
        split_name="observed",
        metric_prefix="observed",
        return_predictions=return_predictions,
        link_metadata=link_metadata,
        true_od_t=true_od_t,
        od_mask_t=od_mask_t,
        extra_metadata={
            "num_input_links": int(np.asarray(input_mask_np).sum()),
            "num_observed_links": int(np.asarray(observed_mask_np).sum()),
            "has_observed": bool(np.asarray(observed_mask_np).sum() > 0),
        },
    )


def evaluate_masked_reconstruction(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    input_mask_np: np.ndarray,
    evaluation_mask_np: np.ndarray,
    device: str,
    split_name: str,
    metric_prefix: str,
    return_predictions: bool = False,
    link_metadata: Optional[pd.DataFrame] = None,
    true_od_t: Optional[torch.Tensor] = None,
    od_mask_t: Optional[torch.Tensor] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> EvaluationResult:
    """
    Generic masked reconstruction evaluator.

    This function is the central implementation used by validation, hold-out and
    full observed-link evaluation. The specific public functions above only
    define the mask and naming convention.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    true_flows_t : torch.Tensor
        Complete link-flow target tensor.

    input_mask_np : np.ndarray
        Binary mask passed to the model as input.

    evaluation_mask_np : np.ndarray
        Binary mask selecting the entries used for metric computation.

    device : str
        Torch device used for inference.

    split_name : str
        Human-readable split name stored in the prediction table.

    metric_prefix : str
        Prefix added to metric names.

    return_predictions : bool, default=False
        If True, return a row-level prediction table.

    link_metadata : Optional[pd.DataFrame], default=None
        Optional link metadata aligned with the flow vector.

    true_od_t : Optional[torch.Tensor], default=None
        Optional OD target tensor.

    od_mask_t : Optional[torch.Tensor], default=None
        Optional OD mask tensor.

    extra_metadata : Optional[Dict[str, Any]], default=None
        Optional additional metadata merged into the output metadata.

    Returns
    -------
    EvaluationResult
        Metrics, optional predictions and metadata.
    """

    input_mask_t = _to_float_tensor(input_mask_np, device=device)

    evaluation_mask_np = _as_binary_1d_mask(
        evaluation_mask_np,
        name="evaluation_mask_np",
    )

    true_flows_t = _ensure_1d_or_batched_tensor(
        true_flows_t,
        device=device,
        name="true_flows_t",
    )

    prediction_t = run_flow_reconstruction(
        model=model,
        observed_flows=true_flows_t,
        flow_mask=input_mask_t,
        device=device,
        true_od_demand=true_od_t,
        od_mask=od_mask_t,
    )

    y_true_all = _tensor_to_1d_numpy(true_flows_t)
    y_pred_all = _tensor_to_1d_numpy(prediction_t)

    _validate_same_length(
        y_true_all,
        y_pred_all,
        "y_true_all",
        "y_pred_all",
    )

    _validate_same_length(
        y_true_all,
        evaluation_mask_np,
        "y_true_all",
        "evaluation_mask_np",
    )

    metrics = compute_masked_regression_metrics(
        y_true=y_true_all,
        y_pred=y_pred_all,
        mask=evaluation_mask_np,
        prefix=metric_prefix,
    )

    metrics[f"{metric_prefix}_flow_balance"] = compute_flow_conservation_summary(
        y_true=y_true_all,
        y_pred=y_pred_all,
        mask=evaluation_mask_np,
    )

    predictions_df = None

    if return_predictions:
        predictions_df = build_prediction_dataframe(
            y_true=y_true_all,
            y_pred=y_pred_all,
            mask=evaluation_mask_np,
            link_metadata=link_metadata,
            split_name=split_name,
        )

    metadata = {
        "evaluation_type": split_name,
        "device": str(device),
        "num_total_links": int(len(y_true_all)),
        "num_evaluated_links": int(evaluation_mask_np.sum()),
    }

    if extra_metadata:
        metadata.update(extra_metadata)

    return EvaluationResult(
        metrics=metrics,
        predictions_df=predictions_df,
        metadata=metadata,
    )


def build_task_evaluation_summary(
    task_name: str,
    fit_result: Any,
    holdout_result: Optional[EvaluationResult] = None,
    validation_result: Optional[EvaluationResult] = None,
    full_result: Optional[EvaluationResult] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a compact task-level summary combining training and evaluation outputs.

    Parameters
    ----------
    task_name : str
        Human-readable task name.

    fit_result : Any
        Result returned by GeneralTrainer.fit().

    holdout_result : Optional[EvaluationResult], default=None
        Hold-out evaluation result.

    validation_result : Optional[EvaluationResult], default=None
        Validation evaluation result.

    full_result : Optional[EvaluationResult], default=None
        Full observed-link evaluation result.

    extra_metadata : Optional[Dict[str, Any]], default=None
        Additional metadata stored under the summary metadata block.

    Returns
    -------
    Dict[str, Any]
        JSON-serializable task summary.
    """

    summary = {
        "task_name": str(task_name),
        "training": {
            "score": _safe_attr(fit_result, "score"),
            "best_epoch": _safe_attr(fit_result, "best_epoch"),
            "last_epoch": _safe_attr(fit_result, "last_epoch"),
            "stopped_early": _safe_attr(fit_result, "stopped_early"),
            "model_path": _safe_attr(fit_result, "model_path"),
            "eval_path": _safe_attr(fit_result, "eval_path"),
            "metadata": _safe_attr(fit_result, "metadata", default={}),
        },
        "metrics": {},
        "metadata": extra_metadata or {},
    }

    if holdout_result is not None:
        summary["metrics"]["holdout"] = holdout_result.metrics
        summary["metadata"]["holdout"] = holdout_result.metadata

    if validation_result is not None:
        summary["metrics"]["validation"] = validation_result.metrics
        summary["metadata"]["validation"] = validation_result.metadata

    if full_result is not None:
        summary["metrics"]["observed"] = full_result.metrics
        summary["metadata"]["observed"] = full_result.metadata

    return to_json_serializable(summary)


# =============================================================================
# Model inference
# =============================================================================


def run_flow_reconstruction(
    model: torch.nn.Module,
    observed_flows: torch.Tensor,
    flow_mask: torch.Tensor,
    device: str,
    true_od_demand: Optional[torch.Tensor] = None,
    od_mask: Optional[torch.Tensor] = None,
    is_pure_inference: bool = True,
) -> torch.Tensor:
    """
    Run model inference and return reconstructed link flows.

    Optional OD tensors are passed only when available. This keeps the evaluator
    compatible with models that require OD-conditioned assignment while still
    allowing simpler models to ignore those arguments.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model.

    observed_flows : torch.Tensor
        Link-flow tensor passed to the model.

    flow_mask : torch.Tensor
        Binary input mask passed to the model.

    device : str
        Torch device used for inference.

    true_od_demand : Optional[torch.Tensor], default=None
        Optional OD target tensor.

    od_mask : Optional[torch.Tensor], default=None
        Optional OD mask tensor.

    is_pure_inference : bool, default=True
        Flag passed to models that distinguish training-style forward passes
        from pure inference.

    Returns
    -------
    torch.Tensor
        Reconstructed link-flow tensor detached from the graph.
    """

    was_training = model.training
    model.eval()

    observed_flows = _ensure_1d_or_batched_tensor(
        observed_flows,
        device=device,
        name="observed_flows",
    )

    flow_mask = _ensure_1d_or_batched_tensor(
        flow_mask,
        device=device,
        name="flow_mask",
    )

    model_kwargs = {
        "observed_flows": observed_flows,
        "flow_mask": flow_mask,
        "warmup": False,
        "is_pure_inference": bool(is_pure_inference),
    }

    if true_od_demand is not None:
        model_kwargs["true_od_demand"] = _ensure_1d_or_batched_tensor(
            true_od_demand,
            device=device,
            name="true_od_demand",
        )

    if od_mask is not None:
        model_kwargs["od_mask"] = _ensure_1d_or_batched_tensor(
            od_mask,
            device=device,
            name="od_mask",
        )

    with torch.no_grad():
        outputs = model(**model_kwargs)

    if was_training:
        model.train()

    outputs = validate_model_inference_output_contract(outputs)

    return outputs["reconstructed_flows"].detach()


# =============================================================================
# Metrics
# =============================================================================


def compute_masked_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Compute standard regression metrics over entries selected by a binary mask.

    Metrics
    -------
    - count
    - MAE
    - MSE
    - RMSE
    - R2
    - MAPE
    - SMAPE
    - mean error
    - descriptive statistics

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth values.

    y_pred : np.ndarray
        Predicted values.

    mask : np.ndarray
        Binary evaluation mask.

    prefix : str, default=""
        Prefix added to metric keys.

    Returns
    -------
    Dict[str, Any]
        Metric dictionary.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = _as_binary_1d_mask(mask, name="evaluation_mask").astype(bool)

    _validate_same_length(y_true, y_pred, "y_true", "y_pred")
    _validate_same_length(y_true, mask, "y_true", "mask")

    metric_prefix = f"{prefix}_" if prefix else ""

    y_true_eval = y_true[mask]
    y_pred_eval = y_pred[mask]

    if y_true_eval.size == 0:
        return {
            f"{metric_prefix}has_data": False,
            f"{metric_prefix}count": 0,
        }

    residual = y_pred_eval - y_true_eval
    abs_error = np.abs(residual)
    squared_error = residual**2

    mse = float(squared_error.mean())
    mae = float(abs_error.mean())
    rmse = float(np.sqrt(mse))

    ss_res = float(np.sum((y_true_eval - y_pred_eval) ** 2))
    ss_tot = float(np.sum((y_true_eval - np.mean(y_true_eval)) ** 2))
    r2 = float(1.0 - ss_res / (ss_tot + 1.0e-8))

    non_zero = np.abs(y_true_eval) > 1.0e-12

    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    (y_true_eval[non_zero] - y_pred_eval[non_zero])
                    / y_true_eval[non_zero]
                )
            )
            * 100.0
        )
    else:
        mape = None

    denominator = np.abs(y_true_eval) + np.abs(y_pred_eval)
    smape_mask = denominator > 1.0e-12

    if np.any(smape_mask):
        smape = float(
            np.mean(
                2.0
                * np.abs(y_pred_eval[smape_mask] - y_true_eval[smape_mask])
                / denominator[smape_mask]
            )
            * 100.0
        )
    else:
        smape = None

    return {
        f"{metric_prefix}has_data": True,
        f"{metric_prefix}count": int(y_true_eval.size),
        f"{metric_prefix}mae": mae,
        f"{metric_prefix}mse": mse,
        f"{metric_prefix}rmse": rmse,
        f"{metric_prefix}r2": r2,
        f"{metric_prefix}mape": mape,
        f"{metric_prefix}smape": smape,
        f"{metric_prefix}mean_error": float(residual.mean()),
        f"{metric_prefix}mean_absolute_error": mae,
        f"{metric_prefix}mean_true": float(y_true_eval.mean()),
        f"{metric_prefix}mean_pred": float(y_pred_eval.mean()),
        f"{metric_prefix}min_true": float(y_true_eval.min()),
        f"{metric_prefix}max_true": float(y_true_eval.max()),
        f"{metric_prefix}min_pred": float(y_pred_eval.min()),
        f"{metric_prefix}max_pred": float(y_pred_eval.max()),
        f"{metric_prefix}nonzero_count_for_mape": int(np.sum(non_zero)),
    }


def compute_flow_conservation_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute simple aggregate flow-balance diagnostics.

    This is not a physical conservation proof. It only checks whether the total
    predicted mass over the evaluated links is close to the target mass.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth link flows.

    y_pred : np.ndarray
        Predicted link flows.

    mask : Optional[np.ndarray], default=None
        Optional binary evaluation mask. If None, all entries are used.

    Returns
    -------
    Dict[str, Any]
        Aggregate flow-balance diagnostics.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    _validate_same_length(y_true, y_pred, "y_true", "y_pred")

    if mask is None:
        mask_bool = np.ones_like(y_true, dtype=bool)
    else:
        mask_bool = _as_binary_1d_mask(mask, name="mask").astype(bool)
        _validate_same_length(y_true, mask_bool, "y_true", "mask")

    y_true_eval = y_true[mask_bool]
    y_pred_eval = y_pred[mask_bool]

    if y_true_eval.size == 0:
        return {
            "has_data": False,
            "count": 0,
        }

    true_total = float(y_true_eval.sum())
    pred_total = float(y_pred_eval.sum())
    absolute_gap = float(pred_total - true_total)
    relative_gap = float(absolute_gap / max(abs(true_total), 1.0e-8))

    return {
        "has_data": True,
        "count": int(y_true_eval.size),
        "true_total_flow": true_total,
        "pred_total_flow": pred_total,
        "absolute_total_flow_gap": absolute_gap,
        "relative_total_flow_gap": relative_gap,
    }


# =============================================================================
# Prediction tables
# =============================================================================


def build_prediction_dataframe(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: Optional[np.ndarray] = None,
    link_metadata: Optional[pd.DataFrame] = None,
    split_name: str = "evaluation",
) -> pd.DataFrame:
    """
    Build a row-level prediction table for diagnostics and export.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth link-flow vector.

    y_pred : np.ndarray
        Predicted link-flow vector.

    mask : Optional[np.ndarray], default=None
        Binary mask indicating which links were evaluated.

    link_metadata : Optional[pd.DataFrame], default=None
        Optional metadata aligned by link position.

    split_name : str, default="evaluation"
        Split name stored in the output table.

    Returns
    -------
    pd.DataFrame
        Prediction table.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    _validate_same_length(y_true, y_pred, "y_true", "y_pred")

    if mask is None:
        mask_arr = np.ones_like(y_true, dtype=np.float32)
    else:
        mask_arr = _as_binary_1d_mask(mask, name="mask")

    _validate_same_length(y_true, mask_arr, "y_true", "mask")

    df = pd.DataFrame(
        {
            "link_pos": np.arange(len(y_true), dtype=int),
            "split": str(split_name),
            "is_evaluated": mask_arr.astype(bool),
            "y_true": y_true,
            "y_pred": y_pred,
            "residual": y_pred - y_true,
            "absolute_error": np.abs(y_pred - y_true),
        }
    )

    non_zero = np.abs(df["y_true"].to_numpy()) > 1.0e-12

    percentage_error = np.full(
        len(df),
        np.nan,
        dtype=float,
    )

    percentage_error[non_zero] = (
        df.loc[non_zero, "absolute_error"].to_numpy()
        / np.abs(df.loc[non_zero, "y_true"].to_numpy())
        * 100.0
    )

    df["absolute_percentage_error"] = percentage_error

    if link_metadata is not None:
        df = _attach_link_metadata(
            prediction_df=df,
            link_metadata=link_metadata,
            expected_num_links=len(y_true),
        )

    return df


def _attach_link_metadata(
    prediction_df: pd.DataFrame,
    link_metadata: pd.DataFrame,
    expected_num_links: int,
) -> pd.DataFrame:
    """
    Attach link metadata to a prediction DataFrame by row order.

    Row i in link_metadata must describe the same directed link as:
        prediction_df.loc[i, 'link_pos']

    This function does not reorder metadata. It only validates that the metadata
    is already aligned with the model link order.

    Parameters
    ----------
    prediction_df : pd.DataFrame
        Prediction table.

    link_metadata : pd.DataFrame
        Link metadata aligned with the flow vector.

    Returns
    -------
    pd.DataFrame
        Enriched prediction table.
    """

    if not isinstance(link_metadata, pd.DataFrame):
        raise ModelInputContractError(
            f"link_metadata must be a pandas DataFrame. Got {type(link_metadata)}."
        )

    if len(link_metadata) != len(prediction_df):
        raise ModelInputContractError(
            "link_metadata length must match prediction_df length. "
            f"Got {len(link_metadata)} and {len(prediction_df)}."
        )

    if len(link_metadata) != int(expected_num_links):
        raise ModelInputContractError(
            "link_metadata length must match expected_num_links. "
            f"Got {len(link_metadata)} and expected {int(expected_num_links)}."
        )

    metadata = link_metadata.reset_index(drop=True).copy()

    if "model_link_position" not in metadata.columns:
        metadata["model_link_position"] = np.arange(len(metadata), dtype=np.int64)

        safe_columns = [
            column
            for column in metadata.columns
            if column not in prediction_df.columns
        ]

        return pd.concat(
            [
                metadata[safe_columns],
                prediction_df.reset_index(drop=True),
            ],
            axis=1,
        )

        if np.isnan(positions).any():
            raise ModelInputContractError(
                "link_metadata['model_link_position'] contains non-numeric values."
            )

        positions = positions.astype(np.int64)
        expected_positions = np.arange(len(metadata), dtype=np.int64)

        if not np.array_equal(positions, expected_positions):
            mismatch_idx = np.where(positions != expected_positions)[0]

            sample = [
                {
                    "row": int(idx),
                    "model_link_position": int(positions[idx]),
                    "expected_position": int(expected_positions[idx]),
                }
                for idx in mismatch_idx[:20]
            ]

            raise ModelInputContractError(
                "link_metadata['model_link_position'] must equal row position. "
                f"num_mismatches={len(mismatch_idx)} | sample={sample}"
            )

    if {"init_node", "term_node"}.issubset(metadata.columns):
        metadata["init_node"] = pd.to_numeric(
            metadata["init_node"],
            errors="raise",
        ).astype(np.int64)

        metadata["term_node"] = pd.to_numeric(
            metadata["term_node"],
            errors="raise",
        ).astype(np.int64)

    safe_columns = [
        column
        for column in metadata.columns
        if column not in prediction_df.columns
    ]

    return pd.concat(
        [
            metadata[safe_columns],
            prediction_df.reset_index(drop=True),
        ],
        axis=1,
    )


# =============================================================================
# Persistence
# =============================================================================


def save_evaluation_result(
    result: EvaluationResult,
    output_dir: str | Path,
    prefix: str,
    save_predictions: bool = True,
) -> Dict[str, Optional[str]]:
    """
    Save evaluation metrics and optional predictions.

    Parameters
    ----------
    result : EvaluationResult
        Evaluation result.

    output_dir : str | Path
        Directory where outputs are saved.

    prefix : str
        File prefix.

    save_predictions : bool, default=True
        If True and predictions exist, save predictions as CSV.

    Returns
    -------
    Dict[str, Optional[str]]
        Paths to saved metrics and prediction files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"{prefix}_metrics.json"

    payload = {
        "metrics": result.metrics,
        "metadata": result.metadata,
    }

    metrics_path.write_text(
        json.dumps(
            to_json_serializable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    predictions_path = None

    if (
        save_predictions
        and result.predictions_df is not None
        and not result.predictions_df.empty
    ):
        predictions_path = output_dir / f"{prefix}_predictions.csv"
        result.predictions_df.to_csv(
            predictions_path,
            index=False,
        )

    return {
        "metrics_path": str(metrics_path),
        "predictions_path": str(predictions_path) if predictions_path else None,
    }


def save_task_evaluation_summary(
    summary: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "task_evaluation_summary.json",
) -> str:
    """
    Save one task-level evaluation summary as JSON.

    Parameters
    ----------
    summary : Dict[str, Any]
        Task summary.

    output_dir : str | Path
        Directory where the summary is saved.

    filename : str, default="task_evaluation_summary.json"
        Output filename.

    Returns
    -------
    str
        Saved summary path.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename

    path.write_text(
        json.dumps(
            to_json_serializable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return str(path)


# =============================================================================
# Internal helpers
# =============================================================================


def _ensure_1d_or_batched_tensor(
    tensor: torch.Tensor,
    device: str,
    name: str,
) -> torch.Tensor:
    """
    Ensure that a tensor is on the correct device and has a batch dimension.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    device : str
        Target device.

    name : str
        Human-readable tensor name used in error messages.

    Returns
    -------
    torch.Tensor
        Tensor on device with shape [1, N] when the input was one-dimensional.
    """

    if not torch.is_tensor(tensor):
        raise ModelInputContractError(
            f"{name} must be a torch.Tensor. Got {type(tensor)}."
        )

    tensor = tensor.to(device)

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    return tensor


def _to_float_tensor(
    value: Any,
    device: str,
) -> torch.Tensor:
    """
    Convert an input value to a float tensor on the target device.

    Parameters
    ----------
    value : Any
        Input value.

    device : str
        Target device.

    Returns
    -------
    torch.Tensor
        Float tensor with a batch dimension when needed.
    """

    if torch.is_tensor(value):
        tensor = value.detach().to(
            device=device,
            dtype=torch.float32,
        )
    else:
        tensor = torch.tensor(
            np.asarray(value, dtype=np.float32),
            dtype=torch.float32,
            device=device,
        )

    if tensor.ndim == 1:
        return tensor.unsqueeze(0)

    return tensor


def _tensor_to_1d_numpy(
    tensor: torch.Tensor,
) -> np.ndarray:
    """
    Convert a tensor to a flattened NumPy array.

    Parameters
    ----------
    tensor : torch.Tensor
        Input tensor.

    Returns
    -------
    np.ndarray
        Flattened NumPy array on CPU.
    """

    if not torch.is_tensor(tensor):
        raise ModelInputContractError(
            f"Expected torch.Tensor, got {type(tensor)}."
        )

    return tensor.detach().cpu().float().reshape(-1).numpy()


def _as_binary_1d_mask(
    value: Any,
    name: str,
) -> np.ndarray:
    """
    Convert an input to a one-dimensional binary float mask.

    Parameters
    ----------
    value : Any
        Input mask.

    name : str
        Human-readable mask name used in error messages.

    Returns
    -------
    np.ndarray
        Binary mask with dtype float32.
    """

    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    arr = arr.astype(np.float32).reshape(-1)

    if not np.isfinite(arr).all():
        raise ModelInputContractError(
            f"{name} contains NaN or infinite values."
        )

    arr = np.clip(arr, 0.0, 1.0)

    unique_values = set(np.unique(arr).tolist())

    if not unique_values.issubset({0.0, 1.0}):
        raise ModelInputContractError(
            f"{name} must be binary after clipping. "
            f"Found values: {sorted(unique_values)}"
        )

    return arr.astype(np.float32)


def _validate_same_length(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    """
    Validate that two flattened arrays have the same length.

    Parameters
    ----------
    left : np.ndarray
        First array.

    right : np.ndarray
        Second array.

    left_name : str
        First array name.

    right_name : str
        Second array name.
    """

    if len(left) != len(right):
        raise ModelInputContractError(
            f"{left_name} and {right_name} must have the same length. "
            f"Got {len(left)} and {len(right)}."
        )


def _safe_attr(
    obj: Any,
    name: str,
    default: Any = None,
) -> Any:
    """
    Safely read an attribute from an object or dictionary.

    Parameters
    ----------
    obj : Any
        Object or dictionary.

    name : str
        Attribute/key name.

    default : Any, default=None
        Fallback value.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)
