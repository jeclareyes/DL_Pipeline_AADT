from __future__ import annotations

"""Centralized evaluation tasks for the testing pipeline.

Each task validates required artifact/static/mask keys with typed exceptions
before performing any computation.
"""

from src.components.assignment_motors.assignment_runner import prepare_and_run_assignment
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import logging

from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys

from src.test._testing_functions import (
    calculate_metrics,
    export_flows_csv,
    export_od_analysis_csv,
    plot_link_histograms,
    plot_scatter_comparison,
    export_spatial_audit_wrapper,
)

def _resolve_assignment_testing_config(
    *,
    static: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any] | None:
    """Resolve AssignmentTesting.assignment_config from static['testing_cfg'].

    Supports both possible shapes:

    1. testing_cfg is already the testing config:
       testing_cfg["AssignmentTesting"]["assignment_config"]

    2. testing_cfg is the full Hydra config:
       testing_cfg["testing"]["AssignmentTesting"]["assignment_config"]

    The returned value is always converted to a plain Python dict because
    assignment_runner expects a mapping, not an OmegaConf DictConfig.
    """
    from collections.abc import Mapping
    from omegaconf import DictConfig, OmegaConf

    testing_cfg = static.get("testing_cfg")

    if testing_cfg is None:
        logging.warning(f"[{model_name}] static['testing_cfg'] is missing.")
        return None

    candidates = []

    if isinstance(testing_cfg, Mapping) or isinstance(testing_cfg, DictConfig):
        candidates.append(testing_cfg)

        try:
            nested_testing = testing_cfg.get("testing")
            if nested_testing is not None:
                candidates.append(nested_testing)
        except Exception:
            pass

    for candidate in candidates:
        try:
            assignment_testing = candidate.get("AssignmentTesting")
        except Exception:
            assignment_testing = None

        if assignment_testing is None:
            continue

        try:
            assignment_config = assignment_testing.get("assignment_config")
        except Exception:
            assignment_config = None

        if assignment_config is None:
            continue

        if isinstance(assignment_config, DictConfig):
            assignment_config = OmegaConf.to_container(
                assignment_config,
                resolve=True,
            )

        if isinstance(assignment_config, dict) and assignment_config:
            return assignment_config

        if isinstance(assignment_config, Mapping):
            return dict(assignment_config)

    fallback_path = Path(__file__).resolve().parents[2] / "configs" / "assignment" / "assignment.yaml"
    if fallback_path.exists():
        try:
            fallback_cfg = OmegaConf.to_container(OmegaConf.load(fallback_path), resolve=True)
            if isinstance(fallback_cfg, dict) and fallback_cfg:
                logging.info(
                    f"[{model_name}] Using fallback assignment config from {fallback_path}"
                )
                return fallback_cfg
        except Exception as exc:
            logging.warning(
                f"[{model_name}] Could not load fallback assignment config from {fallback_path}: {exc}"
            )

    return None

def run_assignment_testing(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    """Run assignment recalculation using the centralized assignment runner.

    This task is intentionally a thin wrapper around prepare_and_run_assignment.
    It should not reconstruct OD matrices, inject VDF parameters, build RouteSet,
    build solvers, or build assignment compositions. Those responsibilities
    belong to src.components.assignment_motors.assignment_runner.

    The task keeps the previous user-facing outputs for compatibility while
    exporting additional route-based diagnostics when available.
    """
    from src.components.assignment_motors.assignment_runner import prepare_and_run_assignment

    if "pred_od" not in artifacts:
        logging.warning(f"[{model_name}] SKIP run_assignment_testing: 'pred_od' missing.")
        return

    _require_keys(
        static,
        ["raw_data", "od_pair_indices", "true_flows"],
        "static for run_assignment_testing",
    )

    assign_cfg_dict = _resolve_assignment_testing_config(
    static=static,
    model_name=model_name,
    )

    if assign_cfg_dict is None:
        logging.warning(
            f"[{model_name}] SKIP run_assignment_testing: "
            "AssignmentTesting.assignment_config is missing or empty."
        )
        return

    assign_cfg_dict = copy.deepcopy(assign_cfg_dict)
    assign_cfg_dict = _inject_synthetic_theta_fallback_if_needed(
        assign_cfg_dict=assign_cfg_dict,
        artifacts=artifacts,
        static=static,
        model_name=model_name,
    )

    try:
        bundle = prepare_and_run_assignment(
            artifacts=artifacts,
            static=static,
            assign_cfg_dict=assign_cfg_dict,
            model_name=model_name,
        )
    except Exception as exc:
        logging.error(f"[{model_name}] run_assignment_testing failed: {exc}")
        return

    result, links_df = _unpack_assignment_runner_output(bundle=bundle, model_name=model_name)
    assigned_flows = _to_numpy_1d(result.final_link_flows, "result.final_link_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")

    if len(assigned_flows) != len(true_flows):
        raise ArtifactSchemaError(
            f"[{model_name}] assigned_flows and true_flows have different lengths: "
            f"{len(assigned_flows)} != {len(true_flows)}."
        )
    if len(links_df) != len(assigned_flows):
        raise ArtifactSchemaError(
            f"[{model_name}] links_df and assigned_flows have different lengths: "
            f"{len(links_df)} != {len(assigned_flows)}."
        )

    _export_assignment_recalculation_results(
        result=result,
        links_df=links_df,
        assigned_flows=assigned_flows,
        true_flows=true_flows,
        artifacts=artifacts,
        static=static,
        output_dir=output_dir,
        model_name=model_name,
    )

    _compare_od_matrix_to_ground_truth(
        assignment_matrix=bundle.assignment_matrix,
        static=static,
        output_dir=output_dir,
        model_name=model_name,
    )

    plot_scatter_comparison(
        pred=assigned_flows,
        target=true_flows,
        mask=None,
        mask_label="All Links",
        title=f"Re-Assigned Flows vs True Flows ({model_name})",
        xlabel="True Flow",
        ylabel="Assigned Flow",
        output_path=os.path.join(output_dir, f"{model_name}_reassigned_scatter_global.png"),
        log_scale=True,
    )

    _plot_reassigned_density_by_group_if_available(
        assigned_flows=assigned_flows,
        true_flows=true_flows,
        links_df=links_df,
        static=static,
        output_dir=output_dir,
        model_name=model_name,
    )

    logging.info(f"[{model_name}] Finished re-assignment testing.")


def _inject_synthetic_theta_fallback_if_needed(
    *,
    assign_cfg_dict: Dict[str, Any],
    artifacts: Dict[str, Any],
    static: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """Switch SUE theta from artifact to explicit when the checkpoint lacks it.

    Synthetic datasets often store the reference theta in the scenario manifest
    even when the saved checkpoint does not expose learned_theta. In that case
    we use the manifest value so route-based assignment can still be executed
    deterministically for comparison against the known ground truth.
    """
    if not isinstance(assign_cfg_dict, dict):
        return assign_cfg_dict

    sue_cfg = assign_cfg_dict.get("stochastic_user_equilibrium")
    if not isinstance(sue_cfg, dict):
        return assign_cfg_dict

    logit_cfg = sue_cfg.get("logit")
    if not isinstance(logit_cfg, dict):
        return assign_cfg_dict

    theta_source = str(logit_cfg.get("theta_source", "")).strip().lower()
    theta_artifact_key = logit_cfg.get("theta_artifact_key")
    if theta_source != "artifact":
        return assign_cfg_dict
    if not theta_artifact_key or theta_artifact_key in artifacts:
        return assign_cfg_dict

    theta_value = _resolve_synthetic_theta_from_manifest(static=static, model_name=model_name)
    if theta_value is None:
        logging.warning(
            f"[{model_name}] SUE theta artifact '{theta_artifact_key}' is missing and no synthetic fallback theta was found."
        )
        return assign_cfg_dict

    logging.info(
        f"[{model_name}] SUE theta artifact '{theta_artifact_key}' is missing; "
        f"using synthetic manifest theta={theta_value} as explicit fallback."
    )
    logit_cfg["theta_source"] = "explicit"
    logit_cfg["theta_value"] = float(theta_value)
    logit_cfg["theta_artifact_key"] = None
    return assign_cfg_dict


def _resolve_synthetic_theta_from_manifest(*, static: Dict[str, Any], model_name: str) -> float | None:
    """Resolve synthetic assignment theta from the dataset creation manifest."""
    raw_data = static.get("raw_data", {})
    if not isinstance(raw_data, dict):
        return None

    source_file = None
    candidate_sources = [
        ("raw", "metadata", "flows", "source_file"),
        ("raw", "metadata", "routes", "source_file"),
        ("paths", "manifest_path"),
        ("metadata", "flows", "source_file"),
    ]
    for path in candidate_sources:
        current: Any = raw_data
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if ok and current:
            source_file = current
            break

    if not source_file:
        return None

    try:
        source_path = Path(str(source_file)).resolve()
    except Exception:
        return None

    manifest_path = source_path.parent / "info" / "dataset_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning(f"[{model_name}] Could not read synthetic manifest {manifest_path}: {exc}")
        return None

    candidate_paths = [
        ("metadata", "flows", "assignment_theta"),
        ("metadata", "flows", "assignment_metadata", "theta"),
        ("metadata", "flows", "assignment_config", "stochastic_user_equilibrium", "logit", "theta_value"),
        ("config", "AssignmentParameters", "SUE_Parameters", "theta"),
        ("assignment_theta",),
    ]

    for path in candidate_paths:
        current: Any = manifest
        ok = True
        for key in path:
            if not isinstance(current, dict) or key not in current:
                ok = False
                break
            current = current[key]
        if not ok:
            continue
        try:
            theta = float(current)
        except (TypeError, ValueError):
            continue
        if np.isfinite(theta) and theta > 0.0:
            return theta

    return None


def _unpack_assignment_runner_output(bundle: Any, model_name: str) -> tuple[Any, pd.DataFrame]:
    """Normalize the output of prepare_and_run_assignment.

    The runner returns AssignmentRunBundle and this helper extracts the
    explicit attributes used by the downstream checks.
    """
    if hasattr(bundle, "result") and hasattr(bundle, "links_df"):
        result = bundle.result
        links_df = bundle.links_df
    else:
        raise TypeError(
            "prepare_and_run_assignment must return AssignmentRunBundle with "
            "result and links_df attributes."
        )

    if not isinstance(links_df, pd.DataFrame):
        raise TypeError(f"[{model_name}] links_df returned by assignment runner must be a pandas DataFrame.")

    required_result_attrs = [
        "final_link_flows",
        "final_route_flows",
        "final_link_costs",
        "final_route_costs",
        "metadata",
    ]
    missing_attrs = [attr for attr in required_result_attrs if not hasattr(result, attr)]
    if missing_attrs:
        raise TypeError(
            f"[{model_name}] assignment result is missing required attributes: {missing_attrs}."
        )

    return result, links_df


def _export_assignment_recalculation_results(
    *,
    result: Any,
    links_df: pd.DataFrame,
    assigned_flows: np.ndarray,
    true_flows: np.ndarray,
    artifacts: Dict[str, Any],
    static: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    """Export link-level and route-level outputs from assignment recalculation."""
    os.makedirs(output_dir, exist_ok=True)

    df = links_df.copy()
    df["assigned_flow"] = assigned_flows
    df["true_flow"] = true_flows

    if "pred_flows" in artifacts:
        model_flow = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
        if len(model_flow) != len(df):
            raise ArtifactSchemaError(
                f"[{model_name}] pred_flows length does not match links_df length: "
                f"{len(model_flow)} != {len(df)}."
            )
        df["model_flow"] = model_flow
    else:
        df["model_flow"] = df["assigned_flow"]

    if hasattr(result, "final_link_costs"):
        final_link_costs = _to_numpy_1d(result.final_link_costs, "result.final_link_costs")
        if len(final_link_costs) != len(df):
            raise ArtifactSchemaError(
                f"[{model_name}] final_link_costs length does not match links_df length: "
                f"{len(final_link_costs)} != {len(df)}."
            )
        df["assigned_link_cost"] = final_link_costs

    df["assigned_absolute_error"] = np.abs(df["assigned_flow"].to_numpy(dtype=float) - df["true_flow"].to_numpy(dtype=float))
    
    # Clean/Theoretical approach: Prevent evaluation on zero-division elements
    true_flow_np = df["true_flow"].to_numpy(dtype=float)
    assigned_flow_np = df["assigned_flow"].to_numpy(dtype=float)
    
    condition = true_flow_np > 0.0

    df["assigned_relative_error"] = np.divide(
        assigned_flow_np - true_flow_np,
        true_flow_np,
        out=np.zeros_like(true_flow_np),
        where=condition,
    )

    # Backward-compatible output expected by earlier testing runs.
    df.to_csv(
        os.path.join(output_dir, f"{model_name}_assignment_recalculation_results.csv"),
        index=False,
    )

    # More explicit output name for the refactored route-based assignment layer.
    df.to_csv(
        os.path.join(output_dir, f"{model_name}_assignment_link_results.csv"),
        index=False,
    )

    if hasattr(result, "final_route_flows") and hasattr(result, "final_route_costs"):
        final_route_flows = _to_numpy_1d(result.final_route_flows, "result.final_route_flows")
        final_route_costs = _to_numpy_1d(result.final_route_costs, "result.final_route_costs")
        if len(final_route_flows) != len(final_route_costs):
            raise ArtifactSchemaError(
                f"[{model_name}] final_route_flows and final_route_costs have different lengths: "
                f"{len(final_route_flows)} != {len(final_route_costs)}."
            )

        route_df = _build_assignment_route_results_df(
            result=result,
            final_route_flows=final_route_flows,
            final_route_costs=final_route_costs,
        )
        route_df.to_csv(
            os.path.join(output_dir, f"{model_name}_assignment_route_results.csv"),
            index=False,
        )

    _export_assignment_convergence_history_if_available(
        result=result,
        output_dir=output_dir,
        model_name=model_name,
    )


def _build_assignment_route_results_df(
    *,
    result: Any,
    final_route_flows: np.ndarray,
    final_route_costs: np.ndarray,
) -> pd.DataFrame:
    """Build a route-level result table using RouteSet metadata when present."""
    route_table = None
    if isinstance(getattr(result, "metadata", None), dict):
        route_table = result.metadata.get("routes_assignment_df")

    if isinstance(route_table, pd.DataFrame) and len(route_table) == len(final_route_flows):
        route_df = route_table.copy()
    else:
        route_df = pd.DataFrame({"route_index": np.arange(len(final_route_flows))})

    route_df["assigned_route_flow"] = final_route_flows
    route_df["assigned_route_cost"] = final_route_costs
    return route_df


def _export_assignment_convergence_history_if_available(
    *,
    result: Any,
    output_dir: str,
    model_name: str,
) -> None:
    """Export convergence history when the assignment motor reports it."""
    metadata = getattr(result, "metadata", {})
    if not isinstance(metadata, dict):
        return

    convergence_history = metadata.get("convergence_history")
    if isinstance(convergence_history, list) and convergence_history:
        pd.DataFrame(convergence_history).to_csv(
            os.path.join(output_dir, f"{model_name}_assignment_convergence_history.csv"),
            index=False,
        )

    metadata_export = {
        key: value
        for key, value in metadata.items()
        if key not in {"routes_assignment_df", "convergence_history"}
    }
    if metadata_export:
        pd.DataFrame([metadata_export]).to_csv(
            os.path.join(output_dir, f"{model_name}_assignment_metadata_summary.csv"),
            index=False,
        )


def _plot_reassigned_density_by_group_if_available(
    *,
    assigned_flows: np.ndarray,
    true_flows: np.ndarray,
    links_df: pd.DataFrame,
    static: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    """Plot reassigned flow densities by link group when group data exists."""
    if "link_group" in links_df.columns:
        link_groups = links_df["link_group"].to_numpy()
    elif "link_group" in static:
        link_groups = _to_numpy_1d(static["link_group"], "link_group")
    else:
        return

    if len(link_groups) != len(assigned_flows):
        raise ArtifactSchemaError(
            f"[{model_name}] link_group length does not match assigned flows length: "
            f"{len(link_groups)} != {len(assigned_flows)}."
        )

    from src.test.plots.density_plots import plot_density_by_group

    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    plot_density_by_group(
        estimated=assigned_flows,
        target=true_flows,
        groups=link_groups,
        group_labels=group_map,
        output_dir=output_dir,
        prefix=f"{model_name}_reassigned",
        log_scale=True,
        x_label="Assigned Flow",
    )


def _require_keys(container: Dict[str, Any], keys: list[str], context: str) -> None:
    require_keys(container, keys, context=context, exc_type=ArtifactSchemaError)


def _to_numpy_1d(value: Any, key_name: str) -> np.ndarray:
    if value is None:
        raise ArtifactSchemaError(f"Required value '{key_name}' is missing")
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    arr = np.asarray(value).reshape(-1)
    return arr


def _resolve_od_known_mask(static: Dict[str, Any], masks: Dict[str, Any]) -> np.ndarray | None:
    """Canonical OD known-mask resolver.

    Canonical source is masks.od_mask.
    Legacy alias static.mask_od_known is still accepted.
    """
    if "od_mask" in masks:
        return _to_numpy_1d(masks["od_mask"], "od_mask").astype(bool)
    if "mask_od_known" in static:
        return _to_numpy_1d(static["mask_od_known"], "mask_od_known").astype(bool)
    return None


def _normalize_epochs_history(epochs_history: Any) -> list[dict[str, Any]]:
    """Convert checkpoint epoch history to a list of row dictionaries."""
    if epochs_history is None:
        return []

    if isinstance(epochs_history, list):
        rows = [row for row in epochs_history if isinstance(row, dict)]
        return rows

    if isinstance(epochs_history, dict):
        rows: list[dict[str, Any]] = []
        for epoch_key, payload in sorted(epochs_history.items(), key=lambda item: int(item[0])):
            if not isinstance(payload, dict):
                continue
            row = dict(payload)
            row.setdefault("epoch", int(epoch_key))
            rows.append(row)
        return rows

    return []


def _extract_dataset_availability(static: Dict[str, Any]) -> Dict[str, bool]:
    """Resolve dataset availability flags from the saved model config."""
    model_cfg = static.get("model_config")
    if model_cfg is None:
        return {}

    from collections.abc import Mapping

    dataset_cfg = None
    if isinstance(model_cfg, Mapping):
        dataset_cfg = model_cfg.get("dataset")
    else:
        dataset_cfg = getattr(model_cfg, "dataset", None)

    if dataset_cfg is None:
        return {}

    availability = None
    if isinstance(dataset_cfg, Mapping):
        availability = dataset_cfg.get("data_availability")
    else:
        availability = getattr(dataset_cfg, "data_availability", None)

    if availability is None:
        return {}

    if isinstance(availability, Mapping):
        return {str(k): bool(v) for k, v in availability.items()}

    return {
        key: bool(getattr(availability, key))
        for key in [
            "has_complete_od_ground_truth",
            "has_observed_link_flows",
            "has_ground_truth_assignment",
        ]
        if hasattr(availability, key)
    }


def _compare_od_matrix_to_ground_truth(
    *,
    assignment_matrix: np.ndarray,
    static: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    """Export a matrix-level OD comparison when complete ground truth exists."""
    raw_data = static.get("raw_data", {})
    if not isinstance(raw_data, dict):
        return

    availability = _extract_dataset_availability(static)
    if not availability.get("has_complete_od_ground_truth", False):
        return

    gt_od = raw_data.get("od_matrix")
    if gt_od is None:
        return

    if hasattr(gt_od, "toarray"):
        gt_od = gt_od.toarray()
    else:
        gt_od = np.asarray(gt_od)

    pred_od = np.asarray(assignment_matrix)
    if pred_od.shape != gt_od.shape:
        raise ArtifactSchemaError(
            f"[{model_name}] Predicted OD matrix shape {pred_od.shape} does not match ground truth {gt_od.shape}."
        )

    gt_vec = gt_od.reshape(-1)
    pred_vec = pred_od.reshape(-1)

    rel_err = np.zeros_like(gt_vec, dtype=float)
    np.divide(
        pred_vec - gt_vec,
        gt_vec,
        out=rel_err,
        where=gt_vec != 0,
    )

    rows = np.repeat(np.arange(gt_od.shape[0]), gt_od.shape[1])
    cols = np.tile(np.arange(gt_od.shape[1]), gt_od.shape[0])
    df = pd.DataFrame(
        {
            "origin_index": rows,
            "destination_index": cols,
            "predicted_od": pred_vec,
            "ground_truth_od": gt_vec,
            "absolute_error": np.abs(pred_vec - gt_vec),
            "relative_error": rel_err,
            "is_intrazonal": rows == cols,
        }
    )

    df.to_csv(os.path.join(output_dir, f"{model_name}_od_matrix_comparison.csv"), index=False)

    summary = pd.DataFrame(
        [
            {
                "model": model_name,
                "pred_total_od": float(np.nansum(pred_vec)),
                "ground_truth_total_od": float(np.nansum(gt_vec)),
                "absolute_total_gap": float(abs(np.nansum(pred_vec) - np.nansum(gt_vec))),
                "relative_total_gap": float(
                    abs(np.nansum(pred_vec) - np.nansum(gt_vec))
                    / max(abs(np.nansum(gt_vec)), 1e-12)
                ),
                "mae": float(np.mean(np.abs(pred_vec - gt_vec))),
                "rmse": float(np.sqrt(np.mean((pred_vec - gt_vec) ** 2))),
            }
        ]
    )
    summary.to_csv(
        os.path.join(output_dir, f"{model_name}_od_matrix_comparison_summary.csv"),
        index=False,
    )


def plot_standard_metrics(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_standard_metrics")
    _require_keys(static, ["true_flows"], "static for plot_standard_metrics")
    _require_keys(masks, ["flow_test"], "masks for plot_standard_metrics")

    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    mask_test = _to_numpy_1d(masks["flow_test"], "flow_test").astype(bool)

    if np.any(mask_test):
        link_metrics = calculate_metrics(pred_flows[mask_test], true_flows[mask_test])
        subset_name = "TEST_SET"
        scatter_mask = mask_test
        scatter_label = "Test Links"
    else:
        link_metrics = calculate_metrics(pred_flows, true_flows)
        subset_name = "FULL_SET"
        scatter_mask = None
        scatter_label = "All Links"

    metrics_df = pd.DataFrame([link_metrics])
    metrics_df["type"] = f"Links_{subset_name}"
    metrics_df["model"] = model_name
    metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)

    plot_scatter_comparison(
        pred=pred_flows,
        target=true_flows,
        mask=scatter_mask,
        mask_label=scatter_label,
        title=f"Traffic Flows: True vs Estimated ({model_name})",
        xlabel="True Flow (veh/h)",
        ylabel="Estimated Flow (veh/h)",
        output_path=os.path.join(output_dir, f"{model_name}_scatter_flows.png"),
        log_scale=True,
    )


def plot_link_scatter_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_scatter_global")
    _require_keys(static, ["true_flows"], "static for plot_link_scatter_global")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    
    plot_scatter_comparison(
        pred=pred_flows,
        target=true_flows,
        mask=None,
        mask_label="All Links",
        title=f"Global Link Scatter ({model_name})",
        xlabel="True Flow",
        ylabel="Estimated Flow",
        output_path=os.path.join(output_dir, f"{model_name}_link_scatter_global.png"),
        log_scale=True,
    )

def plot_link_scatter_by_link_group(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_scatter_by_link_group")
    _require_keys(static, ["true_flows", "link_group"], "static for plot_link_scatter_by_link_group")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    link_groups = _to_numpy_1d(static["link_group"], "link_group")
    
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    unique_groups = np.unique(link_groups)
    
    for g in unique_groups:
        mask = link_groups == g
        if np.any(mask):
            g_label = group_map.get(g, str(g))
            plot_scatter_comparison(
                pred=pred_flows[mask],
                target=true_flows[mask],
                mask=None,
                mask_label=f"Group {g_label}",
                title=f"Link Scatter: {g_label} ({model_name})",
                xlabel="True Flow",
                ylabel="Estimated Flow",
                output_path=os.path.join(output_dir, f"{model_name}_link_scatter_{g_label}.png"),
                log_scale=True,
            )

def plot_od_scatter_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_od"], "artifacts for plot_od_scatter_global")
    _require_keys(static, ["true_od"], "static for plot_od_scatter_global")

    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    true_od = _to_numpy_1d(static["true_od"], "true_od")
    mask_od = _resolve_od_known_mask(static, masks)

    plot_scatter_comparison(
        pred=pred_od,
        target=true_od,
        mask=mask_od,
        mask_label="Known OD",
        title=f"Global OD Scatter ({model_name})",
        xlabel="True OD",
        ylabel="Estimated OD",
        output_path=os.path.join(output_dir, f"{model_name}_od_scatter_global.png"),
        log_scale=True,
    )

def plot_link_density_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.density_plots import plot_density_overlay
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_density_global")
    _require_keys(static, ["true_flows"], "static for plot_link_density_global")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    
    plot_density_overlay(
        estimated=pred_flows,
        target=true_flows,
        title=f"Global Link Density ({model_name})",
        output_path=os.path.join(output_dir, f"{model_name}_link_density_global.png"),
        log_scale=True,
        x_label="Flow",
    )

def plot_link_density_by_group(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.density_plots import plot_density_by_group
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_density_by_group")
    _require_keys(static, ["true_flows", "link_group"], "static for plot_link_density_by_group")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    link_groups = _to_numpy_1d(static["link_group"], "link_group")
    
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    plot_density_by_group(
        estimated=pred_flows,
        target=true_flows,
        groups=link_groups,
        group_labels=group_map,
        output_dir=output_dir,
        prefix=f"{model_name}_link",
        log_scale=True,
        x_label="Flow",
    )

def plot_link_histogram_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.histogram_plots import plot_histogram_comparison
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_histogram_global")
    _require_keys(static, ["true_flows"], "static for plot_link_histogram_global")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    
    plot_histogram_comparison(
        estimated=pred_flows,
        target=true_flows,
        title=f"Global Link Histogram ({model_name})",
        output_path=os.path.join(output_dir, f"{model_name}_link_histogram_global.png"),
        x_label="Flow",
    )

def plot_link_histogram_by_group(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.histogram_plots import plot_histogram_by_group
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_histogram_by_group")
    _require_keys(static, ["true_flows", "link_group"], "static for plot_link_histogram_by_group")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    link_groups = _to_numpy_1d(static["link_group"], "link_group")
    
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    plot_histogram_by_group(
        estimated=pred_flows,
        target=true_flows,
        groups=link_groups,
        group_labels=group_map,
        output_dir=output_dir,
        prefix=f"{model_name}_link",
        x_label="Flow",
    )

def plot_od_density_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.density_plots import plot_density_overlay
    _require_keys(artifacts, ["pred_od"], "artifacts for plot_od_density_global")
    _require_keys(static, ["true_od"], "static for plot_od_density_global")
    
    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    true_od = _to_numpy_1d(static["true_od"], "true_od")
    
    plot_density_overlay(
        estimated=pred_od,
        target=true_od,
        title=f"Global OD Density ({model_name})",
        output_path=os.path.join(output_dir, f"{model_name}_od_density_global.png"),
        log_scale=True,
        x_label="OD Demand",
    )

def plot_vc_density_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.density_plots import plot_density_overlay
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_vc_density_global")
    _require_keys(static, ["true_flows", "capacity"], "static for plot_vc_density_global")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    
    safe_cap = capacity.copy()
    safe_cap[safe_cap == 0] = 1.0
    
    plot_density_overlay(
        estimated=pred_flows / safe_cap,
        target=true_flows / safe_cap,
        title=f"Global V/C Ratio Density ({model_name})",
        output_path=os.path.join(output_dir, f"{model_name}_vc_density_global.png"),
        log_scale=False,
        x_label="V/C Ratio",
    )

def plot_vc_density_by_group(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.density_plots import plot_density_by_group
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_vc_density_by_group")
    _require_keys(static, ["true_flows", "capacity", "link_group"], "static for plot_vc_density_by_group")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    link_groups = _to_numpy_1d(static["link_group"], "link_group")
    
    safe_cap = capacity.copy()
    safe_cap[safe_cap == 0] = 1.0
    
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    plot_density_by_group(
        estimated=pred_flows / safe_cap,
        target=true_flows / safe_cap,
        groups=link_groups,
        group_labels=group_map,
        output_dir=output_dir,
        prefix=f"{model_name}_vc",
        log_scale=False,
        x_label="V/C Ratio",
    )

def plot_vc_histogram_global(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.histogram_plots import plot_histogram_comparison
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_vc_histogram_global")
    _require_keys(static, ["true_flows", "capacity"], "static for plot_vc_histogram_global")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    
    safe_cap = capacity.copy()
    safe_cap[safe_cap == 0] = 1.0
    
    plot_histogram_comparison(
        estimated=pred_flows / safe_cap,
        target=true_flows / safe_cap,
        title=f"Global V/C Ratio Histogram ({model_name})",
        output_path=os.path.join(output_dir, f"{model_name}_vc_histogram_global.png"),
        x_label="V/C Ratio",
    )


def export_global_link_metrics(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.metrics.regression_metrics import calculate_masked_metrics
    
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_global_link_metrics")
    _require_keys(static, ["true_flows"], "static for export_global_link_metrics")
    _require_keys(masks, ["flow_train", "flow_test", "flow_observed"], "masks for export_global_link_metrics")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    
    masks_dict = {
        "train": masks["flow_train"],
        "test": masks["flow_test"],
        "observed_all": masks["flow_observed"]
    }
    
    df = calculate_masked_metrics(pred_flows, true_flows, masks_dict)
    df["model"] = model_name
    
    df.to_csv(os.path.join(output_dir, f"{model_name}_global_link_metrics.csv"), index=False)

def export_grouped_link_metrics(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.metrics.regression_metrics import calculate_grouped_metrics
    
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_grouped_link_metrics")
    _require_keys(static, ["true_flows", "link_group"], "static for export_grouped_link_metrics")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    link_groups = _to_numpy_1d(static["link_group"], "link_group")
    
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    
    df_group = calculate_grouped_metrics(pred_flows, true_flows, link_groups, group_map, label="link_group")
    df_group["model"] = model_name
    df_group.to_csv(os.path.join(output_dir, f"{model_name}_metrics_by_link_group.csv"), index=False)
    
    # If link_type_raw is available, also group by it
    if "link_type_raw" in static:
        link_type = _to_numpy_1d(static["link_type_raw"], "link_type_raw")
        df_type = calculate_grouped_metrics(pred_flows, true_flows, link_type, None, label="link_type")
        df_type["model"] = model_name
        df_type.to_csv(os.path.join(output_dir, f"{model_name}_metrics_by_link_type.csv"), index=False)

def export_od_metrics(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.metrics.regression_metrics import calculate_masked_metrics
    
    if "pred_od" not in artifacts or "true_od" not in static:
        return
        
    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    true_od = _to_numpy_1d(static["true_od"], "true_od")
    
    mask_od_known = _resolve_od_known_mask(static, masks)
    masks_dict = {}
    if mask_od_known is not None:
        masks_dict["supervised_od"] = mask_od_known
    masks_dict["all_od_with_target"] = true_od > 0
    
    df = calculate_masked_metrics(pred_od, true_od, masks_dict)
    df["model"] = model_name
    df.to_csv(os.path.join(output_dir, f"{model_name}_od_metrics.csv"), index=False)


def export_link_results_table(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_link_results_table")
    _require_keys(static, ["true_flows", "raw_data"], "static for export_link_results_table")
    _require_keys(masks, ["flow_train", "flow_test", "flow_observed"], "masks for export_link_results_table")

    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    
    num_links = len(pred_flows)
    raw_data = static.get("raw_data", {})
    link_data = raw_data.get("link_data", pd.DataFrame())
    
    if len(link_data) != num_links:
        link_data = pd.DataFrame({"link_index": range(num_links)})
        
    df = link_data.copy()
    if "link_index" not in df.columns:
        df["link_index"] = np.arange(num_links)
        
    df["estimated_flow"] = pred_flows
    df["target_flow"] = true_flows

    flow_diff = pred_flows - true_flows
    positive_flow = true_flows > 0

    df["absolute_error"] = np.abs(flow_diff)

    relative_error = np.zeros_like(true_flows, dtype=float)
    np.divide(
        flow_diff,
        true_flows,
        out=relative_error,
        where=positive_flow,
    )

    df["relative_error"] = relative_error
    df["squared_error"] = flow_diff ** 2
    df["ape"] = np.abs(relative_error) * 100   

    df["is_train"] = _to_numpy_1d(masks["flow_train"], "flow_train").astype(bool)
    df["is_test"] = _to_numpy_1d(masks["flow_test"], "flow_test").astype(bool)
    df["is_observed"] = _to_numpy_1d(masks["flow_observed"], "flow_observed").astype(bool)
    
    if "capacity" in static:
        df["capacity"] = _to_numpy_1d(static["capacity"], "capacity")
        safe_cap = df["capacity"].copy()
        safe_cap[safe_cap == 0] = 1.0
        df["vc_estimated"] = df["estimated_flow"] / safe_cap
        
    if "link_type_raw" in static:
        df["link_type_raw"] = _to_numpy_1d(static["link_type_raw"], "link_type_raw")
        
    if "link_group" in static:
        df["link_group"] = _to_numpy_1d(static["link_group"], "link_group")
        
    df.to_csv(os.path.join(output_dir, f"{model_name}_link_results.csv"), index=False)


def export_od_results_table(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    if "pred_od" not in artifacts or "true_od" not in static or "od_pair_indices" not in static:
        return
        
    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    true_od = _to_numpy_1d(static["true_od"], "true_od")
    od_indices = static["od_pair_indices"]
    
    if hasattr(od_indices, "detach"):
        od_indices = od_indices.detach().cpu().numpy()
    else:
        od_indices = np.asarray(od_indices)
        
    df = pd.DataFrame({
        "od_index": np.arange(len(pred_od)),
        "origin_node": od_indices[:, 0] if od_indices.ndim == 2 else np.nan,
        "destination_node": od_indices[:, 1] if od_indices.ndim == 2 else np.nan,
        "estimated_od": pred_od,
        "target_od": true_od,
        "absolute_error": np.abs(pred_od - true_od),
        "relative_error": np.where(true_od > 0, (pred_od - true_od) / true_od, 0),
        "squared_error": (pred_od - true_od) ** 2,
        "ape": np.where(true_od > 0, np.abs((pred_od - true_od) / true_od) * 100, 0),
    })
    
    mask_od_known = _resolve_od_known_mask(static, masks)
    if mask_od_known is not None:
        df["is_supervised"] = mask_od_known.astype(bool)
        
    if "od_test" in masks:
        df["is_holdout"] = _to_numpy_1d(masks["od_test"], "od_test").astype(bool)
        
    df["is_intrazonal"] = df["origin_node"] == df["destination_node"]
    
    df.to_csv(os.path.join(output_dir, f"{model_name}_od_results.csv"), index=False)



def plot_cgame_penalty(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["historic_penalty"], "artifacts for plot_cgame_penalty")
    penalty = _to_numpy_1d(artifacts["historic_penalty"], "historic_penalty")
    out_path = os.path.join(output_dir, f"{model_name}_penalty_summary.csv")
    pd.DataFrame({"historic_penalty": penalty}).to_csv(out_path, index=False)


def export_od_link_contributions(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.builders.route_results_builder import build_od_link_contributions
    
    _require_keys(artifacts, ["route_flows", "pred_flows"], "artifacts for export_od_link_contributions")
    _require_keys(static, ["od_pair_indices", "delta_matrix", "route_validity_mask"], "static for export_od_link_contributions")
    
    route_validity_mask = static["route_validity_mask"]
    if hasattr(route_validity_mask, "detach"):
        route_validity_mask = route_validity_mask.detach().cpu().numpy()
    else:
        route_validity_mask = np.asarray(route_validity_mask)
        
    route_flows = _to_numpy_1d(artifacts["route_flows"], "route_flows").reshape(route_validity_mask.shape)
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    
    od_indices = static["od_pair_indices"]
    if hasattr(od_indices, "detach"):
        od_indices = od_indices.detach().cpu().numpy()
    else:
        od_indices = np.asarray(od_indices)
        
    delta_matrix = static["delta_matrix"]
    if hasattr(delta_matrix, "detach"):
        delta_matrix = delta_matrix.detach().cpu().numpy()
    else:
        delta_matrix = np.asarray(delta_matrix)
        
    df = build_od_link_contributions(od_indices, route_flows, route_validity_mask, delta_matrix, pred_flows)
    df.to_csv(os.path.join(output_dir, f"{model_name}_od_link_contributions.csv"), index=False)

def export_route_link_contributions(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.builders.route_results_builder import build_route_link_contributions
    
    _require_keys(artifacts, ["route_flows", "pred_flows"], "artifacts for export_route_link_contributions")
    _require_keys(static, ["od_pair_indices", "delta_matrix", "route_validity_mask"], "static for export_route_link_contributions")
    
    route_validity_mask = static["route_validity_mask"]
    if hasattr(route_validity_mask, "detach"):
        route_validity_mask = route_validity_mask.detach().cpu().numpy()
    else:
        route_validity_mask = np.asarray(route_validity_mask)
        
    route_flows = _to_numpy_1d(artifacts["route_flows"], "route_flows").reshape(route_validity_mask.shape)
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    
    od_indices = static["od_pair_indices"]
    if hasattr(od_indices, "detach"):
        od_indices = od_indices.detach().cpu().numpy()
    else:
        od_indices = np.asarray(od_indices)
        
    delta_matrix = static["delta_matrix"]
    if hasattr(delta_matrix, "detach"):
        delta_matrix = delta_matrix.detach().cpu().numpy()
    else:
        delta_matrix = np.asarray(delta_matrix)
        
    df = build_route_link_contributions(od_indices, route_flows, route_validity_mask, delta_matrix, pred_flows)
    df.to_csv(os.path.join(output_dir, f"{model_name}_route_link_contributions.csv"), index=False)

def export_route_flow_table(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.builders.route_results_builder import build_route_flows_table
    
    _require_keys(artifacts, ["route_flows", "pred_od"], "artifacts for export_route_flow_table")
    _require_keys(static, ["od_pair_indices", "delta_matrix", "route_validity_mask"], "static for export_route_flow_table")
    
    route_validity_mask = static["route_validity_mask"]
    if hasattr(route_validity_mask, "detach"):
        route_validity_mask = route_validity_mask.detach().cpu().numpy()
    else:
        route_validity_mask = np.asarray(route_validity_mask)
        
    route_flows = _to_numpy_1d(artifacts["route_flows"], "route_flows").reshape(route_validity_mask.shape)
    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    
    od_indices = static["od_pair_indices"]
    if hasattr(od_indices, "detach"):
        od_indices = od_indices.detach().cpu().numpy()
    else:
        od_indices = np.asarray(od_indices)
        
    delta_matrix = static["delta_matrix"]
    if hasattr(delta_matrix, "detach"):
        delta_matrix = delta_matrix.detach().cpu().numpy()
    else:
        delta_matrix = np.asarray(delta_matrix)
        
    df = build_route_flows_table(od_indices, route_flows, route_validity_mask, pred_od, delta_matrix)
    df.to_csv(os.path.join(output_dir, f"{model_name}_route_flows.csv"), index=False)

def plot_training_history(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.epoch_plots import plot_loss_curves, plot_lr_curve, plot_loss_and_score
    _require_keys(artifacts, ["epochs_history"], "artifacts for plot_training_history")
    
    epochs_history = _normalize_epochs_history(artifacts["epochs_history"])
    if not epochs_history:
        return
        
    plot_loss_curves(epochs_history, os.path.join(output_dir, f"{model_name}_loss_curves.png"))
    plot_lr_curve(epochs_history, os.path.join(output_dir, f"{model_name}_lr_curve.png"))
    plot_loss_and_score(epochs_history, os.path.join(output_dir, f"{model_name}_loss_score.png"))

def plot_alpha_beta_history(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    from src.test.plots.epoch_plots import plot_alpha_beta_evolution
    _require_keys(artifacts, ["epochs_history"], "artifacts for plot_alpha_beta_history")
    
    epochs_history = _normalize_epochs_history(artifacts["epochs_history"])
    if not epochs_history:
        return
        
    plot_alpha_beta_evolution(epochs_history, os.path.join(output_dir, f"{model_name}_alpha_beta_history.png"))

def export_alpha_beta_table(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["learned_alpha", "learned_beta"], "artifacts for export_alpha_beta_table")
    _require_keys(static, ["link_group", "capacity", "t0"], "static for export_alpha_beta_table")
    
    alpha = _to_numpy_1d(artifacts["learned_alpha"], "learned_alpha")
    beta = _to_numpy_1d(artifacts["learned_beta"], "learned_beta")
    link_group = _to_numpy_1d(static["link_group"], "link_group")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    t0 = _to_numpy_1d(static["t0"], "t0")
    
    # Check if alpha/beta are scalars, vectors per group, or per link
    if alpha.size == 1:
        alpha = np.full_like(link_group, alpha[0])
    if beta.size == 1:
        beta = np.full_like(link_group, beta[0])
        
    # If they are per group, we map them directly
    # Wait, the dimensions of learned_alpha should be either [1], [num_groups] or [num_links]
    df_raw = pd.DataFrame({
        "group_id": link_group,
        "capacity": capacity,
        "t0": t0
    })
    
    if alpha.size == len(np.unique(link_group)):
        # Alpha is per group
        df_raw["alpha"] = alpha[link_group]
    elif alpha.size == len(link_group):
        df_raw["alpha"] = alpha
    else:
        df_raw["alpha"] = alpha[0] if alpha.size > 0 else 0
        
    if beta.size == len(np.unique(link_group)):
        df_raw["beta"] = beta[link_group]
    elif beta.size == len(link_group):
        df_raw["beta"] = beta
    else:
        df_raw["beta"] = beta[0] if beta.size > 0 else 0
        
    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    
    agg_funcs = {
        "alpha": "mean",
        "beta": "mean",
        "group_id": "count",
        "capacity": "mean",
        "t0": "mean"
    }
    
    summary = df_raw.groupby("group_id").agg(agg_funcs).rename(columns={"group_id": "num_links"}).reset_index()
    summary["group_name"] = summary["group_id"].map(lambda x: group_map.get(x, f"Type_{x}"))
    
    # Reorder columns
    cols = ["group_id", "group_name", "alpha", "beta", "num_links", "capacity", "t0"]
    summary = summary[cols].rename(columns={"alpha": "alpha_final", "beta": "beta_final", "capacity": "mean_capacity", "t0": "mean_t0"})
    
    summary.to_csv(os.path.join(output_dir, f"{model_name}_alpha_beta_by_group.csv"), index=False)

# =============================================================================
# FUNCIONES DE VISUALIZACIÓN ESPACIAL (INTERACTIVAS)
# =============================================================================

def plot_spatial_network(
    *,
    artifacts: Dict[str, Any],
    static: Dict[str, Any],
    masks: Dict[str, Any],
    output_dir: str,
    model_name: str,
) -> None:
    """Plot the physical road network using model-ready link geometries."""

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    _require_keys(static, ["link_geometries"], "static for plot_spatial_network")

    geometries = static["link_geometries"]

    if not isinstance(geometries, dict) or not geometries:
        raise ArtifactSchemaError(
            f"[{model_name}] link_geometries must be a non-empty dictionary."
        )

    segments = []
    for _, geom in geometries.items():
        if not isinstance(geom, dict):
            continue

        required = {"x1", "y1", "x2", "y2"}
        if not required.issubset(geom):
            continue

        segments.append(
            [
                (float(geom["x1"]), float(geom["y1"])),
                (float(geom["x2"]), float(geom["y2"])),
            ]
        )

    if not segments:
        raise ArtifactSchemaError(
            f"[{model_name}] No valid drawable link geometries were found."
        )

    fig, ax = plt.subplots(figsize=(12, 12))
    collection = LineCollection(segments, linewidths=0.35, alpha=0.8)
    ax.add_collection(collection)

    ax.autoscale()
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Spatial Network ({model_name})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linewidth=0.3, alpha=0.4)

    output_path = os.path.join(output_dir, f"{model_name}_spatial_network.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)

    logging.info(f"[{model_name}] Spatial network plot saved to: {output_path}")

def export_spatial_audit_wrapper(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    if "link_geometries" not in static:
        logging.warning(f"[SKIP] export_spatial_audit_wrapper: 'link_geometries' missing for {model_name}.")
        return
        
    from src.test.exporters.geopackage_exporter import export_spatial_audit
    
    # We can reconstruct df_links here by calling the global link metrics builder, or just passing pred/true arrays
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_spatial_audit")
    _require_keys(static, ["true_flows", "capacity", "link_group", "link_geometries"], "static for export_spatial_audit")
    
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    
    df = pd.DataFrame({
        "link_index": np.arange(len(pred_flows)),
        "pred_flow": pred_flows,
        "true_flow": true_flows,
        "capacity": capacity,
        "vc_ratio": np.where(capacity > 0, pred_flows / capacity, 0),
        "abs_error": np.abs(pred_flows - true_flows)
    })
    
    out_path = os.path.join(output_dir, f"{model_name}_spatial_audit.gpkg")
    export_spatial_audit(df, static["link_geometries"], out_path)

def export_geopackage_task(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    """
    Evaluation task that delegates spatial export to the export_utils module 
    using the injected raw_data.
    """
    from src.utils.export_utils import export_to_geopackage
    import logging

    # 1. Soft Check for injected raw_data
    if "raw_data" not in static:
        logging.warning(f"[SKIP] export_geopackage_task: 'raw_data' missing in static dictionary for {model_name}.")
        return

    # 2. Hard Contract Check for required flow outputs
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_geopackage_task")
    
    raw_data = static["raw_data"]
    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    
    # 3. Handle optional OD Demand
    pred_od = None
    if "pred_od" in artifacts:
        pred_od = artifacts["pred_od"]
        if hasattr(pred_od, "detach"):
            pred_od = pred_od.detach().cpu().numpy()
        else:
            pred_od = np.asarray(pred_od)

    # 4. Delegate to the GeoPackage exporter
    filename = f"{model_name}_results.gpkg"
    export_to_geopackage(
        raw_data=raw_data,
        reconstructed_flows=pred_flows,
        estimated_demand=pred_od,
        output_dir=output_dir,
        filename=filename
    )
