"""Centralized evaluation tasks for the testing pipeline.

Each task validates required artifact/static/mask keys with typed exceptions
before performing any computation.
"""

from __future__ import annotations

import os
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.contracts.runtime_contracts import ArtifactSchemaError, require_keys

from src.test._testing_functions import (
    calculate_metrics,
    export_flows_csv,
    export_od_analysis_csv,
    plot_link_histograms,
    plot_scatter_comparison,
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


def plot_link_distributions(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for plot_link_distributions")
    _require_keys(static, ["capacity", "link_group"], "static for plot_link_distributions")
    _require_keys(masks, ["flow_test"], "masks for plot_link_distributions")

    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    capacity = _to_numpy_1d(static["capacity"], "capacity")
    link_groups = _to_numpy_1d(static["link_group"], "link_group").astype(int)
    mask_test = _to_numpy_1d(masks["flow_test"], "flow_test").astype(bool)

    safe_capacity = capacity.copy()
    safe_capacity[safe_capacity == 0] = 1.0

    group_map = {0: "Multi-Lane", 1: "Motorway", 2: "Two_Lane", 3: "Rural", 4: "Connectors"}
    link_types_mapped = [group_map.get(g, f"Type_{g}") for g in link_groups]

    df_vis = pd.DataFrame(
        {
            "Pred_Volume": pred_flows,
            "Capacity": capacity,
            "VC_Ratio": pred_flows / safe_capacity,
            "Link_Type": link_types_mapped,
            "Is_Test": mask_test,
        }
    )

    plot_link_histograms(df_vis, model_name, output_dir)


def plot_od_scatter(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_od"], "artifacts for plot_od_scatter")
    _require_keys(static, ["true_od"], "static for plot_od_scatter")

    pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
    true_od = _to_numpy_1d(static["true_od"], "true_od")

    mask_od = _resolve_od_known_mask(static, masks)

    plot_scatter_comparison(
        pred=pred_od,
        target=true_od,
        mask=mask_od,
        mask_label="Known OD (Input)",
        title=f"OD Demand: True vs Estimated ({model_name})",
        xlabel="True Demand",
        ylabel="Estimated Demand",
        output_path=os.path.join(output_dir, f"{model_name}_scatter_od.png"),
        log_scale=True,
    )


def export_standard_csvs(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["pred_flows"], "artifacts for export_standard_csvs")
    _require_keys(static, ["true_flows"], "static for export_standard_csvs")
    _require_keys(masks, ["flow_train", "flow_test"], "masks for export_standard_csvs")

    pred_flows = _to_numpy_1d(artifacts["pred_flows"], "pred_flows")
    true_flows = _to_numpy_1d(static["true_flows"], "true_flows")

    mask_observed_total = (
        _to_numpy_1d(masks["flow_train"], "flow_train").astype(bool)
        | _to_numpy_1d(masks["flow_test"], "flow_test").astype(bool)
    )

    flow_extras = {
        "capacity": _to_numpy_1d(static["capacity"], "capacity") if "capacity" in static else None,
        "t0": _to_numpy_1d(static["t0"], "t0") if "t0" in static else None,
        "link_group": _to_numpy_1d(static["link_group"], "link_group") if "link_group" in static else None,
    }

    export_flows_csv(
        pred_flows=pred_flows,
        true_flows=true_flows,
        mask_observed=mask_observed_total,
        output_path=os.path.join(output_dir, f"{model_name}_flows_detailed.csv"),
        extras=flow_extras,
    )

    if "pred_od" in artifacts and "true_od" in static and "od_pair_indices" in static:
        pred_od = _to_numpy_1d(artifacts["pred_od"], "pred_od")
        true_od = _to_numpy_1d(static["true_od"], "true_od")
        od_indices = static["od_pair_indices"]
        if hasattr(od_indices, "detach"):
            od_indices = od_indices.detach().cpu().numpy()
        else:
            od_indices = np.asarray(od_indices)

        mask_od_known = _resolve_od_known_mask(static, masks)

        export_od_analysis_csv(
            pred_demand=pred_od,
            true_demand=true_od,
            od_indices=od_indices,
            output_path=os.path.join(output_dir, f"{model_name}_od_analysis.csv"),
            mask_known=mask_od_known,
            num_centroids=int(static.get("num_centroids", 0)),
        )


def plot_cgame_penalty(*, artifacts: Dict[str, Any], static: Dict[str, Any], masks: Dict[str, Any], output_dir: str, model_name: str) -> None:
    _require_keys(artifacts, ["historic_penalty"], "artifacts for plot_cgame_penalty")
    penalty = _to_numpy_1d(artifacts["historic_penalty"], "historic_penalty")
    out_path = os.path.join(output_dir, f"{model_name}_penalty_summary.csv")
    pd.DataFrame({"historic_penalty": penalty}).to_csv(out_path, index=False)
