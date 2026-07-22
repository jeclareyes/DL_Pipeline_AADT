"""Reusable assignment runner for model testing and evaluation.

This module is the single entry point for running traffic assignment from
predicted OD artifacts. It prepares the OD matrix, applies learned VDF
parameters when explicitly available, builds the assignment composition, runs
it, and returns a rich bundle for downstream evaluation tasks.

Design principles
-----------------
1. No silent reconstruction of critical mappings.
2. No silent overwriting of duplicated OD predictions.
3. No fallback from invalid learned VDF parameter dimensions to alpha[0]/beta[0].
4. Assignment composition remains delegated to src.components.assignment_motors.
5. The returned bundle exposes all important objects needed by evaluation code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.components.assignment_motors import (
    AssignmentConfig,
    AssignmentResult,
    build_assignment_composition,
    build_assignment_config_from_mapping,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class AssignmentRunBundle:
    """Complete output of one assignment run.

    Attributes
    ----------
    result:
        AssignmentResult returned by the selected behavior model.
    links_df:
        Link table actually used for assignment, including any explicit learned
        VDF parameters injected by this runner.
    assignment_matrix:
        Square OD matrix reconstructed from artifacts["pred_od"] and
        static["od_pair_indices"].
    assignment_config:
        Strict AssignmentConfig built from assign_cfg_dict.
    model_name:
        User-facing model name used in logs and error messages.
    zone_id_to_idx:
        Explicit zone-ID to OD-matrix-index mapping used by the assignment.
    """

    result: AssignmentResult
    links_df: pd.DataFrame
    assignment_matrix: np.ndarray
    assignment_config: AssignmentConfig
    model_name: str
    zone_id_to_idx: dict[int, int]


def prepare_and_run_assignment(
    artifacts: Mapping[str, Any],
    static: Mapping[str, Any],
    assign_cfg_dict: Mapping[str, Any],
    model_name: str,
) -> AssignmentRunBundle:
    """Prepare data and run route-based assignment from model artifacts.

    Parameters
    ----------
    artifacts:
        Model output artifacts. Must contain ``pred_od``. May contain
        ``learned_alpha`` and ``learned_beta`` as either scalars, per-link
        vectors, or per-link-group vectors when ``static["link_group"]`` is
        available.
    static:
        Static testing data. Must contain ``raw_data`` and ``od_pair_indices``.
        The zone mapping must be explicitly available in one of the supported
        artifact paths resolved by ``_resolve_zone_id_to_idx``.
    assign_cfg_dict:
        Raw assignment configuration mapping. It is converted to a strict
        AssignmentConfig before any assignment object is built.
    model_name:
        Name used only for logs and contextual error messages.

    Returns
    -------
    AssignmentRunBundle
        Rich object containing the assignment result and all main prepared
        inputs used for the run.
    """
    _validate_top_level_inputs(
        artifacts=artifacts,
        static=static,
        assign_cfg_dict=assign_cfg_dict,
        model_name=model_name,
    )

    assignment_config = _build_assignment_config(
        assign_cfg_dict=assign_cfg_dict,
        model_name=model_name,
    )

    raw_data = _require_mapping(
        static["raw_data"],
        context=f"[{model_name}] static['raw_data']",
    )

    processed = _resolve_processed_layer(
        raw_data=raw_data,
        model_name=model_name,
    )

    links_df = _get_links_df(
        processed=processed,
        model_name=model_name,
    )

    routes_by_od = _get_routes_by_od(
        processed=processed,
        model_name=model_name,
    )

    zone_id_to_idx = _resolve_zone_id_to_idx(
        static=static,
        raw_data=raw_data,
        model_name=model_name,
    )

    pred_od = _to_numpy_1d(
        artifacts["pred_od"],
        name=f"[{model_name}] pred_od",
    )

    od_pair_indices = _to_numpy_od_pair_indices(
        static["od_pair_indices"],
        name=f"[{model_name}] od_pair_indices",
    )

    assignment_matrix = build_assignment_matrix_from_predictions(
        pred_od=pred_od,
        od_pair_indices=od_pair_indices,
        zone_id_to_idx=zone_id_to_idx,
        model_name=model_name,
    )

    links_df = apply_learned_vdf_parameters_if_present(
        links_df=links_df,
        artifacts=artifacts,
        static=static,
        assignment_config=assignment_config,
        model_name=model_name,
    )

    training_config = _resolve_training_config(
        raw_data=raw_data,
        model_name=model_name,
    )

    LOGGER.info("[%s] Building assignment composition.", model_name)
    composition = build_assignment_composition(
        links_df=links_df,
        routes_by_od=routes_by_od,
        zone_id_to_idx=zone_id_to_idx,
        assignment_config=assignment_config,
        training_config=training_config,
        artifacts=artifacts,
    )

    LOGGER.info("[%s] Solving route-based assignment.", model_name)
    result = composition.behavior_model.solve(
        od_matrix=assignment_matrix,
        config=composition.runtime_config,
    )

    if not isinstance(result, AssignmentResult):
        raise TypeError(
            f"[{model_name}] behavior_model.solve(...) must return AssignmentResult. "
            f"Received {type(result).__name__}."
        )

    return AssignmentRunBundle(
        result=result,
        links_df=links_df,
        assignment_matrix=assignment_matrix,
        assignment_config=assignment_config,
        model_name=model_name,
        zone_id_to_idx=zone_id_to_idx,
    )


def build_assignment_matrix_from_predictions(
    *,
    pred_od: np.ndarray,
    od_pair_indices: np.ndarray,
    zone_id_to_idx: Mapping[int, int],
    model_name: str,
) -> np.ndarray:
    """Reconstruct a square OD matrix from vector predictions and OD indices.

    Duplicate OD indices are rejected. Silent overwriting is not allowed because
    it can hide upstream data-preparation bugs.

    Important
    ---------
    ``od_pair_indices`` are matrix indices, not necessarily real zone IDs.
    ``zone_id_to_idx`` defines the expected matrix size and is validated before
    this function is called.
    """
    zone_id_to_idx = _normalize_zone_id_to_idx(
        zone_id_to_idx=zone_id_to_idx,
        model_name=model_name,
    )
    num_zones = len(zone_id_to_idx)

    if pred_od.ndim != 1:
        raise ValueError(
            f"[{model_name}] pred_od must be one-dimensional. "
            f"Received shape={pred_od.shape}."
        )

    if od_pair_indices.ndim != 2 or od_pair_indices.shape[1] != 2:
        raise ValueError(
            f"[{model_name}] od_pair_indices must have shape [N, 2]. "
            f"Received shape={od_pair_indices.shape}."
        )

    if len(pred_od) != len(od_pair_indices):
        raise ValueError(
            f"[{model_name}] pred_od length must match od_pair_indices rows. "
            f"len(pred_od)={len(pred_od)}, "
            f"len(od_pair_indices)={len(od_pair_indices)}."
        )

    if not np.all(np.isfinite(pred_od)):
        raise ValueError(f"[{model_name}] pred_od contains NaN or infinite values.")

    if np.any(pred_od < 0.0):
        raise ValueError(f"[{model_name}] pred_od contains negative OD demand values.")

    if not np.all(np.isfinite(od_pair_indices)):
        raise ValueError(f"[{model_name}] od_pair_indices contains NaN or infinite values.")

    od_pair_indices_int = od_pair_indices.astype(int, copy=False)

    if not np.array_equal(od_pair_indices, od_pair_indices_int):
        raise ValueError(
            f"[{model_name}] od_pair_indices must contain integer matrix indices only."
        )

    if np.any(od_pair_indices_int < 0):
        raise ValueError(f"[{model_name}] od_pair_indices contains negative matrix indices.")

    if np.any(od_pair_indices_int >= num_zones):
        max_index = int(np.max(od_pair_indices_int))
        raise ValueError(
            f"[{model_name}] od_pair_indices references index {max_index}, "
            f"but len(zone_id_to_idx)={num_zones}."
        )

    pairs_as_tuples = [
        tuple(map(int, pair))
        for pair in od_pair_indices_int.tolist()
    ]

    duplicated_pairs = _find_duplicates(pairs_as_tuples)

    if duplicated_pairs:
        raise ValueError(
            f"[{model_name}] od_pair_indices contains duplicated OD matrix indices. "
            f"First duplicates: {duplicated_pairs[:10]}. "
            "Refusing to overwrite predictions silently."
        )

    assignment_matrix = np.zeros((num_zones, num_zones), dtype=float)
    assignment_matrix[
        od_pair_indices_int[:, 0],
        od_pair_indices_int[:, 1],
    ] = pred_od.astype(float, copy=False)

    return assignment_matrix


def apply_learned_vdf_parameters_if_present(
    *,
    links_df: pd.DataFrame,
    artifacts: Mapping[str, Any],
    static: Mapping[str, Any],
    assignment_config: AssignmentConfig,
    model_name: str,
) -> pd.DataFrame:
    """Inject learned VDF parameters into the configured link-table columns.

    The function is generic over the active VDF contract. It looks at
    ``assignment_config.cost_function.parameters_cols`` and maps each configured
    parameter name to a corresponding artifact key of the form
    ``learned_<parameter_name_without_suffix>``. For example:

    - ``alpha_col`` -> ``learned_alpha``
    - ``beta_col`` -> ``learned_beta``
    - ``J_col`` -> ``learned_j``

    Accepted shapes for each parameter are:
    - scalar: apply the same explicit value to every link;
    - length num_links: apply per link;
    - length num_groups with static['link_group']: apply by group label.
    """
    result = links_df.copy(deep=True)
    link_group = None

    if "link_group" in static and static["link_group"] is not None:
        link_group = _to_numpy_1d(
            static["link_group"],
            name=f"[{model_name}] link_group",
        ).astype(int, copy=False)

        if len(link_group) != len(links_df):
            raise ValueError(
                f"[{model_name}] link_group length must match links_df rows. "
                f"len(link_group)={len(link_group)}, len(links_df)={len(links_df)}."
            )

        if np.any(link_group < 0):
            raise ValueError(f"[{model_name}] link_group contains negative group labels.")

    # Apply learned capacity multiplier if present.
    if "learned_capacity_multiplier" in artifacts:
        cap_mult = _to_numpy_1d(
            artifacts["learned_capacity_multiplier"],
            name=f"[{model_name}] learned_capacity_multiplier",
        )

        if not np.all(np.isfinite(cap_mult)):
            raise ValueError(f"[{model_name}] learned_capacity_multiplier contains NaN or infinite values.")

        if np.any(cap_mult <= 0.0):
            raise ValueError(f"[{model_name}] learned_capacity_multiplier contains non-positive capacity multiplier values.")

        cap_mult_values = _expand_vdf_parameter_to_links(
            parameter=cap_mult,
            parameter_name="learned_capacity_multiplier",
            num_links=len(links_df),
            link_group=link_group,
            model_name=model_name,
        )

        cap_col = assignment_config.cost_function.capacity_col
        result[cap_col] = result[cap_col].astype(float) * cap_mult_values
        result["learned_capacity_multiplier"] = cap_mult_values

        LOGGER.info(
            "[%s] Injected learned capacity multiplier to column %r.",
            model_name,
            cap_col,
        )

    parameter_columns = dict(assignment_config.cost_function.parameters_cols)
    if not parameter_columns:
        return result

    applied_parameters: list[str] = []
    for parameter_name, column_name in parameter_columns.items():
        base_name = parameter_name[:-4] if parameter_name.endswith("_col") else parameter_name
        artifact_key_candidates = (
            f"learned_{base_name}",
            f"learned_{base_name.lower()}",
            f"learned_{parameter_name}",
        )

        artifact_key = next((key for key in artifact_key_candidates if key in artifacts), None)
        if artifact_key is None:
            continue

        parameter = _to_numpy_1d(
            artifacts[artifact_key],
            name=f"[{model_name}] {artifact_key}",
        )

        if not np.all(np.isfinite(parameter)):
            raise ValueError(f"[{model_name}] {artifact_key} contains NaN or infinite values.")

        expanded_values = _expand_vdf_parameter_to_links(
            parameter=parameter,
            parameter_name=artifact_key,
            num_links=len(links_df),
            link_group=link_group,
            model_name=model_name,
        )

        result[column_name] = expanded_values
        result[artifact_key] = expanded_values
        applied_parameters.append(f"{artifact_key}->{column_name}")

    if applied_parameters:
        LOGGER.info(
            "[%s] Injected learned VDF parameters: %s",
            model_name,
            ", ".join(applied_parameters),
        )

    return result


def _resolve_zone_id_to_idx(
    *,
    static: Mapping[str, Any],
    raw_data: Mapping[str, Any],
    model_name: str,
) -> dict[int, int]:
    """Resolve zone_id_to_idx from explicit artifact metadata only.

    Resolution priority
    -------------------
    1. static["zone_id_to_idx"]
    2. raw_data["processed"]["od_indexing"]["zone_id_to_idx"]
    3. raw_data["od_indexing"]["zone_id_to_idx"]
    4. raw_data["metadata"]["trips"]["zone_id_to_idx"]

    The final path is accepted only as a transitional compatibility location.
    This function never derives the mapping from od_pair_indices, OD matrix
    shape, sorted zone IDs, or route keys.
    """
    candidate = static.get("zone_id_to_idx")

    if candidate is None:
        processed = raw_data.get("processed")
        if isinstance(processed, Mapping):
            od_indexing = processed.get("od_indexing")
            if isinstance(od_indexing, Mapping):
                candidate = od_indexing.get("zone_id_to_idx")

    if candidate is None:
        od_indexing = raw_data.get("od_indexing")
        if isinstance(od_indexing, Mapping):
            candidate = od_indexing.get("zone_id_to_idx")

    if candidate is None:
        metadata = raw_data.get("metadata")
        if isinstance(metadata, Mapping):
            trips_metadata = metadata.get("trips")
            if isinstance(trips_metadata, Mapping):
                candidate = trips_metadata.get("zone_id_to_idx")

    if candidate is None:
        raise ValueError(
            f"[{model_name}] zone_id_to_idx is missing. Expected one of: "
            "static['zone_id_to_idx'], "
            "raw_data['processed']['od_indexing']['zone_id_to_idx'], "
            "raw_data['od_indexing']['zone_id_to_idx'], or "
            "raw_data['metadata']['trips']['zone_id_to_idx']. "
            "The assignment runner does not infer zone mappings from "
            "od_pair_indices, OD matrix shape, sorted route keys, or any other fallback."
        )

    return _normalize_zone_id_to_idx(
        zone_id_to_idx=candidate,
        model_name=model_name,
    )


def _normalize_zone_id_to_idx(
    *,
    zone_id_to_idx: Any,
    model_name: str,
) -> dict[int, int]:
    """Normalize and validate a zone-ID to OD-matrix-index mapping."""
    if not isinstance(zone_id_to_idx, Mapping) or len(zone_id_to_idx) == 0:
        raise ValueError(
            f"[{model_name}] zone_id_to_idx must be a non-empty mapping."
        )

    normalized = {
        int(zone_id): int(index)
        for zone_id, index in zone_id_to_idx.items()
    }

    if len(normalized) != len(zone_id_to_idx):
        raise ValueError(
            f"[{model_name}] zone_id_to_idx contains duplicated zone IDs "
            "after integer conversion."
        )

    zone_ids = list(normalized.keys())
    indices = list(normalized.values())

    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError(
            f"[{model_name}] zone_id_to_idx contains duplicated zone IDs."
        )

    if len(indices) != len(set(indices)):
        raise ValueError(
            f"[{model_name}] zone_id_to_idx contains duplicated matrix indices."
        )

    if min(indices) < 0:
        raise ValueError(
            f"[{model_name}] zone_id_to_idx contains negative matrix indices."
        )

    expected_indices = set(range(len(indices)))
    actual_indices = set(indices)

    if actual_indices != expected_indices:
        raise ValueError(
            f"[{model_name}] zone_id_to_idx indices must be contiguous and zero-based. "
            f"Missing={sorted(expected_indices.difference(actual_indices))[:10]}, "
            f"extra={sorted(actual_indices.difference(expected_indices))[:10]}."
        )

    return normalized


def _resolve_processed_layer(
    *,
    raw_data: Mapping[str, Any],
    model_name: str,
) -> Mapping[str, Any]:
    """Resolve the processed artifact layer without guessing its contents.

    Supported shapes
    ----------------
    1. Full artifact-like object:
       raw_data["processed"]["link_df"]

    2. Processed-layer object directly:
       raw_data["link_df"]

    The second shape is useful for TrainingArtifactLoader.load_processed().
    """
    if "processed" in raw_data:
        return _require_mapping(
            raw_data["processed"],
            context=f"[{model_name}] raw_data['processed']",
        )

    if "link_df" in raw_data and "routes_by_od" in raw_data:
        return raw_data

    raise ValueError(
        f"[{model_name}] Could not resolve processed layer. Expected either "
        "raw_data['processed'] or a processed-layer mapping containing "
        "'link_df' and 'routes_by_od'."
    )


def _resolve_training_config(
    *,
    raw_data: Mapping[str, Any],
    model_name: str,
) -> Mapping[str, Any]:
    """Resolve optional training/config mapping from the raw data payload."""
    training_config = raw_data.get("config", {})

    if training_config is None:
        return {}

    if not isinstance(training_config, Mapping):
        raise TypeError(
            f"[{model_name}] raw_data['config'] must be a mapping when present."
        )

    return training_config


def _expand_vdf_parameter_to_links(
    *,
    parameter: np.ndarray,
    parameter_name: str,
    num_links: int,
    link_group: np.ndarray | None,
    model_name: str,
) -> np.ndarray:
    """Expand scalar, per-link, or per-group VDF parameters to link length."""
    if parameter.size == 1:
        return np.full(num_links, float(parameter[0]), dtype=float)

    if parameter.size == num_links:
        return parameter.astype(float, copy=True)

    if link_group is not None:
        unique_groups = np.array(
            sorted(np.unique(link_group).astype(int).tolist()),
            dtype=int,
        )

        if parameter.size == len(unique_groups):
            group_to_position = {
                int(group): position
                for position, group in enumerate(unique_groups.tolist())
            }

            expanded = np.empty(num_links, dtype=float)

            for link_idx, group_label in enumerate(link_group.tolist()):
                expanded[link_idx] = float(
                    parameter[group_to_position[int(group_label)]]
                )

            return expanded

    expected = ["1 scalar", f"{num_links} per-link values"]

    if link_group is not None:
        expected.append(f"{len(np.unique(link_group))} per-group values")

    raise ValueError(
        f"[{model_name}] {parameter_name} has unsupported length {parameter.size}. "
        f"Expected {' or '.join(expected)}."
    )


def _validate_top_level_inputs(
    *,
    artifacts: Mapping[str, Any],
    static: Mapping[str, Any],
    assign_cfg_dict: Mapping[str, Any],
    model_name: str,
) -> None:
    """Validate top-level runner inputs before touching nested objects."""
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    if not isinstance(artifacts, Mapping):
        raise TypeError(f"[{model_name}] artifacts must be a mapping.")

    if not isinstance(static, Mapping):
        raise TypeError(f"[{model_name}] static must be a mapping.")

    if not isinstance(assign_cfg_dict, Mapping):
        raise TypeError(f"[{model_name}] assign_cfg_dict must be a mapping.")

    if "pred_od" not in artifacts:
        raise ValueError(f"[{model_name}] artifacts['pred_od'] is required.")

    if "raw_data" not in static:
        raise ValueError(f"[{model_name}] static['raw_data'] is required.")

    if "od_pair_indices" not in static:
        raise ValueError(f"[{model_name}] static['od_pair_indices'] is required.")


def _build_assignment_config(
    *,
    assign_cfg_dict: Mapping[str, Any],
    model_name: str,
) -> AssignmentConfig:
    """Build strict assignment config with contextual errors."""
    try:
        assignment_config = build_assignment_config_from_mapping(assign_cfg_dict)
    except Exception as exc:
        raise ValueError(
            f"[{model_name}] Failed to build assignment config: {exc}"
        ) from exc

    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError(
            f"[{model_name}] build_assignment_config_from_mapping returned "
            f"{type(assignment_config).__name__}, expected AssignmentConfig."
        )

    assignment_config.validate()
    return assignment_config


def _get_links_df(
    *,
    processed: Mapping[str, Any],
    model_name: str,
) -> pd.DataFrame:
    """Extract and copy the canonical link table."""
    if "link_df" not in processed:
        raise ValueError(f"[{model_name}] processed['link_df'] is required.")

    links_df = processed["link_df"]

    if not isinstance(links_df, pd.DataFrame):
        raise TypeError(f"[{model_name}] processed['link_df'] must be a pandas DataFrame.")

    if links_df.empty:
        raise ValueError(f"[{model_name}] processed['link_df'] cannot be empty.")

    return links_df.copy(deep=True)


def _get_routes_by_od(
    *,
    processed: Mapping[str, Any],
    model_name: str,
) -> Mapping[tuple[int, int], Any]:
    """Extract route alternatives for assignment."""
    if "routes_by_od" not in processed:
        raise ValueError(f"[{model_name}] processed['routes_by_od'] is required.")

    routes_by_od = processed["routes_by_od"]

    if not isinstance(routes_by_od, Mapping):
        raise TypeError(f"[{model_name}] processed['routes_by_od'] must be a mapping.")

    if len(routes_by_od) == 0:
        raise ValueError(f"[{model_name}] processed['routes_by_od'] cannot be empty.")

    return routes_by_od


def _to_numpy_1d(value: Any, name: str) -> np.ndarray:
    """Convert tensors, lists, or arrays into a finite one-dimensional array."""
    array = _to_numpy(value)

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    array = np.asarray(array, dtype=float).reshape(-1)

    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or infinite values.")

    return array


def _to_numpy_od_pair_indices(value: Any, name: str) -> np.ndarray:
    """Convert OD index pairs to a strict two-column numpy array."""
    array = _to_numpy(value)

    if array.size == 0:
        raise ValueError(f"{name} cannot be empty.")

    array = np.asarray(array)

    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(
            f"{name} must have shape [N, 2]. Received shape={array.shape}."
        )

    return array


def _to_numpy(value: Any) -> np.ndarray:
    """Convert common tensor-like objects into numpy without importing torch."""
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()

    return np.asarray(value)


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Validate that a nested object is a mapping."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{context} must be a mapping.")

    return value


def _find_duplicates(values: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return duplicate tuple values while preserving first duplicate discovery order."""
    seen: set[tuple[int, int]] = set()
    duplicates: list[tuple[int, int]] = []
    duplicate_seen: set[tuple[int, int]] = set()

    for value in values:
        if value in seen and value not in duplicate_seen:
            duplicates.append(value)
            duplicate_seen.add(value)

        seen.add(value)

    return duplicates


__all__ = [
    "AssignmentRunBundle",
    "apply_learned_vdf_parameters_if_present",
    "build_assignment_matrix_from_predictions",
    "prepare_and_run_assignment",
]
