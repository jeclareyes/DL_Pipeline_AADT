# src/data_handling/validators/training_artifact_validator.py
from __future__ import annotations

"""
Training Artifact Validator
===========================

This module validates the structural and dimensional consistency of the base
artifact and the downstream training artifact.

Project context
---------------
In the AADT / traffic assignment pipeline, the data-processing stage produces a
base artifact and the asset pipeline can materialize a downstream
training_artifact.joblib file containing:

- raw reader outputs;
- processed transportation objects;
- optionally a model-ready layer;
- flow and OD targets;
- metadata and reproducibility information.

Before the artifact is saved or used by the training pipeline, it should be
validated. This validator checks that:

- required artifact sections exist;
- link dimensions are consistent across link_df, graph, tensors and targets;
- OD dimensions are consistent across routes, targets and model tensors;
- route tensors have compatible shapes;
- masks are binary and aligned with their targets;
- routes are structurally valid and compatible with the graph;
- physical tensors have valid values.

This module does not read TNTP files, build graphs, compute routes, create
targets, move tensors to devices, or save artifacts.

It supports both artifact flavors:

- `base_artifact`
  - Must not contain `model_ready`.
- `training_artifact`
  - Must contain `model_ready`.

Design principles
-----------------
- Validate, do not mutate.
- Fail early for severe inconsistencies.
- Report warnings for suspicious but non-fatal issues.
- Keep validation messages explicit and actionable.
- Support both strict and non-strict validation.
"""

from networkx.generators import spectral_graph_forge

from dataclasses import dataclass, field
import logging
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from scipy import sparse


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationIssue:
    """
    Single validation issue.

    Attributes
    ----------
    severity : str
        Either "error" or "warning".

    code : str
        Short machine-readable issue code.

    message : str
        Human-readable explanation.

    context : Dict[str, Any]
        Optional structured context useful for debugging.
    """

    severity: str
    code: str
    message: str
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrainingArtifactValidationResult:
    """
    Container returned by TrainingArtifactValidator.

    Attributes
    ----------
    is_valid : bool
        True if no validation errors were found.

    errors : List[ValidationIssue]
        Validation errors.

    warnings : List[ValidationIssue]
        Validation warnings.

    summary : Dict[str, Any]
        Lightweight validation summary.
    """

    is_valid: bool
    errors: List[ValidationIssue]
    warnings: List[ValidationIssue]
    summary: Dict[str, Any]


class TrainingArtifactValidator:
    """
    Validate a unified training artifact.

    Parameters
    ----------
    strict : bool, default=True
        If True, validation errors raise ValueError when calling
        validate_or_raise(). The validate() method always returns a structured
        result instead of raising.

    check_route_graph_compatibility : bool, default=True
        If True, each route edge is checked against the graph.

    check_tensor_values : bool, default=True
        If True, model-ready tensor values are checked for finite and
        non-negative values where appropriate.

    check_target_masks : bool, default=True
        If True, target arrays and their observation masks are checked for
        shape, finiteness, non-negativity and binary-mask consistency.

    check_physical_tensors : bool, default=True
        If True, network physical tensors such as travel times, capacities and
        lanes are checked for valid dimensions and values.

    max_reported_items : int, default=20
        Maximum number of problematic examples included in issue contexts.
    """

    REQUIRED_TOP_LEVEL_KEYS = {
        "artifact_type",
        "artifact_version",
        "dataset_name",
        "created_at",
        "config",
        "environment",
        "paths",
        "raw",
        "processed",
        "metadata",
    }

    REQUIRED_RAW_KEYS = {
        "nodes_df",
        "network_df",
        "flow_df",
        "trips_df",
        "od_matrix",
        "primary_source_route_set",
        "primary_source_routes_df",
        "metadata",
    }

    REQUIRED_PROCESSED_KEYS = {
        "link_df",
        "graph",
        "primary_source_route_set",
        "primary_source_routes_df",
        "trips_df",
        "od_matrix",
        "edge_indexing",
        "od_indexing",
    }

    REQUIRED_MODEL_READY_KEYS = {
        "network_params",
        "targets",
        "visualization",
    }

    REQUIRED_NETWORK_PARAM_KEYS = {
        "num_links",
        "num_od_pairs",
        "k_paths",
        "route_masks",
        "delta_matrix",
        "route_validity_mask",
        "od_pairs",
        "od_pair_indices",
        "edge_list",
        "edge_to_idx",
        "link_pair_indices",
        "t0",
        "capacity",
        "lanes",
    }

    REQUIRED_TARGET_KEYS = {
        "flows_target_np",
        "flows_observed_mask_np",
        "od_target_np",
        "od_observed_mask_np",
    }

    def __init__(
        self,
        strict: bool = True,
        check_route_graph_compatibility: bool = True,
        check_tensor_values: bool = True,
        check_target_masks: bool = True,
        check_physical_tensors: bool = True,
        max_reported_items: int = 20,
    ) -> None:
        self.strict = bool(strict)
        self.check_route_graph_compatibility = bool(check_route_graph_compatibility)
        self.check_tensor_values = bool(check_tensor_values)
        self.check_target_masks = bool(check_target_masks)
        self.check_physical_tensors = bool(check_physical_tensors)
        self.max_reported_items = int(max_reported_items)

        self._errors: List[ValidationIssue] = []
        self._warnings: List[ValidationIssue] = []

    def _validate_flow_target_edge_alignment(
        self,
        artifact: Dict[str, Any],
    ) -> None:
        """
        Validate that flow targets are aligned with model link order.

        This check prevents the common failure mode where:
            flows_target_np follows link_df/link_id order,
        while:
            reconstructed_flows follows network_params["link_pair_indices"] order.
        """

        processed = artifact["processed"]
        network_params = artifact["model_ready"]["network_params"]
        targets = artifact["model_ready"]["targets"]

        link_df = processed["link_df"]

        model_link_pairs = network_params.get("link_pair_indices")
        target_link_pairs = targets.get("flow_target_link_pair_indices")

        if model_link_pairs is None:
            self._add_error(
                code="MISSING_MODEL_LINK_PAIR_INDICES",
                message="network_params['link_pair_indices'] is required for alignment validation.",
            )
            return

        if target_link_pairs is None:
            self._add_error(
                code="MISSING_FLOW_TARGET_LINK_PAIR_INDICES",
                message=(
                    "targets['flow_target_link_pair_indices'] is required to verify "
                    "flow target alignment."
                ),
            )
            return

        model_link_pairs = self._to_numpy_array(model_link_pairs)
        target_link_pairs = self._to_numpy_array(target_link_pairs)

        if tuple(model_link_pairs.shape) != tuple(target_link_pairs.shape):
            self._add_error(
                code="FLOW_TARGET_LINK_PAIR_SHAPE_MISMATCH",
                message="Flow target link-pair shape differs from model link-pair shape.",
                context={
                    "model_shape": tuple(model_link_pairs.shape),
                    "target_shape": tuple(target_link_pairs.shape),
                },
            )
            return

        if not np.array_equal(
            model_link_pairs.astype(np.int64),
            target_link_pairs.astype(np.int64),
        ):
            mismatch_idx = np.where(
                np.any(
                    model_link_pairs.astype(np.int64)
                    != target_link_pairs.astype(np.int64),
                    axis=1,
                )
            )[0]

            sample = [
                {
                    "position": int(idx),
                    "model_edge": tuple(map(int, model_link_pairs[idx])),
                    "target_edge": tuple(map(int, target_link_pairs[idx])),
                }
                for idx in mismatch_idx[:self.max_reported_items]
            ]

            self._add_error(
                code="FLOW_TARGET_EDGE_ORDER_MISMATCH",
                message=(
                    "Flow targets are not aligned with model link order. "
                    "targets['flow_target_link_pair_indices'] must equal "
                    "network_params['link_pair_indices'] position by position."
                ),
                context={
                    "num_mismatches": int(len(mismatch_idx)),
                    "sample": sample,
                },
            )
            return

        flow_column = processed.get("selected_flow_column")
        if not isinstance(flow_column, str) or flow_column not in link_df.columns:
            return

        edge_to_flow = {}

        for row in link_df.itertuples(index=False):
            edge = (int(row.init_node), int(row.term_node))
            value = row._asdict()[flow_column]
            edge_to_flow[edge] = float(value) if pd.notna(value) else np.nan

        expected_raw = []

        missing_edges = []

        for idx, (u, v) in enumerate(model_link_pairs.astype(np.int64)):
            edge = (int(u), int(v))

            if edge not in edge_to_volume:
                missing_edges.append(
                    {
                        "position": int(idx),
                        "edge": edge,
                    }
                )
                expected_raw.append(np.nan)
                continue

            expected_raw.append(edge_to_flow[edge])

        if missing_edges:
            self._add_error(
                code="MODEL_EDGE_NOT_FOUND_IN_LINK_DF",
                message="Some model edges cannot be found in processed link_df.",
                context={
                    "count": len(missing_edges),
                    "sample": missing_edges[:self.max_reported_items],
                },
            )
            return

        expected_raw = np.asarray(expected_raw, dtype=np.float32)
        actual_raw = targets.get("flows_raw_np")

        if actual_raw is None:
            actual_raw = targets["flows_target_np"]

        actual_raw = np.asarray(actual_raw, dtype=np.float32)

        comparable_mask = ~np.isnan(expected_raw)

        if comparable_mask.any():
            if not np.allclose(
                actual_raw[comparable_mask],
                expected_raw[comparable_mask],
                rtol=1e-5,
                atol=1e-3,
            ):
                abs_diff = np.abs(
                    actual_raw[comparable_mask] - expected_raw[comparable_mask]
                )

                bad_local = np.where(abs_diff > 1e-3)[0]

                global_indices = np.where(comparable_mask)[0][bad_local]

                sample = [
                    {
                        "position": int(idx),
                        "edge": tuple(map(int, model_link_pairs[idx])),
                        "target_value": float(actual_raw[idx]),
                        "link_df_volume": float(expected_raw[idx]),
                        "abs_diff": float(abs(actual_raw[idx] - expected_raw[idx])),
                    }
                    for idx in global_indices[:self.max_reported_items]
                ]

                self._add_error(
                    code="FLOW_TARGET_VALUES_NOT_EDGE_ALIGNED",
                    message=(
                        "Flow target values do not match link_df volumes after "
                        "alignment to model link order."
                    ),
                    context={
                        "max_abs_diff": float(np.nanmax(abs_diff)),
                        "sample": sample,
                    },
                )


    @staticmethod
    def _to_numpy_array(value: Any) -> np.ndarray:
        """
        Convert tensors, lists or arrays to a NumPy array.
        """

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()

        return np.asarray(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self, artifact: Dict[str, Any]) -> TrainingArtifactValidationResult:
        """
        Validate a training artifact and return a structured result.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact produced by TrainingArtifactBuilder.

        Returns
        -------
        TrainingArtifactValidationResult
            Validation result containing errors, warnings and summary.
        """

        logger.info("Validating training artifact.")

        self._errors = []
        self._warnings = []

        self._validate_top_level_structure(artifact)

        # If the artifact is not even a dictionary or lacks major sections,
        # deeper checks would produce noisy errors.
        if self._errors:
            return self._build_result(artifact)

        self._validate_raw_structure(artifact)
        self._validate_processed_structure(artifact)

        artifact_type = artifact.get("artifact_type")
        has_model_ready = bool(artifact.get("model_ready"))

        if artifact_type == "training_artifact":
            if not has_model_ready:
                self._add_error(
                    code="TRAINING_ARTIFACT_MISSING_MODEL_READY",
                    message="training_artifact must include a non-empty model_ready section.",
                )
            else:
                self._validate_model_ready_structure(artifact)
        elif artifact_type == "base_artifact":
            if has_model_ready:
                self._add_error(
                    code="BASE_ARTIFACT_HAS_MODEL_READY",
                    message="base_artifact must not include a model_ready section.",
                )

        if self._errors:
            return self._build_result(artifact)

        if has_model_ready:
            self._validate_link_consistency(artifact)
            self._validate_od_consistency(artifact)
            
        base_indexing_error_count = len(self._errors)
        self._validate_od_indexing_payload(artifact)
        self._validate_od_matrix_zone_mapping_consistency(artifact)
        self._validate_raw_trips_zone_indexing_consistency(artifact)

        if not has_model_ready:
            if len(self._errors) == base_indexing_error_count:
                logger.info("Base-artifact indexing consistency validation passed.")
            else:
                logger.error("Base-artifact indexing consistency validation failed.")
        
        if has_model_ready:
            self._validate_route_tensor_consistency(artifact)
            if self.check_target_masks:
                self._validate_target_consistency(artifact)
            if self.check_physical_tensors:
                self._validate_physical_tensors(artifact)
            if self.check_tensor_values:
                self._validate_assignment_ground_truth(artifact)

            indexing_error_count = len(self._errors)
            self._validate_indexing_consistency(artifact)
            if len(self._errors) == indexing_error_count:
                logger.info("Training-artifact indexing consistency validation passed.")
            else:
                logger.error("Training-artifact indexing consistency validation failed.")

            self._validate_flow_target_edge_alignment(artifact)

        self._validate_link_df_order_against_processed_edge_indexing(artifact)

        if self.check_route_graph_compatibility:
            self._validate_routes_against_graph(artifact)

        result = self._build_result(artifact)

        if result.is_valid:
            artifact_label = (
                "Training artifact"
                if artifact_type == "training_artifact"
                else "Base artifact"
            )
            logger.info("%s validation passed.", artifact_label)
        else:
            artifact_label = (
                "Training artifact"
                if artifact_type == "training_artifact"
                else "Base artifact"
            )
            logger.error(
                "%s validation failed with %d errors and %d warnings.",
                artifact_label,
                len(result.errors),
                len(result.warnings),
            )

        return result

    def validate_or_raise(self, artifact: Dict[str, Any]) -> TrainingArtifactValidationResult:
        """
        Validate an artifact and raise ValueError if errors are found.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.

        Returns
        -------
        TrainingArtifactValidationResult
            Validation result when valid or when strict=False.

        Raises
        ------
        ValueError
            If validation errors are found and strict=True.
        """

        result = self.validate(artifact)

        if self.strict and not result.is_valid:
            formatted_errors = self.format_issues(result.errors)
            raise ValueError(
                "Training artifact validation failed:\n"
                f"{formatted_errors}"
            )

        return result

    @staticmethod
    def format_issues(issues: List[ValidationIssue]) -> str:
        """
        Format validation issues as readable text.

        Parameters
        ----------
        issues : List[ValidationIssue]
            Validation issues.

        Returns
        -------
        str
            Formatted issue report.
        """

        if not issues:
            return "No issues."

        lines = []

        for idx, issue in enumerate(issues, start=1):
            lines.append(
                f"{idx}. [{issue.severity.upper()}] {issue.code}: {issue.message}"
            )

            if issue.context:
                lines.append(f"   context={issue.context}")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Structure validation
    # ------------------------------------------------------------------

    def _validate_top_level_structure(self, artifact: Any) -> None:
        """
        Validate top-level artifact structure.

        Parameters
        ----------
        artifact : Any
            Object to validate.
        """

        if not isinstance(artifact, dict):
            self._add_error(
                code="ARTIFACT_NOT_DICT",
                message=f"Artifact must be a dictionary. Got {type(artifact)}.",
            )
            return

        self._require_keys(
            obj=artifact,
            required_keys=self.REQUIRED_TOP_LEVEL_KEYS,
            object_name="artifact",
        )

        if artifact.get("artifact_type") not in ("training_artifact", "base_artifact"):
            self._add_error(
                code="INVALID_ARTIFACT_TYPE",
                message="artifact_type must be 'training_artifact' or 'base_artifact'.",
                context={"artifact_type": artifact.get("artifact_type")},
            )

    def _validate_raw_structure(self, artifact: Dict[str, Any]) -> None:
        """
        Validate raw layer structure.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        raw = artifact.get("raw", {})

        if not isinstance(raw, dict):
            self._add_error(
                code="RAW_NOT_DICT",
                message="artifact['raw'] must be a dictionary.",
            )
            return

        self._require_keys(
            obj=raw,
            required_keys=self.REQUIRED_RAW_KEYS,
            object_name="artifact['raw']",
        )

    def _validate_processed_structure(self, artifact: Dict[str, Any]) -> None:
        """
        Validate processed layer structure.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        processed = artifact.get("processed", {})

        if not isinstance(processed, dict):
            self._add_error(
                code="PROCESSED_NOT_DICT",
                message="artifact['processed'] must be a dictionary.",
            )
            return

        self._require_keys(
            obj=processed,
            required_keys=self.REQUIRED_PROCESSED_KEYS,
            object_name="artifact['processed']",
        )

    def _validate_model_ready_structure(self, artifact: Dict[str, Any]) -> None:
        """
        Validate model-ready layer structure.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        model_ready = artifact.get("model_ready", {})

        if not isinstance(model_ready, dict):
            self._add_error(
                code="MODEL_READY_NOT_DICT",
                message="artifact['model_ready'] must be a dictionary.",
            )
            return

        self._require_keys(
            obj=model_ready,
            required_keys=self.REQUIRED_MODEL_READY_KEYS,
            object_name="artifact['model_ready']",
        )

        if "network_params" in model_ready and isinstance(model_ready["network_params"], dict):
            self._require_keys(
                obj=model_ready["network_params"],
                required_keys=self.REQUIRED_NETWORK_PARAM_KEYS,
                object_name="artifact['model_ready']['network_params']",
            )

        if "targets" in model_ready and isinstance(model_ready["targets"], dict):
            self._require_keys(
                obj=model_ready["targets"],
                required_keys=self.REQUIRED_TARGET_KEYS,
                object_name="artifact['model_ready']['targets']",
            )

    # ------------------------------------------------------------------
    # Dimensional consistency
    # ------------------------------------------------------------------

    def _validate_link_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate link dimensions across processed and model-ready sections.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        processed = artifact["processed"]
        model_ready = artifact["model_ready"]

        link_df = processed["link_df"]
        graph = processed["graph"]
        network_params = model_ready["network_params"]
        targets = model_ready["targets"]

        if not isinstance(link_df, pd.DataFrame):
            self._add_error(
                code="LINK_DF_NOT_DATAFRAME",
                message="processed['link_df'] must be a pandas DataFrame.",
            )
            return

        if not isinstance(graph, nx.DiGraph):
            self._add_error(
                code="GRAPH_NOT_DIGRAPH",
                message="processed['graph'] must be a networkx.DiGraph.",
                context={"type": str(type(graph))},
            )
            return

        num_links_df = len(link_df)
        num_links_graph = graph.number_of_edges()
        num_links_params = int(network_params["num_links"])
        num_links_flow_target = len(targets["flows_target_np"])
        num_links_flow_mask = len(targets["flows_observed_mask_np"])

        values = {
            "link_df": num_links_df,
            "graph_edges": num_links_graph,
            "network_params_num_links": num_links_params,
            "flows_target": num_links_flow_target,
            "flows_mask": num_links_flow_mask,
        }

        if len(set(values.values())) != 1:
            self._add_error(
                code="LINK_DIMENSION_MISMATCH",
                message="Link dimensions are inconsistent across artifact sections.",
                context=values,
            )

    def _validate_od_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate OD dimensions across routes, network params and targets.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        processed = artifact["processed"]
        model_ready = artifact["model_ready"]

        routes_by_od = processed["primary_source_route_set"]
        network_params = model_ready["network_params"]
        targets = model_ready["targets"]

        if not isinstance(routes_by_od, dict):
            self._add_error(
                code="ROUTES_BY_OD_NOT_DICT",
                message="processed['primary_source_route_set'] must be a dictionary.",
            )
            return

        num_od_routes = len(routes_by_od)
        num_od_params = int(network_params["num_od_pairs"])
        num_od_target = len(targets["od_target_np"])
        num_od_mask = len(targets["od_observed_mask_np"])

        values = {
            "primary_source_route_set": num_od_routes,
            "network_params_num_od_pairs": num_od_params,
            "od_target": num_od_target,
            "od_mask": num_od_mask,
        }

        if len(set(values.values())) != 1:
            self._add_error(
                code="OD_DIMENSION_MISMATCH",
                message="OD dimensions are inconsistent across artifact sections.",
                context=values,
            )

        if "od_pairs" in targets and "od_pairs" in network_params:
            target_pairs = self._normalize_od_pairs(targets["od_pairs"])
            param_pairs = self._normalize_od_pairs(network_params["od_pairs"])

            if target_pairs != param_pairs:
                self._add_error(
                    code="OD_PAIR_ORDER_MISMATCH",
                    message="OD pair order differs between targets and network_params.",
                    context={
                        "target_sample": target_pairs[:self.max_reported_items],
                        "network_param_sample": param_pairs[:self.max_reported_items],
                    },
                )

    def _validate_od_indexing_payload(self, artifact: Dict[str, Any]) -> None:
        """
        Validate processed['od_indexing'] as the canonical OD indexing payload.

        This check is intentionally strict because route-based assignment needs
        an explicit mapping between real zone IDs and OD-matrix positions. The
        assignment runner must not infer this mapping from sorted OD indices or
        from the OD-matrix shape.
        """
        processed = artifact["processed"]
        model_ready = artifact.get("model_ready", {})
        network_params = model_ready.get("network_params", {})
        targets = model_ready.get("targets", {})

        od_indexing = processed.get("od_indexing")
        if not isinstance(od_indexing, dict):
            self._add_error(
                code="OD_INDEXING_NOT_DICT",
                message="processed['od_indexing'] must be a dictionary.",
                context={"type": str(type(od_indexing))},
            )
            return

        required_keys = {
            "zone_ids",
            "zone_id_to_idx",
            "od_pairs",
            "od_pair_to_idx",
            "idx_to_od_pair",
        }
        missing_keys = required_keys.difference(od_indexing.keys())
        if missing_keys:
            self._add_error(
                code="OD_INDEXING_MISSING_KEYS",
                message="processed['od_indexing'] is missing required keys.",
                context={"missing_keys": sorted(missing_keys)},
            )
            return

        zone_ids_raw = od_indexing["zone_ids"]
        zone_id_to_idx_raw = od_indexing["zone_id_to_idx"]
        od_pairs_raw = od_indexing["od_pairs"]
        od_pair_to_idx_raw = od_indexing["od_pair_to_idx"]
        idx_to_od_pair_raw = od_indexing["idx_to_od_pair"]

        zone_ids = self._normalize_zone_ids_for_validation(zone_ids_raw)
        if not zone_ids:
            self._add_error(
                code="ZONE_IDS_INVALID",
                message="processed['od_indexing']['zone_ids'] must be a non-empty sequence of zone IDs.",
            )
            return

        if len(zone_ids) != len(set(zone_ids)):
            duplicated_zone_ids = self._find_duplicate_scalars(zone_ids)
            self._add_error(
                code="DUPLICATED_ZONE_IDS",
                message="processed['od_indexing']['zone_ids'] contains duplicated zone IDs.",
                context={"duplicates": duplicated_zone_ids[:self.max_reported_items]},
            )

        zone_id_to_idx = self._normalize_scalar_mapping_for_validation(
            zone_id_to_idx_raw,
            mapping_name="processed['od_indexing']['zone_id_to_idx']",
        )
        if zone_id_to_idx is None:
            return

        if set(zone_ids) != set(zone_id_to_idx.keys()):
            self._add_error(
                code="ZONE_IDS_MAPPING_MISMATCH",
                message=(
                    "processed['od_indexing']['zone_ids'] and "
                    "processed['od_indexing']['zone_id_to_idx'].keys() do not match."
                ),
                context={
                    "num_zone_ids": len(zone_ids),
                    "num_mapping_keys": len(zone_id_to_idx),
                    "missing_in_mapping": sorted(set(zone_ids).difference(zone_id_to_idx.keys()))[:self.max_reported_items],
                    "extra_in_mapping": sorted(set(zone_id_to_idx.keys()).difference(zone_ids))[:self.max_reported_items],
                },
            )

        zone_indices = list(zone_id_to_idx.values())
        if len(zone_indices) != len(set(zone_indices)):
            self._add_error(
                code="DUPLICATED_ZONE_MATRIX_INDICES",
                message="zone_id_to_idx contains duplicated OD-matrix indices.",
                context={"duplicates": self._find_duplicate_scalars(zone_indices)[:self.max_reported_items]},
            )
            return

        expected_zone_indices = set(range(len(zone_ids)))
        found_zone_indices = set(zone_indices)
        if found_zone_indices != expected_zone_indices:
            self._add_error(
                code="ZONE_MATRIX_INDICES_NOT_CONTIGUOUS",
                message="zone_id_to_idx values must be contiguous zero-based OD-matrix indices.",
                context={
                    "expected_count": len(expected_zone_indices),
                    "missing_indices": sorted(expected_zone_indices.difference(found_zone_indices))[:self.max_reported_items],
                    "extra_indices": sorted(found_zone_indices.difference(expected_zone_indices))[:self.max_reported_items],
                },
            )

        od_pairs = self._normalize_od_pairs(od_pairs_raw)
        if not od_pairs:
            self._add_error(
                code="OD_PAIRS_INVALID",
                message="processed['od_indexing']['od_pairs'] must be a non-empty sequence of OD pairs.",
            )
            return

        if len(od_pairs) != len(set(od_pairs)):
            self._add_error(
                code="DUPLICATED_OD_PAIRS",
                message="processed['od_indexing']['od_pairs'] contains duplicated OD pairs.",
                context={"duplicates": self._find_duplicate_pairs(od_pairs)[:self.max_reported_items]},
            )

        unknown_zone_records = [
            {"origin_id": int(origin), "destination_id": int(destination)}
            for origin, destination in od_pairs
            if origin not in zone_id_to_idx or destination not in zone_id_to_idx
        ]
        if unknown_zone_records:
            self._add_error(
                code="OD_PAIRS_REFERENCE_UNKNOWN_ZONES",
                message="Some processed OD pairs reference zones absent from zone_id_to_idx.",
                context={
                    "count": len(unknown_zone_records),
                    "sample": unknown_zone_records[:self.max_reported_items],
                },
            )

        routes_by_od_pairs = {
            (int(origin), int(destination))
            for origin, destination in processed["primary_source_route_set"].keys()
        }
        if set(od_pairs) != routes_by_od_pairs:
            self._add_error(
                code="OD_INDEXING_ROUTE_KEYS_MISMATCH",
                message="processed['od_indexing']['od_pairs'] does not match processed['primary_source_route_set'].keys().",
                context={
                    "missing_in_od_indexing": sorted(routes_by_od_pairs.difference(od_pairs))[:self.max_reported_items],
                    "extra_in_od_indexing": sorted(set(od_pairs).difference(routes_by_od_pairs))[:self.max_reported_items],
                },
            )

        od_pair_to_idx = self._normalize_od_pair_to_idx_for_validation(od_pair_to_idx_raw)
        if od_pair_to_idx is None:
            return

        idx_to_od_pair = self._normalize_idx_to_od_pair_for_validation(idx_to_od_pair_raw)
        if idx_to_od_pair is None:
            return

        if set(od_pair_to_idx.keys()) != set(od_pairs):
            self._add_error(
                code="OD_PAIR_TO_IDX_KEYS_MISMATCH",
                message="od_pair_to_idx keys must match od_indexing['od_pairs'].",
                context={
                    "missing_keys": sorted(set(od_pairs).difference(od_pair_to_idx.keys()))[:self.max_reported_items],
                    "extra_keys": sorted(set(od_pair_to_idx.keys()).difference(od_pairs))[:self.max_reported_items],
                },
            )

        od_indices = list(od_pair_to_idx.values())
        if len(od_indices) != len(set(od_indices)):
            self._add_error(
                code="DUPLICATED_OD_PAIR_INDICES",
                message="od_pair_to_idx contains duplicated OD-pair indices.",
                context={"duplicates": self._find_duplicate_scalars(od_indices)[:self.max_reported_items]},
            )
            return

        expected_od_indices = set(range(len(od_pairs)))
        found_od_indices = set(od_indices)
        if found_od_indices != expected_od_indices:
            self._add_error(
                code="OD_PAIR_INDICES_NOT_CONTIGUOUS",
                message="od_pair_to_idx values must be contiguous zero-based OD-pair indices.",
                context={
                    "expected_count": len(expected_od_indices),
                    "missing_indices": sorted(expected_od_indices.difference(found_od_indices))[:self.max_reported_items],
                    "extra_indices": sorted(found_od_indices.difference(expected_od_indices))[:self.max_reported_items],
                },
            )

        if set(idx_to_od_pair.keys()) != expected_od_indices:
            self._add_error(
                code="IDX_TO_OD_PAIR_KEYS_MISMATCH",
                message="idx_to_od_pair keys must be contiguous zero-based OD-pair indices.",
                context={
                    "missing_indices": sorted(expected_od_indices.difference(idx_to_od_pair.keys()))[:self.max_reported_items],
                    "extra_indices": sorted(set(idx_to_od_pair.keys()).difference(expected_od_indices))[:self.max_reported_items],
                },
            )

        inverse_from_od_pair_to_idx = {
            int(index): od_pair
            for od_pair, index in od_pair_to_idx.items()
        }
        mismatched_inverse_records = []
        for index, od_pair in idx_to_od_pair.items():
            expected_pair = inverse_from_od_pair_to_idx.get(index)
            if expected_pair != od_pair:
                mismatched_inverse_records.append(
                    {
                        "index": int(index),
                        "idx_to_od_pair": od_pair,
                        "od_pair_to_idx_inverse": expected_pair,
                    }
                )

        if mismatched_inverse_records:
            self._add_error(
                code="OD_PAIR_INDEXING_INVERSE_MISMATCH",
                message="od_pair_to_idx and idx_to_od_pair are not exact inverses.",
                context={
                    "count": len(mismatched_inverse_records),
                    "sample": mismatched_inverse_records[:self.max_reported_items],
                },
            )

        if "od_pairs" in network_params:
            network_param_pairs = self._normalize_od_pairs(network_params["od_pairs"])
            if network_param_pairs != od_pairs:
                self._add_error(
                    code="OD_INDEXING_NETWORK_PARAMS_ORDER_MISMATCH",
                    message="processed od_indexing['od_pairs'] order differs from network_params['od_pairs'].",
                    context={
                        "processed_sample": od_pairs[:self.max_reported_items],
                        "network_params_sample": network_param_pairs[:self.max_reported_items],
                    },
                )

        if "od_pairs" in targets:
            target_pairs = self._normalize_od_pairs(targets["od_pairs"])
            if target_pairs != od_pairs:
                self._add_error(
                    code="OD_INDEXING_TARGET_ORDER_MISMATCH",
                    message="processed od_indexing['od_pairs'] order differs from targets['od_pairs'].",
                    context={
                        "processed_sample": od_pairs[:self.max_reported_items],
                        "target_sample": target_pairs[:self.max_reported_items],
                    },
                )

    def _validate_od_matrix_zone_mapping_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate that processed['od_matrix'] dimensions match zone_id_to_idx.

        Assignment motors translate OD-matrix row/column positions back to real
        zone IDs through zone_id_to_idx. Therefore the matrix size and the
        explicit mapping size must match exactly.
        """
        processed = artifact["processed"]
        od_matrix = processed.get("od_matrix")
        od_indexing = processed.get("od_indexing", {})

        if not isinstance(od_indexing, dict):
            return
        if "zone_id_to_idx" not in od_indexing:
            return

        shape = getattr(od_matrix, "shape", None)
        if shape is None or len(shape) != 2:
            self._add_error(
                code="OD_MATRIX_NOT_2D",
                message="processed['od_matrix'] must be a two-dimensional matrix-like object.",
                context={"shape": None if shape is None else tuple(shape)},
            )
            return

        shape = tuple(int(value) for value in shape)
        if shape[0] != shape[1]:
            self._add_error(
                code="OD_MATRIX_NOT_SQUARE",
                message="processed['od_matrix'] must be square.",
                context={"shape": shape},
            )
            return

        zone_id_to_idx = self._normalize_scalar_mapping_for_validation(
            od_indexing["zone_id_to_idx"],
            mapping_name="processed['od_indexing']['zone_id_to_idx']",
        )
        if zone_id_to_idx is None:
            return

        if shape[0] != len(zone_id_to_idx):
            self._add_error(
                code="OD_MATRIX_ZONE_MAPPING_MISMATCH",
                message="OD matrix dimension does not match len(zone_id_to_idx).",
                context={
                    "od_matrix_shape": shape,
                    "len_zone_id_to_idx": len(zone_id_to_idx),
                },
            )

    def _validate_raw_trips_zone_indexing_consistency(
        self,
        artifact: Dict[str, Any],
    ) -> None:
        """
        Validate that processed OD indexing preserves trips-reader zone indexing.

        The trips reader defines the compact OD matrix space through:
            raw['metadata']['trips']['zone_ids']
            raw['metadata']['trips']['zone_id_to_idx']

        processed['od_indexing'] must preserve this mapping exactly.
        """

        raw = artifact.get("raw", {})
        processed = artifact.get("processed", {})

        raw_metadata = raw.get("metadata", {})
        trips_metadata = raw_metadata.get("trips", {})

        od_indexing = processed.get("od_indexing", {})

        if not isinstance(trips_metadata, dict):
            self._add_error(
                code="RAW_TRIPS_METADATA_NOT_DICT",
                message="raw['metadata']['trips'] must be a dictionary.",
                context={"type": str(type(trips_metadata))},
            )
            return

        if not isinstance(od_indexing, dict):
            self._add_error(
                code="PROCESSED_OD_INDEXING_NOT_DICT_FOR_TRIPS_CHECK",
                message="processed['od_indexing'] must be a dictionary for trips-zone validation.",
                context={"type": str(type(od_indexing))},
            )
            return

        required_trips_keys = {"zone_ids", "zone_id_to_idx"}
        missing_trips_keys = required_trips_keys - set(trips_metadata.keys())

        if missing_trips_keys:
            self._add_error(
                code="RAW_TRIPS_METADATA_MISSING_ZONE_INDEXING",
                message="raw['metadata']['trips'] is missing zone-indexing keys.",
                context={"missing_keys": sorted(missing_trips_keys)},
            )
            return

        required_processed_keys = {"zone_ids", "zone_id_to_idx"}
        missing_processed_keys = required_processed_keys - set(od_indexing.keys())

        if missing_processed_keys:
            self._add_error(
                code="PROCESSED_OD_INDEXING_MISSING_ZONE_INDEXING_FOR_TRIPS_CHECK",
                message="processed['od_indexing'] is missing zone-indexing keys.",
                context={"missing_keys": sorted(missing_processed_keys)},
            )
            return

        raw_zone_ids = self._normalize_zone_ids_for_validation(
            trips_metadata["zone_ids"]
        )

        processed_zone_ids = self._normalize_zone_ids_for_validation(
            od_indexing["zone_ids"]
        )

        if raw_zone_ids != processed_zone_ids:
            self._add_error(
                code="RAW_TRIPS_PROCESSED_ZONE_ID_ORDER_MISMATCH",
                message=(
                    "processed['od_indexing']['zone_ids'] does not preserve "
                    "raw['metadata']['trips']['zone_ids'] order. This can misalign "
                    "OD matrix positions, routes and OD targets."
                ),
                context={
                    "raw_trips_sample": raw_zone_ids[:self.max_reported_items],
                    "processed_sample": processed_zone_ids[:self.max_reported_items],
                },
            )

        raw_zone_id_to_idx = self._normalize_scalar_mapping_for_validation(
            trips_metadata["zone_id_to_idx"],
            mapping_name="raw['metadata']['trips']['zone_id_to_idx']",
        )

        processed_zone_id_to_idx = self._normalize_scalar_mapping_for_validation(
            od_indexing["zone_id_to_idx"],
            mapping_name="processed['od_indexing']['zone_id_to_idx']",
        )

        if raw_zone_id_to_idx is None or processed_zone_id_to_idx is None:
            return

        if raw_zone_id_to_idx != processed_zone_id_to_idx:
            mismatched_records = []

            all_zone_ids = sorted(
                set(raw_zone_id_to_idx.keys()).union(processed_zone_id_to_idx.keys())
            )

            for zone_id in all_zone_ids:
                raw_idx = raw_zone_id_to_idx.get(zone_id)
                processed_idx = processed_zone_id_to_idx.get(zone_id)

                if raw_idx != processed_idx:
                    mismatched_records.append(
                        {
                            "zone_id": int(zone_id),
                            "raw_trips_idx": None if raw_idx is None else int(raw_idx),
                            "processed_idx": None if processed_idx is None else int(processed_idx),
                        }
                    )

            self._add_error(
                code="RAW_TRIPS_PROCESSED_ZONE_MAPPING_MISMATCH",
                message=(
                    "processed['od_indexing']['zone_id_to_idx'] does not match "
                    "raw['metadata']['trips']['zone_id_to_idx']."
                ),
                context={
                    "num_mismatches": len(mismatched_records),
                    "sample": mismatched_records[:self.max_reported_items],
                },
            )

    @staticmethod
    def _normalize_zone_ids_for_validation(value: Any) -> List[int]:
        """Normalize a zone-ID sequence to integer IDs for validation."""
        if isinstance(value, np.ndarray):
            raw_values = value.reshape(-1).tolist()
        elif torch.is_tensor(value):
            raw_values = value.detach().cpu().numpy().reshape(-1).tolist()
        elif isinstance(value, (list, tuple)):
            raw_values = list(value)
        else:
            return []

        try:
            return [int(item) for item in raw_values]
        except Exception:
            return []

    def _normalize_scalar_mapping_for_validation(
        self,
        value: Any,
        mapping_name: str,
    ) -> Optional[Dict[int, int]]:
        """Normalize a scalar integer mapping and record validation errors."""
        if not isinstance(value, dict) or not value:
            self._add_error(
                code="SCALAR_MAPPING_INVALID",
                message=f"{mapping_name} must be a non-empty dictionary.",
                context={"type": str(type(value))},
            )
            return None

        normalized: Dict[int, int] = {}
        try:
            for raw_key, raw_value in value.items():
                key = int(raw_key)
                item = int(raw_value)
                if key in normalized:
                    self._add_error(
                        code="SCALAR_MAPPING_DUPLICATED_KEYS_AFTER_INT_CONVERSION",
                        message=f"{mapping_name} contains duplicated keys after integer conversion.",
                        context={"duplicated_key": key},
                    )
                    return None
                normalized[key] = item
        except Exception as exc:
            self._add_error(
                code="SCALAR_MAPPING_INT_CONVERSION_FAILED",
                message=f"{mapping_name} keys and values must be convertible to integers.",
                context={"error": str(exc)},
            )
            return None

        if any(index < 0 for index in normalized.values()):
            self._add_error(
                code="SCALAR_MAPPING_NEGATIVE_VALUES",
                message=f"{mapping_name} contains negative indices.",
            )

        return normalized

    def _normalize_od_pair_to_idx_for_validation(self, value: Any) -> Optional[Dict[Tuple[int, int], int]]:
        """Normalize od_pair_to_idx and record validation errors."""
        if not isinstance(value, dict) or not value:
            self._add_error(
                code="OD_PAIR_TO_IDX_INVALID",
                message="processed['od_indexing']['od_pair_to_idx'] must be a non-empty dictionary.",
                context={"type": str(type(value))},
            )
            return None

        normalized: Dict[Tuple[int, int], int] = {}
        try:
            for raw_pair, raw_index in value.items():
                pair = self._normalize_single_od_pair(raw_pair)
                index = int(raw_index)
                if pair in normalized:
                    self._add_error(
                        code="OD_PAIR_TO_IDX_DUPLICATED_KEYS_AFTER_NORMALIZATION",
                        message="od_pair_to_idx contains duplicated OD-pair keys after normalization.",
                        context={"duplicated_pair": pair},
                    )
                    return None
                normalized[pair] = index
        except Exception as exc:
            self._add_error(
                code="OD_PAIR_TO_IDX_NORMALIZATION_FAILED",
                message="Could not normalize od_pair_to_idx keys and values.",
                context={"error": str(exc)},
            )
            return None

        if any(index < 0 for index in normalized.values()):
            self._add_error(
                code="OD_PAIR_TO_IDX_NEGATIVE_INDICES",
                message="od_pair_to_idx contains negative indices.",
            )

        return normalized

    def _normalize_idx_to_od_pair_for_validation(self, value: Any) -> Optional[Dict[int, Tuple[int, int]]]:
        """Normalize idx_to_od_pair and record validation errors."""
        if not isinstance(value, dict) or not value:
            self._add_error(
                code="IDX_TO_OD_PAIR_INVALID",
                message="processed['od_indexing']['idx_to_od_pair'] must be a non-empty dictionary.",
                context={"type": str(type(value))},
            )
            return None

        normalized: Dict[int, Tuple[int, int]] = {}
        try:
            for raw_index, raw_pair in value.items():
                index = int(raw_index)
                pair = self._normalize_single_od_pair(raw_pair)
                if index in normalized:
                    self._add_error(
                        code="IDX_TO_OD_PAIR_DUPLICATED_KEYS_AFTER_INT_CONVERSION",
                        message="idx_to_od_pair contains duplicated indices after integer conversion.",
                        context={"duplicated_index": index},
                    )
                    return None
                normalized[index] = pair
        except Exception as exc:
            self._add_error(
                code="IDX_TO_OD_PAIR_NORMALIZATION_FAILED",
                message="Could not normalize idx_to_od_pair keys and values.",
                context={"error": str(exc)},
            )
            return None

        if any(index < 0 for index in normalized.keys()):
            self._add_error(
                code="IDX_TO_OD_PAIR_NEGATIVE_INDICES",
                message="idx_to_od_pair contains negative indices.",
            )

        return normalized

    @staticmethod
    def _normalize_single_od_pair(value: Any) -> Tuple[int, int]:
        """Normalize one OD-pair-like value to a tuple[int, int]."""
        if isinstance(value, str):
            stripped = value.strip().replace("(", "").replace(")", "")
            parts = [part.strip() for part in stripped.split(",") if part.strip()]
            if len(parts) != 2:
                raise ValueError(f"String OD pair must contain two comma-separated values. Received {value!r}.")
            return int(parts[0]), int(parts[1])

        if not isinstance(value, (list, tuple, np.ndarray)):
            raise TypeError(f"OD pair must be list, tuple, ndarray, or string. Got {type(value)}.")

        pair = list(value)
        if len(pair) != 2:
            raise ValueError(f"OD pair must contain exactly two values. Received {value!r}.")
        return int(pair[0]), int(pair[1])

    @staticmethod
    def _find_duplicate_scalars(values: Iterable[int]) -> List[int]:
        """Return duplicated scalar values preserving first duplicate discovery order."""
        seen = set()
        duplicates = []
        duplicate_seen = set()
        for value in values:
            if value in seen and value not in duplicate_seen:
                duplicates.append(int(value))
                duplicate_seen.add(value)
            seen.add(value)
        return duplicates

    @staticmethod
    def _find_duplicate_pairs(values: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Return duplicated OD pairs preserving first duplicate discovery order."""
        seen = set()
        duplicates = []
        duplicate_seen = set()
        for value in values:
            pair = (int(value[0]), int(value[1]))
            if pair in seen and pair not in duplicate_seen:
                duplicates.append(pair)
                duplicate_seen.add(pair)
            seen.add(pair)
        return duplicates

    def _validate_route_tensor_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate route tensor shapes.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        network_params = artifact["model_ready"]["network_params"]

        num_links = int(network_params["num_links"])
        num_od_pairs = int(network_params["num_od_pairs"])
        k_paths = int(network_params["k_paths"])

        route_masks = network_params["route_masks"]
        delta_matrix = network_params["delta_matrix"]
        route_validity_mask = network_params["route_validity_mask"]

        if not torch.is_tensor(route_masks):
            self._add_error(
                code="ROUTE_MASKS_NOT_TENSOR",
                message="network_params['route_masks'] must be a torch tensor.",
            )
        else:
            expected_shape = (num_od_pairs, k_paths, num_links)

            if tuple(route_masks.shape) != expected_shape:
                self._add_error(
                    code="ROUTE_MASKS_SHAPE_MISMATCH",
                    message="route_masks shape is inconsistent.",
                    context={
                        "found": tuple(route_masks.shape),
                        "expected": expected_shape,
                    },
                )

        if not torch.is_tensor(delta_matrix):
            self._add_error(
                code="DELTA_MATRIX_NOT_TENSOR",
                message="network_params['delta_matrix'] must be a torch tensor.",
            )
        else:
            expected_shape = (num_links, num_od_pairs * k_paths)

            if tuple(delta_matrix.shape) != expected_shape:
                self._add_error(
                    code="DELTA_MATRIX_SHAPE_MISMATCH",
                    message="delta_matrix shape is inconsistent.",
                    context={
                        "found": tuple(delta_matrix.shape),
                        "expected": expected_shape,
                    },
                )

        if not torch.is_tensor(route_validity_mask):
            self._add_error(
                code="ROUTE_VALIDITY_MASK_NOT_TENSOR",
                message="network_params['route_validity_mask'] must be a torch tensor.",
            )
        else:
            expected_shape = (num_od_pairs, k_paths)

            if tuple(route_validity_mask.shape) != expected_shape:
                self._add_error(
                    code="ROUTE_VALIDITY_MASK_SHAPE_MISMATCH",
                    message="route_validity_mask shape is inconsistent.",
                    context={
                        "found": tuple(route_validity_mask.shape),
                        "expected": expected_shape,
                    },
                )

            if route_validity_mask.sum().item() == 0:
                self._add_error(
                    code="NO_VALID_ROUTES",
                    message="route_validity_mask contains no valid route slots.",
                )

    def _validate_target_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate target arrays and masks.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        targets = artifact["model_ready"]["targets"]

        flow_target = targets["flows_target_np"]
        flow_mask = targets["flows_observed_mask_np"]
        od_target = targets["od_target_np"]
        od_mask = targets["od_observed_mask_np"]

        self._validate_numpy_array(
            array=flow_target,
            name="flows_target_np",
            expected_ndim=1,
        )

        self._validate_numpy_array(
            array=flow_mask,
            name="flows_observed_mask_np",
            expected_ndim=1,
        )

        self._validate_numpy_array(
            array=od_target,
            name="od_target_np",
            expected_ndim=1,
        )

        self._validate_numpy_array(
            array=od_mask,
            name="od_observed_mask_np",
            expected_ndim=1,
        )

        self._validate_binary_mask(flow_mask, "flows_observed_mask_np")
        self._validate_binary_mask(od_mask, "od_observed_mask_np")

        if self.check_tensor_values:
            if np.any(flow_target < 0):
                self._add_error(
                    code="NEGATIVE_FLOW_TARGETS",
                    message="flows_target_np contains negative values.",
                )

            if np.any(od_target < 0):
                self._add_error(
                    code="NEGATIVE_OD_TARGETS",
                    message="od_target_np contains negative values.",
                )

            if not np.isfinite(flow_target).all():
                self._add_error(
                    code="NONFINITE_FLOW_TARGETS",
                    message="flows_target_np contains NaN or infinite values.",
                )

            if not np.isfinite(od_target).all():
                self._add_error(
                    code="NONFINITE_OD_TARGETS",
                    message="od_target_np contains NaN or infinite values.",
                )

        if flow_mask.sum() == 0:
            self._add_warning(
                code="NO_OBSERVED_FLOW_TARGETS",
                message="No observed link-flow targets are available.",
            )

        if od_mask.sum() == 0:
            self._add_warning(
                code="NO_OBSERVED_OD_TARGETS",
                message="No observed OD targets are available.",
            )

    def _validate_physical_tensors(self, artifact: Dict[str, Any]) -> None:
        """
        Validate physical network tensors.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        if not self.check_tensor_values:
            return

        network_params = artifact["model_ready"]["network_params"]
        num_links = int(network_params["num_links"])

        tensor_specs = {
            "t0": {"positive": False, "non_negative": True},
            "capacity": {"positive": True, "non_negative": True},
            "lanes": {"positive": True, "non_negative": True},
        }

        optional_specs = {
            "length": {"positive": False, "non_negative": True},
            "speed": {"positive": False, "non_negative": True},
            "b": {"positive": False, "non_negative": False}, # TODO: if negative values means it's unknown or to be estimated
            "power": {"positive": False, "non_negative": False}, # TODO: if negative means unknown or to be estimated
            "toll": {"positive": False, "non_negative": True},
        }

        for key, spec in {**tensor_specs, **optional_specs}.items():
            if key not in network_params:
                if key in tensor_specs:
                    self._add_error(
                        code=f"MISSING_{key.upper()}",
                        message=f"network_params['{key}'] is required.",
                    )
                continue

            tensor = network_params[key]

            if not torch.is_tensor(tensor):
                self._add_error(
                    code=f"{key.upper()}_NOT_TENSOR",
                    message=f"network_params['{key}'] must be a torch tensor.",
                )
                continue

            if tensor.numel() != num_links:
                self._add_error(
                    code=f"{key.upper()}_LENGTH_MISMATCH",
                    message=f"network_params['{key}'] length must match num_links.",
                    context={
                        "found": int(tensor.numel()),
                        "expected": num_links,
                    },
                )

            if not torch.isfinite(tensor).all().item():
                self._add_error(
                    code=f"{key.upper()}_NONFINITE",
                    message=f"network_params['{key}'] contains NaN or infinite values.",
                )

            if spec.get("positive", False) and (tensor <= 0).any().item():
                self._add_error(
                    code=f"{key.upper()}_NONPOSITIVE",
                    message=f"network_params['{key}'] contains values <= 0.",
                )

            elif spec.get("non_negative", False) and (tensor < 0).any().item():
                self._add_error(
                    code=f"{key.upper()}_NEGATIVE",
                    message=f"network_params['{key}'] contains negative values.",
                )

    def _validate_assignment_ground_truth(self, artifact: Dict[str, Any]) -> None:
        """Validate generated assignment outputs without confusing them with dataset targets."""

        targets = artifact["model_ready"]["targets"]
        assignment = targets.get("assignment_ground_truth")
        if assignment is None:
            return
        if not isinstance(assignment, Mapping):
            self._add_error(
                code="ASSIGNMENT_GROUND_TRUTH_NOT_MAPPING",
                message="model_ready.targets['assignment_ground_truth'] must be a mapping.",
            )
            return

        network_params = artifact["model_ready"]["network_params"]
        expected_links = int(network_params["num_links"])
        for key in (
            "final_link_flows",
            "final_link_costs",
        ):
            if key not in assignment:
                self._add_error(
                    code=f"MISSING_ASSIGNMENT_{key.upper()}",
                    message=f"assignment_ground_truth is missing '{key}'.",
                )
                continue
            values = np.asarray(assignment[key])
            if values.ndim != 1 or len(values) != expected_links:
                self._add_error(
                    code=f"ASSIGNMENT_{key.upper()}_LENGTH_MISMATCH",
                    message=f"assignment_ground_truth['{key}'] must have length num_links.",
                    context={"found": int(values.size), "expected": expected_links},
                )
                continue
            if not np.isfinite(values).all():
                self._add_error(
                    code=f"NONFINITE_ASSIGNMENT_{key.upper()}",
                    message=f"assignment_ground_truth['{key}'] contains non-finite values.",
                )
            if key.endswith("flows") and np.any(values < 0.0):
                self._add_error(
                    code=f"NEGATIVE_ASSIGNMENT_{key.upper()}",
                    message=f"assignment_ground_truth['{key}'] contains negative values.",
                )
            if key.endswith("costs") and np.any(values <= 0.0):
                self._add_error(
                    code=f"NONPOSITIVE_ASSIGNMENT_{key.upper()}",
                    message=f"assignment_ground_truth['{key}'] must contain positive costs.",
                )
                continue

    def _validate_indexing_consistency(self, artifact: Dict[str, Any]) -> None:
        """
        Validate indexing payload consistency.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        processed = artifact["processed"]
        network_params = artifact["model_ready"]["network_params"]

        edge_indexing = processed["edge_indexing"]

        if "edge_list" in edge_indexing and "edge_list" in network_params:
            processed_edges = [
                tuple(map(int, edge))
                for edge in edge_indexing["edge_list"]
            ]

            param_edges = [
                tuple(map(int, edge))
                for edge in network_params["edge_list"]
            ]

            if processed_edges != param_edges:
                self._add_error(
                    code="EDGE_ORDER_MISMATCH",
                    message="Edge order differs between processed edge_indexing and network_params.",
                    context={
                        "processed_sample": processed_edges[:self.max_reported_items],
                        "network_params_sample": param_edges[:self.max_reported_items],
                    },
                )

        link_pair_indices = network_params.get("link_pair_indices")

        if isinstance(link_pair_indices, np.ndarray):
            expected_shape = (int(network_params["num_links"]), 2)

            if tuple(link_pair_indices.shape) != expected_shape:
                self._add_error(
                    code="LINK_PAIR_INDICES_SHAPE_MISMATCH",
                    message="link_pair_indices shape is inconsistent.",
                    context={
                        "found": tuple(link_pair_indices.shape),
                        "expected": expected_shape,
                    },
                )

        self._validate_flow_target_edge_alignment(artifact)


    def _validate_link_df_order_against_processed_edge_indexing(
        self,
        artifact: Dict[str, Any],
    ) -> None:
        """
        Validate that processed['edge_indexing'] follows link_df row order.

        In the refactored pipeline, link_df is the canonical source of link order.
        Therefore, processed['edge_indexing']['link_pair_indices'] must match
        link_df[['init_node', 'term_node']] position by position.
        """

        processed = artifact["processed"]

        link_df = processed.get("link_df")
        edge_indexing = processed.get("edge_indexing", {})

        if not isinstance(link_df, pd.DataFrame):
            self._add_error(
                code="LINK_DF_NOT_DATAFRAME_FOR_ORDER_CHECK",
                message="processed['link_df'] must be a pandas DataFrame for link-order validation.",
                context={"type": str(type(link_df))},
            )
            return

        required_columns = {"init_node", "term_node"}
        missing_columns = required_columns - set(link_df.columns)

        if missing_columns:
            self._add_error(
                code="LINK_DF_MISSING_ENDPOINT_COLUMNS_FOR_ORDER_CHECK",
                message="processed['link_df'] is missing endpoint columns required for order validation.",
                context={"missing_columns": sorted(missing_columns)},
            )
            return

        if "link_pair_indices" not in edge_indexing:
            self._add_error(
                code="PROCESSED_EDGE_INDEXING_MISSING_LINK_PAIR_INDICES",
                message=(
                    "processed['edge_indexing']['link_pair_indices'] is required "
                    "to validate canonical link order."
                ),
            )
            return

        link_df_pairs = (
            link_df[["init_node", "term_node"]]
            .astype(np.int64)
            .to_numpy()
        )

        processed_pairs = self._to_numpy_array(
            edge_indexing["link_pair_indices"]
        ).astype(np.int64)

        if processed_pairs.ndim != 2 or processed_pairs.shape[1] != 2:
            self._add_error(
                code="PROCESSED_LINK_PAIR_INDICES_INVALID_SHAPE",
                message="processed['edge_indexing']['link_pair_indices'] must have shape [num_links, 2].",
                context={"shape": tuple(processed_pairs.shape)},
            )
            return

        if tuple(link_df_pairs.shape) != tuple(processed_pairs.shape):
            self._add_error(
                code="LINK_DF_PROCESSED_EDGE_ORDER_SHAPE_MISMATCH",
                message="link_df edge-pair shape differs from processed link_pair_indices shape.",
                context={
                    "link_df_shape": tuple(link_df_pairs.shape),
                    "processed_shape": tuple(processed_pairs.shape),
                },
            )
            return

        if not np.array_equal(link_df_pairs, processed_pairs):
            mismatch_idx = np.where(
                np.any(link_df_pairs != processed_pairs, axis=1)
            )[0]

            sample = [
                {
                    "position": int(idx),
                    "link_df_edge": tuple(map(int, link_df_pairs[idx])),
                    "processed_edge": tuple(map(int, processed_pairs[idx])),
                }
                for idx in mismatch_idx[:self.max_reported_items]
            ]

            self._add_error(
                code="LINK_DF_PROCESSED_EDGE_ORDER_MISMATCH",
                message=(
                    "processed['edge_indexing']['link_pair_indices'] does not follow "
                    "link_df row order. In the refactored pipeline, link_df row order "
                    "is the canonical link order."
                ),
                context={
                    "num_mismatches": int(len(mismatch_idx)),
                    "sample": sample,
                },
            )

    def _validate_routes_against_graph(self, artifact: Dict[str, Any]) -> None:
        """
        Validate that all route edges exist in the graph.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Training artifact.
        """

        graph = artifact["processed"]["graph"]
        routes_by_od = artifact["processed"]["primary_source_route_set"]

        missing_edge_records = []
        invalid_endpoint_records = []

        graph_edges = {
            (int(u), int(v))
            for u, v in graph.edges()
        }

        for od_pair, routes in routes_by_od.items():
            origin, destination = tuple(map(int, od_pair))

            for route_idx, route in enumerate(routes):
                if not route:
                    continue

                route = [int(node) for node in route]

                if route[0] != origin or route[-1] != destination:
                    invalid_endpoint_records.append(
                        {
                            "od_pair": (origin, destination),
                            "route_idx": int(route_idx),
                            "route_start": int(route[0]),
                            "route_end": int(route[-1]),
                        }
                    )

                for u, v in zip(route[:-1], route[1:]):
                    edge = (int(u), int(v))

                    if edge not in graph_edges:
                        missing_edge_records.append(
                            {
                                "od_pair": (origin, destination),
                                "route_idx": int(route_idx),
                                "missing_edge": edge,
                            }
                        )

        if invalid_endpoint_records:
            self._add_error(
                code="ROUTE_ENDPOINT_MISMATCH",
                message="Some routes do not start/end at their OD pair.",
                context={
                    "sample": invalid_endpoint_records[:self.max_reported_items],
                    "count": len(invalid_endpoint_records),
                },
            )

        if missing_edge_records:
            self._add_error(
                code="ROUTE_EDGE_NOT_IN_GRAPH",
                message="Some route edges are not present in the graph.",
                context={
                    "sample": missing_edge_records[:self.max_reported_items],
                    "count": len(missing_edge_records),
                },
            )

    # ------------------------------------------------------------------
    # Generic validation helpers
    # ------------------------------------------------------------------

    def _validate_numpy_array(
        self,
        array: Any,
        name: str,
        expected_ndim: int,
    ) -> None:
        """
        Validate that an object is a NumPy array with expected dimensionality.

        Parameters
        ----------
        array : Any
            Object to validate.

        name : str
            Human-readable array name.

        expected_ndim : int
            Expected number of dimensions.
        """

        if not isinstance(array, np.ndarray):
            self._add_error(
                code=f"{name.upper()}_NOT_NDARRAY",
                message=f"{name} must be a numpy.ndarray.",
                context={"type": str(type(array))},
            )
            return

        if array.ndim != expected_ndim:
            self._add_error(
                code=f"{name.upper()}_NDIM_MISMATCH",
                message=f"{name} has wrong number of dimensions.",
                context={
                    "found": int(array.ndim),
                    "expected": int(expected_ndim),
                },
            )

    def _validate_binary_mask(
        self,
        mask: Any,
        name: str,
    ) -> None:
        """
        Validate that a NumPy mask contains only 0 and 1.

        Parameters
        ----------
        mask : Any
            Mask object.

        name : str
            Mask name.
        """

        if not isinstance(mask, np.ndarray):
            return

        unique_values = set(np.unique(mask).tolist())

        if not unique_values.issubset({0.0, 1.0}):
            self._add_error(
                code=f"{name.upper()}_NOT_BINARY",
                message=f"{name} must contain only 0 and 1.",
                context={"unique_values": sorted(unique_values)},
            )

    def _require_keys(
        self,
        obj: Dict[str, Any],
        required_keys: Iterable[str],
        object_name: str,
    ) -> None:
        """
        Validate that a dictionary contains required keys.

        Parameters
        ----------
        obj : Dict[str, Any]
            Dictionary to validate.

        required_keys : Iterable[str]
            Required keys.

        object_name : str
            Human-readable object name.
        """

        missing = set(required_keys) - set(obj.keys())

        if missing:
            self._add_error(
                code="MISSING_KEYS",
                message=f"{object_name} is missing required keys.",
                context={
                    "object_name": object_name,
                    "missing_keys": sorted(missing),
                },
            )

    @staticmethod
    def _normalize_od_pairs(od_pairs: Iterable[Any]) -> List[Tuple[int, int]]:
        """
        Normalize OD-pair representations to integer tuples.

        Parameters
        ----------
        od_pairs : Iterable[Any]
            OD pair iterable.

        Returns
        -------
        List[Tuple[int, int]]
            Normalized OD pairs.
        """

        return [
            (int(pair[0]), int(pair[1]))
            for pair in od_pairs
        ]

    # ------------------------------------------------------------------
    # Issue handling and result building
    # ------------------------------------------------------------------

    def _add_error(
        self,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a validation error.

        Parameters
        ----------
        code : str
            Issue code.

        message : str
            Issue message.

        context : Optional[Dict[str, Any]], default=None
            Optional issue context.
        """

        self._errors.append(
            ValidationIssue(
                severity="error",
                code=code,
                message=message,
                context=context or {},
            )
        )

    def _add_warning(
        self,
        code: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a validation warning.

        Parameters
        ----------
        code : str
            Issue code.

        message : str
            Issue message.

        context : Optional[Dict[str, Any]], default=None
            Optional issue context.
        """

        self._warnings.append(
            ValidationIssue(
                severity="warning",
                code=code,
                message=message,
                context=context or {},
            )
        )

    def _build_result(
        self,
        artifact: Any,
    ) -> TrainingArtifactValidationResult:
        """
        Build the final validation result.

        Parameters
        ----------
        artifact : Any
            Validated object.

        Returns
        -------
        TrainingArtifactValidationResult
            Validation result.
        """

        summary = self._build_summary(artifact)

        return TrainingArtifactValidationResult(
            is_valid=len(self._errors) == 0,
            errors=list(self._errors),
            warnings=list(self._warnings),
            summary=summary,
        )

    def _build_summary(
        self,
        artifact: Any,
    ) -> Dict[str, Any]:
        """
        Build a lightweight validation summary.

        Parameters
        ----------
        artifact : Any
            Validated object.

        Returns
        -------
        Dict[str, Any]
            Validation summary.
        """

        summary: Dict[str, Any] = {
            "num_errors": int(len(self._errors)),
            "num_warnings": int(len(self._warnings)),
            "is_valid": bool(len(self._errors) == 0),
        }

        if not isinstance(artifact, dict):
            return summary

        summary["artifact_type"] = artifact.get("artifact_type")
        summary["artifact_version"] = artifact.get("artifact_version")
        summary["dataset_name"] = artifact.get("dataset_name")

        try:
            processed = artifact.get("processed", {})
            model_ready = artifact.get("model_ready", {})
            network_params = model_ready.get("network_params", {})
            targets = model_ready.get("targets", {})

            if isinstance(processed.get("graph"), nx.DiGraph):
                summary["num_graph_nodes"] = int(processed["graph"].number_of_nodes())
                summary["num_graph_edges"] = int(processed["graph"].number_of_edges())

            if isinstance(processed.get("link_df"), pd.DataFrame):
                summary["num_link_df_rows"] = int(len(processed["link_df"]))

            if "num_links" in network_params:
                summary["num_links"] = int(network_params["num_links"])

            if "num_od_pairs" in network_params:
                summary["num_od_pairs"] = int(network_params["num_od_pairs"])

            if "flows_observed_mask_np" in targets:
                summary["num_observed_flows"] = int(targets["flows_observed_mask_np"].sum())

            if "od_observed_mask_np" in targets:
                summary["num_observed_od_pairs"] = int(targets["od_observed_mask_np"].sum())

        except Exception as exc:
            summary["summary_error"] = str(exc)

        return summary


def validate_training_artifact(
    artifact: Dict[str, Any],
    strict: bool = True,
    check_route_graph_compatibility: bool = True,
    check_tensor_values: bool = True,
    check_target_masks: bool = True,
    check_physical_tensors: bool = True,
    max_reported_items: int = 20,
) -> TrainingArtifactValidationResult:
    """
    Convenience function to validate a training artifact.

    Parameters
    ----------
    artifact : Dict[str, Any]
        Training artifact.

    strict : bool, default=True
        Whether errors should be considered fatal when using validate_or_raise.

    check_route_graph_compatibility : bool, default=True
        Whether to validate routes against the graph.

    check_tensor_values : bool, default=True
        Whether to validate physical tensor values and target values.

    Returns
    -------
    TrainingArtifactValidationResult
        Structured validation result.
    """

    validator = TrainingArtifactValidator(
        strict=strict,
        check_route_graph_compatibility=check_route_graph_compatibility,
        check_tensor_values=check_tensor_values,
        check_target_masks=check_target_masks,
        check_physical_tensors=check_physical_tensors,
        max_reported_items=max_reported_items,
    )

    return validator.validate(artifact)


def validate_training_artifact_or_raise(
    artifact: Dict[str, Any],
    strict: bool = True,
    check_route_graph_compatibility: bool = True,
    check_tensor_values: bool = True,
    check_target_masks: bool = True,
    check_physical_tensors: bool = True,
    max_reported_items: int = 20,
) -> TrainingArtifactValidationResult:
    """
    Convenience function to validate a training artifact and raise on errors.

    Parameters
    ----------
    artifact : Dict[str, Any]
        Training artifact.

    strict : bool, default=True
        If True, raise ValueError when validation errors are found.

    check_route_graph_compatibility : bool, default=True
        Whether to validate routes against the graph.

    check_tensor_values : bool, default=True
        Whether to validate physical tensor values and target values.

    Returns
    -------
    TrainingArtifactValidationResult
        Validation result.

    Raises
    ------
    ValueError
        If validation errors are found and strict=True.
    """

    validator = TrainingArtifactValidator(
        strict=strict,
        check_route_graph_compatibility=check_route_graph_compatibility,
        check_tensor_values=check_tensor_values,
        check_target_masks=check_target_masks,
        check_physical_tensors=check_physical_tensors,
        max_reported_items=max_reported_items,
    )

    return validator.validate_or_raise(artifact)


def validate_base_artifact_or_raise(
    artifact: Dict[str, Any],
    strict: bool = True,
    check_route_graph_compatibility: bool = True,
    max_reported_items: int = 20,
) -> TrainingArtifactValidationResult:
    """Validate the data-processing output as a base artifact.

    The validator is shared with the downstream training artifact because the
    structural checks overlap, but this entry point makes the boundary
    explicit and prevents the data-processing pipeline from being described as
    a training-artifact producer.
    """

    if not isinstance(artifact, dict):
        raise TypeError("base artifact must be a dictionary.")
    if artifact.get("artifact_type") != "base_artifact":
        raise ValueError(
            "Base-artifact validation requires artifact_type='base_artifact'."
        )

    validator = TrainingArtifactValidator(
        strict=strict,
        check_route_graph_compatibility=check_route_graph_compatibility,
        check_tensor_values=False,
        check_target_masks=False,
        check_physical_tensors=False,
        max_reported_items=max_reported_items,
    )
    return validator.validate_or_raise(artifact)
