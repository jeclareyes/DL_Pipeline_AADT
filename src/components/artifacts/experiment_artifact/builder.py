"""Build an experiment artifact from a base artifact and resolved assets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.components.artifacts.asset_materialization_pipeline import AssetMaterializationResult
from src.data_handling.data_processing.artifact_builders.training_artifact_builder import TrainingArtifactBuilder
from src.data_handling.data_processing.validators.training_artifact_validator import validate_training_artifact_or_raise
from src.utils.flow_columns import FlowColumnContract
from src.utils.serialization import dump, load


class ExperimentArtifactBuilder:
    """Build an experiment artifact from a base artifact and materialized assets."""

    def __init__(
        self,
        *,
        experiment_config: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
        base_artifact_path: str | Path,
    ):
        self.root_config = dict(experiment_config)
        self.dataset_config = dict(dataset_config)
        self.base_artifact_path = Path(base_artifact_path)

    def build(self, assets: AssetMaterializationResult) -> dict[str, Any] | None:
        training_cfg = self.root_config.get("training")
        processing_cfg = self.root_config.get("data_handling", {}).get("data_processing")
        model_cfg = self.root_config.get("model")
        if not isinstance(training_cfg, Mapping) or not isinstance(processing_cfg, Mapping) or not isinstance(model_cfg, Mapping):
            return None

        artifact_path = self._resolve_training_artifact_path(training_cfg)
        base_artifact = load(self.base_artifact_path)
        if "k_paths" not in model_cfg:
            raise KeyError("Configuration is missing required key 'model.k_paths'.")

        builder = TrainingArtifactBuilder(
            cfg=processing_cfg,
            dataset_cfg=self.dataset_config,
            device="cpu",
            artifact_name=artifact_path.name,
            manifest_name="experiment_artifact_manifest.json",
        )
        if assets.route_set is None:
            raise ValueError(
                "An experiment route_set is required to build the experiment_artifact."
            )

        route_asset = load(assets.route_set["path"])
        route_set = getattr(route_asset, "route_set", None)
        if route_set is None:
            raise TypeError("The resolved route_set asset does not contain a materialized RouteSet.")
        assignment_asset = load(assets.assignment_set["path"]) if assets.assignment_set else None

        selected_raw, selected_processed, selection_metadata = self._build_experiment_data_view(base_artifact)
        model_ready = builder.build_model_ready(
            raw=selected_raw,
            processed=selected_processed,
            k_paths=int(model_cfg["k_paths"]),
            route_set=route_set,
            assignment_set=assignment_asset,
        )
        artifact = builder.pack_artifact(
            raw=selected_raw,
            processed=selected_processed,
            artifact_type="training_artifact",
            model_ready=model_ready,
        )
        artifact["metadata"]["experiment_data_selection"] = selection_metadata
        artifact["metadata"]["experiment_provenance"] = {
            "base_artifact": {
                "path": str(self.base_artifact_path),
                "artifact_type": base_artifact.get("artifact_type"),
                "artifact_version": base_artifact.get("artifact_version"),
            },
            "route_set": dict(assets.route_set) if assets.route_set else None,
            "assignment_set": dict(assets.assignment_set) if assets.assignment_set else None,
        }

        validation_cfg = self._nested(training_cfg, ("artifact", "validation"))
        if validation_cfg is None:
            raise KeyError("Configuration is missing required section 'training.artifact.validation'.")
        if bool(validation_cfg["enabled"]):
            validation_result = validate_training_artifact_or_raise(
                artifact,
                strict=bool(validation_cfg["strict"]),
                check_route_graph_compatibility=bool(validation_cfg["check_route_graph_compatibility"]),
                check_tensor_values=bool(validation_cfg["check_tensor_values"]),
                check_target_masks=bool(validation_cfg["check_target_masks"]),
                check_physical_tensors=bool(validation_cfg["check_physical_tensors"]),
                max_reported_items=int(validation_cfg["max_reported_items"]),
            )
            artifact["metadata"]["validation"] = validation_result.summary

        dump(artifact, artifact_path)
        manifest = builder.build_manifest(
            artifact,
            entry_name="experiment_artifact",
            artifact_path=artifact_path,
        )
        manifest_path = artifact_path.with_name("experiment_artifact_manifest.json")
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return {
            "id": "experiment_artifact",
            "spec_id": "experiment_artifact",
            "asset_type": "experiment_artifact",
            "path": str(artifact_path),
            "manifest_path": str(manifest_path),
            "metadata": {"k_paths": int(model_cfg["k_paths"]), "source_base_artifact": str(self.base_artifact_path)},
        }

    def _build_experiment_data_view(self, base_artifact: Mapping[str, Any]):
        data_selection = self.root_config.get("data_selection", {})
        if not isinstance(data_selection, Mapping):
            raise TypeError("experiment.data_selection must be a mapping when provided.")
        requested_year = data_selection.get("volume_year")
        raw = dict(base_artifact["raw"])
        selected_processed = dict(base_artifact["processed"])
        flow_columns = FlowColumnContract.from_mapping(base_artifact.get("metadata", {}).get("flow_columns", {}))
        if requested_year in (None, "", False):
            selected_column = flow_columns.default_training_column()
            selected_processed["selected_flow_column"] = selected_column
            return raw, selected_processed, {"selected_flow_column": selected_column}

        link_df = selected_processed.get("link_df")
        if link_df is None:
            raise KeyError("The base artifact must contain processed.link_df.")
        if str(requested_year).lower() == "all":
            if len(flow_columns.traffic_counts) != 1:
                raise ValueError("Select a specific year when multiple traffic_counts are declared.")
            selected_column = flow_columns.traffic_counts[0]
        else:
            suffix = str(requested_year).strip().lower()
            selectable = list(flow_columns.traffic_counts)
            if not selectable and flow_columns.reference_assignment:
                selectable = [flow_columns.reference_assignment]
            matches = [column for column in selectable if column.lower().endswith(suffix)]
            if len(matches) > 1:
                raise ValueError(f"Experiment volume_year={requested_year!r} matches multiple columns: {matches}.")
            if not matches:
                raise ValueError(f"Experiment volume_year={requested_year!r} was not found in {selectable}.")
            selected_column = matches[0]
        if selected_column not in link_df.columns:
            raise ValueError(f"Selected flow column {selected_column!r} is not available in processed.link_df.")
        selected_processed["selected_flow_column"] = selected_column
        return raw, selected_processed, {
            "volume_year": requested_year,
            "selected_flow_column": selected_column,
            "weight_column": data_selection.get("weight_column"),
        }

    @staticmethod
    def _nested(config: Mapping[str, Any], sections: tuple[str, ...]):
        current: Any = config
        for section in sections:
            if not isinstance(current, Mapping) or section not in current:
                return None
            current = current[section]
        return current

    @staticmethod
    def _resolve_training_artifact_path(training_cfg: Mapping[str, Any]) -> Path:
        artifact_cfg = training_cfg.get("artifact")
        if not isinstance(artifact_cfg, Mapping):
            raise KeyError("Configuration is missing required section 'training.artifact'.")
        path_value = artifact_cfg.get("path")
        if not path_value:
            raise KeyError("Configuration is missing required key 'training.artifact.path'.")
        return Path(str(path_value)).expanduser().resolve(strict=False)


__all__ = ["ExperimentArtifactBuilder"]
