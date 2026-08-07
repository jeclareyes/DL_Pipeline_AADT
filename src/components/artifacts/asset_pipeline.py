"""Independent asset materialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Mapping

from omegaconf import OmegaConf

from .asset_manager import AssetManager
from .asset_materialization_pipeline import AssetMaterializationPipeline
from .config_schemas import (
    load_assets_config,
    load_assignment_set_spec,
    load_dataset_profile,
    load_route_set_spec,
)
from .manifest_store import ManifestStore
from src.data_handling.data_processing.artifact_builders.training_artifact_builder import (
    TrainingArtifactBuilder,
)
from src.data_handling.data_processing.validators.training_artifact_validator import (
    validate_training_artifact_or_raise,
)
from src.utils.serialization import dump, load
from src.utils.paths import resolve_path
from src.utils.flow_columns import FlowColumnContract
from .experiment_artifact.builder import ExperimentArtifactBuilder


@dataclass(frozen=True)
class AssetPipelineResult:
    """Result returned after materializing the requested assets."""

    manifest: dict[str, Any]
    route_set: dict[str, Any] | None
    assignment_set: dict[str, Any] | None
    training_artifact: dict[str, Any] | None


class AssetPipeline:
    """Materialize reusable assets from configuration and a base artifact."""

    def __init__(
        self,
        *,
        experiment_config: Mapping[str, Any] | str | Path,
        dataset_config: Mapping[str, Any] | str | Path,
        manifest_path: str | Path,
        base_artifact_path: str | Path | None = None,
    ) -> None:
        self.root_config = self._load_config(experiment_config, context="experiment_config")
        self.experiment_config = self._extract_experiment_section(self.root_config)
        self.dataset_config = self._load_config(dataset_config, context="dataset_config")
        self.dataset_profile = load_dataset_profile(self.dataset_config)
        assets_source = self.root_config if "assets" in self.root_config else self.experiment_config
        assets_mapping = self._require_section(assets_source, "assets")
        self.assets = load_assets_config(
            self._apply_experiment_data_selection(assets_mapping)
        )
        self.manifest_path = Path(manifest_path)
        self.base_artifact_path = Path(base_artifact_path) if base_artifact_path is not None else None
        creation_artifact_value = self.dataset_config.get("paths", {}).get(
            "artifacts", {}
        ).get("creation")
        self.creation_artifact_path = (
            resolve_path(str(creation_artifact_value))
            if creation_artifact_value is not None
            else None
        )

    def run(self) -> AssetPipelineResult:
        """Materialize assets and build the experiment artifact.

        This remains the compatibility facade for callers that used the old
        combined pipeline. The two lifecycle stages are now delegated to
        dedicated implementations.
        """

        asset_result = AssetMaterializationPipeline(
            experiment_config=self.root_config,
            dataset_config=self.dataset_config,
            manifest_path=self.manifest_path,
            base_artifact_path=self.base_artifact_path,
        ).run()
        if self.base_artifact_path is None:
            raise FileNotFoundError("An experiment artifact requires base_artifact_path.")
        training_artifact_entry = ExperimentArtifactBuilder(
            experiment_config=self.root_config,
            dataset_config=self.dataset_config,
            base_artifact_path=self.base_artifact_path,
        ).build(asset_result)
        manifest = asset_result.manifest_store.read()
        if manifest and training_artifact_entry is not None:
            manifest["experiment_artifact"] = training_artifact_entry
            asset_result.manifest_store.write(manifest)
        return AssetPipelineResult(
            manifest=manifest,
            route_set=asset_result.route_set,
            assignment_set=asset_result.assignment_set,
            training_artifact=training_artifact_entry,
        )

    def _upsert_dataset_profile(self, store: ManifestStore) -> None:
        manifest = store.read()
        if not manifest:
            return
        manifest["dataset_profile"] = {
            "nature": self.dataset_profile.nature.value,
            "data_availability": {
                "has_full_od_ground_truth": self.dataset_profile.data_availability.has_full_od_ground_truth,
                "has_ground_truth_link_flows": self.dataset_profile.data_availability.has_ground_truth_link_flows,
                "has_observed_link_flows": self.dataset_profile.data_availability.has_observed_link_flows,
            },
        }
        store.write(manifest)

    def _apply_experiment_data_selection(
        self,
        assets_mapping: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Propagate the experiment route weight into the route requirement."""

        data_selection = self.root_config.get("data_selection", {})
        if not isinstance(data_selection, Mapping):
            raise TypeError("experiment.data_selection must be a mapping when provided.")
        weight_column = data_selection.get("weight_column")
        if weight_column is None:
            return assets_mapping

        result = dict(assets_mapping)
        requirements = result.get("requirements")
        if not isinstance(requirements, Mapping):
            return assets_mapping
        route_requirement = requirements.get("route_set")
        if not isinstance(route_requirement, Mapping):
            return assets_mapping
        updated_requirements = dict(requirements)
        updated_route_requirement = dict(route_requirement)
        updated_route_requirement.setdefault("weight_column", str(weight_column))
        updated_requirements["route_set"] = updated_route_requirement
        result["requirements"] = updated_requirements
        return result

    def _materialize_training_artifact(
        self,
        store: ManifestStore,
        *,
        route_set_entry: Mapping[str, Any] | None,
        assignment_set_entry: Mapping[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Materialize the training artifact used by the training pipeline."""

        training_cfg = self.root_config.get("training")
        processing_cfg = self._get_nested_section(self.root_config, ("data_handling", "data_processing"))
        model_cfg = self.root_config.get("model")

        if not isinstance(training_cfg, Mapping) or not isinstance(processing_cfg, Mapping) or not isinstance(model_cfg, Mapping):
            return None

        if self.base_artifact_path is None:
            raise FileNotFoundError(
                "Cannot materialize the training artifact because base_artifact_path was not provided."
            )

        artifact_path = self._resolve_training_artifact_path(training_cfg)
        base_artifact = load(self.base_artifact_path)

        if "k_paths" not in model_cfg:
            raise KeyError("Configuration is missing required key 'model.k_paths'.")

        builder = TrainingArtifactBuilder(
            cfg=processing_cfg,
            dataset_cfg=self.dataset_config,
            device="cpu",
            artifact_name=artifact_path.name,
            manifest_name=f"{artifact_path.stem}_manifest.json",
        )

        if route_set_entry is None:
            raise ValueError(
                "An experiment route_set is required to build the training_artifact. "
                "The primary_source_route_set is provenance only."
            )

        route_asset = load(route_set_entry["path"])
        route_set = getattr(route_asset, "route_set", None)
        if route_set is None:
            raise TypeError("The resolved route_set asset does not contain a materialized RouteSet.")

        assignment_asset = None
        if assignment_set_entry is not None:
            assignment_asset = load(assignment_set_entry["path"])

        selected_raw, selected_processed, selection_metadata = self._build_experiment_data_view(
            base_artifact
        )
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
            "route_set": dict(route_set_entry) if route_set_entry is not None else None,
            "assignment_set": (
                dict(assignment_set_entry)
                if assignment_set_entry is not None
                else None
            ),
        }
        validation_cfg = self._get_nested_section(
            training_cfg,
            ("artifact", "validation"),
        )
        if validation_cfg is None:
            raise KeyError(
                "Configuration is missing required section 'training.artifact.validation'."
            )

        if bool(validation_cfg["enabled"]):
            validation_result = validate_training_artifact_or_raise(
                artifact,
                strict=bool(validation_cfg["strict"]),
                check_route_graph_compatibility=bool(
                    validation_cfg["check_route_graph_compatibility"]
                ),
                check_tensor_values=bool(validation_cfg["check_tensor_values"]),
                check_target_masks=bool(validation_cfg["check_target_masks"]),
                check_physical_tensors=bool(validation_cfg["check_physical_tensors"]),
                max_reported_items=int(validation_cfg["max_reported_items"]),
            )
            artifact["metadata"]["validation"] = validation_result.summary
        else:
            logger.warning(
                "Training-artifact validation is disabled by "
                "training.artifact.validation.enabled=false."
            )
        dump(artifact, artifact_path)

        experiment_manifest = builder.build_manifest(
            artifact,
            entry_name="experiment_artifact",
            artifact_path=artifact_path,
        )
        artifact_manifest_path = artifact_path.with_name(
            "experiment_artifact_manifest.json"
        )
        artifact_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_manifest_path.write_text(
            json.dumps(experiment_manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        entry = {
            "id": "training_artifact",
            "spec_id": "training_artifact",
            "asset_type": "training_artifact",
            "path": str(artifact_path),
            "metadata": {
                "k_paths": int(model_cfg["k_paths"]),
                "source_base_artifact": str(self.base_artifact_path),
            },
            "diagnostics": {
                "artifact_version": artifact.get("artifact_version"),
                "dataset_name": artifact.get("dataset_name"),
                "manifest_path": str(artifact_manifest_path),
            },
        }

        manifest = store.read()
        if manifest:
            manifest["training_artifact"] = entry
            store.write(manifest)
        return entry

    def _build_experiment_data_view(
        self,
        base_artifact: Mapping[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Select experiment-specific flow values without rebuilding the base artifact."""

        data_selection = self.root_config.get("data_selection", {})
        if not isinstance(data_selection, Mapping):
            raise TypeError("experiment.data_selection must be a mapping when provided.")
        requested_year = data_selection.get("volume_year")
        raw = dict(base_artifact["raw"])
        selected_processed = dict(base_artifact["processed"])
        flow_columns = FlowColumnContract.from_mapping(
            base_artifact.get("metadata", {}).get("flow_columns", {})
        )
        if requested_year in (None, "", False):
            selected_column = flow_columns.default_training_column()
            selected_processed["selected_flow_column"] = selected_column
            return raw, selected_processed, {"selected_flow_column": selected_column}

        flow_df = raw.get("flow_df")
        link_df = selected_processed.get("link_df")
        if flow_df is None or link_df is None:
            raise KeyError("The base artifact must contain raw.flow_df and processed.link_df.")

        if str(requested_year).lower() == "all":
            if len(flow_columns.traffic_counts) != 1:
                raise ValueError("Select a specific year when multiple traffic_counts are declared.")
            selected_column = flow_columns.traffic_counts[0]
        else:
            requested_suffix = str(requested_year).strip()
            selectable_columns = list(flow_columns.traffic_counts)
            if not selectable_columns and flow_columns.reference_assignment is not None:
                selectable_columns = [flow_columns.reference_assignment]
            matches = [
                column for column in selectable_columns
                if column.lower().endswith(requested_suffix.lower())
            ]
            if len(matches) > 1:
                raise ValueError(
                    f"Experiment volume_year={requested_year!r} matches multiple columns: {matches}."
                )
            if matches:
                selected_column = matches[0]
            elif selectable_columns:
                raise ValueError(
                    f"Experiment volume_year={requested_year!r} was not found. "
                    f"Available flow columns: {selectable_columns}."
                )
            else:
                raise ValueError("The dataset declares no traffic_counts for volume_year selection.")

        if selected_column not in link_df.columns:
            raise ValueError(
                f"Selected flow column {selected_column!r} is not available in processed.link_df. "
                "The base artifact must preserve all candidate flow columns."
            )

        selected_processed["selected_flow_column"] = selected_column
        return raw, selected_processed, {
            "volume_year": requested_year,
            "selected_flow_column": selected_column,
            "weight_column": data_selection.get("weight_column"),
        }

    def _load_spec(self, bank_name: str, spec_id: str) -> dict[str, Any]:
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "configs" / bank_name / f"{spec_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]

    @staticmethod
    def _load_config(config: Mapping[str, Any] | str | Path, *, context: str) -> dict[str, Any]:
        if isinstance(config, (str, Path)):
            return OmegaConf.to_container(OmegaConf.load(Path(config)), resolve=True)  # type: ignore[return-value]
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError(f"{context} must be a mapping or a path.")

    @staticmethod
    def _require_section(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
        if section not in config:
            raise KeyError(f"Configuration is missing required section {section!r}.")
        value = config[section]
        if not isinstance(value, Mapping):
            raise TypeError(f"Configuration section {section!r} must be a mapping.")
        return value

    @classmethod
    def _get_nested_section(
        cls,
        config: Mapping[str, Any],
        sections: tuple[str, ...],
    ) -> Mapping[str, Any] | None:
        current: Mapping[str, Any] | None = config
        for section in sections:
            if current is None:
                return None
            if section not in current:
                return None
            value = current[section]
            if not isinstance(value, Mapping):
                raise TypeError(f"Configuration section {'.'.join(sections)} must be a mapping.")
            current = value
        return current

    @staticmethod
    def _extract_experiment_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
        experiment_section = config.get("experiment")
        if isinstance(experiment_section, Mapping):
            return experiment_section
        return config

    @staticmethod
    def _resolve_training_artifact_path(training_cfg: Mapping[str, Any]) -> Path:
        artifact_cfg = training_cfg.get("artifact")
        if not isinstance(artifact_cfg, Mapping):
            raise KeyError("Configuration is missing required section 'training.artifact'.")
        path_value = artifact_cfg.get("path")
        if not path_value:
            raise KeyError("Configuration is missing required key 'training.artifact.path'.")
        return Path(str(path_value)).expanduser().resolve(strict=False)
