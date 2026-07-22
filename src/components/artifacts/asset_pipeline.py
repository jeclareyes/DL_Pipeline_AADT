"""Independent asset materialization pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from .asset_manager import AssetManager
from .config_schemas import (
    AssetsConfig,
    DatasetProfileConfig,
    load_assets_config,
    load_assignment_set_spec,
    load_dataset_profile,
    load_route_set_spec,
)
from .manifest_store import ManifestStore
from src.data_ingestion.artifact_builders.training_artifact_builder import (
    TrainingArtifactBuilder,
)
from src.data_ingestion.validators.training_artifact_validator import (
    validate_training_artifact_or_raise,
)
from src.utils.serialization import dump, load


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
        self.dataset_profile = load_dataset_profile(self._load_config(dataset_config, context="dataset_config"))
        self.assets = load_assets_config(self._require_section(self.experiment_config, "assets"))
        self.manifest_path = Path(manifest_path)
        self.base_artifact_path = Path(base_artifact_path) if base_artifact_path is not None else None

    def run(self) -> AssetPipelineResult:
        """Materialize the configured assets and persist their manifest entries."""

        manager = AssetManager(
            manifest_path=self.manifest_path,
            base_artifact_path=self.base_artifact_path,
            policy=self.assets.policy,
        )

        route_set_entry = None
        if self.assets.requirements.route_set is not None:
            route_set_spec = load_route_set_spec(self._load_spec("route_bank", self.assets.requirements.route_set.spec_id))
            route_set_entry = manager.resolve_route_set(
                self.assets.requirements.route_set,
                route_set_spec,
            )

        assignment_set_entry = None
        if self.assets.requirements.assignment_set is not None:
            assignment_set_spec = load_assignment_set_spec(
                self._load_spec("assignment_bank", self.assets.requirements.assignment_set.spec_id)
            )
            assignment_set_entry = manager.resolve_assignment_set(assignment_set_spec)

        training_artifact_entry = self._materialize_training_artifact(manager.manifest_store)
        self._upsert_dataset_profile(manager.manifest_store)
        manifest = manager.manifest_store.read()
        return AssetPipelineResult(
            manifest=manifest,
            route_set=route_set_entry,
            assignment_set=assignment_set_entry,
            training_artifact=training_artifact_entry,
        )

    def _upsert_dataset_profile(self, store: ManifestStore) -> None:
        manifest = store.read()
        if not manifest:
            return
        manifest["dataset_profile"] = {
            "nature": self.dataset_profile.nature.value,
            "data_availability": {
                "has_complete_od_ground_truth": self.dataset_profile.data_availability.has_complete_od_ground_truth,
                "has_observed_link_flows": self.dataset_profile.data_availability.has_observed_link_flows,
                "has_ground_truth_assignment": self.dataset_profile.data_availability.has_ground_truth_assignment,
            },
        }
        store.write(manifest)

    def _materialize_training_artifact(self, store: ManifestStore) -> dict[str, Any] | None:
        """Materialize the training artifact used by the training pipeline."""

        training_cfg = self.root_config.get("training")
        processing_cfg = self._get_nested_section(self.root_config, ("data_ingestion", "data_processing"))
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
            device="cpu",
            artifact_name=artifact_path.name,
            manifest_name=f"{artifact_path.stem}_manifest.json",
        )

        model_ready = builder.build_model_ready(
            raw=base_artifact["raw"],
            processed=base_artifact["processed"],
            k_paths=int(model_cfg["k_paths"]),
        )
        artifact = builder.pack_artifact(
            raw=base_artifact["raw"],
            processed=base_artifact["processed"],
            model_ready=model_ready,
        )
        validation_result = validate_training_artifact_or_raise(artifact, strict=True)
        artifact["metadata"]["validation"] = validation_result.summary
        dump(artifact, artifact_path)

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
            },
        }

        manifest = store.read()
        if manifest:
            manifest["training_artifact"] = entry
            store.write(manifest)
        return entry

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
