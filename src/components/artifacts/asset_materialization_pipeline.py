"""Pipeline dedicated to reusable route and assignment assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from .asset_manager import AssetManager
from .config_schemas import (
    load_assets_config,
    load_assignment_set_spec,
    load_dataset_profile,
    load_route_set_spec,
)
from .manifest_store import ManifestStore
from src.utils.paths import resolve_path


@dataclass(frozen=True)
class AssetMaterializationResult:
    """Reusable assets resolved for one experiment."""

    manifest: dict[str, Any]
    manifest_store: ManifestStore
    route_set: dict[str, Any] | None
    assignment_set: dict[str, Any] | None


class AssetMaterializationPipeline:
    """Resolve or build reusable route and assignment assets only."""

    def __init__(
        self,
        *,
        experiment_config: Mapping[str, Any] | str | Path,
        dataset_config: Mapping[str, Any] | str | Path,
        manifest_path: str | Path,
        base_artifact_path: str | Path | None = None,
    ) -> None:
        self.root_config = self._load_config(experiment_config, "experiment_config")
        self.dataset_config = self._load_config(dataset_config, "dataset_config")
        self.dataset_profile = load_dataset_profile(self.dataset_config)
        assets_source = (
            self.root_config
            if "assets" in self.root_config
            else self._extract_experiment_section(self.root_config)
        )
        assets_mapping = self._require_section(assets_source, "assets")
        self.assets = load_assets_config(
            self._apply_experiment_data_selection(assets_mapping)
        )
        self.manifest_path = Path(manifest_path)
        self.base_artifact_path = (
            Path(base_artifact_path) if base_artifact_path is not None else None
        )
        creation_artifact_value = self.dataset_config.get("paths", {}).get(
            "artifacts", {}
        ).get("creation")
        self.creation_artifact_path = (
            resolve_path(str(creation_artifact_value))
            if creation_artifact_value is not None
            else None
        )

    def run(self) -> AssetMaterializationResult:
        manager = AssetManager(
            manifest_path=self.manifest_path,
            base_artifact_path=self.base_artifact_path,
            creation_artifact_path=self.creation_artifact_path,
            policy=self.assets.policy,
        )

        route_set_entry = None
        route_requirement = self.assets.requirements.route_set
        if route_requirement is not None:
            route_spec = load_route_set_spec(
                self._load_spec("route_bank", route_requirement.spec_id)
            )
            route_set_entry = manager.resolve_route_set(route_requirement, route_spec)

        assignment_set_entry = None
        assignment_requirement = self.assets.requirements.assignment_set
        if assignment_requirement is not None:
            availability = self.dataset_profile.data_availability
            if not availability.has_full_od_ground_truth:
                raise ValueError(
                    "The experiment requests an assignment_set, but the selected "
                    f"dataset {self.dataset_profile.name!r} does not declare "
                    "has_full_od_ground_truth=true."
                )
            if not availability.has_ground_truth_link_flows:
                raise ValueError(
                    "The experiment requests an assignment_set, but the selected "
                    f"dataset {self.dataset_profile.name!r} does not declare "
                    "has_ground_truth_link_flows=true."
                )
            assignment_spec = load_assignment_set_spec(
                self._load_spec("assignment_bank", assignment_requirement.spec_id)
            )
            assignment_set_entry = manager.resolve_assignment_set(
                assignment_spec,
                route_set_requirement=route_requirement,
            )

        self._upsert_dataset_profile(manager.manifest_store)
        return AssetMaterializationResult(
            manifest=manager.manifest_store.read(),
            manifest_store=manager.manifest_store,
            route_set=route_set_entry,
            assignment_set=assignment_set_entry,
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
        self, assets_mapping: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        data_selection = self.root_config.get("data_selection", {})
        if not isinstance(data_selection, Mapping):
            raise TypeError("experiment.data_selection must be a mapping when provided.")
        weight_column = data_selection.get("weight_column")
        if weight_column is None:
            return assets_mapping
        requirements = assets_mapping.get("requirements")
        if not isinstance(requirements, Mapping):
            return assets_mapping
        route_requirement = requirements.get("route_set")
        if not isinstance(route_requirement, Mapping):
            return assets_mapping
        result = dict(assets_mapping)
        result_requirements = dict(requirements)
        result_route = dict(route_requirement)
        result_route.setdefault("weight_column", str(weight_column))
        result_requirements["route_set"] = result_route
        result["requirements"] = result_requirements
        return result

    def _load_spec(self, bank_name: str, spec_id: str) -> dict[str, Any]:
        # ``asset_materialization_pipeline.py`` lives at
        # ``src/components/artifacts``; the repository root is three parents
        # above the file.
        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "configs" / bank_name / f"{spec_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Spec file not found: {path}")
        return OmegaConf.to_container(OmegaConf.load(path), resolve=True)  # type: ignore[return-value]

    @staticmethod
    def _load_config(config: Mapping[str, Any] | str | Path, context: str) -> dict[str, Any]:
        if isinstance(config, (str, Path)):
            return OmegaConf.to_container(OmegaConf.load(Path(config)), resolve=True)  # type: ignore[return-value]
        if isinstance(config, Mapping):
            return dict(config)
        raise TypeError(f"{context} must be a mapping or a path.")

    @staticmethod
    def _require_section(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
        value = config.get(section)
        if not isinstance(value, Mapping):
            raise KeyError(f"Configuration section {section!r} must be a mapping.")
        return value

    @staticmethod
    def _extract_experiment_section(config: Mapping[str, Any]) -> Mapping[str, Any]:
        experiment = config.get("experiment")
        return experiment if isinstance(experiment, Mapping) else config


__all__ = ["AssetMaterializationPipeline", "AssetMaterializationResult"]
