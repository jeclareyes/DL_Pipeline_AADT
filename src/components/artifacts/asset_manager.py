"""Resolve, materialize and register reusable artifact assets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

from .asset_materializer import AssetMaterializer
from .asset_registry import AssetRegistry
from .config_schemas import (
    AssignmentSetRequirementConfig,
    AssignmentSetSpecConfig,
    AssetPolicyConfig,
    RouteSetRequirementConfig,
    RouteSetSpecConfig,
)
from .fingerprints import (
    compute_link_order_fingerprint,
    compute_network_fingerprint,
    compute_od_space_fingerprint,
    compute_zone_order_fingerprint,
)
from .manifest_store import ManifestStore
from src.utils.serialization import load


logger = logging.getLogger(__name__)


class AssetManager:
    """Coordinate asset reuse and materialization."""

    def __init__(
        self,
        manifest_path: str | Path,
        base_artifact_path: str | Path | None = None,
        creation_artifact_path: str | Path | None = None,
        policy: Mapping[str, Any] | AssetPolicyConfig | None = None,
    ) -> None:
        self.manifest_store = ManifestStore(manifest_path)
        self.registry = AssetRegistry(self.manifest_store)
        self.base_artifact_path = Path(base_artifact_path) if base_artifact_path is not None else None
        self.creation_artifact_path = (
            Path(creation_artifact_path) if creation_artifact_path is not None else None
        )
        self.policy = self._normalize_policy(policy)
        self._materializer: AssetMaterializer | None = None

    def resolve_route_set(
        self,
        requirement: Mapping[str, Any] | RouteSetRequirementConfig,
        spec: Mapping[str, Any] | RouteSetSpecConfig,
    ) -> dict[str, Any]:
        """Resolve or materialize a route-set asset."""

        requirement_cfg = self._coerce_route_requirement(requirement)
        spec_cfg = self._coerce_route_set_spec(spec)
        description = self._materializer_for_base().route_set_builder.describe(
            spec=spec_cfg,
            requirement=requirement_cfg,
        )

        existing_any = self.registry.find_route_set(
            spec_id=spec_cfg.id,
            required_fingerprint=None,
            minimum_k_active=requirement_cfg.k_active,
        )
        existing = self.registry.find_route_set(
            spec_id=spec_cfg.id,
            required_fingerprint=description["fingerprint"] if self.policy.require_exact_fingerprint else None,
            minimum_k_active=requirement_cfg.k_active,
        )
        if existing is not None:
            logger.info("Reusing existing route set: %s", spec_cfg.id)
            return existing

        if existing_any is not None:
            if self.policy.on_stale == "use_existing":
                logger.info("Reusing stale route set due to policy: %s", spec_cfg.id)
                return existing_any
            if self.policy.on_stale == "fail":
                raise ValueError(
                    f"Route set {spec_cfg.id!r} exists but is stale under the current fingerprint policy."
                )
        if self.policy.on_stale == "use_existing":
            logger.info("No matching route set found; continuing with missing policy.")
        elif self.policy.on_stale == "rebuild" and existing_any is not None:
            logger.info("Rebuilding stale route set: %s", spec_cfg.id)
            result = self._materializer_for_base().build_route_set(spec_cfg, requirement_cfg)
            self._register_manifest_entry("route_sets", spec_cfg.id, result.manifest_entry)
            return result.manifest_entry

        if existing_any is None and self.policy.on_missing != "build":
            raise ValueError(
                f"Route set {spec_cfg.id!r} is missing or stale and policy.on_missing={self.policy.on_missing!r}."
            )

        logger.info("Materializing route set: %s", spec_cfg.id)
        result = self._materializer_for_base().build_route_set(spec_cfg, requirement_cfg)
        self._register_manifest_entry("route_sets", spec_cfg.id, result.manifest_entry)
        return result.manifest_entry

    def resolve_assignment_set(
        self,
        spec: Mapping[str, Any] | AssignmentSetSpecConfig,
        route_set_requirement: Mapping[str, Any] | RouteSetRequirementConfig | None = None,
    ) -> dict[str, Any]:
        """Resolve or materialize an assignment-set recipe asset."""

        spec_cfg = self._coerce_assignment_set_spec(spec)
        route_requirement = spec_cfg.requires
        if route_set_requirement is not None:
            requested_route_requirement = self._coerce_route_requirement(route_set_requirement)
            if requested_route_requirement.spec_id != route_requirement.spec_id:
                raise ValueError(
                    "Experiment route_set spec does not match the route_set required by "
                    f"assignment_set {spec_cfg.id!r}: "
                    f"experiment={requested_route_requirement.spec_id!r}, "
                    f"assignment={route_requirement.spec_id!r}."
                )
            if requested_route_requirement.k_active != route_requirement.k_active:
                raise ValueError(
                    "Experiment route_set k_active does not match the assignment_set requirement: "
                    f"experiment={requested_route_requirement.k_active}, "
                    f"assignment={route_requirement.k_active}."
                )
            route_requirement = requested_route_requirement
        route_set_spec = self._load_route_set_spec(route_requirement.spec_id)
        route_set_entry = self.resolve_route_set(route_requirement, route_set_spec)

        description = self._materializer_for_base().assignment_set_builder.describe(
            spec=spec_cfg,
            route_set_entry=route_set_entry,
        )

        existing_any = self.registry.find_assignment_set(spec_id=spec_cfg.id, required_fingerprint=None)
        existing = self.registry.find_assignment_set(
            spec_id=spec_cfg.id,
            required_fingerprint=description["fingerprint"] if self.policy.require_exact_fingerprint else None,
        )
        if existing is not None:
            logger.info("Reusing existing assignment set: %s", spec_cfg.id)
            return existing

        if existing_any is not None:
            if self.policy.on_stale == "use_existing":
                logger.info("Reusing stale assignment set due to policy: %s", spec_cfg.id)
                return existing_any
            if self.policy.on_stale == "fail":
                raise ValueError(
                    f"Assignment set {spec_cfg.id!r} exists but is stale under the current fingerprint policy."
                )
        if self.policy.on_stale == "use_existing":
            logger.info("No matching assignment set found; continuing with missing policy.")
        elif self.policy.on_stale == "rebuild" and existing_any is not None:
            logger.info("Rebuilding stale assignment set: %s", spec_cfg.id)
            result = self._materializer_for_base().build_assignment_set(spec_cfg, route_set_entry)
            self._register_manifest_entry("assignment_sets", spec_cfg.id, result.manifest_entry)
            return result.manifest_entry

        if existing_any is None and self.policy.on_missing != "build":
            raise ValueError(
                f"Assignment set {spec_cfg.id!r} is missing or stale and policy.on_missing={self.policy.on_missing!r}."
            )

        logger.info("Materializing assignment set: %s", spec_cfg.id)
        result = self._materializer_for_base().build_assignment_set(spec_cfg, route_set_entry)
        self._register_manifest_entry("assignment_sets", spec_cfg.id, result.manifest_entry)
        return result.manifest_entry

    def _materializer_for_base(self) -> AssetMaterializer:
        if self._materializer is not None:
            return self._materializer
        base_artifact = self._load_base_artifact()
        output_root = self.manifest_store.manifest_path.parent
        self._materializer = AssetMaterializer(
            base_artifact=base_artifact,
            output_root=output_root,
            creation_artifact=self.creation_artifact_path,
        )
        return self._materializer

    def _load_base_artifact(self) -> dict[str, Any]:
        if self.base_artifact_path is not None:
            return load(self.base_artifact_path)

        base_entry = self.registry.get_base_artifact_entry()
        base_path = base_entry.get("path")
        if not base_path:
            raise KeyError("Manifest base_artifact entry does not expose a path.")
        return load(Path(base_path))

    def _register_manifest_entry(self, section: str, spec_id: str, entry: dict[str, Any]) -> None:
        manifest = self._ensure_manifest_root()
        section_entries = manifest.setdefault(section, {})
        if not isinstance(section_entries, dict):
            raise ValueError(f"Manifest section {section!r} must be a dictionary.")
        section_entries[spec_id] = entry
        self.manifest_store.write(manifest)

    def _ensure_manifest_root(self) -> dict[str, Any]:
        manifest = self.manifest_store.read()
        if manifest:
            return manifest

        if self.base_artifact_path is None:
            raise FileNotFoundError(
                "Manifest is missing and base_artifact_path was not provided. "
                "Cannot bootstrap the artifact bundle manifest."
            )

        base_artifact = load(self.base_artifact_path)
        base_entry = self._build_base_artifact_entry(base_artifact)
        manifest = {
            "schema_version": "artifact_bundle.v1",
            "base_artifact": base_entry,
            "route_sets": {},
            "assignment_sets": {},
        }
        self.manifest_store.write(manifest)
        return manifest

    def _build_base_artifact_entry(self, base_artifact: Mapping[str, Any]) -> dict[str, Any]:
        processed = base_artifact["processed"]
        raw = base_artifact["raw"]
        network_fp = compute_network_fingerprint(processed["link_df"], raw["nodes_df"])
        od_fp = compute_od_space_fingerprint(processed["od_indexing"].get("od_pairs", []))
        link_order_fp = compute_link_order_fingerprint(processed["edge_indexing"]["link_pair_indices"])
        zone_order_fp = compute_zone_order_fingerprint(processed["od_indexing"].get("zone_ids", []))
        return {
            "path": str(self.base_artifact_path) if self.base_artifact_path is not None else None,
            "artifact_type": base_artifact.get("artifact_type"),
            "artifact_version": base_artifact.get("artifact_version"),
            "network_fingerprint": network_fp,
            "od_space_fingerprint": od_fp,
            "link_order_fingerprint": link_order_fp,
            "zone_order_fingerprint": zone_order_fp,
            "metadata": {
                "dataset_name": base_artifact.get("dataset_name"),
                "processed_summary": base_artifact.get("metadata", {}).get(
                    "processed_summary", {}
                ),
                "reader_metadata": self._summarize_manifest_value(
                    base_artifact.get("metadata", {}).get("reader_metadata", {}),
                    artifact_key="raw.metadata",
                ),
            },
        }

    @staticmethod
    def _summarize_manifest_value(value: Any, artifact_key: str) -> Any:
        """Keep registry metadata inspectable without duplicating the artifact."""

        if isinstance(value, Mapping):
            return {
                str(key): AssetManager._summarize_manifest_value(
                    child, f"{artifact_key}.{key}"
                )
                for key, child in value.items()
            }
        if isinstance(value, (list, tuple, set)):
            if len(value) > 20:
                return {
                    "kind": type(value).__name__,
                    "length": int(len(value)),
                    "artifact_key": artifact_key,
                }
            return [
                AssetManager._summarize_manifest_value(
                    child, f"{artifact_key}[{index}]"
                )
                for index, child in enumerate(value)
            ]
        return value

    def _load_route_set_spec(self, spec_id: str) -> RouteSetSpecConfig:
        from .config_schemas import load_route_set_spec

        project_root = Path(__file__).resolve().parents[3]
        path = project_root / "configs" / "route_bank" / f"{spec_id}.yaml"
        if not path.exists():
            raise FileNotFoundError(f"Route set spec file not found: {path}")
        from omegaconf import OmegaConf

        return load_route_set_spec(OmegaConf.to_container(OmegaConf.load(path), resolve=True))

    def _coerce_route_requirement(self, value: Mapping[str, Any] | RouteSetRequirementConfig) -> RouteSetRequirementConfig:
        if isinstance(value, RouteSetRequirementConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("route_set requirement must be a mapping or RouteSetRequirementConfig.")
        return RouteSetRequirementConfig(
            spec_id=str(value["spec_id"]),
            k_active=int(value["k_active"]),
            weight_column=(
                None
                if value.get("weight_column") is None
                else str(value["weight_column"])
            ),
        )

    def _coerce_route_set_spec(self, value: Mapping[str, Any] | RouteSetSpecConfig) -> RouteSetSpecConfig:
        if isinstance(value, RouteSetSpecConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("route_set spec must be a mapping or RouteSetSpecConfig.")
        from .config_schemas import load_route_set_spec

        return load_route_set_spec(value)

    def _coerce_assignment_set_spec(
        self,
        value: Mapping[str, Any] | AssignmentSetSpecConfig,
    ) -> AssignmentSetSpecConfig:
        if isinstance(value, AssignmentSetSpecConfig):
            return value
        if not isinstance(value, Mapping):
            raise TypeError("assignment_set spec must be a mapping or AssignmentSetSpecConfig.")
        from .config_schemas import load_assignment_set_spec

        return load_assignment_set_spec(value)

    def _normalize_policy(
        self,
        policy: Mapping[str, Any] | AssetPolicyConfig | None,
    ) -> AssetPolicyConfig:
        if policy is None:
            return AssetPolicyConfig()
        if isinstance(policy, AssetPolicyConfig):
            return policy
        if not isinstance(policy, Mapping):
            raise TypeError("policy must be a mapping or AssetPolicyConfig.")
        return AssetPolicyConfig(
            on_missing=policy["on_missing"],
            on_stale=policy["on_stale"],
            overwrite_existing=policy["overwrite_existing"],
            require_exact_fingerprint=policy["require_exact_fingerprint"],
        )
