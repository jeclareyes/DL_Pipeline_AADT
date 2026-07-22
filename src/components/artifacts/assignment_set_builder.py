"""Materialize reusable assignment-set recipe assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .config_schemas import AssignmentSetSpecConfig
from .fingerprints import (
    compute_assignment_set_fingerprint,
    compute_assignment_set_signature,
    compute_network_fingerprint,
    compute_od_space_fingerprint,
)
from src.utils.serialization import dump, load


@dataclass(frozen=True)
class AssignmentSetAsset:
    """Serialized assignment-set asset payload."""

    asset_type: str
    spec_id: str
    fingerprint: str
    signature: str
    recipe: dict[str, Any]
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class AssignmentSetBuildResult:
    """Build result returned by AssignmentSetBuilder."""

    asset: AssignmentSetAsset
    path: Path
    manifest_entry: dict[str, Any]


class AssignmentSetBuilder:
    """Build and persist assignment-set recipe assets."""

    def __init__(self, base_artifact: Mapping[str, Any], output_root: str | Path) -> None:
        self.base_artifact = self._validate_base_artifact(base_artifact)
        self.output_root = Path(output_root)

    @classmethod
    def from_artifact_path(cls, artifact_path: str | Path, output_root: str | Path) -> "AssignmentSetBuilder":
        base_artifact = load(Path(artifact_path))
        return cls(base_artifact=base_artifact, output_root=output_root)

    def build(
        self,
        *,
        spec: AssignmentSetSpecConfig,
        route_set_entry: Mapping[str, Any],
    ) -> AssignmentSetBuildResult:
        """Build and persist an assignment-set asset."""

        if not isinstance(spec, AssignmentSetSpecConfig):
            raise TypeError("spec must be an AssignmentSetSpecConfig instance.")
        if not isinstance(route_set_entry, Mapping):
            raise TypeError("route_set_entry must be a mapping.")

        description = self.describe(spec=spec, route_set_entry=route_set_entry)

        recipe = {
            "spec": description["signature_payload"],
            "route_set_reference": {
                "spec_id": description["route_set_spec_id"],
                "fingerprint": description["route_set_fingerprint"],
                "path": route_set_entry.get("path"),
            },
        }

        metadata = {
            "base_fingerprint": description["base_fingerprint"],
            "route_set_fingerprint": description["route_set_fingerprint"],
            "route_set_spec_id": description["route_set_spec_id"],
        }
        diagnostics = {
            "route_set_required_k_active": int(spec.requires.k_active),
        }

        asset = AssignmentSetAsset(
            asset_type="assignment_set_asset",
            spec_id=spec.id,
            fingerprint=description["fingerprint"],
            signature=description["signature"],
            recipe=recipe,
            metadata=metadata,
            diagnostics=diagnostics,
        )

        path = description["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(asset, path)

        manifest_entry = {
            "id": spec.id,
            "spec_id": spec.id,
            "asset_type": "assignment_set",
            "path": str(path),
            "fingerprint": description["fingerprint"],
            "signature": description["signature"],
            "compatibility": {
                "base_fingerprint": description["base_fingerprint"],
                "route_set_fingerprint": description["route_set_fingerprint"],
            },
            "metadata": metadata,
            "diagnostics": diagnostics,
        }
        return AssignmentSetBuildResult(asset=asset, path=path, manifest_entry=manifest_entry)

    def describe(
        self,
        *,
        spec: AssignmentSetSpecConfig,
        route_set_entry: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Describe the assignment-set asset without materializing it."""

        if not isinstance(spec, AssignmentSetSpecConfig):
            raise TypeError("spec must be an AssignmentSetSpecConfig instance.")
        if not isinstance(route_set_entry, Mapping):
            raise TypeError("route_set_entry must be a mapping.")

        route_set_fingerprint = str(route_set_entry.get("fingerprint"))
        if not route_set_fingerprint:
            raise ValueError("route_set_entry must expose a non-empty fingerprint.")

        route_set_spec_id = str(route_set_entry.get("spec_id", route_set_entry.get("id", "")))
        if route_set_spec_id != spec.requires.spec_id:
            raise ValueError(
                f"assignment_set spec requires route_set {spec.requires.spec_id!r}, "
                f"but the resolved route_set entry is {route_set_spec_id!r}."
            )

        base_fingerprint = self._base_fingerprint()
        signature_payload = self._build_signature_payload(spec=spec, route_set_entry=route_set_entry)
        signature = compute_assignment_set_signature(signature_payload)
        fingerprint = compute_assignment_set_fingerprint(
            signature=signature_payload,
            base_fingerprint=base_fingerprint,
            route_set_fingerprint=route_set_fingerprint,
        )
        return {
            "signature_payload": signature_payload,
            "signature": signature,
            "fingerprint": fingerprint,
            "base_fingerprint": base_fingerprint,
            "route_set_fingerprint": route_set_fingerprint,
            "route_set_spec_id": route_set_spec_id,
            "path": self._resolve_output_path(spec_id=spec.id, fingerprint=fingerprint),
        }

    def _build_signature_payload(
        self,
        *,
        spec: AssignmentSetSpecConfig,
        route_set_entry: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "spec": {
                "id": spec.id,
                "requires": {
                    "spec_id": spec.requires.spec_id,
                    "k_active": int(spec.requires.k_active),
                },
                "behavior_model": {
                    "name": spec.behavior_model.name,
                    "theta": float(spec.behavior_model.theta),
                },
                "solver": {
                    "name": spec.solver.name,
                    "max_iterations": int(spec.solver.max_iterations),
                    "convergence_gap": float(spec.solver.convergence_gap),
                },
                "vdf": {
                    "name": spec.vdf.name,
                },
            },
            "route_set": {
                "spec_id": route_set_entry.get("spec_id", route_set_entry.get("id")),
                "fingerprint": route_set_entry.get("fingerprint"),
            },
        }

    def _resolve_output_path(self, *, spec_id: str, fingerprint: str) -> Path:
        return self.output_root / "asset_cache" / "assignment_banks" / f"{spec_id}__{fingerprint}.joblib"

    def _base_fingerprint(self) -> str:
        processed = self.base_artifact["processed"]
        raw = self.base_artifact["raw"]
        network_fp = compute_network_fingerprint(processed["link_df"], raw["nodes_df"])
        od_fp = compute_od_space_fingerprint(processed["od_indexing"].get("od_pairs", []))
        return f"{network_fp}:{od_fp}"

    @staticmethod
    def _validate_base_artifact(base_artifact: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(base_artifact, Mapping):
            raise TypeError("base_artifact must be a mapping.")
        artifact = dict(base_artifact)
        if "processed" not in artifact or "raw" not in artifact:
            raise ValueError("base_artifact must contain raw and processed sections.")
        return artifact
