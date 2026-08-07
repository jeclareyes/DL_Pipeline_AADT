"""Materialize reusable assignment-set recipe assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from omegaconf import OmegaConf

from src.components.assignment_motors import build_assignment_composition
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
    assignment_result: dict[str, Any]
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

    def __init__(
        self,
        base_artifact: Mapping[str, Any],
        output_root: str | Path,
        creation_artifact: Mapping[str, Any] | str | Path,
    ) -> None:
        self.base_artifact = self._validate_base_artifact(base_artifact)
        self.output_root = Path(output_root)
        self.creation_artifact = self._load_creation_artifact(creation_artifact)

    @classmethod
    def from_artifact_path(
        cls,
        artifact_path: str | Path,
        output_root: str | Path,
        creation_artifact: Mapping[str, Any] | str | Path,
    ) -> "AssignmentSetBuilder":
        base_artifact = load(Path(artifact_path))
        return cls(
            base_artifact=base_artifact,
            output_root=output_root,
            creation_artifact=creation_artifact,
        )

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

        route_asset = load(route_set_entry["path"])
        route_set = getattr(route_asset, "route_set", None)
        if route_set is None:
            raise TypeError("route_set_entry does not point to a materialized RouteSet asset.")

        theta = description["theta"]
        assignment_config = self._load_assignment_config(spec, theta=theta)
        links_df = self.base_artifact["processed"]["link_df"]
        routes_by_od = self._routes_by_od(route_set)
        zone_ids = [
            int(value)
            for value in self.base_artifact["processed"]["od_indexing"]["zone_ids"]
        ]
        zone_id_to_idx = {zone_id: index for index, zone_id in enumerate(zone_ids)}
        od_matrix = self.base_artifact["raw"]["od_matrix"]
        if hasattr(od_matrix, "toarray"):
            od_matrix = od_matrix.toarray()
        od_matrix = np.asarray(od_matrix, dtype=float)

        composition = build_assignment_composition(
            links_df=links_df,
            routes_by_od=routes_by_od,
            zone_id_to_idx=zone_id_to_idx,
            assignment_config=assignment_config,
            training_config={},
            artifacts={},
        )
        result = composition.behavior_model.solve(
            od_matrix=od_matrix,
            config=composition.runtime_config,
        )
        assignment_result = {
            "final_route_flows": np.asarray(result.final_route_flows, dtype=float),
            "final_link_flows": np.asarray(result.final_link_flows, dtype=float),
            "final_link_costs": np.asarray(result.final_link_costs, dtype=float),
            "final_route_costs": np.asarray(result.final_route_costs, dtype=float),
            "metadata": {
                **dict(result.metadata),
                "creation_artifact_theta": theta,
            },
        }

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
            "creation_artifact_fingerprint": description["creation_artifact_fingerprint"],
            "theta": theta,
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
            assignment_result=assignment_result,
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

    @staticmethod
    def _routes_by_od(route_set: Any) -> dict[tuple[int, int], list[list[int]]]:
        routes_df = route_set.routes_df
        return {
            (int(origin), int(destination)): [
                list(routes_df.iloc[int(route_index)]["route_nodes"])
                for route_index in route_indices
            ]
            for (origin, destination), route_indices in route_set.od_to_route_indices.items()
        }

    @staticmethod
    def _load_assignment_config(spec: AssignmentSetSpecConfig, *, theta: float | None) -> Any:
        project_root = Path(__file__).resolve().parents[3]
        config_path = project_root / spec.base_config
        if not config_path.exists():
            raise FileNotFoundError(f"Assignment base config not found: {config_path}")

        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        flow_cfg = spec
        config["behavior_model"]["name"] = flow_cfg.behavior_model.name
        config["common"]["max_iterations"] = int(flow_cfg.solver.max_iterations)
        config["cost_function"]["vdf_name"] = flow_cfg.vdf.name
        config["solvers"]["active_solver"] = flow_cfg.solver.name
        if flow_cfg.behavior_model.name == "stochastic_user_equilibrium":
            if theta is None:
                raise ValueError(
                    "A stochastic_user_equilibrium assignment requires theta from "
                    "the dataset creation artifact."
                )
            config["stochastic_user_equilibrium"]["logit"]["theta_source"] = "explicit"
            config["stochastic_user_equilibrium"]["logit"]["theta_value"] = float(theta)
            config["stochastic_user_equilibrium"]["logit"]["theta_artifact_key"] = None
            config["stochastic_user_equilibrium"]["convergence"]["max_relative_gap_threshold"] = float(
                flow_cfg.solver.convergence_gap
            )

        from src.components.assignment_motors import build_assignment_config_from_mapping

        return build_assignment_config_from_mapping(config)

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

        theta = self._resolve_theta(spec)
        creation_artifact_fingerprint = self._creation_artifact_fingerprint()
        base_fingerprint = self._base_fingerprint(creation_artifact_fingerprint)
        signature_payload = self._build_signature_payload(
            spec=spec,
            route_set_entry=route_set_entry,
            theta=theta,
            creation_artifact_fingerprint=creation_artifact_fingerprint,
        )
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
            "theta": theta,
            "creation_artifact_fingerprint": creation_artifact_fingerprint,
            "path": self._resolve_output_path(spec_id=spec.id, fingerprint=fingerprint),
        }

    def _build_signature_payload(
        self,
        *,
        spec: AssignmentSetSpecConfig,
        route_set_entry: Mapping[str, Any],
        theta: float | None,
        creation_artifact_fingerprint: str,
    ) -> dict[str, Any]:
        return {
            "spec": {
                "id": spec.id,
                "materialization_version": 2,
                "requires": {
                    "spec_id": spec.requires.spec_id,
                    "k_active": int(spec.requires.k_active),
                },
                "behavior_model": {
                    "name": spec.behavior_model.name,
                    "theta_source": "dataset_creation_artifact",
                    "theta": theta,
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
            "creation_artifact": {
                "fingerprint": creation_artifact_fingerprint,
            },
        }

    def _resolve_output_path(self, *, spec_id: str, fingerprint: str) -> Path:
        return self.output_root / "asset_cache" / "assignment_banks" / f"{spec_id}__{fingerprint}.joblib"

    def _base_fingerprint(self, creation_artifact_fingerprint: str) -> str:
        processed = self.base_artifact["processed"]
        raw = self.base_artifact["raw"]
        network_fp = compute_network_fingerprint(processed["link_df"], raw["nodes_df"])
        od_fp = compute_od_space_fingerprint(processed["od_indexing"].get("od_pairs", []))
        return f"{network_fp}:{od_fp}:creation:{creation_artifact_fingerprint}"

    def _resolve_theta(self, spec: AssignmentSetSpecConfig) -> float | None:
        if spec.behavior_model.name != "stochastic_user_equilibrium":
            return None

        try:
            theta = self.creation_artifact["config"]["AssignmentParameters"]["SUE_Parameters"]["theta"]
        except KeyError as exc:
            raise KeyError(
                "Creation artifact is missing config.AssignmentParameters."
                "SUE_Parameters.theta required by stochastic_user_equilibrium."
            ) from exc

        try:
            theta_value = float(theta)
        except (TypeError, ValueError) as exc:
            raise TypeError("Dataset creation theta must be numeric.") from exc
        if not np.isfinite(theta_value) or theta_value <= 0.0:
            raise ValueError("Dataset creation theta must be finite and greater than zero.")
        return theta_value

    def _creation_artifact_fingerprint(self) -> str:
        config_hash = self.creation_artifact.get("config_hash")
        artifact_id = self.creation_artifact.get("artifact_id")
        if not config_hash and not artifact_id:
            raise ValueError(
                "Creation artifact must expose config_hash or artifact_id for assignment provenance."
            )
        return f"{config_hash or ''}:{artifact_id or ''}"

    @staticmethod
    def _load_creation_artifact(
        creation_artifact: Mapping[str, Any] | str | Path,
    ) -> dict[str, Any]:
        if isinstance(creation_artifact, (str, Path)):
            creation_artifact = load(Path(creation_artifact))
        if not isinstance(creation_artifact, Mapping):
            raise TypeError("creation_artifact must be a mapping or an artifact path.")
        artifact = dict(creation_artifact)
        if artifact.get("artifact_type") != "dataset_master":
            raise ValueError(
                "creation_artifact must have artifact_type='dataset_master'."
            )
        return artifact

    @staticmethod
    def _validate_base_artifact(base_artifact: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(base_artifact, Mapping):
            raise TypeError("base_artifact must be a mapping.")
        artifact = dict(base_artifact)
        if "processed" not in artifact or "raw" not in artifact:
            raise ValueError("base_artifact must contain raw and processed sections.")
        return artifact
