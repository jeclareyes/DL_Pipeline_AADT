"""Materialize reusable route-set assets from a base artifact and a recipe."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import networkx as nx
import pandas as pd

from src.components.assignment_motors.route_set import RouteInputFormat, RouteSet, RouteSetBuildConfig
from src.components.route_engines import get_route_engine
from src.utils.serialization import dump, load

from .config_schemas import RouteSetSpecConfig, RouteSetRequirementConfig
from .fingerprints import (
    compute_link_order_fingerprint,
    compute_od_space_fingerprint,
    compute_route_set_fingerprint,
    compute_route_set_signature,
    compute_network_fingerprint,
    compute_zone_order_fingerprint,
)


@dataclass(frozen=True)
class RouteSetAsset:
    """Serialized route-set asset payload."""

    asset_type: str
    spec_id: str
    fingerprint: str
    signature: str
    route_set: RouteSet
    metadata: dict[str, Any]
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class RouteSetBuildResult:
    """Build result returned by RouteSetBuilder."""

    asset: RouteSetAsset
    path: Path
    manifest_entry: dict[str, Any]


class RouteSetBuilder:
    """Build a reusable route-set asset from a base artifact."""

    def __init__(self, base_artifact: Mapping[str, Any], output_root: str | Path) -> None:
        self.base_artifact = self._validate_base_artifact(base_artifact)
        self.output_root = Path(output_root)

    @classmethod
    def from_artifact_path(cls, artifact_path: str | Path, output_root: str | Path) -> "RouteSetBuilder":
        base_artifact = load(Path(artifact_path))
        return cls(base_artifact=base_artifact, output_root=output_root)

    def build(
        self,
        *,
        spec: RouteSetSpecConfig,
        requirement: RouteSetRequirementConfig | None = None,
    ) -> RouteSetBuildResult:
        """Build and persist a route-set asset."""

        if not isinstance(spec, RouteSetSpecConfig):
            raise TypeError("spec must be a RouteSetSpecConfig instance.")
        if requirement is not None and not isinstance(requirement, RouteSetRequirementConfig):
            raise TypeError("requirement must be a RouteSetRequirementConfig instance or None.")

        description = self.describe(spec=spec, requirement=requirement)
        processed = self.base_artifact["processed"]
        graph = processed["graph"]
        link_df = processed["link_df"]
        od_pairs = self._extract_od_pairs(processed)
        zone_ids = self._extract_zone_ids(processed)

        routes_by_od = self._generate_routes_by_od(
            graph=graph,
            od_pairs=od_pairs,
            spec=spec,
        )
        routes_by_od = self._apply_ordering(routes_by_od=routes_by_od, link_df=link_df, spec=spec)

        route_set_config = self._build_route_set_config(spec=spec)
        route_set = RouteSet.from_routes_by_od(
            links_df=link_df,
            routes_by_od=routes_by_od,
            config=route_set_config,
        )

        metadata = {
            "spec": description["signature_payload"],
            "route_count": int(route_set.number_of_routes),
            "od_pair_count": int(len(route_set.od_to_route_indices)),
            "k_generate": int(spec.builder.k_generate),
            "k_active": None if requirement is None else int(requirement.k_active),
            "canonical_link_id_order": list(route_set.canonical_link_id_order),
            "route_ordering": spec.ordering.route_rank_policy,
        }

        diagnostics = self._build_diagnostics(
            routes_by_od=routes_by_od,
            k_generate=int(spec.builder.k_generate),
        )
        asset = RouteSetAsset(
            asset_type="route_set_asset",
            spec_id=spec.id,
            fingerprint=description["fingerprint"],
            signature=description["signature"],
            route_set=route_set,
            metadata=metadata,
            diagnostics=diagnostics,
        )

        path = description["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(asset, path)

        manifest_entry = {
            "id": spec.id,
            "spec_id": spec.id,
            "asset_type": "route_set",
            "path": str(path),
            "fingerprint": description["fingerprint"],
            "signature": description["signature"],
            "compatibility": {
                "network_fingerprint": description["network_fingerprint"],
                "od_space_fingerprint": description["od_space_fingerprint"],
                "link_order_fingerprint": compute_link_order_fingerprint(route_set.canonical_link_id_order),
                "zone_order_fingerprint": compute_zone_order_fingerprint(zone_ids),
            },
            "metadata": metadata,
            "diagnostics": diagnostics,
        }
        return RouteSetBuildResult(asset=asset, path=path, manifest_entry=manifest_entry)

    def describe(
        self,
        *,
        spec: RouteSetSpecConfig,
        requirement: RouteSetRequirementConfig | None = None,
    ) -> dict[str, Any]:
        """Describe the route-set asset without materializing it."""

        if not isinstance(spec, RouteSetSpecConfig):
            raise TypeError("spec must be a RouteSetSpecConfig instance.")
        if requirement is not None and not isinstance(requirement, RouteSetRequirementConfig):
            raise TypeError("requirement must be a RouteSetRequirementConfig instance or None.")

        if spec.builder.k_generate < 1:
            raise ValueError("Route set builder requires k_generate >= 1.")
        if requirement is not None and requirement.k_active > spec.builder.k_generate:
            raise ValueError(
                f"route_set requirement k_active={requirement.k_active} exceeds "
                f"spec.builder.k_generate={spec.builder.k_generate}."
            )

        processed = self.base_artifact["processed"]
        zone_ids = self._extract_zone_ids(processed)
        signature_payload = self._build_signature_payload(spec=spec, requirement=requirement, zone_ids=zone_ids)
        signature = compute_route_set_signature(signature_payload)
        network_fp = self._network_fingerprint()
        od_fp = self._od_space_fingerprint()
        fingerprint = compute_route_set_fingerprint(
            signature=signature_payload,
            network_fp=network_fp,
            od_fp=od_fp,
        )
        return {
            "signature_payload": signature_payload,
            "signature": signature,
            "fingerprint": fingerprint,
            "network_fingerprint": network_fp,
            "od_space_fingerprint": od_fp,
            "path": self._resolve_output_path(spec=spec, fingerprint=fingerprint),
        }

    def _generate_routes_by_od(
        self,
        *,
        graph: nx.DiGraph,
        od_pairs: list[tuple[int, int]],
        spec: RouteSetSpecConfig,
    ) -> dict[tuple[int, int], list[list[int]]]:
        engine = get_route_engine(spec.builder.engine, graph)
        constraints = {
            "allow_duplicates": bool(spec.constraints.allow_duplicates),
            "allow_loops": bool(spec.constraints.allow_loops),
            "allow_auto_routes": bool(spec.constraints.allow_auto_routes),
        }
        connector_link_types = {int(value) for value in spec.connectors.connector_link_types}

        routes_by_od: dict[tuple[int, int], list[list[int]]] = {}
        missing_pairs: list[tuple[int, int]] = []
        for origin_id, destination_id in od_pairs:
            routes = engine.get_k_routes(
                origin_id=origin_id,
                destination_id=destination_id,
                k=int(spec.builder.k_generate),
                weight=str(spec.builder.weight),
                constraints=constraints,
                connector_link_types=connector_link_types,
            )
            route_lists = [list(route) for route in routes]
            if not route_lists:
                missing_pairs.append((int(origin_id), int(destination_id)))
            routes_by_od[(int(origin_id), int(destination_id))] = route_lists

        if missing_pairs:
            raise ValueError(
                "Route-set materialization failed because some OD pairs produced no routes. "
                f"First missing pairs: {missing_pairs[:10]}"
            )

        return routes_by_od

    def _apply_ordering(
        self,
        *,
        routes_by_od: dict[tuple[int, int], list[list[int]]],
        link_df: pd.DataFrame,
        spec: RouteSetSpecConfig,
    ) -> dict[tuple[int, int], list[list[int]]]:
        if spec.ordering.route_rank_policy == "engine_order":
            return routes_by_od
        if spec.ordering.route_rank_policy != "free_flow_cost_ascending":
            raise ValueError(
                f"Unsupported route_rank_policy={spec.ordering.route_rank_policy!r}. "
                "Supported values: 'engine_order', 'free_flow_cost_ascending'."
            )

        link_cost_by_edge = {
            (int(row.init_node), int(row.term_node)): float(getattr(row, spec.ordering.cost_field))
            for row in link_df.itertuples(index=False)
        }

        ordered: dict[tuple[int, int], list[list[int]]] = {}
        for od_pair, routes in routes_by_od.items():
            ordered[od_pair] = sorted(
                routes,
                key=lambda route: sum(
                    link_cost_by_edge[(int(u), int(v))]
                    for u, v in zip(route[:-1], route[1:])
                ),
            )
        return ordered

    def _build_route_set_config(self, *, spec: RouteSetSpecConfig) -> RouteSetBuildConfig:
        return RouteSetBuildConfig(
            route_input_format=RouteInputFormat.NODE_SEQUENCE,
            link_id_col="link_id",
            init_node_col="init_node",
            term_node_col="term_node",
            route_cost_col=spec.ordering.cost_field,
            fail_on_duplicate_routes=not bool(spec.constraints.allow_duplicates),
            fail_on_empty_route_set=True,
            fail_on_missing_od_routes=False,
            require_simple_node_routes=not bool(spec.constraints.allow_loops),
            require_unique_link_ids=True,
            require_unique_directed_edges=True,
        )

    def _build_signature_payload(
        self,
        *,
        spec: RouteSetSpecConfig,
        requirement: RouteSetRequirementConfig | None,
        zone_ids: list[int],
    ) -> dict[str, Any]:
        return {
            "spec": {
                "id": spec.id,
                "builder": {
                    "engine": spec.builder.engine,
                    "weight": spec.builder.weight,
                    "k_generate": int(spec.builder.k_generate),
                },
                "constraints": {
                    "allow_duplicates": spec.constraints.allow_duplicates,
                    "allow_loops": spec.constraints.allow_loops,
                    "allow_auto_routes": spec.constraints.allow_auto_routes,
                },
                "connectors": list(spec.connectors.connector_link_types),
                "ordering": {
                    "route_rank_policy": spec.ordering.route_rank_policy,
                    "cost_field": spec.ordering.cost_field,
                },
            },
            "requirement": None
            if requirement is None
            else {
                "spec_id": requirement.spec_id,
                "k_active": int(requirement.k_active),
            },
            "zone_ids": [int(zone_id) for zone_id in zone_ids],
        }

    def _build_diagnostics(
        self,
        *,
        routes_by_od: Mapping[tuple[int, int], list[list[int]]],
        k_generate: int,
    ) -> dict[str, Any]:
        route_counts = [len(routes) for routes in routes_by_od.values()]
        partial_pairs = [
            od_pair
            for od_pair, routes in routes_by_od.items()
            if len(routes) < k_generate
        ]
        return {
            "num_od_pairs": int(len(routes_by_od)),
            "min_routes_found": int(min(route_counts)) if route_counts else 0,
            "max_routes_found": int(max(route_counts)) if route_counts else 0,
            "od_pairs_with_less_than_k": int(len(partial_pairs)),
            "od_pairs_with_less_than_k_sample": [list(pair) for pair in partial_pairs[:20]],
        }

    def _resolve_output_path(self, *, spec: RouteSetSpecConfig, fingerprint: str) -> Path:
        return self.output_root / spec.storage.directory / f"{spec.id}__{fingerprint}.joblib"

    def _network_fingerprint(self) -> str:
        processed = self.base_artifact["processed"]
        raw = self.base_artifact["raw"]
        return compute_network_fingerprint(processed["link_df"], raw["nodes_df"])

    def _od_space_fingerprint(self) -> str:
        processed = self.base_artifact["processed"]
        od_indexing = processed["od_indexing"]
        return compute_od_space_fingerprint(od_indexing.get("od_pairs", []))

    def _extract_od_pairs(self, processed: Mapping[str, Any]) -> list[tuple[int, int]]:
        od_indexing = processed.get("od_indexing", {})
        od_pairs = od_indexing.get("od_pairs")
        if not od_pairs:
            raise ValueError("Base artifact does not expose processed['od_indexing']['od_pairs'].")
        return [(int(origin), int(destination)) for origin, destination in od_pairs]

    def _extract_zone_ids(self, processed: Mapping[str, Any]) -> list[int]:
        od_indexing = processed.get("od_indexing", {})
        zone_ids = od_indexing.get("zone_ids")
        if not zone_ids:
            raise ValueError("Base artifact does not expose processed['od_indexing']['zone_ids'].")
        return [int(zone_id) for zone_id in zone_ids]

    @staticmethod
    def _validate_base_artifact(base_artifact: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(base_artifact, Mapping):
            raise TypeError("base_artifact must be a mapping.")
        artifact = dict(base_artifact)
        if "processed" not in artifact or "raw" not in artifact:
            raise ValueError("base_artifact must contain raw and processed sections.")
        return artifact
