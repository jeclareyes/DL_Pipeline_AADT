from __future__ import annotations

import joblib
import networkx as nx
import numpy as np
import pandas as pd

from src.components.artifacts.asset_manager import AssetManager
from src.components.artifacts.config_schemas import load_route_set_spec, RouteSetRequirementConfig


def _build_base_artifact() -> dict[str, object]:
    graph = nx.DiGraph()
    graph.add_edge(1, 2, link_id=1, free_flow_time=1.0, effective_capacity=1000.0, lanes=1)
    graph.add_edge(2, 3, link_id=2, free_flow_time=1.0, effective_capacity=1000.0, lanes=1)
    graph.add_edge(1, 3, link_id=3, free_flow_time=3.0, effective_capacity=1000.0, lanes=1)

    nodes_df = pd.DataFrame(
        [
            {"node_id": 1, "x": 0.0, "y": 0.0, "type": "zone"},
            {"node_id": 2, "x": 1.0, "y": 0.0, "type": "intersection"},
            {"node_id": 3, "x": 2.0, "y": 0.0, "type": "zone"},
        ]
    )
    link_df = pd.DataFrame(
        [
            {"link_id": 1, "init_node": 1, "term_node": 2, "free_flow_time": 1.0, "effective_capacity": 1000.0, "lanes": 1},
            {"link_id": 2, "init_node": 2, "term_node": 3, "free_flow_time": 1.0, "effective_capacity": 1000.0, "lanes": 1},
            {"link_id": 3, "init_node": 1, "term_node": 3, "free_flow_time": 3.0, "effective_capacity": 1000.0, "lanes": 1},
        ]
    )

    return {
        "artifact_type": "base_artifact",
        "artifact_version": "1.0",
        "dataset_name": "UnitTest",
        "created_at": "2026-01-01T00:00:00",
        "config": {"dataset_name": "UnitTest"},
        "environment": {},
        "paths": {},
        "raw": {
            "nodes_df": nodes_df,
            "network_df": link_df,
            "flow_df": pd.DataFrame(),
            "trips_df": pd.DataFrame(),
            "od_matrix": np.zeros((2, 2), dtype=float),
            "routes_by_od": {(1, 3): [[1, 3], [1, 2, 3]]},
            "routes_df": pd.DataFrame(),
            "metadata": {"trips": {"zone_ids": [1, 3]}},
        },
        "processed": {
            "link_df": link_df,
            "graph": graph,
            "edge_indexing": {"link_pair_indices": np.array([[1, 2], [2, 3], [1, 3]], dtype=np.int64)},
            "node_indexing": {},
            "od_indexing": {"od_pairs": [(1, 3)], "zone_ids": [1, 3]},
            "metadata": {},
            "routes_by_od": {(1, 3): [[1, 3], [1, 2, 3]]},
            "routes_df": pd.DataFrame(),
            "trips_df": pd.DataFrame(),
            "od_matrix": np.zeros((2, 2), dtype=float),
        },
        "metadata": {
            "dataset_name": "UnitTest",
            "processed_summary": {},
            "reader_metadata": {},
        },
    }


def test_asset_manager_materializes_and_reuses_route_set(tmp_path):
    manifest_path = tmp_path / "training_manifest.json"
    artifact_path = tmp_path / "base_artifact.joblib"
    joblib.dump(_build_base_artifact(), artifact_path)

    spec = load_route_set_spec(
        {
            "id": "fft_k20",
            "asset_type": "route_set",
            "builder": {"engine": "networkx", "weight": "free_flow_time", "k_generate": 2},
            "constraints": {"allow_duplicates": False, "allow_loops": False, "allow_auto_routes": False},
            "connectors": {"connector_link_types": [99]},
            "ordering": {"route_rank_policy": "engine_order", "cost_field": "free_flow_time"},
            "compatibility": {"requires_network_fingerprint": True, "requires_od_space_fingerprint": True},
            "storage": {"format": "joblib", "directory": "asset_cache/route_banks"},
        }
    )
    requirement = RouteSetRequirementConfig(spec_id="fft_k20", k_active=1)

    manager = AssetManager(
        manifest_path=manifest_path,
        base_artifact_path=artifact_path,
        policy={
            "on_missing": "build",
            "on_stale": "fail",
            "overwrite_existing": False,
            "require_exact_fingerprint": True,
        },
    )

    first = manager.resolve_route_set(requirement, spec)
    second = manager.resolve_route_set(requirement, spec)

    assert first["fingerprint"] == second["fingerprint"]
    assert first["path"] == second["path"]
    assert manifest_path.exists()

