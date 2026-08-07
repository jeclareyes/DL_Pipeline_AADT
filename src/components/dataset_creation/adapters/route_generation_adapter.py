from __future__ import annotations
from src.components.dataset_creation.config import DatasetConfig

import networkx as nx
import logging
from typing import Any

import pandas as pd

from ..exporters.tntp_exporter import save_routes_as_tntp

LOGGER = logging.getLogger(__name__)

def _path_has_repeated_links(path: list[int]) -> bool:
    links = list(zip(path[:-1], path[1:]))
    return len(links) != len(set(links))


def _path_uses_invalid_connector(
    *,
    graph: nx.DiGraph,
    path: list[int],
    origin_id: int,
    destination_id: int,
    connector_link_types: set[int],
) -> bool:
    origin_id = int(origin_id)
    destination_id = int(destination_id)

    for init_node, term_node in zip(path[:-1], path[1:]):
        edge_data = graph[int(init_node)][int(term_node)]
        link_type = int(edge_data.get("link_type", -1))
        if link_type not in connector_link_types:
            continue
        if not (int(init_node) == origin_id or int(term_node) == destination_id):
            return True
    return False


def _compute_route_weight(graph: nx.DiGraph, route: list[int], weight: str) -> float:
    total_weight = 0.0
    for init_node, term_node in zip(route[:-1], route[1:]):
        total_weight += float(graph[int(init_node)][int(term_node)][weight])
    return total_weight


def _build_routes_dataframe(
    *,
    graph: nx.DiGraph,
    routes_by_od: dict[tuple[int, int], list[list[int]]],
    weight: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (origin_id, destination_id), routes in routes_by_od.items():
        for route_idx, route in enumerate(routes):
            rows.append(
                {
                    "origin_id": int(origin_id),
                    "destination_id": int(destination_id),
                    "route_idx": int(route_idx),
                    "route": route,
                    "num_nodes": len(route),
                    "num_links": len(route) - 1,
                    "weight_name": weight,
                    "route_weight": _compute_route_weight(graph, route, weight),
                }
            )
    return pd.DataFrame(rows)


def build_routes(config: DatasetConfig, data: dict[str, Any], metadata: dict[str, Any]) -> tuple[dict[tuple[int, int], list[list[int]]], dict[str, Any]]:
    from src.components.route_engines.base_engine import generate_routes_by_od

    if data["graph"] is None:
        raise ValueError("Scenario graph is missing. Run network generation before routes.")
    if not metadata["trips"] or "zone_ids" not in metadata["trips"]:
        raise ValueError("Trips metadata is missing zone_ids. Run demand generation before routes.")

    route_cfg = config.RouteParameters
    k_routes = int(route_cfg.K_paths)
    weight = str(route_cfg.Weight)
    constraints = {
        "allow_duplicates": bool(route_cfg.allow_duplicates),
        "allow_loops": bool(route_cfg.allow_loops),
        "allow_auto_routes": bool(route_cfg.allow_auto_routes),
    }
    connector_link_types = set(int(v) for v in metadata["network"].get("connector_link_types", []))
    engine_name = str(getattr(route_cfg, "Engine", "igraph_native"))

    zone_ids = metadata["trips"]["zone_ids"]
    od_pairs = [
        (int(origin_id), int(destination_id))
        for origin_id in zone_ids
        for destination_id in zone_ids
        if int(origin_id) != int(destination_id) or constraints["allow_auto_routes"]
    ]
    routes_by_od = generate_routes_by_od(
        graph=data["graph"],
        od_pairs=od_pairs,
        engine_name=engine_name,
        k=k_routes,
        weight=weight,
        constraints=constraints,
        connector_link_types=connector_link_types,
        parallel=True,
        show_progress=True,
    )

    for od_pair, candidate_routes in routes_by_od.items():
        origin_id, destination_id = od_pair
        filtered_routes: list[list[int]] = []
        seen_routes: set[tuple[int, ...]] = set()
        for route in candidate_routes:
            route = [int(node) for node in route]
            route_key = tuple(route)
            if route_key in seen_routes and not constraints["allow_duplicates"]:
                continue
            if _path_has_repeated_links(route) and not constraints["allow_loops"]:
                continue
            if _path_uses_invalid_connector(
                graph=data["graph"],
                path=route,
                origin_id=origin_id,
                destination_id=destination_id,
                connector_link_types=connector_link_types,
            ):
                continue
            filtered_routes.append(route)
            seen_routes.add(route_key)
        routes_by_od[od_pair] = filtered_routes

    routes_path = save_routes_as_tntp(routes_by_od, config.paths.export_filepaths.routes)
    routes_df = _build_routes_dataframe(graph=data["graph"], routes_by_od=routes_by_od, weight=weight)

    data["routes_df"] = routes_df
    
    route_counts = {od_pair: len(routes) for od_pair, routes in routes_by_od.items()}
    metadata: dict[str, Any] = {
        "k_requested": k_routes,
        "weight": weight,
        "zone_ids": zone_ids,
        "num_od_pairs": len(routes_by_od),
        "route_counts": route_counts,
        "od_pairs_with_less_than_k_routes": [
            od_pair for od_pair, count in route_counts.items() if count < k_routes
        ],
        "od_pairs_without_routes": [
            od_pair for od_pair, count in route_counts.items() if count == 0
        ],
        "routes_path": str(routes_path),
        "constraints": constraints,
        "engine": engine_name,
    }
    return routes_by_od, metadata
