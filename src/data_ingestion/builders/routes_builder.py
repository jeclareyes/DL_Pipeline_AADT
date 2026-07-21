from __future__ import annotations

import heapq
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from src.data_ingestion.builders.graph_builder import GraphBuilder
from src.data_ingestion.readers.tntp_network_reader import read_tntp_network
from src.data_ingestion.readers.tntp_node_reader import read_tntp_nodes
from src.data_ingestion.readers.tntp_trips_reader import read_tntp_trips
from src.components.route_engines import get_route_engine, RouteEngine

logger = logging.getLogger(__name__)

ODPair = Tuple[int, int]
Route = List[int]

_ROUTE_WORKER_ENGINE: RouteEngine | None = None
_ROUTE_WORKER_K: int = 10
_ROUTE_WORKER_WEIGHT: str = "free_flow_time"
_ROUTE_WORKER_CONSTRAINTS: dict[str, bool] | None = None
_ROUTE_WORKER_CONNECTOR_LINK_TYPES: set[int] | None = None


def _initialize_routes_worker(
    engine_name: str,
    graph: nx.DiGraph,
    k: int,
    weight: str,
    constraints: dict[str, bool],
    connector_link_types: set[int],
) -> None:
    global _ROUTE_WORKER_ENGINE, _ROUTE_WORKER_K, _ROUTE_WORKER_WEIGHT, _ROUTE_WORKER_CONSTRAINTS, _ROUTE_WORKER_CONNECTOR_LINK_TYPES
    _ROUTE_WORKER_ENGINE = get_route_engine(engine_name, graph)
    _ROUTE_WORKER_K = k
    _ROUTE_WORKER_WEIGHT = weight
    _ROUTE_WORKER_CONSTRAINTS = constraints
    _ROUTE_WORKER_CONNECTOR_LINK_TYPES = connector_link_types


def _solve_routes_batch_worker(
    batch_tasks: List[Tuple[int, int]]
) -> List[Tuple[Tuple[int, int], List[Route]]]:
    if _ROUTE_WORKER_ENGINE is None:
        raise RuntimeError("Worker engine not initialized.")

    results = []
    for origin_id, destination_id in batch_tasks:
        routes = _ROUTE_WORKER_ENGINE.get_k_routes(
            origin_id=origin_id,
            destination_id=destination_id,
            k=_ROUTE_WORKER_K,
            weight=_ROUTE_WORKER_WEIGHT,
            constraints=_ROUTE_WORKER_CONSTRAINTS,
            connector_link_types=_ROUTE_WORKER_CONNECTOR_LINK_TYPES,
        )
        results.append(((origin_id, destination_id), routes))
    return results


def _chunk_tasks(tasks: List[Tuple[int, int]], batch_size: int) -> List[List[Tuple[int, int]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    return [tasks[start:start + batch_size] for start in range(0, len(tasks), batch_size)]


def recompute_routes_from_tntp(cfg: DictConfig) -> Dict[str, object]:
    """
    Recompute the TNTP routes file from the current nodes/network/trips inputs.

    Diagnostics meaning:
        Rebuilds the K-shortest route sets so the routes file stays consistent
        with the latest network and trips inputs.

    Storage:
        Writes the routes file at cfg.input_routes.routes_route.
    """
    routes_cfg = dict(cfg.get("routes_recompute", {}) or {})
    enabled = bool(routes_cfg.get("enabled", False))
    if not enabled:
        return {"enabled": False, "skipped": True}

    routes_path = Path(cfg.input_routes.routes_route)
    overwrite_existing = bool(routes_cfg.get("overwrite_existing", True))

    if routes_path.exists() and not overwrite_existing:
        logger.info("Routes recompute skipped (exists): %s", routes_path)
        return {
            "enabled": True,
            "skipped": True,
            "routes_path": str(routes_path),
            "reason": "exists",
        }

    nodes_cfg = cfg.readers.get("nodes", {})
    network_cfg = cfg.readers.get("network", {})
    trips_cfg = cfg.readers.get("trips", {})

    node_result = read_tntp_nodes(
        path=cfg.input_routes.node_route,
        strict=bool(nodes_cfg.get("strict", True)),
        preserve_extra_columns=bool(nodes_cfg.get("preserve_extra_columns", True)),
    )

    network_result = read_tntp_network(
        path=cfg.input_routes.network_route,
        strict=bool(network_cfg.get("strict", True)),
        preserve_extra_columns=bool(network_cfg.get("preserve_extra_columns", True)),
    )

    trips_result = read_tntp_trips(
        path=cfg.input_routes.trips_route,
        aggregation=str(trips_cfg.get("aggregation", "average_daily")),
        matrix_format=str(trips_cfg.get("matrix_format", "csr")),
        include_zero_flows=bool(trips_cfg.get("include_zero_flows", False)),
        strict=bool(trips_cfg.get("strict", True)),
    )

    graph_cfg = cfg.get("graph", {})
    graph_builder = GraphBuilder(
        strict=bool(graph_cfg.get("strict", True)),
        add_missing_link_nodes=bool(graph_cfg.get("add_missing_link_nodes", False)),
        preserve_extra_attributes=bool(graph_cfg.get("preserve_extra_attributes", True)),
        weight_column=str(graph_cfg.get("weight_column", "free_flow_time")),
    )

    graph = graph_builder.build(
        link_df=network_result.network_df,
        node_df=node_result.nodes_df,
    ).graph

    zone_ids = trips_result.metadata.get("zone_ids", [])
    if not zone_ids:
        raise ValueError("Routes recompute failed: no zone_ids found in trips metadata.")

    k_paths = int(routes_cfg.get("k_paths", cfg.readers.routes.max_routes_per_od))
    weight = str(routes_cfg.get("weight", graph_cfg.get("weight_column", "free_flow_time")))
    engine_name = str(routes_cfg.get("engine", "networkx"))


    allow_duplicates = bool(routes_cfg.get("allow_duplicates"))
    allow_loops = bool(routes_cfg.get("allow_loops"))
    allow_auto_routes = bool(routes_cfg.get("allow_auto_routes"))
    connector_link_types = _as_int_set(routes_cfg.get("connector_link_types", []))

    constraints = {
        "allow_duplicates": allow_duplicates,
        "allow_loops": allow_loops,
        "allow_auto_routes": allow_auto_routes,
    }

    parallel = bool(routes_cfg.get("parallel", True))
    parallel_workers = routes_cfg.get("parallel_workers", 10)
    if parallel_workers is not None:
        parallel_workers = int(parallel_workers)
    od_batch_size = int(routes_cfg.get("od_batch_size", 10))
    show_progress = bool(routes_cfg.get("show_progress", True))
    require_exact_k_routes = bool(routes_cfg.get("require_exact_k_routes", False))

    all_od_tasks = []
    for origin_id in zone_ids:
        for destination_id in zone_ids:
            origin_id = int(origin_id)
            destination_id = int(destination_id)

            if origin_id == destination_id and not allow_auto_routes:
                continue
            all_od_tasks.append((origin_id, destination_id))

    total_od_pairs = len(all_od_tasks)
    routes_by_od: Dict[ODPair, List[Route]] = {}
    od_pairs_without_routes = []
    od_pairs_with_less_than_k = []
    total_routes_found = 0

    logging.info(
        f"Recomputing routes with engine {engine_name}, weight {weight}, k={k_paths}\n"
        f"using parallel={parallel}, parallel_workers={parallel_workers}, od_batch_size={od_batch_size}\n"
        f"Constrains: {[(k, v) for k, v in constraints.items()]}"
    )

    if parallel:
        if parallel_workers is None:
            parallel_workers = max(1, (os.cpu_count() or 2) - 1)
        if parallel_workers <= 1:
            parallel = False

    if parallel:
        batches = _chunk_tasks(all_od_tasks, od_batch_size)
        if show_progress:
            pbar = tqdm(total=total_od_pairs, desc="Recomputing routes", unit="OD pair")
        else:
            pbar = None

        try:
            with ProcessPoolExecutor(
                max_workers=parallel_workers,
                initializer=_initialize_routes_worker,
                initargs=(engine_name, graph, k_paths, weight, constraints, connector_link_types),
            ) as executor:
                futures = [executor.submit(_solve_routes_batch_worker, batch) for batch in batches]
                for future in as_completed(futures):
                    for od_pair, routes in future.result():
                        routes_by_od[od_pair] = routes
                        num_routes = len(routes)
                        total_routes_found += num_routes
                        if num_routes == 0:
                            od_pairs_without_routes.append(od_pair)
                        if num_routes < k_paths:
                            od_pairs_with_less_than_k.append(od_pair)

                        if pbar is not None:
                            pbar.update(1)
                            pbar.set_postfix({
                                "routes": total_routes_found,
                                "no_route": len(od_pairs_without_routes),
                                "less_k": len(od_pairs_with_less_than_k),
                            })
        finally:
            if pbar is not None:
                pbar.close()
    else:
        if show_progress:
            pbar = tqdm(total=total_od_pairs, desc="Recomputing routes", unit="OD pair")
        else:
            pbar = None

        try:
            engine = get_route_engine(engine_name, graph)
            for origin_id, destination_id in all_od_tasks:
                routes = engine.get_k_routes(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    k=k_paths,
                    weight=weight,
                    constraints=constraints,
                    connector_link_types=connector_link_types,
                )
                routes_by_od[(origin_id, destination_id)] = routes
                num_routes = len(routes)
                total_routes_found += num_routes
                if num_routes == 0:
                    od_pairs_without_routes.append((origin_id, destination_id))
                if num_routes < k_paths:
                    od_pairs_with_less_than_k.append((origin_id, destination_id))

                if pbar is not None:
                    pbar.update(1)
                    pbar.set_postfix({
                        "routes": total_routes_found,
                        "no_route": len(od_pairs_without_routes),
                        "less_k": len(od_pairs_with_less_than_k),
                    })
        finally:
            if pbar is not None:
                pbar.close()

    # Reorder routes_by_od to match deterministic original nested-loop order
    ordered_routes_by_od = {}
    for od_pair in all_od_tasks:
        if od_pair in routes_by_od:
            ordered_routes_by_od[od_pair] = routes_by_od[od_pair]
    routes_by_od = ordered_routes_by_od

    _save_routes_as_tntp(routes_by_od=routes_by_od, routes_path=routes_path)

    counts = [len(routes) for routes in routes_by_od.values()]
    metadata = {
        "enabled": True,
        "routes_path": str(routes_path),
        "num_od_pairs": int(len(routes_by_od)),
        "k_requested": int(k_paths),
        "min_routes_found": int(min(counts)) if counts else 0,
        "max_routes_found": int(max(counts)) if counts else 0,
        "od_pairs_with_less_than_k": int(len(od_pairs_with_less_than_k)),
        "od_pairs_without_routes": int(len(od_pairs_without_routes)),
        "od_pairs_with_less_than_k_sample": od_pairs_with_less_than_k[:20],
        "od_pairs_without_routes_sample": od_pairs_without_routes[:20],
        "weight": weight,
        "engine": engine_name,
        "constraints": constraints,
        # Not so relevant. Perhaps remove later
        # "parallel": parallel,
        # "parallel_workers": parallel_workers,
        # "od_batch_size": od_batch_size,
        # Until here.
    }

    logger.info(
        "Routes recomputed | od_pairs=%d | k=%d | min_routes=%d | max_routes=%d | path=%s",
        metadata["num_od_pairs"],
        metadata["k_requested"],
        metadata["min_routes_found"],
        metadata["max_routes_found"],
        metadata["routes_path"],
    )

    if require_exact_k_routes and od_pairs_with_less_than_k:
        sample = od_pairs_with_less_than_k[:20]
        raise ValueError(
            "Route generation did not find the required number of routes for all OD pairs. "
            f"Required k_paths={k_paths}. "
            f"Pairs with less than K: {len(od_pairs_with_less_than_k)}. "
            f"Sample: {sample}. "
            "This usually means the graph does not contain enough distinct simple paths "
            "for those OD pairs."
        )

    return metadata


def _as_int_set(values: Optional[Iterable[object]]) -> set[int]:
    if values is None:
        return set()
    return {int(v) for v in values}


def _save_routes_as_tntp(
    routes_by_od: Dict[ODPair, List[Route]],
    routes_path: Path,
) -> None:
    lines = []

    """
    for _, routes in routes_by_od.items():
        if routes:
            lines.append(str(routes))
        else:
            lines.append("[]")
    """

    for (origin, destination), routes in routes_by_od.items():
        route_text = str(routes) if routes else "[]"
        lines.append(f"{int(origin)} {int(destination)} : {route_text}")

    routes_path.parent.mkdir(parents=True, exist_ok=True)
    routes_path.write_text("\n".join(lines))
