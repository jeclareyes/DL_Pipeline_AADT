from __future__ import annotations

import heapq
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import count
from typing import Any

import networkx as nx
import pandas as pd

try:
    from .od_utils import get_zone_ids_from_nodes_tntp
    from .tntp_builders import build_export_graph_from_network_tntp
except (ImportError, ValueError):
    try:
        from data_handling.data_from_pickle.od_utils import get_zone_ids_from_nodes_tntp
        from data_handling.data_from_pickle.tntp_builders import build_export_graph_from_network_tntp
    except ImportError:
        from od_utils import get_zone_ids_from_nodes_tntp  # type: ignore
        from tntp_builders import build_export_graph_from_network_tntp  # type: ignore

_ROUTE_WORKER_GRAPH: nx.DiGraph | None = None
_ROUTE_WORKER_REACHABILITY: dict[str, set[str]] | None = None
_ROUTE_WORKER_CONFIG: dict[str, Any] | None = None


def path_has_repeated_directed_links(path: list[str]) -> bool:
    directed_links = list(zip(path[:-1], path[1:]))
    return len(directed_links) != len(set(directed_links))


def route_weight(graph: nx.DiGraph, route: list[str], weight: str) -> float:
    return float(sum(graph[str(u)][str(v)][weight] for u, v in zip(route[:-1], route[1:])))


def initialize_route_worker(graph: nx.DiGraph, reachable_by_origin: dict[str, set[str]], route_config: dict[str, Any]) -> None:
    global _ROUTE_WORKER_GRAPH, _ROUTE_WORKER_REACHABILITY, _ROUTE_WORKER_CONFIG
    _ROUTE_WORKER_GRAPH = graph
    _ROUTE_WORKER_REACHABILITY = reachable_by_origin
    _ROUTE_WORKER_CONFIG = route_config


def print_routes_progress(processed_pairs: int, total_pairs: int, routes_found: int, pairs_without_routes: int, pairs_with_less_than_k: int) -> None:
    if total_pairs <= 0:
        return
    percent = 100.0 * processed_pairs / total_pairs
    bar_width = 30
    filled = int(bar_width * processed_pairs / total_pairs)
    bar = "#" * filled + "-" * (bar_width - filled)
    message = (
        f"\rRoutes [{bar}] {processed_pairs}/{total_pairs} OD pairs ({percent:6.2f}%) "
        f"found routes={routes_found} | pairs without routes ={pairs_without_routes} | pairs with less than k routes ={pairs_with_less_than_k}"
    )
    sys.stdout.write(message)
    sys.stdout.flush()
    if processed_pairs == total_pairs:
        sys.stdout.write("\n")
        sys.stdout.flush()


def build_reachability_by_origin(graph: nx.DiGraph, origins: list[str]) -> dict[str, set[str]]:
    return {str(origin): set(nx.descendants(graph, str(origin))) | {str(origin)} for origin in origins if graph.has_node(str(origin))}


def get_next_valid_intrazonal_route_from_generator(origin_id: str, successor: str, generator: Any, graph: nx.DiGraph, weight: str, allow_loops: bool) -> list[str] | None:
    for tail_path in generator:
        route = [str(origin_id)] + [str(node) for node in tail_path]
        if path_has_repeated_directed_links(route) and not allow_loops:
            continue
        return route
    return None


def generate_intrazonal_cycle_routes(graph: nx.DiGraph, origin_id: str, k_routes: int, weight: str, allow_loops: bool = False) -> list[list[str]]:
    origin_id = str(origin_id)
    if not graph.has_node(origin_id):
        return []

    heap = []
    tie_breaker = count()
    generators = {}
    for successor in graph.successors(origin_id):
        successor = str(successor)
        try:
            generator = nx.shortest_simple_paths(graph, source=successor, target=origin_id, weight=weight)
            route = get_next_valid_intrazonal_route_from_generator(origin_id, successor, generator, graph, weight, allow_loops)
            if route is None:
                continue
            generators[successor] = generator
            heapq.heappush(heap, (route_weight(graph, route, weight), next(tie_breaker), successor, route))
        except (nx.NetworkXNoPath, nx.NodeNotFound, StopIteration):
            continue

    routes = []
    seen_routes = set()
    while heap and len(routes) < k_routes:
        _, _, successor, route = heapq.heappop(heap)
        route_key = tuple(route)
        if route_key not in seen_routes:
            routes.append(route)
            seen_routes.add(route_key)

        generator = generators.get(successor)
        if generator is None:
            continue

        next_route = get_next_valid_intrazonal_route_from_generator(origin_id, successor, generator, graph, weight, allow_loops)
        if next_route is None:
            continue
        heapq.heappush(heap, (route_weight(graph, next_route, weight), next(tie_breaker), successor, next_route))

    return routes


def validate_route_generation_config(allow_intrazonal: bool, intrazonal_policy: str, k_routes: int) -> None:
    if k_routes <= 0:
        raise ValueError("k_routes must be greater than zero.")
    valid_intrazonal_policies = {"cycle", "zero_length"}
    if intrazonal_policy not in valid_intrazonal_policies:
        raise ValueError(f"Unsupported intrazonal_policy='{intrazonal_policy}'. Valid options are: {sorted(valid_intrazonal_policies)}.")
    if not allow_intrazonal:
        raise ValueError(
            "Invalid route configuration: allow_intrazonal is false, but the route builder generates the full zone x zone OD matrix, including origin==destination pairs. Set routes.allow_intrazonal=true and routes.intrazonal_policy='cycle' if intrazonal routes must be generated as real loops."
        )
    if allow_intrazonal and intrazonal_policy != "cycle":
        raise ValueError("Invalid route configuration for this pipeline: intrazonal routes must be generated as real loops. Set routes.intrazonal_policy='cycle'.")


def get_k_routes_from_export_graph(
    graph: nx.DiGraph,
    origin_id: str,
    destination_id: str,
    k_routes: int,
    weight: str,
    allow_intrazonal: bool = False,
    allow_loops: bool = False,
    intrazonal_policy: str = "cycle",
    reachable_by_origin: dict[str, set[str]] | None = None,
) -> list[list[str]]:
    origin_id = str(origin_id)
    destination_id = str(destination_id)
    if not graph.has_node(origin_id) or not graph.has_node(destination_id):
        return []
    if origin_id == destination_id:
        if not allow_intrazonal:
            return []
        if intrazonal_policy != "cycle":
            raise ValueError("Intrazonal OD pairs must be generated as real cycle routes. Set intrazonal_policy='cycle'.")
        return generate_intrazonal_cycle_routes(graph=graph, origin_id=origin_id, k_routes=k_routes, weight=weight, allow_loops=allow_loops)
    if reachable_by_origin is not None:
        reachable_nodes = reachable_by_origin.get(origin_id, set())
        if destination_id not in reachable_nodes:
            return []

    routes: list[list[str]] = []
    try:
        candidate_paths = nx.shortest_simple_paths(graph, source=origin_id, target=destination_id, weight=weight)
        for path in candidate_paths:
            normalized_path = [str(node) for node in path]
            if path_has_repeated_directed_links(normalized_path) and not allow_loops:
                continue
            routes.append(normalized_path)
            if len(routes) >= k_routes:
                break
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []
    return routes


def route_nodes_to_link_positions(graph: nx.DiGraph, route: list[str]) -> list[int]:
    link_positions = []
    for init_node, term_node in zip(route[:-1], route[1:]):
        edge_data = graph[str(init_node)][str(term_node)]
        link_positions.append(int(edge_data["link_pos"]))
    return link_positions


def solve_single_od_route_task(
    graph: nx.DiGraph,
    reachable_by_origin: dict[str, set[str]] | None,
    od_pair_position: int,
    origin_id: str,
    destination_id: str,
    k_routes: int,
    weight: str,
    allow_intrazonal: bool,
    allow_loops: bool,
    intrazonal_policy: str,
) -> dict[str, Any]:
    routes = get_k_routes_from_export_graph(
        graph=graph,
        origin_id=origin_id,
        destination_id=destination_id,
        k_routes=k_routes,
        weight=weight,
        allow_intrazonal=allow_intrazonal,
        allow_loops=allow_loops,
        intrazonal_policy=intrazonal_policy,
        reachable_by_origin=reachable_by_origin,
    )
    metadata_rows = []
    for route_idx, route in enumerate(routes):
        link_positions = route_nodes_to_link_positions(graph=graph, route=route)
        metadata_rows.append({
            "od_pair_position": od_pair_position,
            "origin_id": origin_id,
            "destination_id": destination_id,
            "route_idx": route_idx,
            "route_nodes": route,
            "route_link_pos": link_positions,
            "num_nodes": len(route),
            "num_links": len(link_positions),
            "weight": weight,
            "route_weight": route_weight(graph=graph, route=route, weight=weight),
        })
    return {
        "od_pair_position": od_pair_position,
        "origin_id": origin_id,
        "destination_id": destination_id,
        "routes": routes,
        "line": str(routes) if routes else "[]",
        "num_routes": len(routes),
        "metadata_rows": metadata_rows,
    }


def solve_route_batch_worker(batch_tasks: list[tuple[int, str, str]]) -> list[dict[str, Any]]:
    if _ROUTE_WORKER_GRAPH is None:
        raise RuntimeError("Route worker graph was not initialized.")
    if _ROUTE_WORKER_CONFIG is None:
        raise RuntimeError("Route worker configuration was not initialized.")

    results = []
    for od_pair_position, origin_id, destination_id in batch_tasks:
        results.append(
            solve_single_od_route_task(
                graph=_ROUTE_WORKER_GRAPH,
                reachable_by_origin=_ROUTE_WORKER_REACHABILITY,
                od_pair_position=od_pair_position,
                origin_id=origin_id,
                destination_id=destination_id,
                k_routes=int(_ROUTE_WORKER_CONFIG["k_routes"]),
                weight=str(_ROUTE_WORKER_CONFIG["weight"]),
                allow_intrazonal=bool(_ROUTE_WORKER_CONFIG["allow_intrazonal"]),
                allow_loops=bool(_ROUTE_WORKER_CONFIG["allow_loops"]),
                intrazonal_policy=str(_ROUTE_WORKER_CONFIG["intrazonal_policy"]),
            )
        )
    return results


def chunk_tasks(tasks: list[tuple[int, str, str]], batch_size: int) -> list[list[tuple[int, str, str]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")
    return [tasks[start:start + batch_size] for start in range(0, len(tasks), batch_size)]

def build_less_than_k_routes_temporaldiagnostics(
    graph: nx.DiGraph,
    od_pairs_with_less_than_k: list[tuple[str, str]],
    route_counts_by_od: dict[tuple[str, str], int],
    k_routes: int,
    node_type_by_id: dict[str, str] | None = None,
    max_samples: int = 100,
) -> list[dict[str, Any]]:
    """
    Build diagnostics for OD pairs that did not reach the required number of routes.

    This is especially useful for auxiliary/connector nodes, because they often
    have low in-degree/out-degree and therefore limited path diversity.
    """
    diagnostics = []

    for origin_id, destination_id in od_pairs_with_less_than_k[:max_samples]:
        origin_id = str(origin_id)
        destination_id = str(destination_id)

        diagnostics.append({
            "origin_id": origin_id,
            "destination_id": destination_id,
            "num_routes_found": int(
                route_counts_by_od.get((origin_id, destination_id), 0)
            ),
            "required_k_routes": int(k_routes),
            "origin_in_degree": int(graph.in_degree(origin_id)) if graph.has_node(origin_id) else None,
            "origin_out_degree": int(graph.out_degree(origin_id)) if graph.has_node(origin_id) else None,
            "destination_in_degree": int(graph.in_degree(destination_id)) if graph.has_node(destination_id) else None,
            "destination_out_degree": int(graph.out_degree(destination_id)) if graph.has_node(destination_id) else None,
            "origin_successors": (
                [str(node) for node in graph.successors(origin_id)]
                if graph.has_node(origin_id)
                else []
            ),
            "origin_predecessors": (
                [str(node) for node in graph.predecessors(origin_id)]
                if graph.has_node(origin_id)
                else []
            ),
            "destination_successors": (
                [str(node) for node in graph.successors(destination_id)]
                if graph.has_node(destination_id)
                else []
            ),
            "destination_predecessors": (
                [str(node) for node in graph.predecessors(destination_id)]
                if graph.has_node(destination_id)
                else []
            ),
            "is_intrazonal": origin_id == destination_id,

            "origin_type": (
                node_type_by_id.get(origin_id)
                if node_type_by_id is not None
                else None
            ),
            "destination_type": (
                node_type_by_id.get(destination_id)
                if node_type_by_id is not None
                else None
            ),
        })

    return diagnostics

def build_routes_tntp_text(
    data: dict[str, Any],
    nodes_tntp: pd.DataFrame,
    network_tntp: pd.DataFrame,
    k_routes: int = 10,
    weight: str = "free_flow_time",
    allow_intrazonal: bool = False,
    allow_loops: bool = False,
    intrazonal_policy: str = "cycle",
    require_exact_k_routes: bool = True,
    show_progress: bool = True,
    parallel: bool = False,
    parallel_workers: int | None = None,
    od_batch_size: int = 100,
) -> str:
    validate_route_generation_config(
        allow_intrazonal=allow_intrazonal,
        intrazonal_policy=intrazonal_policy,
        k_routes=k_routes,
    )

    export_graph = build_export_graph_from_network_tntp(network_tntp=network_tntp, weight_col=weight)
    zone_ids = get_zone_ids_from_nodes_tntp(nodes_tntp)

    node_class_by_id = (
    nodes_tntp.set_index("node_id")["type"]
    .astype(str)
    .to_dict()
)

    route_metadata_rows = []
    od_pairs_without_routes = []
    od_pairs_with_less_than_k = []
    route_counts_by_od: dict[tuple[str, str], int] = {}

    all_od_tasks = [
        (od_pair_position, str(origin_id), str(destination_id))
        for od_pair_position, (origin_id, destination_id) in enumerate(
            (origin_id, destination_id)
            for origin_id in zone_ids
            for destination_id in zone_ids
        )
    ]

    total_od_pairs = len(all_od_tasks)
    lines_by_position: list[str | None] = [None] * total_od_pairs
    reachable_by_origin = build_reachability_by_origin(graph=export_graph, origins=[str(origin_id) for origin_id in zone_ids])
    processed_pairs = 0
    total_routes_found = 0

    if od_batch_size <= 0:
        raise ValueError("od_batch_size must be greater than zero.")

    route_config = {
        "k_routes": int(k_routes),
        "weight": weight,
        "allow_intrazonal": bool(allow_intrazonal),
        "allow_loops": bool(allow_loops),
        "intrazonal_policy": intrazonal_policy,
    }

    def consume_route_result(result: dict[str, Any]) -> None:
        nonlocal processed_pairs, total_routes_found

        od_pair_position = int(result["od_pair_position"])
        origin_id = str(result["origin_id"])
        destination_id = str(result["destination_id"])
        num_routes = int(result["num_routes"])
        route_counts_by_od[(origin_id, destination_id)] = num_routes

        lines_by_position[od_pair_position] = str(result["line"])
        route_metadata_rows.extend(result["metadata_rows"])

        if num_routes == 0:
            od_pairs_without_routes.append((origin_id, destination_id))
        if num_routes < k_routes and not (origin_id == destination_id and not allow_intrazonal):
            od_pairs_with_less_than_k.append((origin_id, destination_id))

        processed_pairs += 1
        total_routes_found += num_routes
        if show_progress:
            print_routes_progress(
                processed_pairs=processed_pairs,
                total_pairs=total_od_pairs,
                routes_found=total_routes_found,
                pairs_without_routes=len(od_pairs_without_routes),
                pairs_with_less_than_k=len(od_pairs_with_less_than_k),
            )

    if parallel:
        if parallel_workers is None:
            parallel_workers = max(1, (os.cpu_count() or 2) - 1)
        if parallel_workers <= 1:
            parallel = False

    if parallel:
        batches = chunk_tasks(tasks=all_od_tasks, batch_size=od_batch_size)
        with ProcessPoolExecutor(max_workers=parallel_workers, initializer=initialize_route_worker, initargs=(export_graph, reachable_by_origin, route_config)) as executor:
            futures = [executor.submit(solve_route_batch_worker, batch) for batch in batches]
            for future in as_completed(futures):
                for result in future.result():
                    consume_route_result(result)
    else:
        for od_pair_position, origin_id, destination_id in all_od_tasks:
            result = solve_single_od_route_task(
                graph=export_graph,
                reachable_by_origin=reachable_by_origin,
                od_pair_position=od_pair_position,
                origin_id=origin_id,
                destination_id=destination_id,
                k_routes=k_routes,
                weight=weight,
                allow_intrazonal=allow_intrazonal,
                allow_loops=allow_loops,
                intrazonal_policy=intrazonal_policy,
            )
            consume_route_result(result)

    if any(line is None for line in lines_by_position):
        missing_positions = [position for position, line in enumerate(lines_by_position) if line is None]
        raise RuntimeError(f"Some OD route lines were not generated. Missing positions sample: {missing_positions[:20]}")

    lines = [str(line) for line in lines_by_position]

    routes_metadata_df = pd.DataFrame(route_metadata_rows)
    if not routes_metadata_df.empty:
        routes_metadata_df = routes_metadata_df.sort_values(by=["od_pair_position", "route_idx"], ascending=True).reset_index(drop=True)

    intrazonal_pairs_without_routes = [(origin_id, destination_id) for origin_id, destination_id in od_pairs_without_routes if origin_id == destination_id]

    less_than_k_routes_diagnostics = build_less_than_k_routes_temporaldiagnostics(
    graph=export_graph,
    od_pairs_with_less_than_k=od_pairs_with_less_than_k,
    route_counts_by_od=route_counts_by_od,
    k_routes=k_routes,
    node_type_by_id=node_class_by_id,
    max_samples=100,
    )

    route_count_distribution = {}

    for count_value in route_counts_by_od.values():
        route_count_distribution[int(count_value)] = (
            route_count_distribution.get(int(count_value), 0) + 1
        )

    data["routes_reconstruction_metadata"] = {
        "source": "generated_from_repaired_network_tntp",
        "exact_reconstruction": False,
        "format": "node_sequence_per_od_line",
        "k_routes": int(k_routes),
        "weight": weight,
        "allow_intrazonal": bool(allow_intrazonal),
        "num_intrazonal_od_pairs": int(len(zone_ids)),
        "intrazonal_routes_required": bool(allow_intrazonal),
        "intrazonal_policy": intrazonal_policy,
        "allow_loops": bool(allow_loops),
        "num_zone_ids": int(len(zone_ids)),
        "num_od_pairs": int(len(zone_ids) * len(zone_ids)),
        "num_route_rows": int(len(routes_metadata_df)),
        "num_od_pairs_without_routes": int(len(od_pairs_without_routes)),
        "num_od_pairs_with_less_than_k": int(len(od_pairs_with_less_than_k)),
        "od_pairs_without_routes_sample": od_pairs_without_routes[:20],
        "od_pairs_with_less_than_k_sample": od_pairs_with_less_than_k[:20],
        "require_exact_k_routes": bool(require_exact_k_routes),
        "show_progress": bool(show_progress),
        "parallel": bool(parallel),
        "parallel_workers": None if parallel_workers is None else int(parallel_workers),
        "od_batch_size": int(od_batch_size),
        "num_intrazonal_pairs_without_routes": int(len(intrazonal_pairs_without_routes)),
        "intrazonal_pairs_without_routes_sample": intrazonal_pairs_without_routes[:20],
        "less_than_k_routes_diagnostics": less_than_k_routes_diagnostics,
        "route_count_distribution": route_count_distribution,
    }
    data["routes_metadata_df"] = routes_metadata_df

    if require_exact_k_routes and od_pairs_with_less_than_k:
        sample = od_pairs_with_less_than_k[:20]

        raise ValueError(
            "Route generation did not find the required number of routes for all OD pairs. "
            f"Required k_routes={k_routes}. "
            f"Pairs with less than K: {len(od_pairs_with_less_than_k)}. "
            f"Sample: {sample}. "
            "Route diagnostics were stored in data['routes_reconstruction_metadata']"
            "['less_than_k_routes_diagnostics']. "
            "This usually means the graph does not contain enough distinct simple paths "
            "for those OD pairs."
        )

    return "\n".join(lines)
