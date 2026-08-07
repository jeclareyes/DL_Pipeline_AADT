"""Native igraph K-shortest route engine."""

from __future__ import annotations

import logging
from typing import Dict, List, Set

import igraph as ig
import networkx as nx

from .base_engine import Route, RouteEngine
from .igraph_connector_topology_compiler import CompiledRoutingGraph, ConnectorTopologyCompiler

logger = logging.getLogger(__name__)


class IgraphNativeRouteEngine(RouteEngine):
    """Route engine backed by igraph's native K-shortest-path implementation."""

    def __init__(self, graph: nx.DiGraph):
        self.nx_graph = graph
        zone_ids = {
            int(node)
            for node, data in graph.nodes(data=True)
            if data.get("class") == "Zones"
        }
        if not zone_ids:
            # Reference/synthetic graphs generally have no dataset-specific
            # node class.  Every graph node must then be queryable as an OD
            # endpoint; the compiler keeps these nodes in the core and adds
            # zero-cost source/sink terminals around them.
            zone_ids = {int(node) for node in graph.nodes()}
        connector_types = {
            int(data.get("link_type"))
            for _, _, data in graph.edges(data=True)
            if int(data.get("link_type", -1)) == 99
        }
        self._compiled_by_connector_types: dict[frozenset[int], CompiledRoutingGraph] = {}
        self._zone_ids = zone_ids
        self._default_connector_types = connector_types
        self._compiler = ConnectorTopologyCompiler()
        self._compiled_by_connector_types[frozenset(connector_types)] = self._compiler.compile(
            graph, zone_ids, connector_types
        )
        logger.info(
            "IgraphNativeRouteEngine initialized: compiled %d vertices and %d edges.",
            self._compiled_by_connector_types[frozenset(connector_types)].graph.vcount(),
            self._compiled_by_connector_types[frozenset(connector_types)].graph.ecount(),
        )

    def _compiled(self, connector_link_types: Set[int]) -> CompiledRoutingGraph:
        key = frozenset(int(value) for value in connector_link_types)
        compiled = self._compiled_by_connector_types.get(key)
        if compiled is None:
            compiled = self._compiler.compile(self.nx_graph, self._zone_ids, set(key))
            self._compiled_by_connector_types[key] = compiled
        return compiled

    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: Dict[str, bool],
        connector_link_types: Set[int],
    ) -> List[Route]:
        if int(k) < 1:
            raise ValueError("k must be >= 1.")
        origin_id, destination_id = int(origin_id), int(destination_id)
        is_intrazonal = origin_id == destination_id
        if is_intrazonal and not constraints.get("allow_auto_routes", False):
            return []
        if not is_intrazonal and constraints.get("allow_loops", False):
            raise ValueError("IgraphNativeRouteEngine supports simple paths only; allow_loops must be false.")
        if is_intrazonal and not constraints.get("allow_loops", False):
            return []
        if constraints.get("allow_duplicates", False):
            raise ValueError("IgraphNativeRouteEngine never returns duplicate routes; allow_duplicates must be false.")

        compiled = self._compiled(connector_link_types)
        if origin_id not in compiled.source_terminal_by_zone:
            raise KeyError(f"Origin zone {origin_id} is not present in the compiled routing graph.")
        if destination_id not in compiled.sink_terminal_by_zone:
            raise KeyError(f"Destination zone {destination_id} is not present in the compiled routing graph.")
        if weight not in compiled.weights_by_attribute:
            raise ValueError(f"Weight attribute '{weight}' is unavailable in the compiled routing graph.")

        source = compiled.source_terminal_by_zone[origin_id]
        target = compiled.sink_terminal_by_zone[destination_id]
        paths = compiled.graph.get_k_shortest_paths(
            source,
            to=target,
            k=int(k),
            weights=compiled.weights_by_attribute[weight],
            mode=ig.OUT,
            output="vpath",
        )
        routes: list[Route] = []
        seen: set[tuple[int, ...]] = set()
        for path in paths:
            route = self._decode_path(path, compiled, origin_id, destination_id)
            if route is None:
                continue
            key = tuple(route)
            if key not in seen:
                routes.append(route)
                seen.add(key)
            if len(routes) >= int(k):
                break
        return routes

    @staticmethod
    def _decode_path(
        path: list[int],
        compiled: CompiledRoutingGraph,
        origin_id: int,
        destination_id: int,
    ) -> Route | None:
        if not path or path[0] != compiled.source_terminal_by_zone[origin_id] or path[-1] != compiled.sink_terminal_by_zone[destination_id]:
            raise ValueError("igraph returned a path with unexpected terminal endpoints.")
        core_nodes = [
            int(compiled.graph.vs[vertex]["original_id"])
            for vertex in path
            if compiled.graph.vs[vertex]["kind"] == "core"
        ]
        # In graphs without explicit zone-node metadata the compiler keeps
        # the endpoint vertices in the core and surrounds them with zero-cost
        # terminal edges.  Those helper visits are not part of the decoded
        # route itself.
        if core_nodes and core_nodes[0] == origin_id:
            core_nodes = core_nodes[1:]
        if core_nodes and core_nodes[-1] == destination_id:
            core_nodes = core_nodes[:-1]
        route = [origin_id, *core_nodes, destination_id]
        if origin_id == destination_id and len(route) < 3:
            return None
        if len(route) < 2:
            return None
        internal = route[1:-1]
        if len(internal) != len(set(internal)):
            return None
        return route
