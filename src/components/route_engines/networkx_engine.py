import heapq
import logging
from itertools import count
from typing import List, Tuple, Dict, Set, Any, Optional

import networkx as nx

from .base_engine import RouteEngine, Route

logger = logging.getLogger(__name__)

class NetworkXEngine(RouteEngine):
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def _filtered_graph(
        self,
        origin_id: int,
        destination_id: int,
        connector_link_types: Set[int],
    ) -> nx.DiGraph:
        """Return a view that excludes intermediate connector edges.

        Connector validity is an OD-level constraint.  Filtering only after
        ``shortest_simple_paths`` is unsafe on real networks because zero-cost
        connector hubs can yield an effectively unbounded stream of rejected
        candidates.
        """
        if not connector_link_types:
            return self.graph

        origin_id = int(origin_id)
        destination_id = int(destination_id)

        def edge_allowed(u: int, v: int) -> bool:
            link_type = int(self.graph[int(u)][int(v)].get("link_type", -1))
            return (
                link_type not in connector_link_types
                or int(u) == origin_id
                or int(v) == destination_id
            )

        return nx.subgraph_view(self.graph, filter_edge=edge_allowed)

    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: Dict[str, bool],
        connector_link_types: Set[int]
    ) -> List[Route]:
        
        if origin_id == destination_id:
            if not constraints.get("allow_auto_routes", False):
                return []
            return self._generate_intrazonal_cycle_routes(
                origin_id=origin_id,
                k_routes=k,
                weight=weight,
                allow_loops=constraints.get("allow_loops", False),
                connector_link_types=connector_link_types,
            )

        routes: List[Route] = []
        seen_routes: Set[Tuple[int, ...]] = set()

        try:
            routing_graph = self._filtered_graph(
                origin_id, destination_id, connector_link_types
            )
            candidate_paths = nx.shortest_simple_paths(
                routing_graph,
                source=origin_id,
                target=destination_id,
                weight=weight,
            )

            for path in candidate_paths:
                path = [int(node) for node in path]
                path_key = tuple(path)

                if len(path) < 2:
                    continue

                if (path_key in seen_routes) and (not constraints.get("allow_duplicates", False)):
                    continue

                if self._path_has_repeated_links(path) and (not constraints.get("allow_loops", False)):
                    continue

                if self._path_uses_invalid_connector(
                    path=path,
                    origin_id=origin_id,
                    destination_id=destination_id,
                    connector_link_types=connector_link_types,
                ):
                    continue

                routes.append(path)
                seen_routes.add(path_key)

                if len(routes) >= k:
                    break

        except (nx.NetworkXNoPath, nx.NodeNotFound):
            logger.debug(
                "No path found between origin %s and destination %s.",
                origin_id,
                destination_id,
            )
            routes = []

        return routes

    def _path_has_repeated_links(self, path: List[int]) -> bool:
        links = list(zip(path[:-1], path[1:]))
        return len(links) != len(set(links))

    def _path_uses_invalid_connector(
        self,
        path: List[int],
        origin_id: int,
        destination_id: int,
        connector_link_types: Set[int],
    ) -> bool:
        if not connector_link_types:
            return False

        for init_node, term_node in zip(path[:-1], path[1:]):
            edge_data = self.graph[int(init_node)][int(term_node)]
            link_type = int(edge_data.get("link_type", -1))

            if link_type not in connector_link_types:
                continue

            init_node = int(init_node)
            term_node = int(term_node)

            is_origin_exit = init_node == origin_id
            is_destination_entry = term_node == destination_id

            if not (is_origin_exit or is_destination_entry):
                return True

        return False

    def _route_weight(self, route: List[int], weight: str) -> float:
        return float(sum(self.graph[int(u)][int(v)][weight] for u, v in zip(route[:-1], route[1:])))

    def _path_has_repeated_directed_links(self, path: List[int]) -> bool:
        directed_links = list(zip(path[:-1], path[1:]))
        return len(directed_links) != len(set(directed_links))

    def _get_next_valid_intrazonal_route_from_generator(
        self,
        origin_id: int,
        successor: int,
        generator: Any,
        weight: str,
        allow_loops: bool,
    ) -> Optional[List[int]]:
        for tail_path in generator:
            route = [int(origin_id)] + [int(node) for node in tail_path]
            if self._path_has_repeated_directed_links(route) and not allow_loops:
                continue
            return route
        return None

    def _generate_intrazonal_cycle_routes(
        self,
        origin_id: int,
        k_routes: int,
        weight: str,
        allow_loops: bool = False,
        connector_link_types: Optional[Set[int]] = None,
    ) -> List[List[int]]:
        origin_id = int(origin_id)
        if not self.graph.has_node(origin_id):
            return []

        heap = []
        tie_breaker = count()
        generators = {}
        connector_link_types = connector_link_types or set()
        routing_graph = self._filtered_graph(
            origin_id, origin_id, connector_link_types
        )
        for successor in routing_graph.successors(origin_id):
            successor = int(successor)
            try:
                generator = nx.shortest_simple_paths(routing_graph, source=successor, target=origin_id, weight=weight)
                route = self._get_next_valid_intrazonal_route_from_generator(origin_id, successor, generator, weight, allow_loops)
                if route is None:
                    continue
                generators[successor] = generator
                heapq.heappush(heap, (self._route_weight(route, weight), next(tie_breaker), successor, route))
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

            next_route = self._get_next_valid_intrazonal_route_from_generator(origin_id, successor, generator, weight, allow_loops)
            if next_route is None:
                continue
            heapq.heappush(heap, (self._route_weight(next_route, weight), next(tie_breaker), successor, next_route))

        return routes
