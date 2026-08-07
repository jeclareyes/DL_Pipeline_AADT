import heapq
import logging
from itertools import count
from typing import List, Tuple, Dict, Set, Optional, Any

import networkx as nx
import rustworkx as rx



from .base_engine import RouteEngine, Route

logger = logging.getLogger(__name__)

class RustworkXEngine(RouteEngine):
    def __init__(self, graph: nx.DiGraph):
        if rx is None:
            raise ImportError("rustworkx is not installed. Please install it to use RustworkXEngine.")

        self.nx_graph = graph
        # NetworkX DiGraph is simple; disabling multigraph prevents accidental parallel edges.
        self.rx_graph = rx.PyDiGraph(multigraph=False)
        
        # Bi-directional mapping
        self.node_to_rx = {}
        self.rx_to_node = {}

        # Build mapping and add nodes
        for node in graph.nodes():
            node_id = int(node)
            rx_node = self.rx_graph.add_node(node_id)
            self.node_to_rx[node_id] = rx_node
            self.rx_to_node[rx_node] = node_id

        # Edge mapping for weight extraction, structure is (u_rx, v_rx) -> edge_data
        # rustworkx allows multi-edges, but we assume simple directed graph from NetworkX
        self.edge_data_map = {}
        for u, v, data in graph.edges(data=True):
            u_id = int(u)
            v_id = int(v)
            u_rx = self.node_to_rx[u_id]
            v_rx = self.node_to_rx[v_id]
            edge_data = dict(data)
            self.rx_graph.add_edge(u_rx, v_rx, edge_data)
            self.edge_data_map[(u_rx, v_rx)] = edge_data

    def _dijkstra_shortest_path(
        self,
        graph: rx.PyDiGraph,
        source_rx: int,
        target_rx: int,
        weight_fn,
        forbidden_nodes: Set[int],
        forbidden_edges: Set[Tuple[int, int]],
        connector_link_types: Optional[Set[int]] = None,
        connector_origin_rx: Optional[int] = None,
        connector_destination_rx: Optional[int] = None,
    ) -> Optional[List[int]]:
        """
        Runs Dijkstra on a subgraph ignoring forbidden nodes and edges.
        Instead of physically removing nodes/edges, we filter them using node/edge subgraphs
        if possible, or build a temporary filtered graph.
        rustworkx has subgraph filtering.
        """
        def node_filter(n_idx):
            return n_idx not in forbidden_nodes

        def edge_filter(source_idx, target_idx, edge_data):
            # source_idx and target_idx are already rx indices
            return (source_idx, target_idx) not in forbidden_edges

        connector_link_types = connector_link_types or set()
        if connector_origin_rx is None:
            connector_origin_rx = source_rx
        if connector_destination_rx is None:
            connector_destination_rx = target_rx

        # Create a filtered subgraph view (only works on rx >= 0.12 roughly via subgraph)
        # However, building a temporary graph is safer if filtering functions are not fully supported
        # for all path algorithms. But `rx.dijkstra_shortest_paths` supports custom weight functions.
        # We can return float('inf') for forbidden edges. For forbidden nodes, we can forbid all their edges.
        
        def safe_weight_fn(edge):
            # rustworkx custom weight functions for path algorithms usually take the edge payload
            # but we need to know the endpoints to forbid specific edges/nodes.
            # actually rx.dijkstra_shortest_paths doesn't pass endpoints to weight_fn.
            pass

        # Building a temporary graph is safest and respects the requirements strictly
        temp_graph = graph.copy()

        # Connector legality belongs to the complete OD, not the current Yen
        # spur.  Remove intermediate connectors before Dijkstra so invalid
        # zero-cost hub paths never enter Yen's candidate heap.
        for (u_rx, v_rx), edge_data in self.edge_data_map.items():
            if int(edge_data.get("link_type", -1)) in connector_link_types:
                if u_rx != connector_origin_rx and v_rx != connector_destination_rx:
                    forbidden_edges.add((u_rx, v_rx))
        
        # Remove forbidden edges.
        # `edge_indices_from_endpoints()` returns integer edge IDs, while
        # `remove_edges_from()` expects endpoint tuples. Removing by index avoids
        # the TypeError: "int object cannot be converted to PyTuple".
        for u_rx, v_rx in forbidden_edges:
            if not temp_graph.has_edge(u_rx, v_rx):
                continue

            edge_indices = list(temp_graph.edge_indices_from_endpoints(u_rx, v_rx))
            for edge_idx in edge_indices:
                temp_graph.remove_edge_from_index(int(edge_idx))

        # Remove forbidden nodes
        # If we remove nodes, their indices might change if we use remove_node, BUT PyDiGraph
        # with remove_node keeps other indices intact (it creates a hole).
        nodes_to_remove = [n for n in forbidden_nodes if temp_graph.has_node(n)]
        if nodes_to_remove:
            temp_graph.remove_nodes_from(nodes_to_remove)

        if not temp_graph.has_node(source_rx) or not temp_graph.has_node(target_rx):
            return None

        paths = rx.dijkstra_shortest_paths(temp_graph, source=source_rx, target=target_rx, weight_fn=weight_fn)
        
        if target_rx in paths:
            # rustworkx returns rx node indices (NodeIndices), convert to python list
            return list(paths[target_rx])
        return None

    def _yen_k_shortest_paths(
        self,
        source: int,
        target: int,
        k: int,
        weight: str,
        connector_link_types: Optional[Set[int]] = None,
        connector_origin: Optional[int] = None,
        connector_destination: Optional[int] = None,
    ) -> List[List[int]]:
        """
        Strict implementation of Yen's algorithm.
        """
        if source not in self.node_to_rx or target not in self.node_to_rx:
            return []

        source_rx = self.node_to_rx[source]
        target_rx = self.node_to_rx[target]
        connector_link_types = connector_link_types or set()
        connector_origin_rx = self.node_to_rx.get(
            source if connector_origin is None else int(connector_origin)
        )
        connector_destination_rx = self.node_to_rx.get(
            target if connector_destination is None else int(connector_destination)
        )
        if connector_origin_rx is None or connector_destination_rx is None:
            return []

        def weight_fn(edge_data):
            return float(edge_data.get(weight, 1.0))

        # 1. Find shortest path A^1
        first_path_rx = self._dijkstra_shortest_path(
            self.rx_graph,
            source_rx,
            target_rx,
            weight_fn,
            set(),
            set(),
            connector_link_types,
            connector_origin_rx,
            connector_destination_rx,
        )

        if not first_path_rx:
            return []

        A = [first_path_rx]
        B = []
        B_set = set() # To prevent duplicate paths in B
        tie_breaker = count()

        for k_idx in range(1, k):
            for i in range(len(A[k_idx - 1]) - 1):
                spur_node_rx = A[k_idx - 1][i]
                root_path_rx = A[k_idx - 1][:i + 1]

                forbidden_edges = set()
                for p in A:
                    if len(p) > i and p[:i + 1] == root_path_rx:
                        forbidden_edges.add((p[i], p[i + 1]))

                forbidden_nodes = set(root_path_rx[:-1])

                spur_path_rx = self._dijkstra_shortest_path(
                    self.rx_graph,
                    spur_node_rx,
                    target_rx,
                    weight_fn,
                    forbidden_nodes,
                    forbidden_edges,
                    connector_link_types,
                    connector_origin_rx,
                    connector_destination_rx,
                )

                if spur_path_rx:
                    total_path_rx = root_path_rx[:-1] + spur_path_rx
                    total_path_tuple = tuple(total_path_rx)
                    if total_path_tuple not in B_set:
                        B_set.add(total_path_tuple)
                        cost = self._route_weight_rx(total_path_rx, weight)
                        heapq.heappush(B, (cost, next(tie_breaker), total_path_rx))

            if not B:
                break

            appended = False
            while B:
                cost, _, next_path_rx = heapq.heappop(B)
                B_set.remove(tuple(next_path_rx))

                if next_path_rx not in A:
                    A.append(next_path_rx)
                    appended = True
                    break

            if not appended:
                break

        # Convert back to original node IDs
        A_original = []
        for path_rx in A:
            A_original.append([self.rx_to_node[n] for n in path_rx])
            
        return A_original

    def _route_weight_rx(self, route_rx: List[int], weight: str) -> float:
        total_weight = 0.0
        for u, v in zip(route_rx[:-1], route_rx[1:]):
            edge_data = self.edge_data_map.get((u, v), {})
            total_weight += float(edge_data.get(weight, 1.0))
        return total_weight

    def _route_weight(self, route: List[int], weight: str) -> float:
        return float(sum(self.nx_graph[int(u)][int(v)][weight] for u, v in zip(route[:-1], route[1:])))

    def _path_has_repeated_links(self, path: List[int]) -> bool:
        links = list(zip(path[:-1], path[1:]))
        return len(links) != len(set(links))

    def _path_has_repeated_directed_links(self, path: List[int]) -> bool:
        directed_links = list(zip(path[:-1], path[1:]))
        return len(directed_links) != len(set(directed_links))

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
            edge_data = self.nx_graph[int(init_node)][int(term_node)]
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

    def _get_next_valid_intrazonal_route_from_generator(
        self,
        origin_id: int,
        successor: int,
        generator: Any,
        weight: str,
        allow_loops: bool,
        connector_link_types: Set[int],
    ) -> Optional[List[int]]:
        for tail_path in generator:
            route = [int(origin_id)] + [int(node) for node in tail_path]

            if self._path_has_repeated_directed_links(route) and not allow_loops:
                continue

            if self._path_uses_invalid_connector(
                path=route,
                origin_id=origin_id,
                destination_id=origin_id,
                connector_link_types=connector_link_types,
            ):
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
        """
        Intrazonal cycle routes rely on finding paths back to the origin.
        We can use NetworkX here because Yen's logic is specifically for simple paths between different nodes.
        Alternatively, we could implement Yen's here too, but since the user requested Yen specifically 
        to mimic `nx.shortest_simple_paths` equivalent behavior, and intrazonal loops 
        use `nx.shortest_simple_paths` under the hood for `successor -> origin`, we should mimic that.
        Since we want RustworkX entirely for performance, we should do Yen from successor to origin.
        """
        origin_id = int(origin_id)
        connector_link_types = connector_link_types or set()

        if not self.nx_graph.has_node(origin_id):
            return []

        heap = []
        tie_breaker = count()
        
        # We need a lazy generator equivalent for Yen's to mimic the logic.
        # Since Yen computes paths iteratively, we can pre-compute up to k_routes for each successor.
        # It's an approximation to lazy generation but works well if k is small.
        
        # Actually, let's keep a state for each successor.
        generators = {}
        for successor in self.nx_graph.successors(origin_id):
            successor = int(successor)
            
            # Precompute k paths for this successor to origin using Yen
            paths = self._yen_k_shortest_paths(
                successor,
                origin_id,
                k_routes,
                weight,
                connector_link_types=connector_link_types,
                connector_origin=origin_id,
                connector_destination=origin_id,
            )
            
            # Create a simple iterator over these paths
            generator = iter(paths)
            
            route = self._get_next_valid_intrazonal_route_from_generator(
                origin_id, successor, generator, weight, allow_loops, connector_link_types
            )
            if route is None:
                continue
            generators[successor] = generator
            heapq.heappush(heap, (self._route_weight(route, weight), next(tie_breaker), successor, route))

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

            next_route = self._get_next_valid_intrazonal_route_from_generator(
                origin_id, successor, generator, weight, allow_loops, connector_link_types
            )
            if next_route is None:
                continue
            heapq.heappush(heap, (self._route_weight(next_route, weight), next(tie_breaker), successor, next_route))

        return routes

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
            # Generate more paths than k to filter out invalid ones
            # In Yen, paths are simple by definition. So allow_loops doesn't apply to the paths themselves,
            # but maybe constraints applied later do.
            # We will generate up to K valid paths.
            
            # Since Yen produces simple paths, and we need to check constraints, we might need
            # to keep calling Yen until we have k valid paths.
            # However, Yen algorithm produces simple paths (no repeated nodes, thus no repeated links).
            # So `allow_loops` is trivially satisfied (they never have loops).
            # The connector links check is the main filter.
            
            # For simplicity, we just ask for k * 2 paths, filter them, and take k.
            # Or better, we can modify the Yen implementation to yield lazily, but returning a list is easier.
            candidate_paths = self._yen_k_shortest_paths(
                source=origin_id,
                target=destination_id,
                k=k,
                weight=weight,
                connector_link_types=connector_link_types,
                connector_origin=origin_id,
                connector_destination=destination_id,
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

        except Exception as e:
            logger.debug(
                "Error finding path between origin %s and destination %s: %s",
                origin_id,
                destination_id,
                str(e)
            )
            routes = []

        return routes
