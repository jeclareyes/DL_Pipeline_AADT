import heapq
import logging
from itertools import count
from typing import List, Tuple, Dict, Set, Optional, Any

import networkx as nx
import rustworkx as rx

from .base_engine import RouteEngine, Route

logger = logging.getLogger(__name__)

class RustworkXOptimizedEngine(RouteEngine):
    def __init__(self, graph: nx.DiGraph):
        self.nx_graph = graph

        # Bi-directional mappings
        self.node_to_rx = {}
        self.rx_to_node = {}

        for rx_idx, node in enumerate(graph.nodes()):
            node_id = int(node)
            self.node_to_rx[node_id] = rx_idx
            self.rx_to_node[rx_idx] = node_id

        self.edge_lookup = {}
        self.edge_weight = {}
        self.edge_link_type = {}
        self.edge_pair_by_id = {}
        self.raw_edge_attrs = {}

        for edge_id, (u, v, data) in enumerate(graph.edges(data=True)):
            u_rx = self.node_to_rx[int(u)]
            v_rx = self.node_to_rx[int(v)]
            self.edge_lookup[(u_rx, v_rx)] = edge_id
            self.edge_pair_by_id[edge_id] = (u_rx, v_rx)

            link_type = int(data.get("link_type", -1))
            self.edge_link_type[edge_id] = link_type
            self.raw_edge_attrs[edge_id] = data

        # Adjacency for path validation (outgoing edges)
        self.adj: dict[int, list[tuple[int, int, float, int]]] = {
            rx_idx: [] for rx_idx in self.rx_to_node
        }
        # Reverse adjacency for computing forbidden incident edges (incoming)
        self._reverse_adj: dict[int, list[tuple[int, int, float, int]]] = {
            rx_idx: [] for rx_idx in self.rx_to_node
        }

        # Native rustworkx graph for fast Dijkstra
        self.rx_graph = rx.PyDiGraph(multigraph=False)
        for rx_idx in range(len(self.rx_to_node)):
            self.rx_graph.add_node(self.rx_to_node[rx_idx])
        self._rx_edge_data: dict[int, dict] = {}

        for edge_id, (u_rx, v_rx) in self.edge_pair_by_id.items():
            link_type = self.edge_link_type[edge_id]
            self.adj[u_rx].append((v_rx, edge_id, 1.0, link_type))
            self._reverse_adj[v_rx].append((u_rx, edge_id, 1.0, link_type))
            rx_edge_data: dict = {"edge_id": edge_id, "weight": 1.0}
            self.rx_graph.add_edge(u_rx, v_rx, rx_edge_data)
            self._rx_edge_data[edge_id] = rx_edge_data

        self._current_weight_attr: str | None = None
        self._current_forbidden_edges: set[int] = set()

        logger.info(
            "RustworkXOptimizedEngine initialized: converted %d nodes and %d edges.",
            len(self.node_to_rx),
            len(self.edge_lookup)
        )

    def _prepare_weight_and_adj(self, weight: str) -> None:
        """
        Prepares self.edge_weight, self.adj and self._reverse_adj dynamically
        for a specific weight key, and updates the rustworkx graph edge data in-place.
        Raises ValueError if any weight is negative or missing.
        """
        self.edge_weight = {}
        for edge_id, data in self.raw_edge_attrs.items():
            if weight not in data:
                raise ValueError(f"Weight attribute '{weight}' is missing from edge data.")
            val = float(data[weight])
            if val < 0.0:
                raise ValueError(f"Negative edge weight detected for attribute '{weight}': {val}")
            self.edge_weight[edge_id] = val

        self.adj = {rx_idx: [] for rx_idx in self.rx_to_node}
        self._reverse_adj = {rx_idx: [] for rx_idx in self.rx_to_node}
        for edge_id, (u_rx, v_rx) in self.edge_pair_by_id.items():
            weight_val = self.edge_weight[edge_id]
            link_type = self.edge_link_type[edge_id]
            self.adj[u_rx].append((v_rx, edge_id, weight_val, link_type))
            self._reverse_adj[v_rx].append((u_rx, edge_id, weight_val, link_type))
            self._rx_edge_data[edge_id]["weight"] = weight_val

        self._current_weight_attr = weight

    def _path_to_edge_ids(self, path_rx: List[int]) -> List[int]:
        edge_ids = []
        for u, v in zip(path_rx[:-1], path_rx[1:]):
            edge_id = self.edge_lookup.get((u, v))
            if edge_id is None:
                raise ValueError(f"Edge ({u}, {v}) not found in lookup.")
            edge_ids.append(edge_id)
        return edge_ids

    def _route_cost_rx(self, path_rx: List[int]) -> float:
        edge_ids = self._path_to_edge_ids(path_rx)
        return sum(self.edge_weight[eid] for eid in edge_ids)

    def _convert_path_to_original_ids(self, path_rx: List[int]) -> List[int]:
        return [self.rx_to_node[n] for n in path_rx]

    def _is_valid_connector_usage(
        self,
        path_rx: List[int],
        origin_rx: int,
        destination_rx: int,
        connector_link_types: Set[int]
    ) -> bool:
        if not connector_link_types:
            return True
        edge_ids = self._path_to_edge_ids(path_rx)
        for edge_id, u, v in zip(edge_ids, path_rx[:-1], path_rx[1:]):
            link_type = self.edge_link_type[edge_id]
            if link_type in connector_link_types:
                if u != origin_rx and v != destination_rx:
                    return False
        return True

    def _is_path_valid(
        self,
        path_rx: List[int],
        source_rx: int,
        target_rx: int,
        connector_link_types: Set[int],
        cycle_origin_rx: Optional[int]
    ) -> bool:
        if cycle_origin_rx is not None:
            full_path = [cycle_origin_rx] + path_rx
            orig = cycle_origin_rx
            dest = cycle_origin_rx
        else:
            full_path = path_rx
            orig = source_rx
            dest = target_rx
        return self._is_valid_connector_usage(full_path, orig, dest, connector_link_types)

    def _edge_weight_fn(self, edge_data: dict) -> float:
        if edge_data["edge_id"] in self._current_forbidden_edges:
            return float("inf")
        return float(edge_data.get("weight", 1.0))

    def _dijkstra_filtered(
        self,
        source_rx: int,
        target_rx: int,
        forbidden_nodes: Set[int],
        forbidden_edges: Set[int],
        connector_link_types: Optional[Set[int]] = None,
        connector_origin_rx: Optional[int] = None,
        connector_destination_rx: Optional[int] = None,
    ) -> Optional[List[int]]:
        """
        Runs Dijkstra via rustworkx (Rust native) using weight poisoning:
        forbidden edges are assigned infinite weight.

        When ``connector_link_types`` is provided, any connector edge whose
        *tail* node is not the complete route origin and whose *head* node is
        not the complete route destination is treated as forbidden.  This is
        the key invariant for real networks: connector links are only legal at
        the very first and very last step of a route.  The complete endpoints
        are passed separately because Yen also invokes Dijkstra from spur
        nodes.
        """
        if source_rx == target_rx:
            return [source_rx]

        # Yen invokes Dijkstra from spur nodes, but connector legality is
        # defined by the complete OD route, not by the current spur.  If these
        # are omitted, every intermediate connector hub becomes a new
        # temporary origin and produces an unbounded stream of invalid paths.
        connector_origin_rx = (
            source_rx if connector_origin_rx is None else connector_origin_rx
        )
        connector_destination_rx = (
            target_rx
            if connector_destination_rx is None
            else connector_destination_rx
        )

        # Collect forbidden edges from forbidden nodes
        all_forbidden = set(forbidden_edges)
        for node in forbidden_nodes:
            for v, eid, w, lt in self.adj.get(node, []):
                all_forbidden.add(eid)
            for u, eid, w, lt in self._reverse_adj.get(node, []):
                all_forbidden.add(eid)

        # Pre-exclude connector edges that are not incident to source or target.
        # This guarantees Dijkstra can only produce paths that satisfy the
        # connector-usage constraint, so every candidate handed to Yen's loop
        # is already valid.  Without this, on real networks with central
        # connector hubs, the algorithm enters a near-infinite loop because
        # virtually every short path passes through an intermediate connector.
        if connector_link_types:
            for edge_id, (u_rx, v_rx) in self.edge_pair_by_id.items():
                if self.edge_link_type[edge_id] in connector_link_types:
                    if (
                        u_rx != connector_origin_rx
                        and v_rx != connector_destination_rx
                    ):
                        all_forbidden.add(edge_id)

        self._current_forbidden_edges = all_forbidden

        paths = rx.dijkstra_shortest_paths(
            self.rx_graph,
            source=source_rx,
            target=target_rx,
            weight_fn=self._edge_weight_fn,
        )

        if target_rx in paths:
            candidate_path = list(paths[target_rx])
            # rustworkx may still return a path containing an edge whose
            # callback weight is ``inf`` when no finite alternative exists.
            # Such a path is not a valid filtered result: returning it would
            # make Yen reject it later and keep expanding invalid candidates.
            candidate_edge_ids = self._path_to_edge_ids(candidate_path)
            if any(edge_id in all_forbidden for edge_id in candidate_edge_ids):
                return None
            return candidate_path
        return None

    def _yen_k_shortest_paths_rx(
        self,
        source_rx: int,
        target_rx: int,
        k: int,
        connector_link_types: Set[int],
        cycle_origin_rx: Optional[int] = None
    ) -> List[List[int]]:
        """
        Calculates K shortest simple paths from source_rx to target_rx.

        Connector constraints are enforced *inside* each Dijkstra call so
        that only topologically-valid spur paths are ever generated.  This
        prevents the algorithm from spending exponential time exploring
        intermediate-connector paths that would only be discarded later.
        """
        connector_origin_rx = cycle_origin_rx if cycle_origin_rx is not None else source_rx
        connector_destination_rx = cycle_origin_rx if cycle_origin_rx is not None else target_rx

        first_path = self._dijkstra_filtered(
            source_rx,
            target_rx,
            set(),
            set(),
            connector_link_types,
            connector_origin_rx,
            connector_destination_rx,
        )
        if not first_path:
            return []

        A = [first_path]
        accepted_set = {tuple(first_path)}

        valid_routes = []
        if self._is_path_valid(first_path, source_rx, target_rx, connector_link_types, cycle_origin_rx):
            valid_routes.append(first_path)

        B: List = []
        candidate_set: Set[tuple] = set()
        tie_breaker = count()

        current_branch_path_idx = 0

        while len(valid_routes) < k:
            if current_branch_path_idx < len(A):
                prev_path = A[current_branch_path_idx]
                current_branch_path_idx += 1

                for i in range(len(prev_path) - 1):
                    spur_node = prev_path[i]
                    root_path = prev_path[:i + 1]

                    forbidden_nodes = set(root_path[:-1])
                    forbidden_edges: Set[int] = set()

                    for p in A:
                        if len(p) > i + 1 and p[:i + 1] == root_path:
                            edge_to_forbid = (p[i], p[i + 1])
                            edge_id = self.edge_lookup.get(edge_to_forbid)
                            if edge_id is not None:
                                forbidden_edges.add(edge_id)

                    # Pass connector_link_types so spur Dijkstra never routes
                    # through intermediate connector hubs.
                    spur_path = self._dijkstra_filtered(
                        spur_node, target_rx,
                        forbidden_nodes, forbidden_edges,
                        connector_link_types,
                        connector_origin_rx,
                        connector_destination_rx,
                    )
                    if spur_path:
                        total_path = root_path[:-1] + spur_path
                        total_tuple = tuple(total_path)
                        if total_tuple not in accepted_set and total_tuple not in candidate_set:
                            cost = self._route_cost_rx(total_path)
                            heapq.heappush(B, (cost, next(tie_breaker), total_path))
                            candidate_set.add(total_tuple)

            if not B:
                break

            cost, _, next_path = heapq.heappop(B)
            candidate_set.remove(tuple(next_path))

            A.append(next_path)
            accepted_set.add(tuple(next_path))

            if self._is_path_valid(next_path, source_rx, target_rx, connector_link_types, cycle_origin_rx):
                valid_routes.append(next_path)

        logger.debug(
            "Yen path search: generated %d candidates to find %d valid paths.",
            len(A),
            len(valid_routes)
        )
        return valid_routes

    def _generate_intrazonal_cycle_routes(
        self,
        origin_rx: int,
        k: int,
        connector_link_types: Set[int]
    ) -> List[List[int]]:
        """
        Generates real cycles for intrazonal routing (origin == destination).
        For each successor, finds paths back to origin and validates connector constraints.
        """
        successors = []
        for v, edge_id, weight_val, link_type in self.adj.get(origin_rx, []):
            if v != origin_rx:
                successors.append((v, edge_id, weight_val, link_type))

        candidates_heap = []
        tie_breaker = count()
        seen_cycles = set()

        for s_rx, edge_id, first_weight, first_link_type in successors:
            tail_paths = self._yen_k_shortest_paths_rx(
                source_rx=s_rx,
                target_rx=origin_rx,
                k=k,
                connector_link_types=connector_link_types,
                cycle_origin_rx=origin_rx
            )
            for tail_path in tail_paths:
                cycle = [origin_rx] + tail_path
                cycle_tuple = tuple(cycle)
                if cycle_tuple in seen_cycles:
                    continue
                seen_cycles.add(cycle_tuple)

                cost = self._route_cost_rx(cycle)
                heapq.heappush(candidates_heap, (cost, next(tie_breaker), cycle))

        valid_cycles = []
        while candidates_heap and len(valid_cycles) < k:
            cost, _, cycle = heapq.heappop(candidates_heap)
            valid_cycles.append(cycle)

        return valid_cycles

    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: Dict[str, bool],
        connector_link_types: Set[int]
    ) -> List[Route]:
        
        # Ensure weight is prepared (checks if the requested weight attribute matches the cached one)
        if not hasattr(self, "_current_weight_attr") or self._current_weight_attr != weight:
            self._prepare_weight_and_adj(weight)

        origin_rx = self.node_to_rx.get(origin_id)
        destination_rx = self.node_to_rx.get(destination_id)

        if origin_rx is None or destination_rx is None:
            logger.warning("Origin %s or destination %s not found in graph.", origin_id, destination_id)
            return []

        if origin_id == destination_id:
            if not constraints.get("allow_auto_routes", False):
                return []
            cycles = self._generate_intrazonal_cycle_routes(origin_rx, k, connector_link_types)
            if not cycles:
                logger.warning("No cycle found for intrazonal origin %s.", origin_id)
            elif len(cycles) < k:
                # TODO: Remove debug log
                logger.debug("Only %d cycles found for intrazonal origin %s (requested k=%d).", len(cycles), origin_id, k)
            return [self._convert_path_to_original_ids(cycle) for cycle in cycles]

        # Standard routing
        routes_rx = self._yen_k_shortest_paths_rx(origin_rx, destination_rx, k, connector_link_types)
        
        if not routes_rx:
            logger.warning("No path found between origin %s and destination %s.", origin_id, destination_id)
        elif len(routes_rx) < k:
            logger.warning("Only %d routes found between origin %s and destination %s (requested k=%d).", len(routes_rx), origin_id, destination_id, k)

        return [self._convert_path_to_original_ids(route) for route in routes_rx]
