from __future__ import annotations
# src/data_ingestion/adapters/route_model_adapter.py

"""
Route Model Adapter
===================

This module converts processed transportation network objects into model-ready
PyTorch tensors.

Project context
---------------
In the AADT / traffic assignment pipeline, routes are stored as node sequences:

    (origin, destination): [
        [origin, ..., destination],
        [origin, ..., destination],
    ]

The neural network and route-based traffic assignment models cannot directly
operate on these Python lists. They need tensor representations that describe:

- which links belong to each route;
- which route slots are valid;
- physical link attributes such as effective capacity and free-flow time;
- OD-pair indexing in model space.

This adapter converts:

    graph + routes_by_od

into:

- route_masks: sparse 3D legacy incidence tensor [OD, K, Links];
- delta_matrix: sparse 2D route-link incidence tensor [Links, OD*K];
- route_validity_mask: dense boolean mask [OD, K];
- od_pair_indices: tensor [OD, 2];
- physical link tensors: t0, effective_capacity, length, lanes, speed, link_group.

This module does not read files, build graphs, create targets, or save artifacts.
It only adapts processed network objects into tensors.

Design principles
-----------------
- Use the graph edge order as canonical link order.
- Preserve OD-pair order from routes_by_od.
- Keep both legacy and optimized incidence formats.
- Validate route links against graph edges.
- Use safe defaults only when explicitly allowed.
- Return CPU tensors by default for artifact portability.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np
import torch


logger = logging.getLogger(__name__)


Edge = Tuple[int, int]
ODPair = Tuple[int, int]
Route = List[int]
RoutesByOD = Dict[ODPair, List[Route]]


@dataclass(frozen=True)
class RouteModelAdaptResult:
    """
    Container returned by RouteModelAdapter.

    Attributes
    ----------
    network_params : Dict[str, Any]
        Model-ready network parameters and tensors.

    metadata : Dict[str, Any]
        Metadata describing tensor construction, route validity, skipped links
        and dimensional summaries.
    """

    network_params: Dict[str, Any]
    metadata: Dict[str, Any]


class RouteModelAdapter:
    """
    Convert graph and route data into PyTorch tensors for route-based models.

    Parameters
    ----------
    device : str, default="cpu"
        Device used for tensor allocation. For artifact generation, "cpu" is
        recommended.

    k_paths : int, default=10
        Maximum number of routes retained per OD pair.

    strict : bool, default=True
        If True, routes containing links not present in the graph raise errors.
        If False, invalid route links are skipped and reported in metadata.

    allow_safe_defaults : bool, default=True
        If True, missing optional edge attributes are filled with safe defaults.
        Required attributes are still validated when strict=True.

    coalesce_sparse_tensors : bool, default=True
        If True, sparse tensors are coalesced before returning.
    """

    REQUIRED_EDGE_ATTRIBUTES = {
        "free_flow_time",
        "effective_capacity",
        "lanes"
    }

    DEFAULT_EDGE_VALUES = {
        "free_flow_time": 1.0,
        "effective_capacity": 1000.0,
        "length": 1.0,
        "lanes": 1,
        "speed": 15.0,
        "link_type": 0,
        "b": 0.15,
        "power": 4.0,
        "toll": 0.0,
        
    }

    def __init__(
        self,
        device: str = "cpu",
        k_paths: int = 10,
        strict: bool = True,
        allow_safe_defaults: bool = True,
        coalesce_sparse_tensors: bool = True,
    ) -> None:
        self.device = str(device)
        self.k_paths = int(k_paths)
        self.strict = bool(strict)
        self.allow_safe_defaults = bool(allow_safe_defaults)
        self.coalesce_sparse_tensors = bool(coalesce_sparse_tensors)

        if self.k_paths <= 0:
            raise ValueError("k_paths must be a positive integer.")

    def transform(
        self,
        graph: nx.DiGraph,
        routes_by_od: RoutesByOD,
        edge_order: Optional[Sequence[Edge]] = None,
    ) -> Dict[str, Any]:
        """
        Convert graph and routes into model-ready tensors.

        The optional edge_order argument defines the canonical model link order.
        If provided, all link-level tensors and route-link incidence structures
        are built in exactly that order.
        """

        return self.adapt(
            graph=graph,
            routes_by_od=routes_by_od,
            edge_order=edge_order,
        ).network_params

    def adapt(
        self,
        graph: nx.DiGraph,
        routes_by_od: RoutesByOD,
        edge_order: Optional[Sequence[Edge]] = None,
    ) -> RouteModelAdaptResult:
        """
        Convert graph and routes into model-ready tensors and metadata.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph with edge attributes.

        routes_by_od : RoutesByOD
            Dictionary mapping each OD pair to a list of node-based routes.

        Returns
        -------
        RouteModelAdaptResult
            Network parameters and metadata.
        """

        logger.info("Adapting graph and routes into model-ready tensors.")

        self._validate_inputs(graph=graph, routes_by_od=routes_by_od)

        edge_list = self._resolve_edge_list(
    graph=graph,
            edge_order=edge_order,
        )

        edge_to_idx = self._build_edge_to_idx(
            edge_list=edge_list,
        )   

        physics_payload = self._extract_physics(
            graph=graph,
            edge_list=edge_list,
        )

        route_payload = self._build_route_incidence_tensors(
            routes_by_od=routes_by_od,
            edge_to_idx=edge_to_idx,
            num_links=len(edge_list),
        )

        node_indexing = self._build_node_indexing(graph)

        od_pair_indices = self._build_od_pair_indices(
            od_pairs=route_payload["od_pairs"],
            node_id_to_idx=node_indexing["node_id_to_idx"],
        )

        link_pair_indices = np.array(
            edge_list,
            dtype=np.int64,
        )

        network_params: Dict[str, Any] = {
            "num_links": int(len(edge_list)),
            "num_od_pairs": int(len(route_payload["od_pairs"])),
            "k_paths": int(self.k_paths),

            # Topology tensors
            "route_masks": route_payload["route_masks"],
            "delta_matrix": route_payload["delta_matrix"],
            "route_validity_mask": route_payload["route_validity_mask"],

            # OD indexing
            "od_pairs": route_payload["od_pairs"],
            "od_pair_indices": od_pair_indices,
            "od_pair_node_labels": [
                [str(origin), str(destination)]
                for origin, destination in route_payload["od_pairs"]
            ],

            # Link indexing
            "edge_list": edge_list,
            "edge_to_idx": edge_to_idx,
            "link_pair_indices": link_pair_indices,

            # Node indexing
            "node_id_to_idx": node_indexing["node_id_to_idx"],
            "idx_to_node_id": node_indexing["idx_to_node_id"],

            # Route diagnostics
            "route_lengths_links": route_payload["route_lengths_links"],

            # Physical tensors
            **physics_payload,
        }

        metadata = self._build_metadata(
            graph=graph,
            edge_list=edge_list,
            route_payload=route_payload,
            physics_payload=physics_payload,
        )

        logger.info(
            "Route model adaptation completed | links=%d | od_pairs=%d | routes=%d",
            metadata["num_links"],
            metadata["num_od_pairs"],
            metadata["num_valid_routes"],
        )

        return RouteModelAdaptResult(
            network_params=network_params,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Input validation and indexing
    # ------------------------------------------------------------------

    def _validate_inputs(
        self,
        graph: nx.DiGraph,
        routes_by_od: RoutesByOD,
    ) -> None:
        """
        Validate graph and route inputs.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        routes_by_od : RoutesByOD
            Route dictionary.
        """

        if not isinstance(graph, nx.DiGraph):
            raise TypeError(
                f"graph must be a networkx.DiGraph. Got: {type(graph)}"
            )

        if graph.number_of_edges() == 0:
            raise ValueError("graph contains no directed edges.")

        if graph.number_of_nodes() == 0:
            raise ValueError("graph contains no nodes.")

        if not isinstance(routes_by_od, dict):
            raise TypeError(
                f"routes_by_od must be a dictionary. Got: {type(routes_by_od)}"
            )

        if not routes_by_od:
            raise ValueError("routes_by_od is empty.")

        if self.strict:
            self._validate_required_edge_attributes(graph)

    def _validate_required_edge_attributes(
        self,
        graph: nx.DiGraph,
    ) -> None:
        """
        Validate required edge attributes.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        Raises
        ------
        KeyError
            If required edge attributes are missing.
        """

        missing_records = []

        for u, v, data in graph.edges(data=True):
            missing = self.REQUIRED_EDGE_ATTRIBUTES - set(data.keys())

            if missing:
                missing_records.append(
                    {
                        "edge": (int(u), int(v)),
                        "missing": sorted(missing),
                    }
                )

        if missing_records:
            sample = missing_records[:10]

            raise KeyError(
                "Some graph edges are missing required attributes. "
                f"sample={sample}"
            )

    @staticmethod
    def _get_edge_list(graph: nx.DiGraph) -> List[Edge]:
        """
        Get canonical edge order from the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        Returns
        -------
        List[Edge]
            Ordered directed edges.
        """

        return [
            (int(u), int(v))
            for u, v in graph.edges()
        ]

    @staticmethod
    def _resolve_edge_list(
        graph: nx.DiGraph,
        edge_order: Optional[Sequence[Edge]] = None,
    ) -> List[Edge]:
        """
        Resolve the canonical link order used by the model.

        If edge_order is provided, it becomes the source of truth. This is the
        preferred behavior because the canonical link order should be inherited
        from GraphBuilder / processed["edge_indexing"], not inferred again from
        graph.edges().

        If edge_order is not provided, the method falls back to graph.edges() for
        backward compatibility.
        """

        if edge_order is None:
            return RouteModelAdapter._get_edge_list(graph)

        resolved_edge_list = [
            (int(u), int(v))
            for u, v in edge_order
        ]

        if not resolved_edge_list:
            raise ValueError("edge_order was provided but it is empty.")

        if len(resolved_edge_list) != len(set(resolved_edge_list)):
            duplicated_edges = RouteModelAdapter._find_duplicated_edges(
                resolved_edge_list
            )

            raise ValueError(
                "edge_order contains duplicated directed edges. "
                f"Sample duplicated edges: {duplicated_edges[:20]}"
            )

        graph_edge_set = {
            (int(u), int(v))
            for u, v in graph.edges()
        }

        edge_order_set = set(resolved_edge_list)

        if edge_order_set != graph_edge_set:
            missing_in_graph = sorted(edge_order_set - graph_edge_set)
            extra_in_graph = sorted(graph_edge_set - edge_order_set)

            raise ValueError(
                "Provided edge_order does not match graph edges. "
                "The model cannot build aligned link tensors. "
                f"missing_in_graph_sample={missing_in_graph[:20]} | "
                f"extra_in_graph_sample={extra_in_graph[:20]} | "
                f"num_missing_in_graph={len(missing_in_graph)} | "
                f"num_extra_in_graph={len(extra_in_graph)}"
            )

        return resolved_edge_list

    @staticmethod
    def _find_duplicated_edges(
        edge_list: Sequence[Edge],
    ) -> List[Edge]:
        """
        Return duplicated directed edges from an edge list.
        """

        seen = set()
        duplicated = []

        for edge in edge_list:
            if edge in seen:
                duplicated.append(edge)
            else:
                seen.add(edge)

        return duplicated

    @staticmethod
    def _build_edge_to_idx(edge_list: Sequence[Edge]) -> Dict[Edge, int]:
        """
        Build edge-to-index mapping.

        Parameters
        ----------
        edge_list : Sequence[Edge]
            Ordered edge list.

        Returns
        -------
        Dict[Edge, int]
            Mapping from directed edge to link index.
        """

        return {
            (int(u), int(v)): int(idx)
            for idx, (u, v) in enumerate(edge_list)
        }

    @staticmethod
    def _build_node_indexing(graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Build node indexing dictionaries.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        Returns
        -------
        Dict[str, Any]
            Node indexing payload.
        """

        node_list = [
            int(node)
            for node in graph.nodes()
        ]

        node_id_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(node_list)
        }

        idx_to_node_id = {
            idx: node_id
            for node_id, idx in node_id_to_idx.items()
        }

        return {
            "node_list": node_list,
            "node_id_to_idx": node_id_to_idx,
            "idx_to_node_id": idx_to_node_id,
        }

    # ------------------------------------------------------------------
    # Physics tensors
    # ------------------------------------------------------------------

    def _extract_physics(
        self,
        graph: nx.DiGraph,
        edge_list: Sequence[Edge],
    ) -> Dict[str, Any]:
        """
        Extract physical and operational edge attributes into tensors.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        edge_list : Sequence[Edge]
            Canonical edge order.

        Returns
        -------
        Dict[str, Any]
            Physical link tensors.
        """

        t0_list = []
        capacity_list = []
        length_list = []
        lanes_list = []
        speed_list = []
        link_type_list = []
        b_list = []
        power_list = []
        toll_list = []

        for edge in edge_list:
            u, v = edge
            data = graph[u][v]

            t0_list.append(
                self._get_edge_value(data, "free_flow_time", edge)
            )
            capacity_list.append(
                self._get_edge_value(data, "effective_capacity", edge)
            )
            length_list.append(
                self._get_edge_value(data, "length", edge)
            )
            lanes_list.append(
                self._get_edge_value(data, "lanes", edge)
            )
            speed_list.append(
                self._get_edge_value(data, "speed", edge)
            )
            link_type_list.append(
                self._get_edge_value(data, "link_type", edge)
            )
            b_list.append(
                self._get_edge_value(data, "b", edge)
            )
            power_list.append(
                self._get_edge_value(data, "power", edge)
            )
            toll_list.append(
                self._get_edge_value(data, "toll", edge)
            )

        link_type_array = np.asarray(link_type_list, dtype=np.int64)
        unique_link_types = np.unique(link_type_array)

        link_type_to_group = {
            int(link_type): group_idx
            for group_idx, link_type in enumerate(unique_link_types)
        }

        link_group_indices = np.array(
            [
                link_type_to_group[int(link_type)]
                for link_type in link_type_array
            ],
            dtype=np.int64,
        )

        return {
            "t0": self._float_tensor(t0_list),
            "effective_capacity": self._float_tensor(capacity_list),
            "length": self._float_tensor(length_list),
            "lanes": self._long_tensor(lanes_list),
            "speed": self._float_tensor(speed_list),
            "b": self._float_tensor(b_list),
            "power": self._float_tensor(power_list),
            "toll": self._float_tensor(toll_list),
            "link_type_raw": self._long_tensor(link_type_list),
            "link_group": self._long_tensor(link_group_indices),
            "num_link_groups": int(len(unique_link_types)),
            "link_type_to_group": link_type_to_group,
        }

    def _get_edge_value(
        self,
        data: Dict[str, Any],
        key: str,
        edge: Edge,
    ) -> Any:
        """
        Get an edge attribute, applying safe defaults when allowed.

        Parameters
        ----------
        data : Dict[str, Any]
            Edge attribute dictionary.

        key : str
            Attribute name.

        edge : Edge
            Edge used for error messages.

        Returns
        -------
        Any
            Edge attribute value.
        """

        value = data.get(key, None)

        if value is None:
            if not self.allow_safe_defaults:
                raise KeyError(
                    f"Edge {edge} is missing attribute '{key}' and "
                    "allow_safe_defaults=False."
                )

            return self.DEFAULT_EDGE_VALUES[key]

        return value

    # ------------------------------------------------------------------
    # Route incidence tensors
    # ------------------------------------------------------------------

    def _build_route_incidence_tensors(
        self,
        routes_by_od: RoutesByOD,
        edge_to_idx: Dict[Edge, int],
        num_links: int,
    ) -> Dict[str, Any]:
        """
        Build sparse route-link incidence tensors.

        The method creates two complementary representations:

        1. route_masks:
           Sparse 3D tensor with shape [num_od_pairs, k_paths, num_links].
           This is kept for legacy compatibility.

        2. delta_matrix:
           Sparse 2D tensor with shape [num_links, num_od_pairs * k_paths].
           This is efficient for link-flow projection using sparse matrix
           multiplication.

        Parameters
        ----------
        routes_by_od : RoutesByOD
            Route dictionary.

        edge_to_idx : Dict[Edge, int]
            Mapping from graph edge to link index.

        num_links : int
            Number of graph links.

        Returns
        -------
        Dict[str, Any]
            Route incidence tensors and diagnostics.
        """

        od_pairs = [
            (int(origin), int(destination))
            for origin, destination in routes_by_od.keys()
        ]

        num_od_pairs = len(od_pairs)

        indices_3d: List[List[int]] = []
        values_3d: List[float] = []

        indices_2d: List[List[int]] = []
        values_2d: List[float] = []

        validity_mask = torch.zeros(
            (num_od_pairs, self.k_paths),
            dtype=torch.bool,
            device=self.device,
        )

        route_lengths_links = torch.zeros(
            (num_od_pairs, self.k_paths),
            dtype=torch.long,
            device=self.device,
        )

        invalid_route_records: List[Dict[str, Any]] = []
        skipped_route_link_records: List[Dict[str, Any]] = []

        for od_idx, od_pair in enumerate(od_pairs):
            routes = routes_by_od.get(od_pair, [])

            if len(routes) > self.k_paths:
                logger.warning(
                    "OD pair %s has %d routes, but only first %d will be retained.",
                    od_pair,
                    len(routes),
                    self.k_paths,
                )

            for k_idx, route in enumerate(routes[: self.k_paths]):
                route = [int(node) for node in route]

                is_valid, invalid_reason = self._validate_route_structure(
                    route=route,
                    od_pair=od_pair,
                )

                if not is_valid:
                    invalid_record = {
                        "od_pair": od_pair,
                        "k_idx": int(k_idx),
                        "route": route,
                        "reason": invalid_reason,
                    }

                    if self.strict:
                        raise ValueError(f"Invalid route found: {invalid_record}")

                    invalid_route_records.append(invalid_record)
                    continue

                link_indices = self._route_to_link_indices(
                    route=route,
                    edge_to_idx=edge_to_idx,
                    od_pair=od_pair,
                    k_idx=k_idx,
                    skipped_route_link_records=skipped_route_link_records,
                )

                if len(link_indices) == 0:
                    invalid_record = {
                        "od_pair": od_pair,
                        "k_idx": int(k_idx),
                        "route": route,
                        "reason": "route contains no valid graph links",
                    }

                    if self.strict:
                        raise ValueError(f"Invalid route found: {invalid_record}")

                    invalid_route_records.append(invalid_record)
                    continue

                validity_mask[od_idx, k_idx] = True
                route_lengths_links[od_idx, k_idx] = len(link_indices)

                flattened_route_idx = (od_idx * self.k_paths) + k_idx

                for link_idx in link_indices:
                    indices_3d.append(
                        [
                            int(od_idx),
                            int(k_idx),
                            int(link_idx),
                        ]
                    )
                    values_3d.append(1.0)

                    indices_2d.append(
                        [
                            int(link_idx),
                            int(flattened_route_idx),
                        ]
                    )
                    values_2d.append(1.0)

        route_masks = self._build_sparse_tensor(
            indices=indices_3d,
            values=values_3d,
            size=(num_od_pairs, self.k_paths, num_links),
        )

        delta_matrix = self._build_sparse_tensor(
            indices=indices_2d,
            values=values_2d,
            size=(num_links, num_od_pairs * self.k_paths),
        )

        return {
            "od_pairs": od_pairs,
            "route_masks": route_masks,
            "delta_matrix": delta_matrix,
            "route_validity_mask": validity_mask,
            "route_lengths_links": route_lengths_links,
            "invalid_route_records": invalid_route_records,
            "skipped_route_link_records": skipped_route_link_records,
            "num_valid_routes": int(validity_mask.sum().item()),
            "num_invalid_routes": int(len(invalid_route_records)),
            "num_skipped_route_links": int(len(skipped_route_link_records)),
        }

    def _validate_route_structure(
        self,
        route: Route,
        od_pair: ODPair,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate a node-based route structure.

        Parameters
        ----------
        route : Route
            Node sequence.

        od_pair : ODPair
            Expected origin and destination.

        Returns
        -------
        Tuple[bool, Optional[str]]
            Whether the route is valid and invalid reason if applicable.
        """

        origin, destination = od_pair

        if len(route) < 2:
            return False, "route has fewer than two nodes"

        if int(route[0]) != int(origin):
            return False, f"route starts at {route[0]}, expected {origin}"

        if int(route[-1]) != int(destination):
            return False, f"route ends at {route[-1]}, expected {destination}"

        # TODO: a route may have repeated nodes, in cae of auto routes
        #if len(route) != len(set(route)):
        #    return False, "route contains repeated nodes"

        return True, None

    def _route_to_link_indices(
        self,
        route: Route,
        edge_to_idx: Dict[Edge, int],
        od_pair: ODPair,
        k_idx: int,
        skipped_route_link_records: List[Dict[str, Any]],
    ) -> List[int]:
        """
        Convert a node-based route into graph link indices.

        Parameters
        ----------
        route : Route
            Node sequence.

        edge_to_idx : Dict[Edge, int]
            Edge-to-index mapping.

        od_pair : ODPair
            OD pair used for diagnostics.

        k_idx : int
            Route index within the OD pair.

        skipped_route_link_records : List[Dict[str, Any]]
            Mutable list where skipped link diagnostics are appended.

        Returns
        -------
        List[int]
            Link indices in canonical graph edge order.
        """

        link_indices: List[int] = []

        for u, v in zip(route[:-1], route[1:]):
            edge = (int(u), int(v))

            if edge not in edge_to_idx:
                record = {
                    "od_pair": od_pair,
                    "k_idx": int(k_idx),
                    "missing_edge": edge,
                    "route": route,
                }

                if self.strict:
                    raise KeyError(
                        "Route contains an edge not present in the graph: "
                        f"{record}"
                    )

                skipped_route_link_records.append(record)
                continue

            link_indices.append(int(edge_to_idx[edge]))

        return link_indices

    def _build_od_pair_indices(
        self,
        od_pairs: Sequence[ODPair],
        node_id_to_idx: Dict[int, int],
    ) -> torch.Tensor:
        """
        Build OD-pair indices in graph node-index space.

        Parameters
        ----------
        od_pairs : Sequence[ODPair]
            Ordered OD pairs in node-label space.

        node_id_to_idx : Dict[int, int]
            Mapping from node ID to graph node index.

        Returns
        -------
        torch.Tensor
            Long tensor with shape [num_od_pairs, 2].
        """

        indices = []

        for origin, destination in od_pairs:
            if origin not in node_id_to_idx or destination not in node_id_to_idx:
                message = (
                    f"OD pair ({origin}, {destination}) references nodes not "
                    "present in graph node indexing."
                )

                if self.strict:
                    raise KeyError(message)

                logger.warning(message)
                indices.append([-1, -1])
                continue

            indices.append(
                [
                    int(node_id_to_idx[origin]),
                    int(node_id_to_idx[destination]),
                ]
            )

        return torch.tensor(
            indices,
            dtype=torch.long,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Sparse and dense tensor helpers
    # ------------------------------------------------------------------

    def _build_sparse_tensor(
        self,
        indices: List[List[int]],
        values: List[float],
        size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Build a sparse COO tensor.

        Parameters
        ----------
        indices : List[List[int]]
            Sparse indices as list of coordinate rows.

        values : List[float]
            Sparse values.

        size : Tuple[int, ...]
            Tensor size.

        Returns
        -------
        torch.Tensor
            Sparse COO tensor.
        """

        if not indices:
            index_tensor = torch.empty(
                (len(size), 0),
                dtype=torch.long,
                device=self.device,
            )

            value_tensor = torch.empty(
                (0,),
                dtype=torch.float32,
                device=self.device,
            )

        else:
            index_tensor = torch.tensor(
                indices,
                dtype=torch.long,
                device=self.device,
            ).T

            value_tensor = torch.tensor(
                values,
                dtype=torch.float32,
                device=self.device,
            )

        sparse_tensor = torch.sparse_coo_tensor(
            indices=index_tensor,
            values=value_tensor,
            size=size,
            dtype=torch.float32,
            device=self.device,
        )

        if self.coalesce_sparse_tensors:
            sparse_tensor = sparse_tensor.coalesce()

        return sparse_tensor

    def _float_tensor(self, values: Sequence[Any]) -> torch.Tensor:
        """
        Build a float tensor on the configured device.

        Parameters
        ----------
        values : Sequence[Any]
            Input values.

        Returns
        -------
        torch.Tensor
            Float tensor.
        """

        return torch.tensor(
            values,
            dtype=torch.float32,
            device=self.device,
        )

    def _long_tensor(self, values: Sequence[Any]) -> torch.Tensor:
        """
        Build a long tensor on the configured device.

        Parameters
        ----------
        values : Sequence[Any]
            Input values.

        Returns
        -------
        torch.Tensor
            Long tensor.
        """

        return torch.tensor(
            values,
            dtype=torch.long,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        graph: nx.DiGraph,
        edge_list: Sequence[Edge],
        route_payload: Dict[str, Any],
        physics_payload: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build metadata describing the adaptation result.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        edge_list : Sequence[Edge]
            Canonical edge list.

        route_payload : Dict[str, Any]
            Route incidence payload.

        physics_payload : Dict[str, Any]
            Physical tensor payload.

        Returns
        -------
        Dict[str, Any]
            Adaptation metadata.
        """

        validity_mask = route_payload["route_validity_mask"]
        route_lengths = route_payload["route_lengths_links"]

        valid_route_lengths = route_lengths[validity_mask]

        metadata: Dict[str, Any] = {
            "num_nodes": int(graph.number_of_nodes()),
            "num_links": int(len(edge_list)),
            "num_od_pairs": int(len(route_payload["od_pairs"])),
            "k_paths": int(self.k_paths),
            "num_route_slots": int(validity_mask.numel()),
            "num_valid_routes": int(validity_mask.sum().item()),
            "num_invalid_routes": int(route_payload["num_invalid_routes"]),
            "num_skipped_route_links": int(route_payload["num_skipped_route_links"]),
            "route_masks_shape": tuple(route_payload["route_masks"].shape),
            "route_masks_nnz": int(route_payload["route_masks"]._nnz()),
            "delta_matrix_shape": tuple(route_payload["delta_matrix"].shape),
            "delta_matrix_nnz": int(route_payload["delta_matrix"]._nnz()),
            "route_validity_mask_shape": tuple(validity_mask.shape),
            "device": self.device,
            "strict": self.strict,
            "allow_safe_defaults": self.allow_safe_defaults,
            "num_link_groups": int(physics_payload["num_link_groups"]),
            "link_type_to_group": physics_payload["link_type_to_group"],
        }

        if valid_route_lengths.numel() > 0:
            metadata.update(
                {
                    "min_valid_route_length_links": int(valid_route_lengths.min().item()),
                    "max_valid_route_length_links": int(valid_route_lengths.max().item()),
                    "mean_valid_route_length_links": float(
                        valid_route_lengths.float().mean().item()
                    ),
                }
            )
        else:
            metadata.update(
                {
                    "min_valid_route_length_links": 0,
                    "max_valid_route_length_links": 0,
                    "mean_valid_route_length_links": 0.0,
                }
            )

        if route_payload["invalid_route_records"]:
            metadata["invalid_route_records_sample"] = (
                route_payload["invalid_route_records"][:20]
            )

        if route_payload["skipped_route_link_records"]:
            metadata["skipped_route_link_records_sample"] = (
                route_payload["skipped_route_link_records"][:20]
            )

        return metadata


def adapt_routes_to_model(
    graph: nx.DiGraph,
    routes_by_od: RoutesByOD,
    device: str = "cpu",
    k_paths: int = 10,
    strict: bool = True,
    allow_safe_defaults: bool = True,
    coalesce_sparse_tensors: bool = True,
) -> RouteModelAdaptResult:
    """
    Convenience function to adapt graph and routes into model-ready tensors.

    Parameters
    ----------
    graph : nx.DiGraph
        Directed graph.

    routes_by_od : RoutesByOD
        Dictionary mapping OD pairs to route lists.

    device : str, default="cpu"
        Tensor device.

    k_paths : int, default=10
        Maximum number of paths per OD pair.

    strict : bool, default=True
        Whether to use strict validation behavior.

    allow_safe_defaults : bool, default=True
        Whether missing optional edge attributes may use safe defaults.

    coalesce_sparse_tensors : bool, default=True
        Whether sparse tensors should be coalesced.

    Returns
    -------
    RouteModelAdaptResult
        Network parameters and metadata.
    """

    adapter = RouteModelAdapter(
        device=device,
        k_paths=k_paths,
        strict=strict,
        allow_safe_defaults=allow_safe_defaults,
        coalesce_sparse_tensors=coalesce_sparse_tensors,
    )

    return adapter.adapt(
        graph=graph,
        routes_by_od=routes_by_od,
    )
