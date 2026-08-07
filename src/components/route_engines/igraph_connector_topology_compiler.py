"""Compile zone connector rules into a static igraph routing topology."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import igraph as ig
import networkx as nx


@dataclass(frozen=True)
class CompiledRoutingGraph:
    """Immutable routing graph and mappings back to the source network."""

    graph: ig.Graph
    source_terminal_by_zone: dict[int, int]
    sink_terminal_by_zone: dict[int, int]
    internal_node_by_original_id: dict[int, int]
    original_node_by_internal_id: dict[int, int]
    original_edge_by_internal_id: dict[int, tuple[int, int]]
    weights_by_attribute: dict[str, list[float]]
    zone_ids: frozenset[int]
    terminal_vertices: frozenset[int]


class ConnectorTopologyCompiler:
    """Create a terminal-expanded graph without dynamic connector filters.

    For a connector ``u -> v`` the original edge is removed.  If ``u`` is a
    zone, ``SOURCE[u] -> v`` is added; if ``v`` is a zone, ``u -> SINK[v]`` is
    added.  Thus a connector can only be used at the complete OD boundary.
    Zone nodes are removed from the core when the graph explicitly marks them
    with ``class=Zones`` (as the TNTP-derived artifacts do).
    """

    def compile(
        self,
        graph: nx.DiGraph,
        zone_ids: Iterable[int] | None,
        connector_link_types: set[int],
    ) -> CompiledRoutingGraph:
        if not isinstance(graph, nx.DiGraph):
            raise TypeError("ConnectorTopologyCompiler requires nx.DiGraph.")

        zones = self._resolve_zone_ids(graph, zone_ids, connector_link_types)
        explicit_zone_nodes = bool(connector_link_types) and any(
            isinstance(data, dict) and data.get("class") == "Zones"
            for _, data in graph.nodes(data=True)
        )
        core_nodes = [
            int(node)
            for node, data in graph.nodes(data=True)
            if not (explicit_zone_nodes and int(node) in zones)
        ]

        source_by_zone: dict[int, int] = {}
        sink_by_zone: dict[int, int] = {}
        vertex_original: dict[int, int | None] = {}
        vertex_kind: dict[int, str] = {}
        next_vertex = 0

        def add_vertex(original_id: int | None, kind: str) -> int:
            nonlocal next_vertex
            vertex = next_vertex
            next_vertex += 1
            vertex_original[vertex] = original_id
            vertex_kind[vertex] = kind
            return vertex

        internal_by_original: dict[int, int] = {}
        original_by_internal: dict[int, int] = {}
        for node in core_nodes:
            vertex = add_vertex(node, "core")
            internal_by_original[node] = vertex
            original_by_internal[vertex] = node

        for zone in sorted(zones):
            source_by_zone[zone] = add_vertex(None, "source")
            sink_by_zone[zone] = add_vertex(None, "sink")

        edges: list[tuple[int, int]] = []
        edge_attrs: list[dict[str, Any]] = []
        original_edge_by_internal: dict[int, tuple[int, int]] = {}

        def add_compiled_edge(
            u_vertex: int,
            v_vertex: int,
            data: dict[str, Any],
            original_edge: tuple[int, int] | None,
        ) -> None:
            edge_id = len(edges)
            edges.append((u_vertex, v_vertex))
            edge_attrs.append(dict(data))
            if original_edge is not None:
                original_edge_by_internal[edge_id] = original_edge

        for u_raw, v_raw, raw_data in graph.edges(data=True):
            u, v = int(u_raw), int(v_raw)
            data = dict(raw_data)
            link_type = int(data.get("link_type", -1))
            is_connector = link_type in connector_link_types
            u_is_zone = u in zones
            v_is_zone = v in zones

            if explicit_zone_nodes and (u_is_zone or v_is_zone):
                if u_is_zone:
                    if v_is_zone:
                        add_compiled_edge(
                            source_by_zone[u], sink_by_zone[v], data, (u, v)
                        )
                    elif v in internal_by_original:
                        add_compiled_edge(
                            source_by_zone[u], internal_by_original[v], data, (u, v)
                        )
                elif v_is_zone and u in internal_by_original:
                    add_compiled_edge(
                        internal_by_original[u], sink_by_zone[v], data, (u, v)
                    )
                continue

            if is_connector:
                if u_is_zone and v in internal_by_original:
                    add_compiled_edge(
                        source_by_zone[u], internal_by_original[v], data, (u, v)
                    )
                if v_is_zone and u in internal_by_original:
                    add_compiled_edge(
                        internal_by_original[u], sink_by_zone[v], data, (u, v)
                    )
                if not u_is_zone and not v_is_zone:
                    raise ValueError(
                        "Connector edge is not incident to a known zone: "
                        f"edge=({u}, {v}), link_type={link_type}."
                    )
                continue

            # Synthetic/reference graphs do not mark zone nodes.  Their source
            # and sink terminals are attached below with zero-cost edges.
            if u in internal_by_original and v in internal_by_original:
                add_compiled_edge(
                    internal_by_original[u], internal_by_original[v], data, (u, v)
                )

        if not explicit_zone_nodes:
            for zone in sorted(zones):
                if zone not in internal_by_original:
                    raise ValueError(f"Zone {zone} is not present in graph nodes.")
                zero_data = {"_terminal": True}
                add_compiled_edge(
                    source_by_zone[zone], internal_by_original[zone], zero_data, None
                )
                add_compiled_edge(
                    internal_by_original[zone], sink_by_zone[zone], zero_data, None
                )

        compiled = ig.Graph(n=next_vertex, edges=edges, directed=True)
        weight_names = self._weight_names(graph)
        for name in weight_names:
            values: list[float] = []
            for data in edge_attrs:
                if data.get("_terminal"):
                    values.append(0.0)
                elif name not in data:
                    raise ValueError(f"Weight attribute '{name}' is missing from a compiled edge.")
                else:
                    value = float(data[name])
                    if value < 0:
                        raise ValueError(f"Negative edge weight detected for '{name}': {value}")
                    values.append(value)
            compiled.es[name] = values

        compiled.vs["original_id"] = [vertex_original[v] for v in range(next_vertex)]
        compiled.vs["kind"] = [vertex_kind[v] for v in range(next_vertex)]
        compiled.es["original_edge"] = [
            original_edge_by_internal.get(edge_id) for edge_id in range(len(edges))
        ]

        internal_by_original = dict(internal_by_original)
        return CompiledRoutingGraph(
            graph=compiled,
            source_terminal_by_zone=source_by_zone,
            sink_terminal_by_zone=sink_by_zone,
            internal_node_by_original_id=internal_by_original,
            original_node_by_internal_id=original_by_internal,
            original_edge_by_internal_id=original_edge_by_internal,
            weights_by_attribute={name: list(compiled.es[name]) for name in weight_names},
            zone_ids=frozenset(zones),
            terminal_vertices=frozenset((*source_by_zone.values(), *sink_by_zone.values())),
        )

    @staticmethod
    def _resolve_zone_ids(
        graph: nx.DiGraph,
        zone_ids: Iterable[int] | None,
        connector_link_types: set[int],
    ) -> set[int]:
        if zone_ids is not None:
            zones = {int(zone) for zone in zone_ids}
        else:
            zones = {
                int(node)
                for node, data in graph.nodes(data=True)
                if data.get("class") == "Zones"
            }
        for u, v, data in graph.edges(data=True):
            if int(data.get("link_type", -1)) in connector_link_types:
                if int(u) not in zones and int(v) not in zones:
                    raise ValueError(
                        "Connector edge is not incident to a known zone: "
                        f"edge=({u}, {v})."
                    )
        return zones

    @staticmethod
    def _weight_names(graph: nx.DiGraph) -> list[str]:
        if graph.number_of_edges() == 0:
            return []
        common = set.intersection(
            *[set(data.keys()) for _, _, data in graph.edges(data=True)]
        )
        excluded = {"link_type", "link_id", "reverse_link_id", "_terminal"}
        names: list[str] = []
        for name in sorted(common - excluded):
            try:
                values = [float(data[name]) for _, _, data in graph.edges(data=True)]
            except (TypeError, ValueError):
                continue
            if all(value >= 0 for value in values):
                names.append(name)
        return names
