"""Route-set domain objects for route-based traffic assignment.

This module converts raw route definitions into a canonical, validated route
representation that can be reused by different assignment behavior models and
solution algorithms.

Design goals
------------
1. Avoid silent fallbacks: missing or inconsistent inputs raise explicit errors.
2. Keep route preparation outside assignment motors: SUE and Gradient Projection
   should consume a prepared RouteSet instead of rebuilding route structures.
3. Preserve link-order consistency: route-link positions are always tied to the
   row order of the provided links_df.
4. Support future extensibility: routes can be built from node sequences or
   link-id sequences, and additional route sources can be added later without
   changing assignment motors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


class RouteInputFormat(str, Enum):
    """Supported raw route representations.

    NODE_SEQUENCE means each route is a sequence of node IDs, for example:
    [origin_node, intermediate_node, destination_node].

    LINK_SEQUENCE means each route is a sequence of link IDs, for example:
    [link_id_1, link_id_2, link_id_3].
    """

    NODE_SEQUENCE = "node_sequence"
    LINK_SEQUENCE = "link_sequence"


@dataclass(frozen=True)
class RouteSetBuildConfig:
    """Configuration used to build and validate a RouteSet.

    No field has a default value on purpose. Every value must be injected from
    configuration or from an upstream validated artifact. This avoids silent
    assumptions during testing and assignment.

    Attributes
    ----------
    route_input_format:
        Raw route representation expected by the builder.
    link_id_col:
        Column in links_df containing the stable link identifier.
    init_node_col:
        Column in links_df containing the directed link start node.
    term_node_col:
        Column in links_df containing the directed link end node.
    route_cost_col:
        Column used to compute free-flow route cost at preparation time.
        This can later be changed to another static route attribute if needed.
    fail_on_duplicate_routes:
        If True, duplicated routes inside the same OD pair raise an error.
    fail_on_empty_route_set:
        If True, an empty route dictionary raises an error.
    fail_on_missing_od_routes:
        If True, positive-demand OD pairs passed to validate_positive_od_coverage
        must have at least one route.
    require_simple_node_routes:
        If True, node-based routes cannot repeat nodes. This is useful for
        standard simple-path K-shortest route sets.
    require_unique_link_ids:
        If True, links_df cannot contain duplicated link IDs.
    require_unique_directed_edges:
        If True, links_df cannot contain duplicated directed pairs
        (init_node, term_node). This should be True for node-sequence routes
        because node-to-link conversion would otherwise be ambiguous.
    """

    route_input_format: RouteInputFormat
    link_id_col: str
    init_node_col: str
    term_node_col: str
    route_cost_col: str
    fail_on_duplicate_routes: bool
    fail_on_empty_route_set: bool
    fail_on_missing_od_routes: bool
    require_simple_node_routes: bool
    require_unique_link_ids: bool
    require_unique_directed_edges: bool

    def __post_init__(self) -> None:
        if not isinstance(self.route_input_format, RouteInputFormat):
            raise TypeError(
                "route_input_format must be a RouteInputFormat enum value. "
                "Convert external YAML strings before creating RouteSetBuildConfig."
            )

        text_fields = {
            "link_id_col": self.link_id_col,
            "init_node_col": self.init_node_col,
            "term_node_col": self.term_node_col,
            "route_cost_col": self.route_cost_col,
        }
        for field_name, value in text_fields.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")

        bool_fields = {
            "fail_on_duplicate_routes": self.fail_on_duplicate_routes,
            "fail_on_empty_route_set": self.fail_on_empty_route_set,
            "fail_on_missing_od_routes": self.fail_on_missing_od_routes,
            "require_simple_node_routes": self.require_simple_node_routes,
            "require_unique_link_ids": self.require_unique_link_ids,
            "require_unique_directed_edges": self.require_unique_directed_edges,
        }
        for field_name, value in bool_fields.items():
            if not isinstance(value, bool):
                raise TypeError(f"{field_name} must be a bool.")

        if self.route_input_format == RouteInputFormat.NODE_SEQUENCE and not self.require_unique_directed_edges:
            raise ValueError(
                "Node-sequence routes require unique directed edges in links_df. "
                "Otherwise, converting consecutive nodes into link IDs is ambiguous."
            )


@dataclass(frozen=True)
class PreparedRoute:
    """Single prepared route record.

    Attributes
    ----------
    route_index:
        Global zero-based route index in the prepared RouteSet.
    origin_id:
        Origin zone/node ID used as the OD key.
    destination_id:
        Destination zone/node ID used as the OD key.
    route_number:
        One-based route number within the OD pair.
    route_nodes:
        Node sequence when available. For link-sequence inputs this can be
        reconstructed from links_df and is therefore stored as a tuple too.
    route_link_ids:
        Stable link IDs traversed by the route.
    route_link_positions:
        Zero-based row positions of route links in the current canonical
        links_df order. These positions must only be used with the same link
        order used to build this RouteSet.
    free_flow_route_cost:
        Static route cost computed using config.route_cost_col.
    """

    route_index: int
    origin_id: int
    destination_id: int
    route_number: int
    route_nodes: tuple[int, ...]
    route_link_ids: tuple[int, ...]
    route_link_positions: tuple[int, ...]
    free_flow_route_cost: float

    def __post_init__(self) -> None:
        if self.route_index < 0:
            raise ValueError("route_index must be non-negative.")
        if self.route_number < 1:
            raise ValueError("route_number must be one-based and therefore >= 1.")
        if len(self.route_link_ids) != len(self.route_link_positions):
            raise ValueError("route_link_ids and route_link_positions must have the same length.")
        if not np.isfinite(self.free_flow_route_cost):
            raise ValueError("free_flow_route_cost must be finite.")
        if self.free_flow_route_cost < 0.0:
            raise ValueError("free_flow_route_cost cannot be negative.")


@dataclass(frozen=True)
class RouteSet:
    """Canonical route representation for route-based assignment.

    A RouteSet is immutable after construction. Assignment motors can safely use
    it without revalidating raw route structures. If links_df changes order, a
    new RouteSet must be built because route_link_positions are order-dependent.
    """

    routes_df: pd.DataFrame
    od_to_route_indices: dict[tuple[int, int], list[int]]
    route_link_positions: list[list[int]]
    route_link_ids: list[list[int]]
    canonical_link_id_order: tuple[int, ...]
    config: RouteSetBuildConfig

    @classmethod
    def from_routes_by_od(
        cls,
        *,
        links_df: pd.DataFrame,
        routes_by_od: Mapping[tuple[int, int], Sequence[Sequence[int]]],
        config: RouteSetBuildConfig,
    ) -> "RouteSet":
        """Build a RouteSet from a dictionary keyed by OD pair.

        Parameters
        ----------
        links_df:
            Link table whose row order defines the canonical link-flow order.
        routes_by_od:
            Mapping from (origin_id, destination_id) to a list of routes. Each
            route must follow config.route_input_format.
        config:
            Strict route-set construction configuration.

        Returns
        -------
        RouteSet
            Prepared and validated route representation.
        """
        _validate_links_df_for_route_set(links_df=links_df, config=config)
        _validate_routes_container(routes_by_od=routes_by_od, config=config)

        link_id_to_position = _build_link_id_to_position(links_df=links_df, config=config)
        directed_edge_to_link_id = _build_directed_edge_to_link_id(links_df=links_df, config=config)
        link_id_to_nodes = _build_link_id_to_nodes(links_df=links_df, config=config)

        link_cost_series = _build_link_cost_series(links_df=links_df, config=config)
        canonical_link_id_order = tuple(int(value) for value in links_df[config.link_id_col].tolist())

        prepared_routes: list[PreparedRoute] = []
        od_to_route_indices: dict[tuple[int, int], list[int]] = {}
        seen_routes_by_od: dict[tuple[int, int], set[tuple[int, ...]]] = {}

        for raw_od_pair, raw_routes in routes_by_od.items():
            od_pair = _normalize_od_pair(raw_od_pair)
            od_to_route_indices[od_pair] = []
            seen_routes_by_od[od_pair] = set()

            if not isinstance(raw_routes, Sequence):
                raise TypeError(f"Routes for OD pair {od_pair} must be a sequence of routes.")

            for route_number, raw_route in enumerate(raw_routes, start=1):
                route_values = _normalize_route_values(raw_route=raw_route, od_pair=od_pair, route_number=route_number)

                if config.route_input_format == RouteInputFormat.NODE_SEQUENCE:
                    route_nodes = route_values
                    _validate_node_sequence_route(route_nodes=route_nodes, od_pair=od_pair, route_number=route_number, config=config)
                    route_link_ids = _node_route_to_link_ids(
                        route_nodes=route_nodes,
                        directed_edge_to_link_id=directed_edge_to_link_id,
                        od_pair=od_pair,
                        route_number=route_number,
                    )
                elif config.route_input_format == RouteInputFormat.LINK_SEQUENCE:
                    route_link_ids = route_values
                    _validate_link_sequence_route(
                        route_link_ids=route_link_ids,
                        link_id_to_position=link_id_to_position,
                        od_pair=od_pair,
                        route_number=route_number,
                    )
                    route_nodes = _link_route_to_node_sequence(
                        route_link_ids=route_link_ids,
                        link_id_to_nodes=link_id_to_nodes,
                        od_pair=od_pair,
                        route_number=route_number,
                    )
                else:
                    raise ValueError(f"Unsupported route input format: {config.route_input_format}")

                duplicate_key = tuple(route_link_ids)
                if duplicate_key in seen_routes_by_od[od_pair] and config.fail_on_duplicate_routes:
                    raise ValueError(
                        f"Duplicated route detected for OD pair {od_pair}, route_number={route_number}. "
                        "Duplicate detection is based on the link-id sequence."
                    )
                seen_routes_by_od[od_pair].add(duplicate_key)

                route_link_positions = tuple(link_id_to_position[int(link_id)] for link_id in route_link_ids)
                free_flow_route_cost = float(link_cost_series.loc[list(route_link_ids)].sum())
                route_index = len(prepared_routes)

                prepared_route = PreparedRoute(
                    route_index=route_index,
                    origin_id=od_pair[0],
                    destination_id=od_pair[1],
                    route_number=route_number,
                    route_nodes=tuple(route_nodes),
                    route_link_ids=tuple(route_link_ids),
                    route_link_positions=route_link_positions,
                    free_flow_route_cost=free_flow_route_cost,
                )
                prepared_routes.append(prepared_route)
                od_to_route_indices[od_pair].append(route_index)

        if not prepared_routes and config.fail_on_empty_route_set:
            raise ValueError("RouteSet contains zero prepared routes.")

        routes_df = _prepared_routes_to_dataframe(prepared_routes)
        route_link_ids = [list(route.route_link_ids) for route in prepared_routes]
        route_link_positions = [list(route.route_link_positions) for route in prepared_routes]

        _validate_route_set_internal_consistency(
            routes_df=routes_df,
            od_to_route_indices=od_to_route_indices,
            route_link_positions=route_link_positions,
            route_link_ids=route_link_ids,
            canonical_link_id_order=canonical_link_id_order,
        )

        return cls(
            routes_df=routes_df,
            od_to_route_indices=od_to_route_indices,
            route_link_positions=route_link_positions,
            route_link_ids=route_link_ids,
            canonical_link_id_order=canonical_link_id_order,
            config=config,
        )

    def validate_positive_od_coverage(self, positive_od_pairs: Iterable[tuple[int, int]]) -> None:
        """Validate that positive-demand OD pairs are covered by this RouteSet.

        This method is intentionally separate from construction because demand
        is not always known when routes are prepared. Assignment testing can call
        it after reconstructing the predicted OD matrix.
        """
        missing_pairs: list[tuple[int, int]] = []
        for raw_od_pair in positive_od_pairs:
            od_pair = _normalize_od_pair(raw_od_pair)
            if od_pair not in self.od_to_route_indices or not self.od_to_route_indices[od_pair]:
                missing_pairs.append(od_pair)

        if missing_pairs and self.config.fail_on_missing_od_routes:
            preview = missing_pairs[:10]
            raise ValueError(
                f"RouteSet is missing routes for {len(missing_pairs)} positive-demand OD pairs. "
                f"First missing pairs: {preview}"
            )

    def require_same_link_order(self, links_df: pd.DataFrame) -> None:
        """Ensure that a links_df still matches this RouteSet's link order.

        route_link_positions are only valid if the current links_df row order is
        exactly the same as the order used during RouteSet construction.
        """
        if self.config.link_id_col not in links_df.columns:
            raise ValueError(
                f"links_df is missing configured link ID column '{self.config.link_id_col}'."
            )

        current_order = tuple(int(value) for value in links_df[self.config.link_id_col].tolist())
        if current_order != self.canonical_link_id_order:
            raise ValueError(
                "Current links_df order does not match RouteSet.canonical_link_id_order. "
                "Rebuild the RouteSet or reorder links_df before assignment."
            )

    def get_route_indices_for_od(self, origin_id: int, destination_id: int) -> list[int]:
        """Return prepared route indices for one OD pair.

        The returned list is a copy to prevent callers from mutating internal
        RouteSet state.
        """
        od_pair = (int(origin_id), int(destination_id))
        return list(self.od_to_route_indices.get(od_pair, []))

    @property
    def number_of_routes(self) -> int:
        """Return the number of prepared routes."""
        return len(self.route_link_positions)

    @property
    def number_of_links(self) -> int:
        """Return the number of links in the canonical link order."""
        return len(self.canonical_link_id_order)


# ---------------------------------------------------------------------------
# Validation and construction helpers
# ---------------------------------------------------------------------------


def _validate_links_df_for_route_set(*, links_df: pd.DataFrame, config: RouteSetBuildConfig) -> None:
    """Validate link-table columns and structural assumptions for RouteSet."""
    if not isinstance(links_df, pd.DataFrame):
        raise TypeError("links_df must be a pandas DataFrame.")
    if links_df.empty:
        raise ValueError("links_df cannot be empty.")

    required_columns = {
        config.link_id_col,
        config.init_node_col,
        config.term_node_col,
        config.route_cost_col,
    }
    missing_columns = sorted(required_columns.difference(links_df.columns))
    if missing_columns:
        raise ValueError(f"links_df is missing required columns for RouteSet: {missing_columns}")

    if links_df[config.link_id_col].isna().any():
        raise ValueError(f"links_df column '{config.link_id_col}' contains missing values.")
    if links_df[config.init_node_col].isna().any() or links_df[config.term_node_col].isna().any():
        raise ValueError(
            f"links_df columns '{config.init_node_col}' and '{config.term_node_col}' cannot contain missing values."
        )
    if links_df[config.route_cost_col].isna().any():
        raise ValueError(f"links_df column '{config.route_cost_col}' contains missing route-cost values.")

    route_cost_values = links_df[config.route_cost_col].to_numpy(dtype=float)
    if not np.all(np.isfinite(route_cost_values)):
        raise ValueError(f"links_df column '{config.route_cost_col}' must contain finite numeric values.")
    if np.any(route_cost_values < 0.0):
        raise ValueError(f"links_df column '{config.route_cost_col}' cannot contain negative values.")

    if config.require_unique_link_ids and links_df[config.link_id_col].duplicated().any():
        duplicated = links_df.loc[links_df[config.link_id_col].duplicated(), config.link_id_col].head(10).tolist()
        raise ValueError(f"links_df contains duplicated link IDs. First duplicates: {duplicated}")

    if config.require_unique_directed_edges:
        edge_columns = [config.init_node_col, config.term_node_col]
        duplicated_edges_mask = links_df.duplicated(subset=edge_columns, keep=False)
        if duplicated_edges_mask.any():
            duplicated_edges = links_df.loc[duplicated_edges_mask, edge_columns].head(10).to_dict("records")
            raise ValueError(
                "links_df contains duplicated directed edges, so node-route conversion is ambiguous. "
                f"First duplicated edges: {duplicated_edges}"
            )


def _validate_routes_container(
    *,
    routes_by_od: Mapping[tuple[int, int], Sequence[Sequence[int]]],
    config: RouteSetBuildConfig,
) -> None:
    """Validate the top-level routes_by_od object before processing routes."""
    if not isinstance(routes_by_od, Mapping):
        raise TypeError("routes_by_od must be a mapping from OD pairs to route sequences.")
    if len(routes_by_od) == 0 and config.fail_on_empty_route_set:
        raise ValueError("routes_by_od is empty.")


def _build_link_id_to_position(*, links_df: pd.DataFrame, config: RouteSetBuildConfig) -> dict[int, int]:
    """Map each stable link ID to its zero-based row position in links_df."""
    return {
        int(link_id): int(position)
        for position, link_id in enumerate(links_df[config.link_id_col].tolist())
    }


def _build_directed_edge_to_link_id(*, links_df: pd.DataFrame, config: RouteSetBuildConfig) -> dict[tuple[int, int], int]:
    """Map each directed edge (init_node, term_node) to its stable link ID."""
    edge_to_link_id: dict[tuple[int, int], int] = {}
    for _, row in links_df.iterrows():
        edge_key = (int(row[config.init_node_col]), int(row[config.term_node_col]))
        edge_to_link_id[edge_key] = int(row[config.link_id_col])
    return edge_to_link_id


def _build_link_id_to_nodes(*, links_df: pd.DataFrame, config: RouteSetBuildConfig) -> dict[int, tuple[int, int]]:
    """Map each stable link ID to its directed node pair."""
    link_id_to_nodes: dict[int, tuple[int, int]] = {}
    for _, row in links_df.iterrows():
        link_id_to_nodes[int(row[config.link_id_col])] = (
            int(row[config.init_node_col]),
            int(row[config.term_node_col]),
        )
    return link_id_to_nodes


def _build_link_cost_series(*, links_df: pd.DataFrame, config: RouteSetBuildConfig) -> pd.Series:
    """Build a link-id-indexed static cost series used for route preparation."""
    link_cost_frame = links_df[[config.link_id_col, config.route_cost_col]].copy()
    link_cost_frame[config.link_id_col] = link_cost_frame[config.link_id_col].astype(int)
    link_cost_frame[config.route_cost_col] = link_cost_frame[config.route_cost_col].astype(float)
    return link_cost_frame.set_index(config.link_id_col)[config.route_cost_col]


def _normalize_od_pair(raw_od_pair: Any) -> tuple[int, int]:
    """Convert an OD key into a strict pair of integer IDs."""
    if not isinstance(raw_od_pair, Sequence) or isinstance(raw_od_pair, (str, bytes)):
        raise TypeError(f"OD pair must be a two-item sequence, got {raw_od_pair!r}.")
    if len(raw_od_pair) != 2:
        raise ValueError(f"OD pair must contain exactly two values, got {raw_od_pair!r}.")
    return int(raw_od_pair[0]), int(raw_od_pair[1])


def _normalize_route_values(raw_route: Any, od_pair: tuple[int, int], route_number: int) -> tuple[int, ...]:
    """Convert a raw route sequence into a strict tuple of integer values."""
    if not isinstance(raw_route, Sequence) or isinstance(raw_route, (str, bytes)):
        raise TypeError(
            f"Route {route_number} for OD pair {od_pair} must be a sequence of integers, got {raw_route!r}."
        )
    if len(raw_route) == 0:
        raise ValueError(f"Route {route_number} for OD pair {od_pair} cannot be empty.")
    return tuple(int(value) for value in raw_route)


def _validate_node_sequence_route(
    *,
    route_nodes: tuple[int, ...],
    od_pair: tuple[int, int],
    route_number: int,
    config: RouteSetBuildConfig,
) -> None:
    """Validate a route represented as a node sequence."""
    if len(route_nodes) < 2:
        raise ValueError(f"Node route {route_number} for OD pair {od_pair} must contain at least two nodes.")
    if config.require_simple_node_routes and len(set(route_nodes)) != len(route_nodes):
        raise ValueError(
            f"Node route {route_number} for OD pair {od_pair} repeats nodes. "
            "Set require_simple_node_routes=False only if cyclic routes are intentionally allowed."
        )


def _validate_link_sequence_route(
    *,
    route_link_ids: tuple[int, ...],
    link_id_to_position: Mapping[int, int],
    od_pair: tuple[int, int],
    route_number: int,
) -> None:
    """Validate a route represented as a link-id sequence."""
    if len(route_link_ids) < 1:
        raise ValueError(f"Link route {route_number} for OD pair {od_pair} must contain at least one link.")
    missing_link_ids = [int(link_id) for link_id in route_link_ids if int(link_id) not in link_id_to_position]
    if missing_link_ids:
        raise ValueError(
            f"Link route {route_number} for OD pair {od_pair} contains link IDs not present in links_df. "
            f"Missing link IDs: {missing_link_ids[:10]}"
        )


def _node_route_to_link_ids(
    *,
    route_nodes: tuple[int, ...],
    directed_edge_to_link_id: Mapping[tuple[int, int], int],
    od_pair: tuple[int, int],
    route_number: int,
) -> tuple[int, ...]:
    """Convert a node-sequence route into a link-id sequence."""
    route_link_ids: list[int] = []
    for init_node, term_node in zip(route_nodes[:-1], route_nodes[1:]):
        directed_edge = (int(init_node), int(term_node))
        if directed_edge not in directed_edge_to_link_id:
            raise ValueError(
                f"Node route {route_number} for OD pair {od_pair} contains directed edge {directed_edge}, "
                "but that edge does not exist in links_df."
            )
        route_link_ids.append(int(directed_edge_to_link_id[directed_edge]))
    return tuple(route_link_ids)


def _link_route_to_node_sequence(
    *,
    route_link_ids: tuple[int, ...],
    link_id_to_nodes: Mapping[int, tuple[int, int]],
    od_pair: tuple[int, int],
    route_number: int,
) -> tuple[int, ...]:
    """Reconstruct and validate a node sequence from a link-id route."""
    nodes: list[int] = []
    previous_term_node: int | None = None

    for local_position, link_id in enumerate(route_link_ids):
        init_node, term_node = link_id_to_nodes[int(link_id)]
        if local_position == 0:
            nodes.append(int(init_node))
        elif previous_term_node != init_node:
            raise ValueError(
                f"Link route {route_number} for OD pair {od_pair} is not topologically continuous. "
                f"Previous term node was {previous_term_node}, but next init node is {init_node}."
            )
        nodes.append(int(term_node))
        previous_term_node = int(term_node)

    return tuple(nodes)


def _prepared_routes_to_dataframe(prepared_routes: Sequence[PreparedRoute]) -> pd.DataFrame:
    """Convert prepared route records to a DataFrame used by diagnostics and solvers."""
    rows = [
        {
            "route_index": route.route_index,
            "origin_id": route.origin_id,
            "destination_id": route.destination_id,
            "route_number": route.route_number,
            "route_nodes": list(route.route_nodes),
            "route_link_ids": list(route.route_link_ids),
            "route_link_positions": list(route.route_link_positions),
            "free_flow_route_cost": route.free_flow_route_cost,
        }
        for route in prepared_routes
    ]
    return pd.DataFrame(rows)


def _validate_route_set_internal_consistency(
    *,
    routes_df: pd.DataFrame,
    od_to_route_indices: Mapping[tuple[int, int], list[int]],
    route_link_positions: Sequence[Sequence[int]],
    route_link_ids: Sequence[Sequence[int]],
    canonical_link_id_order: Sequence[int],
) -> None:
    """Validate consistency between RouteSet internal representations."""
    number_of_routes = len(routes_df)
    if len(route_link_positions) != number_of_routes:
        raise ValueError("route_link_positions length does not match routes_df length.")
    if len(route_link_ids) != number_of_routes:
        raise ValueError("route_link_ids length does not match routes_df length.")

    valid_indices = set(range(number_of_routes))
    referenced_indices: set[int] = set()
    for od_pair, indices in od_to_route_indices.items():
        for route_index in indices:
            if route_index not in valid_indices:
                raise ValueError(f"OD pair {od_pair} references invalid route_index={route_index}.")
            referenced_indices.add(route_index)

    if referenced_indices != valid_indices:
        missing = sorted(valid_indices.difference(referenced_indices))[:10]
        raise ValueError(f"Some routes are not referenced by od_to_route_indices. First missing: {missing}")

    number_of_links = len(canonical_link_id_order)
    for route_index, positions in enumerate(route_link_positions):
        if not positions:
            raise ValueError(f"Prepared route_index={route_index} has no link positions.")
        invalid_positions = [int(position) for position in positions if int(position) < 0 or int(position) >= number_of_links]
        if invalid_positions:
            raise ValueError(
                f"Prepared route_index={route_index} contains invalid link positions: {invalid_positions[:10]}"
            )
