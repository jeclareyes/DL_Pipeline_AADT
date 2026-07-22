"""Core contracts for full route-based traffic assignment.

This module defines the shared abstractions used by route-based traffic
assignment behavior models and route-based numerical solvers.

The central architectural rule is:

    route flows are the primary state;
    link flows are always derived from route flows through RouteSet.

Therefore, behavior models and solvers must never treat link flows as the
canonical assignment state. Link flows are accepted only when they are checked
against the current route-flow vector and the prepared RouteSet.

Design principles
-----------------
1. No silent fallbacks or backward-compatibility aliases.
2. No optional route-flow or route-cost fields in final results.
3. Solvers update route-flow vectors, not link-flow vectors.
4. Behavior models own route-choice logic; solvers own numerical updates.
5. RouteSet owns route preparation and link-position alignment.
6. Public contracts validate shape, finiteness, positivity/non-negativity,
   OD-demand conservation, link-flow aggregation, and route-cost consistency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .route_set import RouteSet


# =============================================================================
# Solver-step contracts
# =============================================================================


@dataclass(frozen=True)
class RouteBasedSolverStepRequest:
    """Input contract for one route-based solver update.

    A behavior model builds this object after computing the current assignment
    state and an auxiliary route-flow solution. The numerical solver receives a
    complete route-space state and returns the next route-flow vector.

    Attributes
    ----------
    iteration:
        One-based iteration counter. Iteration zero is rejected because rules
        such as MSA 1/n are defined from iteration 1 onward.
    current_route_flows:
        Current feasible route-flow vector in RouteSet route order.
    auxiliary_route_flows:
        Feasible auxiliary route-flow vector produced by the behavior model.
        For deterministic UE this is usually shortest-route loading. For SUE it
        is usually Logit route loading.
    current_link_flows:
        Link-flow vector derived from current_route_flows. Its order must match
        the RouteSet canonical link order and the link_table row order.
    auxiliary_link_flows:
        Link-flow vector derived from auxiliary_route_flows.
    current_link_costs:
        Link costs evaluated at current_link_flows using the configured VDF.
    current_route_costs:
        Route costs obtained by summing current_link_costs along each route.
    link_table:
        Per-solve link table. It must preserve the same row order used to build
        the RouteSet. Derived columns, such as scaled assignment capacity, may
        be added before this request is built.
    od_demands:
        Positive OD demands represented in route space. Keys are
        (origin_id, destination_id), values are strictly positive demands.
    route_set:
        Prepared canonical route representation aligned with link_table.
    metadata:
        Diagnostics or contextual information. Required numerical configuration
        should not be hidden here; it belongs in explicit config objects.
    """

    iteration: int
    current_route_flows: np.ndarray
    auxiliary_route_flows: np.ndarray
    current_link_flows: np.ndarray
    auxiliary_link_flows: np.ndarray
    current_link_costs: np.ndarray
    current_route_costs: np.ndarray
    link_table: pd.DataFrame
    od_demands: Mapping[tuple[int, int], float]
    route_set: RouteSet
    metadata: Mapping[str, Any]

    def validate(self) -> None:
        """Validate a complete route-based solver request."""
        if not isinstance(self.iteration, int) or self.iteration < 1:
            raise ValueError(f"iteration must be an integer >= 1. Received {self.iteration!r}.")
        if not isinstance(self.route_set, RouteSet):
            raise TypeError("route_set must be a RouteSet instance.")
        if not isinstance(self.link_table, pd.DataFrame):
            raise TypeError("link_table must be a pandas DataFrame.")
        if self.link_table.empty:
            raise ValueError("link_table cannot be empty.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        self.route_set.require_same_link_order(self.link_table)

        number_of_routes = self.route_set.number_of_routes
        number_of_links = self.route_set.number_of_links
        if number_of_routes <= 0:
            raise ValueError("route_set must contain at least one route.")
        if number_of_links <= 0:
            raise ValueError("route_set must contain at least one link.")
        if len(self.link_table) != number_of_links:
            raise ValueError(
                "link_table row count must match route_set.number_of_links. "
                f"Received len(link_table)={len(self.link_table)}, number_of_links={number_of_links}."
            )

        validate_vector("current_route_flows", self.current_route_flows, number_of_routes)
        validate_vector("auxiliary_route_flows", self.auxiliary_route_flows, number_of_routes)
        validate_vector("current_link_flows", self.current_link_flows, number_of_links)
        validate_vector("auxiliary_link_flows", self.auxiliary_link_flows, number_of_links)
        validate_vector("current_link_costs", self.current_link_costs, number_of_links)
        validate_vector("current_route_costs", self.current_route_costs, number_of_routes)

        validate_non_negative_vector("current_route_flows", self.current_route_flows)
        validate_non_negative_vector("auxiliary_route_flows", self.auxiliary_route_flows)
        validate_non_negative_vector("current_link_flows", self.current_link_flows)
        validate_non_negative_vector("auxiliary_link_flows", self.auxiliary_link_flows)
        validate_strictly_positive_vector("current_link_costs", self.current_link_costs)
        validate_strictly_positive_vector("current_route_costs", self.current_route_costs)

        validate_od_demands(self.od_demands)
        validate_route_flow_demand_conservation(
            route_flows=self.current_route_flows,
            od_demands=self.od_demands,
            route_set=self.route_set,
            vector_name="current_route_flows",
        )
        validate_route_flow_demand_conservation(
            route_flows=self.auxiliary_route_flows,
            od_demands=self.od_demands,
            route_set=self.route_set,
            vector_name="auxiliary_route_flows",
        )

        expected_current_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=self.current_route_flows,
            route_set=self.route_set,
        )
        expected_auxiliary_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=self.auxiliary_route_flows,
            route_set=self.route_set,
        )
        assert_vectors_close(
            name="current_link_flows",
            actual=self.current_link_flows,
            expected=expected_current_link_flows,
            message="current_link_flows is not consistent with current_route_flows and RouteSet.",
        )
        assert_vectors_close(
            name="auxiliary_link_flows",
            actual=self.auxiliary_link_flows,
            expected=expected_auxiliary_link_flows,
            message="auxiliary_link_flows is not consistent with auxiliary_route_flows and RouteSet.",
        )

        expected_current_route_costs = compute_route_costs_from_link_costs(
            link_costs=self.current_link_costs,
            route_set=self.route_set,
        )
        assert_vectors_close(
            name="current_route_costs",
            actual=self.current_route_costs,
            expected=expected_current_route_costs,
            message="current_route_costs is not consistent with current_link_costs and RouteSet.",
        )


@dataclass(frozen=True)
class RouteBasedSolverStepResult:
    """Output contract for one route-based solver update.

    A solver must return only the next route-flow vector plus metadata. Scalar
    quantities such as step size are diagnostics and therefore must live in
    metadata, not as public fields in this contract.
    """

    new_route_flows: np.ndarray
    metadata: Mapping[str, Any]

    def validate(self, route_set: RouteSet, od_demands: Mapping[tuple[int, int], float]) -> None:
        """Validate solver output before a behavior model accepts it."""
        if not isinstance(route_set, RouteSet):
            raise TypeError("route_set must be a RouteSet instance.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        validate_vector("new_route_flows", self.new_route_flows, route_set.number_of_routes)
        validate_non_negative_vector("new_route_flows", self.new_route_flows)
        validate_od_demands(od_demands)
        validate_route_flow_demand_conservation(
            route_flows=self.new_route_flows,
            od_demands=od_demands,
            route_set=route_set,
            vector_name="new_route_flows",
        )


@dataclass(frozen=True)
class AssignmentResult:
    """Standard output contract for full route-based assignment.

    None is intentionally not allowed for any final vector. A behavior model
    that cannot produce route flows, link flows, link costs, and route costs is
    not compatible with the full route-based architecture.

    Attributes
    ----------
    final_link_flows:
        Final link-flow vector in the canonical links_df row order.
    final_route_flows:
        Final route-flow vector in RouteSet route order.
    final_link_costs:
        Final link-cost vector in the canonical links_df row order.
    final_route_costs:
        Final route-cost vector in RouteSet route order.
    metadata:
        Diagnostics, convergence history, skipped OD pairs, and any
        behavior-model-specific reports.
    """

    final_link_flows: np.ndarray
    final_route_flows: np.ndarray
    final_link_costs: np.ndarray
    final_route_costs: np.ndarray
    metadata: Mapping[str, Any]

    def validate(self, route_set: RouteSet, od_demands: Mapping[tuple[int, int], float]) -> None:
        """Validate final assignment outputs against RouteSet and OD demands."""
        if not isinstance(route_set, RouteSet):
            raise TypeError("route_set must be a RouteSet instance.")
        if not isinstance(self.metadata, Mapping):
            raise TypeError("metadata must be a mapping.")

        validate_od_demands(od_demands)

        validate_vector("final_route_flows", self.final_route_flows, route_set.number_of_routes)
        validate_vector("final_link_flows", self.final_link_flows, route_set.number_of_links)
        validate_vector("final_link_costs", self.final_link_costs, route_set.number_of_links)
        validate_vector("final_route_costs", self.final_route_costs, route_set.number_of_routes)

        validate_non_negative_vector("final_route_flows", self.final_route_flows)
        validate_non_negative_vector("final_link_flows", self.final_link_flows)
        validate_strictly_positive_vector("final_link_costs", self.final_link_costs)
        validate_strictly_positive_vector("final_route_costs", self.final_route_costs)

        validate_route_flow_demand_conservation(
            route_flows=self.final_route_flows,
            od_demands=od_demands,
            route_set=route_set,
            vector_name="final_route_flows",
        )

        expected_final_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=self.final_route_flows,
            route_set=route_set,
        )
        expected_final_route_costs = compute_route_costs_from_link_costs(
            link_costs=self.final_link_costs,
            route_set=route_set,
        )
        assert_vectors_close(
            name="final_link_flows",
            actual=self.final_link_flows,
            expected=expected_final_link_flows,
            message="final_link_flows is not consistent with final_route_flows and RouteSet.",
        )
        assert_vectors_close(
            name="final_route_costs",
            actual=self.final_route_costs,
            expected=expected_final_route_costs,
            message="final_route_costs is not consistent with final_link_costs and RouteSet.",
        )


# =============================================================================
# Abstract base classes
# =============================================================================


class BaseRouteBasedSolver(ABC):
    """Abstract base class for route-based numerical solvers.

    This abstraction represents how route flows are numerically updated. It does
    not represent traveler behavior. Traveler behavior belongs to assignment
    models such as deterministic UE or SUE.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the canonical solver name used in configs and reports."""
        raise NotImplementedError

    @abstractmethod
    def compute_route_flow_update(
        self,
        request: RouteBasedSolverStepRequest,
    ) -> RouteBasedSolverStepResult:
        """Return a new feasible route-flow vector for one iteration."""
        raise NotImplementedError


class BaseRouteBasedAssignmentModel(ABC):
    """Abstract base class for full route-based behavior models.

    This abstraction represents how travelers are assumed to choose or update
    routes. The numerical route-flow update is injected separately through a
    BaseRouteBasedSolver instance.
    """

    REQUIRED_TOPOLOGY_COLUMNS: tuple[str, str, str] = ("link_id", "init_node", "term_node")

    def __init__(
        self,
        links_df: pd.DataFrame,
        route_set: RouteSet,
        solver: BaseRouteBasedSolver,
        zone_id_to_idx: Mapping[int, int],
    ) -> None:
        """Initialize a behavior model with explicit dependencies.

        Parameters
        ----------
        links_df:
            Canonical directed link table. Final link-flow vectors follow this
            exact row order.
        route_set:
            Prepared RouteSet aligned with links_df row order.
        solver:
            Explicit route-based numerical solver. No fallback solver is
            allowed.
        zone_id_to_idx:
            Explicit mapping from zone ID to OD-matrix index.
        """
        self._validate_constructor_inputs(
            links_df=links_df,
            route_set=route_set,
            solver=solver,
            zone_id_to_idx=zone_id_to_idx,
        )
        route_set.require_same_link_order(links_df)

        self.links_df = links_df.copy(deep=True)
        self.route_set = route_set
        self.solver = solver
        self.zone_id_to_idx = {int(zone_id): int(idx) for zone_id, idx in zone_id_to_idx.items()}
        self.idx_to_zone_id = {int(idx): int(zone_id) for zone_id, idx in self.zone_id_to_idx.items()}

    @property
    @abstractmethod
    def behavior_model_name(self) -> str:
        """Return the canonical behavior-model name used in configs/reports."""
        raise NotImplementedError

    @abstractmethod
    def solve(self, od_matrix: np.ndarray, config: Any) -> AssignmentResult:
        """Run route-based assignment with an explicit runtime config."""
        raise NotImplementedError

    @classmethod
    def _validate_constructor_inputs(
        cls,
        links_df: pd.DataFrame,
        route_set: RouteSet,
        solver: BaseRouteBasedSolver,
        zone_id_to_idx: Mapping[int, int],
    ) -> None:
        """Validate dependencies common to all route-based behavior models."""
        validate_links_df(links_df=links_df, required_topology_columns=cls.REQUIRED_TOPOLOGY_COLUMNS)

        if not isinstance(route_set, RouteSet):
            raise TypeError("route_set must be a RouteSet instance.")
        if route_set.number_of_routes <= 0:
            raise ValueError("route_set must contain at least one route.")
        if route_set.number_of_links != len(links_df):
            raise ValueError(
                "route_set.number_of_links must match links_df row count. "
                f"Received route_set.number_of_links={route_set.number_of_links}, len(links_df)={len(links_df)}."
            )
        route_set.require_same_link_order(links_df)

        if not isinstance(solver, BaseRouteBasedSolver):
            raise TypeError("solver must be an instance of BaseRouteBasedSolver. No implicit solver is allowed.")
        if not isinstance(solver.name, str) or not solver.name.strip():
            raise ValueError("solver.name must be a non-empty string.")

        validate_zone_mapping(zone_id_to_idx=zone_id_to_idx)

    def validate_assignment_result(
        self,
        result: AssignmentResult,
        od_demands: Mapping[tuple[int, int], float],
    ) -> None:
        """Validate a concrete model result before returning it."""
        if not isinstance(result, AssignmentResult):
            raise TypeError("result must be an AssignmentResult instance.")
        result.validate(route_set=self.route_set, od_demands=od_demands)


# =============================================================================
# Shared validation helpers
# =============================================================================


def validate_links_df(links_df: pd.DataFrame, required_topology_columns: tuple[str, str, str]) -> None:
    """Validate a canonical directed link table.

    The topology columns are intentionally checked here because every assignment
    model depends on stable link IDs and unambiguous directed edges.
    """
    if not isinstance(links_df, pd.DataFrame):
        raise TypeError("links_df must be a pandas DataFrame.")
    if links_df.empty:
        raise ValueError("links_df cannot be empty.")
    if not isinstance(required_topology_columns, tuple) or len(required_topology_columns) != 3:
        raise TypeError("required_topology_columns must be a three-item tuple.")

    missing_columns = [column for column in required_topology_columns if column not in links_df.columns]
    if missing_columns:
        raise ValueError(f"links_df is missing required topology columns: {missing_columns}.")

    topology_frame = links_df[list(required_topology_columns)]
    null_mask = topology_frame.isna().any(axis=1)
    if null_mask.any():
        bad_rows = topology_frame.index[null_mask].tolist()[:20]
        raise ValueError(f"Topology columns contain null values at rows: {bad_rows}.")

    link_id_col, init_node_col, term_node_col = required_topology_columns
    link_ids = links_df[link_id_col].astype(int)
    duplicated_link_ids = link_ids.duplicated(keep=False)
    if duplicated_link_ids.any():
        duplicated_values = sorted(link_ids.loc[duplicated_link_ids].unique().tolist())
        raise ValueError(f"links_df contains duplicated link IDs: {duplicated_values[:20]}.")

    directed_edges = links_df[[init_node_col, term_node_col]].astype(int)
    duplicated_edges = directed_edges.duplicated(keep=False)
    if duplicated_edges.any():
        edge_preview = directed_edges.loc[duplicated_edges].drop_duplicates().values.tolist()[:20]
        raise ValueError(f"links_df contains duplicated directed edges: {edge_preview}.")


def validate_zone_mapping(zone_id_to_idx: Mapping[int, int]) -> None:
    """Validate an explicit zone-ID to OD-matrix-index mapping."""
    if not isinstance(zone_id_to_idx, Mapping):
        raise TypeError("zone_id_to_idx must be a mapping.")
    if len(zone_id_to_idx) == 0:
        raise ValueError("zone_id_to_idx cannot be empty.")

    zone_ids = [int(zone_id) for zone_id in zone_id_to_idx.keys()]
    matrix_indices = [int(idx) for idx in zone_id_to_idx.values()]

    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError("zone_id_to_idx contains duplicated zone IDs after integer conversion.")
    if len(matrix_indices) != len(set(matrix_indices)):
        raise ValueError("zone_id_to_idx contains duplicated matrix indices.")
    if min(matrix_indices) < 0:
        raise ValueError("zone_id_to_idx matrix indices must be non-negative.")

    sorted_indices = sorted(matrix_indices)
    expected_indices = list(range(len(sorted_indices)))
    if sorted_indices != expected_indices:
        raise ValueError(
            "zone_id_to_idx matrix indices must be contiguous and zero-based. "
            f"Received first indices={sorted_indices[:20]}, expected first indices={expected_indices[:20]}."
        )


def validate_vector(name: str, value: np.ndarray, expected_length: int) -> None:
    """Validate a finite one-dimensional numpy vector with known length."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string.")
    if not isinstance(expected_length, int) or expected_length <= 0:
        raise ValueError(f"expected_length must be an integer > 0. Received {expected_length!r}.")
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, not {type(value).__name__}.")
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional. Received shape={value.shape}.")
    if len(value) != expected_length:
        raise ValueError(f"{name} length must be {expected_length}. Received {len(value)}.")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} contains NaN or infinite values.")


def validate_non_negative_vector(name: str, value: np.ndarray, tolerance: float = 1e-10) -> None:
    """Validate that a numeric vector has no materially negative entries.

    The small tolerance exists only for floating-point noise. Values below
    -tolerance are treated as real modeling errors.
    """
    validate_tolerance(name="tolerance", value=tolerance, allow_zero=True)
    if np.any(value < -float(tolerance)):
        min_value = float(np.min(value))
        raise ValueError(f"{name} contains negative values. Minimum={min_value}.")


def validate_strictly_positive_vector(name: str, value: np.ndarray) -> None:
    """Validate that a numeric vector is strictly positive."""
    if np.any(value <= 0.0):
        min_value = float(np.min(value))
        raise ValueError(f"{name} must be strictly positive. Minimum={min_value}.")


def validate_od_demands(od_demands: Mapping[tuple[int, int], float]) -> None:
    """Validate positive OD demands keyed by integer OD pairs."""
    if not isinstance(od_demands, Mapping):
        raise TypeError("od_demands must be a mapping keyed by (origin_id, destination_id).")

    for raw_od_pair, raw_demand in od_demands.items():
        if not isinstance(raw_od_pair, tuple) or len(raw_od_pair) != 2:
            raise TypeError(f"OD key must be a two-item tuple. Received {raw_od_pair!r}.")
        origin_id = int(raw_od_pair[0])
        destination_id = int(raw_od_pair[1])
        demand = float(raw_demand)
        if not np.isfinite(demand):
            raise ValueError(f"Demand for OD {(origin_id, destination_id)} is not finite: {raw_demand!r}.")
        if demand <= 0.0:
            raise ValueError(f"Demand for OD {(origin_id, destination_id)} must be positive. Received {demand}.")


def validate_route_flow_demand_conservation(
    route_flows: np.ndarray,
    od_demands: Mapping[tuple[int, int], float],
    route_set: RouteSet,
    vector_name: str,
    *,
    rtol: float = 1e-8,
    atol: float = 1e-6,
) -> None:
    """Validate that route flows conserve demand for every represented OD."""
    if not isinstance(route_set, RouteSet):
        raise TypeError("route_set must be a RouteSet instance.")
    validate_vector(vector_name, route_flows, route_set.number_of_routes)
    validate_non_negative_vector(vector_name, route_flows)
    validate_od_demands(od_demands)

    for od_pair, demand in od_demands.items():
        route_indices = route_set.get_route_indices_for_od(int(od_pair[0]), int(od_pair[1]))
        if not route_indices:
            raise ValueError(f"RouteSet has no routes for positive-demand OD pair {od_pair}.")

        assigned_demand = float(route_flows[route_indices].sum())
        if not np.isclose(assigned_demand, float(demand), rtol=rtol, atol=atol):
            raise ValueError(
                f"{vector_name} violates OD demand conservation for {od_pair}. "
                f"Assigned={assigned_demand}, demand={float(demand)}."
            )


def validate_tolerance(name: str, value: float, *, allow_zero: bool) -> None:
    """Validate tolerance-like numerical parameters."""
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite. Received {value!r}.")
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} must be non-negative. Received {value!r}.")
    elif value <= 0.0:
        raise ValueError(f"{name} must be strictly positive. Received {value!r}.")


def assert_vectors_close(
    *,
    name: str,
    actual: np.ndarray,
    expected: np.ndarray,
    message: str,
    rtol: float = 1e-8,
    atol: float = 1e-8,
) -> None:
    """Raise a detailed error when two validated vectors are not close."""
    validate_vector(f"{name}.actual", actual, len(actual))
    validate_vector(f"{name}.expected", expected, len(expected))
    if actual.shape != expected.shape:
        raise ValueError(f"{name} shape mismatch. actual={actual.shape}, expected={expected.shape}.")
    if not np.allclose(actual, expected, rtol=rtol, atol=atol):
        absolute_difference = np.abs(actual - expected)
        max_position = int(np.argmax(absolute_difference))
        raise ValueError(
            f"{message} "
            f"max_abs_difference={float(absolute_difference[max_position])}, "
            f"position={max_position}, actual={float(actual[max_position])}, "
            f"expected={float(expected[max_position])}."
        )


# =============================================================================
# Route-link transformation helpers
# =============================================================================


def aggregate_route_flows_to_link_flows(route_flows: np.ndarray, route_set: RouteSet) -> np.ndarray:
    """Aggregate route flows into link flows using RouteSet link positions."""
    if not isinstance(route_set, RouteSet):
        raise TypeError("route_set must be a RouteSet instance.")
    validate_vector("route_flows", route_flows, route_set.number_of_routes)
    validate_non_negative_vector("route_flows", route_flows)

    link_flows = np.zeros(route_set.number_of_links, dtype=float)
    for route_index, link_positions in enumerate(route_set.route_link_positions):
        if len(link_positions) == 0:
            raise ValueError(f"Route index {route_index} has no link positions.")
        link_flows[np.asarray(link_positions, dtype=int)] += float(route_flows[route_index])

    validate_vector("link_flows", link_flows, route_set.number_of_links)
    validate_non_negative_vector("link_flows", link_flows)
    return link_flows


def compute_route_costs_from_link_costs(link_costs: np.ndarray, route_set: RouteSet) -> np.ndarray:
    """Compute route costs by summing link costs along each prepared route."""
    if not isinstance(route_set, RouteSet):
        raise TypeError("route_set must be a RouteSet instance.")
    validate_vector("link_costs", link_costs, route_set.number_of_links)
    validate_strictly_positive_vector("link_costs", link_costs)

    route_costs = np.empty(route_set.number_of_routes, dtype=float)
    for route_index, link_positions in enumerate(route_set.route_link_positions):
        if len(link_positions) == 0:
            raise ValueError(f"Route index {route_index} has no link positions.")
        route_costs[route_index] = float(link_costs[np.asarray(link_positions, dtype=int)].sum())

    validate_vector("route_costs", route_costs, route_set.number_of_routes)
    validate_strictly_positive_vector("route_costs", route_costs)
    return route_costs


def print_assignment_progress(
    model_name: str,
    solver_name: str,
    iteration: int,
    max_iterations: int,
    error_info: str,
    bar_length: int = 20,
    is_finished: bool = False,
) -> None:
    """Print a dynamic console progress bar with convergence info."""
    import sys
    if is_finished:
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    percent = float(iteration) / float(max_iterations) if max_iterations > 0 else 0.0
    filled_length = int(round(bar_length * percent))
    bar = "█" * filled_length + "░" * (bar_length - filled_length)
    
    line = f"\r[{model_name}] ({solver_name}) |{bar}| {percent*100:.1f}% ({iteration}/{max_iterations}) | {error_info}"
    sys.stdout.write(line + "          ")
    sys.stdout.flush()


__all__ = [
    "AssignmentResult",
    "BaseRouteBasedAssignmentModel",
    "BaseRouteBasedSolver",
    "RouteBasedSolverStepRequest",
    "RouteBasedSolverStepResult",
    "aggregate_route_flows_to_link_flows",
    "assert_vectors_close",
    "compute_route_costs_from_link_costs",
    "print_assignment_progress",
    "validate_links_df",
    "validate_non_negative_vector",
    "validate_od_demands",
    "validate_route_flow_demand_conservation",
    "validate_strictly_positive_vector",
    "validate_tolerance",
    "validate_vector",
    "validate_zone_mapping",
]