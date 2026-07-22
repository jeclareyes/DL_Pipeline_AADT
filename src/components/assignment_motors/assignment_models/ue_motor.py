"""Route-based deterministic User Equilibrium behavior model.

This module implements deterministic User Equilibrium (UE) over a prepared
RouteSet. The model is full route-based: the primary state is the route-flow
vector, and link flows are always derived from route flows through RouteSet.

Conceptual separation
---------------------
- UE is a behavioral assignment model.
- MSA, Frank-Wolfe, and Gradient Projection are numerical route-flow solvers.
- RouteSet owns route preparation and canonical route/link alignment.
- BPR costs are currently evaluated here through an explicit config; this can
  later be moved into a dedicated LinkCostFunction/VDF object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..assignment_configs import (
    RouteBasedUEConvergenceConfig,
    RouteBasedUEInitializationConfig,
    RouteBasedUEPolicyConfig,
)
from src.components.vdf.config import VDFConfig
from ..base import (
    AssignmentResult,
    BaseRouteBasedAssignmentModel,
    RouteBasedSolverStepRequest,
    aggregate_route_flows_to_link_flows,
    compute_route_costs_from_link_costs,
    print_assignment_progress,
    validate_non_negative_vector,
    validate_vector,
)


@dataclass(frozen=True)
class RouteBasedUserEquilibriumRuntimeConfig:
    """Runtime configuration required by route-based deterministic UE.

    This object is intentionally separate from raw YAML. The composition layer
    builds it from the validated assignment configuration and injects it into
    ``RouteBasedUserEquilibriumModel.solve``.

    Attributes
    ----------
    max_iterations:
        Maximum number of UE iterations.
    capacity_scaling_factor:
        Positive factor applied to the raw capacity column before BPR costs are
        evaluated. The derived capacity is written back into
        ``cost_function.capacity_col`` in the per-solve link table.
    cost_function:
        Explicit BPR column contract from ``assignment_configs.py``.
    convergence:
        Deterministic UE relative-gap thresholds.
    initialization:
        Initial route-flow policy, either free-flow shortest route or uniform
        split across available routes.
    policy:
        Edge-case policy for missing routes, skipped OD pairs, intrazonals and
        expected solver.
    """

    max_iterations: int
    capacity_scaling_factor: float
    cost_function: VDFConfig
    convergence: RouteBasedUEConvergenceConfig
    initialization: RouteBasedUEInitializationConfig
    policy: RouteBasedUEPolicyConfig

    def validate(self) -> None:
        """Validate the complete UE runtime configuration."""
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError(f"max_iterations must be an integer >= 1. Received {self.max_iterations!r}.")
        if not np.isfinite(self.capacity_scaling_factor) or self.capacity_scaling_factor <= 0.0:
            raise ValueError(
                "capacity_scaling_factor must be finite and strictly positive. "
                f"Received {self.capacity_scaling_factor!r}."
            )
        if not isinstance(self.cost_function, VDFConfig):
            raise TypeError("cost_function must be a VDFConfig instance.")
        if not isinstance(self.convergence, RouteBasedUEConvergenceConfig):
            raise TypeError("convergence must be a RouteBasedUEConvergenceConfig instance.")
        if not isinstance(self.initialization, RouteBasedUEInitializationConfig):
            raise TypeError("initialization must be a RouteBasedUEInitializationConfig instance.")
        if not isinstance(self.policy, RouteBasedUEPolicyConfig):
            raise TypeError("policy must be a RouteBasedUEPolicyConfig instance.")

        self.cost_function.validate()
        self.convergence.validate()
        self.initialization.validate()
        self.policy.validate()

        if self.convergence.minimum_iterations > self.max_iterations:
            raise ValueError("convergence.minimum_iterations cannot exceed max_iterations.")


class RouteBasedUserEquilibriumModel(BaseRouteBasedAssignmentModel):
    """Deterministic route-based User Equilibrium model.

    The iterative loop follows the route-based UE sequence:

    1. Keep current route flows as the primary assignment state.
    2. Aggregate current route flows into link flows.
    3. Evaluate BPR link costs.
    4. Compute route costs by summing link costs along each route.
    5. Build a deterministic all-or-nothing auxiliary route assignment by OD.
    6. Compute deterministic UE relative gap.
    7. Delegate the route-flow update to the injected solver.
    8. Return a strict AssignmentResult after final consistency validation.
    """

    @property
    def behavior_model_name(self) -> str:
        """Return the canonical behavior-model name."""
        return "route_based_user_equilibrium"

    def solve(
        self,
        od_matrix: np.ndarray,
        config: RouteBasedUserEquilibriumRuntimeConfig,
    ) -> AssignmentResult:
        """Solve deterministic route-based UE for one OD matrix.

        Parameters
        ----------
        od_matrix:
            Square OD matrix indexed according to ``zone_id_to_idx``.
        config:
            Fully validated runtime configuration created by the composition
            layer. Raw YAML and artifacts are not read inside this motor.
        """
        if not isinstance(config, RouteBasedUserEquilibriumRuntimeConfig):
            raise TypeError("config must be a RouteBasedUserEquilibriumRuntimeConfig instance.")
        config.validate()
        self._validate_expected_solver(config=config)

        od_matrix_array = np.asarray(od_matrix, dtype=float)
        self._validate_od_matrix(od_matrix_array)
        self._validate_cost_columns(links_df=self.links_df, cost_config=config.cost_function)

        positive_od_pairs, intrazonal_records = self._build_positive_od_pairs(
            od_matrix=od_matrix_array,
            intrazonal_policy=config.policy.intrazonal_policy,
        )
        route_ready_od_pairs, missing_route_records = self._filter_od_pairs_with_routes(
            positive_od_pairs=positive_od_pairs,
            fail_on_missing_routes=config.policy.fail_on_missing_routes,
        )
        skipped_records = [*intrazonal_records, *missing_route_records]
        if skipped_records and config.policy.fail_on_skipped_od_pairs:
            raise ValueError(
                "Some positive-demand OD pairs were skipped by route-based UE, while "
                "policy.fail_on_skipped_od_pairs=True. "
                f"First skipped records: {skipped_records[:10]}."
            )
        if not route_ready_od_pairs:
            raise ValueError(
                "No positive-demand OD pair with at least one route is available for route-based UE. "
                "Check OD demand, intrazonal_policy, route coverage, and missing-route policies."
            )

        od_demands = self._build_od_demands(route_ready_od_pairs)
        link_table = self._build_assignment_link_table(config=config)

        current_route_flows = self._compute_initial_route_flows(
            od_pairs=route_ready_od_pairs,
            link_costs_by_position=self._compute_costs(
                link_flows=np.zeros(len(link_table), dtype=float),
                link_table=link_table,
                cost_config=config.cost_function,
            ),
            initialization_policy=config.initialization.policy,
        )

        convergence_history: list[dict[str, Any]] = []
        solver_history: list[Mapping[str, Any]] = []
        converged = False

        for iteration in range(1, config.max_iterations + 1):
            current_link_flows = aggregate_route_flows_to_link_flows(
                route_flows=current_route_flows,
                route_set=self.route_set,
            )
            current_link_costs = self._compute_costs(
                link_flows=current_link_flows,
                link_table=link_table,
                cost_config=config.cost_function,
            )
            current_route_costs = compute_route_costs_from_link_costs(
                link_costs=current_link_costs,
                route_set=self.route_set,
            )
            auxiliary_route_flows, shortest_path_cost = self._compute_deterministic_auxiliary_route_flows(
                od_pairs=route_ready_od_pairs,
                current_route_costs=current_route_costs,
            )
            auxiliary_link_flows = aggregate_route_flows_to_link_flows(
                route_flows=auxiliary_route_flows,
                route_set=self.route_set,
            )

            total_current_cost = float(np.dot(current_link_flows, current_link_costs))
            gap_metrics = self._compute_deterministic_relative_gap(
                total_current_cost=total_current_cost,
                shortest_path_cost=shortest_path_cost,
                zero_total_cost_tolerance=config.convergence.zero_total_cost_tolerance,
            )

            error_info = f"RelGap: {gap_metrics['relative_gap']:.6e} (Threshold: {config.convergence.relative_gap_threshold:.6e})"
            print_assignment_progress(
                model_name=self.behavior_model_name,
                solver_name=self.solver.name,
                iteration=iteration,
                max_iterations=config.max_iterations,
                error_info=error_info,
            )

            history_item = {
                "iteration": iteration,
                "deterministic_relative_gap": gap_metrics["relative_gap"],
                "absolute_gap": gap_metrics["absolute_gap"],
                "total_current_cost": total_current_cost,
                "total_shortest_path_cost": float(shortest_path_cost),
                "total_current_route_flow": float(np.sum(current_route_flows)),
                "total_current_link_flow": float(np.sum(current_link_flows)),
                "total_auxiliary_route_flow": float(np.sum(auxiliary_route_flows)),
                "total_auxiliary_link_flow": float(np.sum(auxiliary_link_flows)),
            }
            convergence_history.append(history_item)

            if self._has_converged(
                relative_gap=gap_metrics["relative_gap"],
                iteration=iteration,
                convergence=config.convergence,
            ):
                converged = True
                break

            request = RouteBasedSolverStepRequest(
                iteration=iteration,
                current_route_flows=current_route_flows,
                auxiliary_route_flows=auxiliary_route_flows,
                current_link_flows=current_link_flows,
                auxiliary_link_flows=auxiliary_link_flows,
                current_link_costs=current_link_costs,
                current_route_costs=current_route_costs,
                link_table=link_table,
                od_demands=od_demands,
                route_set=self.route_set,
                metadata={
                    "behavior_model": self.behavior_model_name,
                    "deterministic_relative_gap": gap_metrics["relative_gap"],
                },
            )
            step_result = self.solver.compute_route_flow_update(request)
            solver_history.append(step_result.metadata)
            current_route_flows = step_result.new_route_flows

        print_assignment_progress(
            model_name=self.behavior_model_name,
            solver_name=self.solver.name,
            iteration=len(convergence_history),
            max_iterations=config.max_iterations,
            error_info="",
            is_finished=True,
        )

        if not convergence_history:
            raise RuntimeError("UE solve finished without recording convergence history.")

        final_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=current_route_flows,
            route_set=self.route_set,
        )
        final_link_costs = self._compute_costs(
            link_flows=final_link_flows,
            link_table=link_table,
            cost_config=config.cost_function,
        )
        final_route_costs = compute_route_costs_from_link_costs(
            link_costs=final_link_costs,
            route_set=self.route_set,
        )

        metadata = {
            "behavior_model": self.behavior_model_name,
            "solver": self.solver.name,
            "solver_class": type(self.solver).__name__,
            "max_iterations": config.max_iterations,
            "iterations_run": len(convergence_history),
            "converged": bool(converged),
            "convergence_metric": "deterministic_relative_gap",
            "convergence_history": convergence_history,
            "solver_history": solver_history,
            "final_deterministic_relative_gap": convergence_history[-1]["deterministic_relative_gap"],
            "final_absolute_gap": convergence_history[-1]["absolute_gap"],
            "intrazonal_records": intrazonal_records,
            "intrazonal_demand_skipped": float(sum(float(record["demand"]) for record in intrazonal_records)),
            "missing_route_records": missing_route_records,
            "missing_route_demand_skipped": float(sum(float(record["demand"]) for record in missing_route_records)),
            "skipped_od_pairs": skipped_records,
            "skipped_demand_total": float(sum(float(record["demand"]) for record in skipped_records)),
            "assigned_od_pairs": len(route_ready_od_pairs),
            "assigned_od_demand_total": float(sum(float(demand) for _, _, demand in route_ready_od_pairs)),
            "routes_assignment_df": self.route_set.routes_df,
            "route_set_link_order": list(self.route_set.canonical_link_id_order),
        }

        result = AssignmentResult(
            final_route_flows=current_route_flows,
            final_link_flows=final_link_flows,
            final_link_costs=final_link_costs,
            final_route_costs=final_route_costs,
            metadata=metadata,
        )
        self.validate_assignment_result(result=result, od_demands=od_demands)
        return result

    def _validate_expected_solver(self, config: RouteBasedUserEquilibriumRuntimeConfig) -> None:
        """Ensure that the injected solver matches the UE policy."""
        expected_solver = config.policy.expected_solver.value
        if self.solver.name != expected_solver:
            raise ValueError(
                "Injected solver does not match route_based_user_equilibrium.policy.expected_solver. "
                f"solver.name={self.solver.name!r}, expected_solver={expected_solver!r}."
            )

    def _validate_od_matrix(self, od_matrix: np.ndarray) -> None:
        """Validate OD matrix shape, values and coverage by zone_id_to_idx."""
        if not isinstance(od_matrix, np.ndarray):
            raise TypeError("od_matrix must be a numpy.ndarray after conversion.")
        if od_matrix.ndim != 2:
            raise ValueError(f"od_matrix must be two-dimensional. Received shape={od_matrix.shape}.")
        if od_matrix.shape[0] != od_matrix.shape[1]:
            raise ValueError(f"od_matrix must be square. Received shape={od_matrix.shape}.")
        if od_matrix.size == 0:
            raise ValueError("od_matrix cannot be empty.")
        if not np.all(np.isfinite(od_matrix)):
            raise ValueError("od_matrix contains NaN or infinite values.")
        if np.any(od_matrix < 0.0):
            raise ValueError("od_matrix contains negative demand values.")

        expected_indices = set(range(od_matrix.shape[0]))
        mapped_indices = set(self.idx_to_zone_id.keys())
        missing_indices = sorted(expected_indices.difference(mapped_indices))
        extra_indices = sorted(mapped_indices.difference(expected_indices))
        if missing_indices:
            raise ValueError(f"zone_id_to_idx does not cover all OD matrix indices. Missing: {missing_indices[:10]}.")
        if extra_indices:
            raise ValueError(f"zone_id_to_idx references indices outside od_matrix. Extra: {extra_indices[:10]}.")

    @staticmethod
    def _validate_cost_columns(links_df: pd.DataFrame, cost_config: VDFConfig) -> None:
        """Validate configured VDF input columns before assignment starts."""
        required_columns = cost_config.get_vdf_class().get_required_columns()
        
        # We assume VDF methods extract what they need through kwargs mapping
        kwargs_mapping = cost_config.get_columns_kwargs()
        
        # Check that all required columns are actually in kwargs mapping or have defaults
        actual_columns = []
        for req in required_columns:
            if req in kwargs_mapping:
                actual_columns.append(kwargs_mapping[req])
            elif req == "free_flow_time_col":
                actual_columns.append(cost_config.free_flow_time_col)
            else:
                actual_columns.append(req) # Fallback to generic name

        missing_columns = [column for column in actual_columns if column not in links_df.columns]
        if missing_columns:
            raise ValueError(f"links_df is missing required UE cost columns: {missing_columns}.")

        for column in actual_columns:
            values = links_df[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"links_df column {column!r} contains NaN or infinite values.")

        if np.any(links_df[cost_config.free_flow_time_col].to_numpy(dtype=float) <= 0.0):
            raise ValueError(f"Column {cost_config.free_flow_time_col!r} must be strictly positive.")

    def _build_assignment_link_table(self, config: RouteBasedUserEquilibriumRuntimeConfig) -> pd.DataFrame:
        """Create the per-solve link table with explicitly scaled effective capacity."""
        link_table = self.links_df.copy(deep=True)
        capacity_col = config.cost_function.capacity_col
        link_table[capacity_col] = (
            link_table[capacity_col].astype(float) * float(config.capacity_scaling_factor)
        )
        scaled_capacity = link_table[capacity_col].to_numpy(dtype=float)
        if not np.all(np.isfinite(scaled_capacity)):
            raise ValueError("Derived capacity contains NaN or infinite values.")
        if np.any(scaled_capacity <= 0.0):
            raise ValueError("Derived capacity must be strictly positive for every link.")
        return link_table

    def _build_positive_od_pairs(
        self,
        od_matrix: np.ndarray,
        intrazonal_policy: str,
    ) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]]]:
        """Extract positive OD demand according to the intrazonal policy."""
        if intrazonal_policy not in {"raise", "skip_and_report", "assign_if_routes_exist"}:
            raise ValueError(f"Unsupported intrazonal_policy={intrazonal_policy!r}.")

        od_pairs: list[tuple[int, int, float]] = []
        intrazonal_records: list[dict[str, Any]] = []

        for origin_idx in range(od_matrix.shape[0]):
            for destination_idx in range(od_matrix.shape[1]):
                demand = float(od_matrix[origin_idx, destination_idx])
                if demand <= 0.0:
                    continue

                origin_id = int(self.idx_to_zone_id[origin_idx])
                destination_id = int(self.idx_to_zone_id[destination_idx])
                if origin_id == destination_id:
                    record = {
                        "origin_id": origin_id,
                        "destination_id": destination_id,
                        "origin_idx": origin_idx,
                        "destination_idx": destination_idx,
                        "demand": demand,
                        "reason": "positive_intrazonal_demand",
                    }
                    if intrazonal_policy == "raise":
                        raise ValueError(
                            "Positive intrazonal demand found and intrazonal_policy='raise'. "
                            f"Record: {record}."
                        )
                    if intrazonal_policy == "skip_and_report":
                        intrazonal_records.append(record)
                        continue
                    if intrazonal_policy == "assign_if_routes_exist":
                        od_pairs.append((origin_id, destination_id, demand))
                        continue

                od_pairs.append((origin_id, destination_id, demand))

        return od_pairs, intrazonal_records

    def _filter_od_pairs_with_routes(
        self,
        positive_od_pairs: list[tuple[int, int, float]],
        fail_on_missing_routes: bool,
    ) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]]]:
        """Keep assignable OD pairs and report positive-demand pairs without routes."""
        if not isinstance(fail_on_missing_routes, bool):
            raise TypeError("fail_on_missing_routes must be a boolean.")

        ready_pairs: list[tuple[int, int, float]] = []
        missing_records: list[dict[str, Any]] = []
        for origin_id, destination_id, demand in positive_od_pairs:
            route_indices = self.route_set.get_route_indices_for_od(origin_id, destination_id)
            if not route_indices:
                missing_records.append(
                    {
                        "origin_id": int(origin_id),
                        "destination_id": int(destination_id),
                        "demand": float(demand),
                        "reason": "no_routes_available_in_route_set",
                    }
                )
                continue
            ready_pairs.append((int(origin_id), int(destination_id), float(demand)))

        if missing_records and fail_on_missing_routes:
            raise ValueError(
                "RouteSet is missing routes for positive-demand OD pairs. "
                f"First missing records: {missing_records[:10]}."
            )
        return ready_pairs, missing_records

    @staticmethod
    def _build_od_demands(od_pairs: list[tuple[int, int, float]]) -> dict[tuple[int, int], float]:
        """Build the strict OD-demand mapping required by base.py validators."""
        od_demands: dict[tuple[int, int], float] = {}
        for origin_id, destination_id, demand in od_pairs:
            od_key = (int(origin_id), int(destination_id))
            if od_key in od_demands:
                raise ValueError(f"Duplicated positive OD pair after extraction: {od_key}.")
            if not np.isfinite(float(demand)) or float(demand) <= 0.0:
                raise ValueError(f"OD demand must be positive and finite. OD={od_key}, demand={demand!r}.")
            od_demands[od_key] = float(demand)
        return od_demands

    @staticmethod
    def _compute_costs(
        link_flows: np.ndarray,
        link_table: pd.DataFrame,
        cost_config: VDFConfig,
    ) -> np.ndarray:
        """Compute link costs using the configured VDF and explicit column mapping."""
        validate_vector(name="link_flows", value=np.asarray(link_flows, dtype=float), expected_length=len(link_table))
        validate_non_negative_vector(name="link_flows", value=np.asarray(link_flows, dtype=float))

        vdf_class = cost_config.get_vdf_class()
        kwargs = cost_config.get_columns_kwargs()

        costs = vdf_class.evaluate_costs_numpy(link_flows, link_table, **kwargs)
        
        if not np.all(np.isfinite(costs)):
            raise ValueError("Computed costs contain NaN or infinite values.")
        if np.any(costs <= 0.0):
            raise ValueError("Computed costs must be strictly positive.")
        return costs.astype(float)

    def _compute_initial_route_flows(
        self,
        od_pairs: list[tuple[int, int, float]],
        link_costs_by_position: np.ndarray,
        initialization_policy: str,
    ) -> np.ndarray:
        """Compute feasible initial route flows for deterministic UE."""
        if initialization_policy not in {"free_flow_shortest", "uniform"}:
            raise ValueError(f"Unsupported UE initialization policy={initialization_policy!r}.")
        validate_vector(
            name="link_costs_by_position",
            value=np.asarray(link_costs_by_position, dtype=float),
            expected_length=self.route_set.number_of_links,
        )

        route_flows = np.zeros(self.route_set.number_of_routes, dtype=float)
        for origin_id, destination_id, demand in od_pairs:
            od_pair = (int(origin_id), int(destination_id))
            route_indices = self.route_set.get_route_indices_for_od(origin_id, destination_id)
            if not route_indices:
                raise ValueError(f"RouteSet has no routes for assignable OD pair {od_pair}.")

            if initialization_policy == "uniform":
                route_flows[route_indices] = route_flows[route_indices] + float(demand) / float(len(route_indices))
                continue

            route_costs = np.array(
                [float(link_costs_by_position[self.route_set.route_link_positions[route_idx]].sum()) for route_idx in route_indices],
                dtype=float,
            )
            validate_vector(name="initial_od_route_costs", value=route_costs, expected_length=len(route_indices))
            if np.any(route_costs <= 0.0):
                raise ValueError(f"Initial route costs for OD {od_pair} must be strictly positive.")
            best_local_index = int(np.argmin(route_costs))
            route_flows[route_indices[best_local_index]] += float(demand)

        return route_flows

    def _compute_deterministic_auxiliary_route_flows(
        self,
        od_pairs: list[tuple[int, int, float]],
        current_route_costs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Build deterministic all-or-nothing route flows over available routes."""
        validate_vector(
            name="current_route_costs",
            value=np.asarray(current_route_costs, dtype=float),
            expected_length=self.route_set.number_of_routes,
        )
        if np.any(current_route_costs <= 0.0):
            raise ValueError("current_route_costs must be strictly positive for deterministic UE auxiliary assignment.")

        auxiliary_route_flows = np.zeros(self.route_set.number_of_routes, dtype=float)
        total_shortest_path_cost = 0.0

        for origin_id, destination_id, demand in od_pairs:
            od_pair = (int(origin_id), int(destination_id))
            route_indices = self.route_set.get_route_indices_for_od(origin_id, destination_id)
            if not route_indices:
                raise ValueError(f"RouteSet has no routes for assignable OD pair {od_pair}.")

            od_route_costs = current_route_costs[route_indices]
            best_local_index = int(np.argmin(od_route_costs))
            best_route_index = int(route_indices[best_local_index])
            best_route_cost = float(od_route_costs[best_local_index])
            auxiliary_route_flows[best_route_index] += float(demand)
            total_shortest_path_cost += best_route_cost * float(demand)

        return auxiliary_route_flows, float(total_shortest_path_cost)

    @staticmethod
    def _compute_deterministic_relative_gap(
        total_current_cost: float,
        shortest_path_cost: float,
        zero_total_cost_tolerance: float,
    ) -> dict[str, float]:
        """Compute the deterministic UE relative gap."""
        if not np.isfinite(total_current_cost) or total_current_cost < 0.0:
            raise ValueError(f"total_current_cost must be finite and non-negative. Received {total_current_cost}.")
        if not np.isfinite(shortest_path_cost) or shortest_path_cost < 0.0:
            raise ValueError(f"shortest_path_cost must be finite and non-negative. Received {shortest_path_cost}.")
        if not np.isfinite(zero_total_cost_tolerance) or zero_total_cost_tolerance < 0.0:
            raise ValueError(
                "zero_total_cost_tolerance must be finite and non-negative. "
                f"Received {zero_total_cost_tolerance}."
            )

        absolute_gap = max(float(total_current_cost) - float(shortest_path_cost), 0.0)
        if total_current_cost > zero_total_cost_tolerance:
            relative_gap = absolute_gap / float(total_current_cost)
        else:
            relative_gap = 0.0
        return {"absolute_gap": float(absolute_gap), "relative_gap": float(relative_gap)}

    @staticmethod
    def _has_converged(
        relative_gap: float,
        iteration: int,
        convergence: RouteBasedUEConvergenceConfig,
    ) -> bool:
        """Evaluate deterministic UE convergence thresholds."""
        if iteration < convergence.minimum_iterations:
            return False
        return float(relative_gap) <= float(convergence.relative_gap_threshold)


__all__ = [
    "RouteBasedUserEquilibriumModel",
    "RouteBasedUserEquilibriumRuntimeConfig",
]
