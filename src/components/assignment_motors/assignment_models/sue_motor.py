"""Route-based Stochastic User Equilibrium behavior model.

This module implements a full route-based Stochastic User Equilibrium (SUE)
fixed-point assignment. The primary assignment state is always represented in
route space. Link flows are derived from route flows through the prepared
``RouteSet``.

Conceptual separation
---------------------
- SUE is a behavioral assignment model: it creates Logit auxiliary route flows.
- The injected solver is a numerical route-flow updater. For this SUE
  fixed-point implementation, only MSA is currently supported.
- ``RouteSet`` owns route preparation and link-position alignment.
- BPR costs are currently evaluated here through an explicit runtime config.
  A future VDF refactor should move this logic to a shared LinkCostFunction.

The motor does not read raw YAML, Hydra/OmegaConf objects, training configs or
artifacts. The composition layer must inject a fully validated runtime config
with an already-resolved numeric Logit theta.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

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


from src.components.vdf.config import VDFConfig


@dataclass(frozen=True)
class SUEConvergenceConfig:
    """Convergence thresholds for the SUE fixed-point process.

    These metrics compare current link flows against Logit auxiliary link
    flows. They are fixed-point diagnostics, not deterministic UE relative gap
    metrics.
    """

    equilibrium_l1_threshold: float
    max_absolute_gap_threshold: float
    max_relative_gap_threshold: float
    min_flow_for_relative_gap: float
    minimum_iterations: int

    def validate(self) -> None:
        """Validate SUE fixed-point convergence thresholds."""
        numeric_fields = {
            "equilibrium_l1_threshold": self.equilibrium_l1_threshold,
            "max_absolute_gap_threshold": self.max_absolute_gap_threshold,
            "max_relative_gap_threshold": self.max_relative_gap_threshold,
            "min_flow_for_relative_gap": self.min_flow_for_relative_gap,
        }
        for field_name, value in numeric_fields.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{field_name} must be finite. Received {value!r}.")
            if float(value) < 0.0:
                raise ValueError(f"{field_name} must be non-negative. Received {value!r}.")

        if float(self.min_flow_for_relative_gap) <= 0.0:
            raise ValueError("min_flow_for_relative_gap must be strictly positive.")
        if not isinstance(self.minimum_iterations, int) or self.minimum_iterations < 1:
            raise ValueError(f"minimum_iterations must be an integer >= 1. Received {self.minimum_iterations!r}.")


@dataclass(frozen=True)
class SUELogitConfig:
    """Runtime Logit route-choice configuration for SUE.

    ``theta`` must already be resolved by the composition layer. This motor does
    not read artifacts or configuration paths.
    """

    theta: float
    fail_on_invalid_probability: bool

    def validate(self) -> None:
        """Validate Logit sensitivity and probability policy."""
        if not np.isfinite(float(self.theta)) or float(self.theta) <= 0.0:
            raise ValueError(f"theta must be a positive finite value. Received {self.theta!r}.")
        if not isinstance(self.fail_on_invalid_probability, bool):
            raise TypeError("fail_on_invalid_probability must be a bool.")


@dataclass(frozen=True)
class RouteBasedSUEPolicyConfig:
    """Explicit edge-case policies for route-based SUE.

    SUE + MSA is the only supported combination in this fixed-point
    implementation. Frank-Wolfe and Gradient Projection require dedicated
    SUE-compatible mathematical formulations before being enabled.
    """

    fail_on_missing_routes: bool
    fail_on_skipped_od_pairs: bool
    intrazonal_policy: str
    expected_solver: str

    def validate(self) -> None:
        """Validate edge-case policies and solver compatibility."""
        if not isinstance(self.fail_on_missing_routes, bool):
            raise TypeError("fail_on_missing_routes must be a bool.")
        if not isinstance(self.fail_on_skipped_od_pairs, bool):
            raise TypeError("fail_on_skipped_od_pairs must be a bool.")

        allowed_intrazonal_policies = {"raise", "skip_and_report", "assign_if_routes_exist"}
        if self.intrazonal_policy not in allowed_intrazonal_policies:
            raise ValueError(
                "Unsupported intrazonal_policy for SUE. "
                f"Received {self.intrazonal_policy!r}. "
                f"Allowed values are {sorted(allowed_intrazonal_policies)}."
            )

        if not isinstance(self.expected_solver, str) or not self.expected_solver.strip():
            raise ValueError("expected_solver must be a non-empty string.")
        if self.expected_solver.strip().lower() != "msa":
            raise ValueError(
                "This SUE fixed-point implementation currently supports only expected_solver='msa'. "
                "Do not enable SUE + Frank-Wolfe or SUE + Gradient Projection until their "
                "SUE-specific formulations are implemented. "
                f"Received {self.expected_solver!r}."
            )


@dataclass(frozen=True)
class RouteBasedStochasticUserEquilibriumRuntimeConfig:
    """Complete solve-time configuration for route-based SUE."""

    max_iterations: int
    capacity_scaling_factor: float
    cost_function: VDFConfig
    convergence: SUEConvergenceConfig
    logit: SUELogitConfig
    policy: RouteBasedSUEPolicyConfig

    def validate(self) -> None:
        """Validate the complete SUE runtime configuration."""
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError(f"max_iterations must be an integer >= 1. Received {self.max_iterations!r}.")
        if not np.isfinite(float(self.capacity_scaling_factor)) or float(self.capacity_scaling_factor) <= 0.0:
            raise ValueError(
                "capacity_scaling_factor must be finite and strictly positive. "
                f"Received {self.capacity_scaling_factor!r}."
            )
        if not isinstance(self.cost_function, VDFConfig):
            raise TypeError("cost_function must be a VDFConfig instance.")
        if not isinstance(self.convergence, SUEConvergenceConfig):
            raise TypeError("convergence must be a SUEConvergenceConfig instance.")
        if not isinstance(self.logit, SUELogitConfig):
            raise TypeError("logit must be a SUELogitConfig instance.")
        if not isinstance(self.policy, RouteBasedSUEPolicyConfig):
            raise TypeError("policy must be a RouteBasedSUEPolicyConfig instance.")

        self.cost_function.validate()
        self.convergence.validate()
        self.logit.validate()
        self.policy.validate()

        if self.convergence.minimum_iterations > self.max_iterations:
            raise ValueError("convergence.minimum_iterations cannot exceed max_iterations.")


class RouteBasedStochasticUserEquilibriumModel(BaseRouteBasedAssignmentModel):
    """Route-based Stochastic User Equilibrium behavioral model.

    The iterative loop follows the SUE fixed-point sequence:

    1. Keep current route flows as the primary assignment state.
    2. Aggregate current route flows into link flows.
    3. Evaluate BPR link costs.
    4. Compute route costs by summing link costs along each route.
    5. Build a Logit auxiliary route assignment by OD.
    6. Compute fixed-point gap between current and auxiliary link flows.
    7. Delegate the route-flow update to the injected MSA solver.
    8. Return a strict AssignmentResult after final consistency validation.
    """

    @property
    def behavior_model_name(self) -> str:
        """Return the canonical behavior-model name."""
        return "stochastic_user_equilibrium"

    def solve(
        self,
        od_matrix: np.ndarray,
        config: RouteBasedStochasticUserEquilibriumRuntimeConfig,
    ) -> AssignmentResult:
        """Solve route-based SUE using a prepared RouteSet and explicit config."""
        if not isinstance(config, RouteBasedStochasticUserEquilibriumRuntimeConfig):
            raise TypeError("config must be a RouteBasedStochasticUserEquilibriumRuntimeConfig instance.")
        config.validate()
        self._validate_expected_solver(config.policy)

        od_matrix_array = np.asarray(od_matrix, dtype=float)
        self._validate_od_matrix(od_matrix_array)
        self._validate_cost_columns(links_df=self.links_df, cost_config=config.cost_function)

        positive_od_pairs, intrazonal_records = self._build_positive_od_pairs(
            od_matrix=od_matrix_array,
            intrazonal_policy=config.policy.intrazonal_policy,
        )
        assignable_od_pairs, missing_route_records = self._filter_positive_od_pairs_by_route_coverage(
            positive_od_pairs=positive_od_pairs,
            fail_on_missing_routes=config.policy.fail_on_missing_routes,
        )
        if missing_route_records and config.policy.fail_on_skipped_od_pairs:
            raise ValueError(
                "Some positive-demand OD pairs were skipped because no routes were available. "
                f"First skipped records: {missing_route_records[:10]}."
            )

        od_demands = self._build_od_demands(assignable_od_pairs)
        link_table = self._build_assignment_link_table(config=config)

        initial_link_flows = np.zeros(self.route_set.number_of_links, dtype=float)
        initial_link_costs = self._compute_costs(
            link_flows=initial_link_flows,
            link_table=link_table,
            cost_config=config.cost_function,
        )
        initial_route_costs = compute_route_costs_from_link_costs(
            link_costs=initial_link_costs,
            route_set=self.route_set,
        )
        current_route_flows, last_route_choice_diagnostics = self._compute_logit_auxiliary_route_flows(
            od_demands=od_demands,
            route_costs=initial_route_costs,
            theta=float(config.logit.theta),
            fail_on_invalid_probability=config.logit.fail_on_invalid_probability,
        )

        convergence_history: list[dict[str, Any]] = []
        converged = False
        last_solver_metadata: Mapping[str, Any] = {}

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

            auxiliary_route_flows, route_choice_diagnostics = self._compute_logit_auxiliary_route_flows(
                od_demands=od_demands,
                route_costs=current_route_costs,
                theta=float(config.logit.theta),
                fail_on_invalid_probability=config.logit.fail_on_invalid_probability,
            )
            auxiliary_link_flows = aggregate_route_flows_to_link_flows(
                route_flows=auxiliary_route_flows,
                route_set=self.route_set,
            )

            fixed_point_gap_metrics = self._compute_fixed_point_gap_metrics(
                current_link_flows=current_link_flows,
                auxiliary_link_flows=auxiliary_link_flows,
                min_flow_for_relative_gap=float(config.convergence.min_flow_for_relative_gap),
            )

            error_info = (
                f"L1: {fixed_point_gap_metrics['sue_fixed_point_gap_l1']:.4e}/{config.convergence.equilibrium_l1_threshold:.4e} | "
                f"MaxAbs: {fixed_point_gap_metrics['sue_fixed_point_max_absolute_gap']:.4e}/{config.convergence.max_absolute_gap_threshold:.4e} | "
                f"MaxRel: {fixed_point_gap_metrics['sue_fixed_point_max_relative_gap']:.4e}/{config.convergence.max_relative_gap_threshold:.4e}"
            )
            print_assignment_progress(
                model_name=self.behavior_model_name,
                solver_name=self.solver.name,
                iteration=iteration,
                max_iterations=config.max_iterations,
                error_info=error_info,
            )

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
                    "theta": float(config.logit.theta),
                    "current_gap": fixed_point_gap_metrics["sue_fixed_point_gap_l1"],
                    "current_max_absolute_gap": fixed_point_gap_metrics["sue_fixed_point_max_absolute_gap"],
                    "current_max_relative_gap": fixed_point_gap_metrics["sue_fixed_point_max_relative_gap"],
                },
            )
            step_result = self.solver.compute_route_flow_update(request)
            current_route_flows = step_result.new_route_flows
            last_solver_metadata = step_result.metadata
            last_route_choice_diagnostics = route_choice_diagnostics

            new_link_flows = aggregate_route_flows_to_link_flows(
                route_flows=current_route_flows,
                route_set=self.route_set,
            )
            route_step_l1 = float(np.linalg.norm(step_result.new_route_flows - request.current_route_flows, ord=1))
            link_step_l1 = float(np.linalg.norm(new_link_flows - request.current_link_flows, ord=1))

            history_item = {
                "iteration": int(iteration),
                "step_size": float(step_result.metadata.get("step_size", np.nan)),
                "sue_fixed_point_gap_l1": fixed_point_gap_metrics["sue_fixed_point_gap_l1"],
                "sue_fixed_point_max_absolute_gap": fixed_point_gap_metrics["sue_fixed_point_max_absolute_gap"],
                "sue_fixed_point_max_relative_gap": fixed_point_gap_metrics["sue_fixed_point_max_relative_gap"],
                "route_step_l1_norm": route_step_l1,
                "link_step_l1_norm": link_step_l1,
                "total_current_link_flow": float(np.sum(request.current_link_flows)),
                "total_auxiliary_link_flow": float(np.sum(request.auxiliary_link_flows)),
                "total_new_link_flow": float(np.sum(new_link_flows)),
                "assigned_od_pairs": int(len(od_demands)),
            }
            convergence_history.append(history_item)

            converged = self._has_converged(
                gap_metrics=fixed_point_gap_metrics,
                convergence=config.convergence,
                iteration=iteration,
            )
            if converged:
                break

        print_assignment_progress(
            model_name=self.behavior_model_name,
            solver_name=self.solver.name,
            iteration=len(convergence_history),
            max_iterations=config.max_iterations,
            error_info="",
            is_finished=True,
        )

        if not convergence_history:
            raise RuntimeError("SUE solve finished without recording convergence history.")

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
            "max_iterations": int(config.max_iterations),
            "iterations_run": int(len(convergence_history)),
            "converged": bool(converged),
            "theta": float(config.logit.theta),
            # "convergence_history": convergence_history, # TODO: this may not be necessary (verbous)
            "final_sue_fixed_point_gap_l1": convergence_history[-1]["sue_fixed_point_gap_l1"],
            "final_sue_fixed_point_max_absolute_gap": convergence_history[-1]["sue_fixed_point_max_absolute_gap"],
            "final_sue_fixed_point_max_relative_gap": convergence_history[-1]["sue_fixed_point_max_relative_gap"],
            "intrazonal_records": intrazonal_records,
            "intrazonal_demand_skipped": float(sum(record["demand"] for record in intrazonal_records)),
            "missing_route_records": missing_route_records,
            "missing_route_demand_skipped": float(sum(record["demand"] for record in missing_route_records)),
            "skipped_demand_total": float(
                sum(record["demand"] for record in intrazonal_records)
                + sum(record["demand"] for record in missing_route_records)
            ),
            "assigned_od_pairs": int(len(od_demands)),
            "last_solver_metadata": dict(last_solver_metadata),
            # "last_route_choice_diagnostics": last_route_choice_diagnostics, # TODO: this may not be necessary, unless auditing (verbous)
            "routes_assignment_df": self.route_set.routes_df,
            "route_set_link_order": list(self.route_set.canonical_link_id_order),
        }

        result = AssignmentResult(
            final_link_flows=final_link_flows,
            final_route_flows=current_route_flows,
            final_link_costs=final_link_costs,
            final_route_costs=final_route_costs,
            metadata=metadata,
        )
        self.validate_assignment_result(result=result, od_demands=od_demands)
        return result

    def _validate_expected_solver(self, policy: RouteBasedSUEPolicyConfig) -> None:
        """Ensure the injected solver matches the SUE policy."""
        if self.solver.name != policy.expected_solver:
            raise ValueError(
                "Injected solver does not match SUE policy.expected_solver. "
                f"solver.name={self.solver.name!r}, expected_solver={policy.expected_solver!r}."
            )
        if self.solver.name != "msa":
            raise ValueError(
                "Route-based SUE currently supports only the MSA solver. "
                f"Received solver.name={self.solver.name!r}."
            )

    def _validate_od_matrix(self, od_matrix: np.ndarray) -> None:
        """Validate OD matrix shape, values and zone-index coverage."""
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
            raise ValueError(
                "zone_id_to_idx does not cover all OD matrix indices. "
                f"Missing indices: {missing_indices[:10]}."
            )
        if extra_indices:
            raise ValueError(
                "zone_id_to_idx references indices outside the OD matrix. "
                f"Extra indices: {extra_indices[:10]}."
            )

    @staticmethod
    def _validate_cost_columns(links_df: pd.DataFrame, cost_config: VDFConfig) -> None:
        """Validate configured VDF input columns before assignment starts."""
        if not isinstance(links_df, pd.DataFrame):
            raise TypeError("links_df must be a pandas DataFrame.")

        required_columns = cost_config.get_vdf_class().get_required_columns()
        kwargs_mapping = cost_config.get_columns_kwargs()

        actual_columns = []
        for req in required_columns:
            if req in kwargs_mapping:
                actual_columns.append(kwargs_mapping[req])
            elif req == "free_flow_time_col":
                actual_columns.append(cost_config.free_flow_time_col)
            else:
                actual_columns.append(req)

        missing_columns = [column for column in actual_columns if column not in links_df.columns]
        if missing_columns:
            raise ValueError(f"links_df is missing required SUE cost columns: {missing_columns}.")

        for column in actual_columns:
            values = links_df[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"links_df column {column!r} contains NaN or infinite values.")

        if np.any(links_df[cost_config.free_flow_time_col].to_numpy(dtype=float) <= 0.0):
            raise ValueError(f"Column {cost_config.free_flow_time_col!r} must be strictly positive.")

    def _build_assignment_link_table(self, config: RouteBasedStochasticUserEquilibriumRuntimeConfig) -> pd.DataFrame:
        """Create a per-solve link table with explicitly scaled effective capacity."""
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
        self.route_set.require_same_link_order(link_table)
        return link_table

    def _build_positive_od_pairs(
        self,
        od_matrix: np.ndarray,
        intrazonal_policy: str,
    ) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]]]:
        """Extract positive OD pairs and apply the explicit intrazonal policy."""
        positive_od_pairs: list[tuple[int, int, float]] = []
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
                        "origin_idx": int(origin_idx),
                        "destination_idx": int(destination_idx),
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
                        positive_od_pairs.append((origin_id, destination_id, demand))
                        continue
                    raise ValueError(f"Unsupported intrazonal_policy={intrazonal_policy!r}.")

                positive_od_pairs.append((origin_id, destination_id, demand))

        return positive_od_pairs, intrazonal_records

    def _filter_positive_od_pairs_by_route_coverage(
        self,
        positive_od_pairs: list[tuple[int, int, float]],
        fail_on_missing_routes: bool,
    ) -> tuple[list[tuple[int, int, float]], list[dict[str, Any]]]:
        """Keep only OD pairs with available routes, or fail when configured."""
        assignable_od_pairs: list[tuple[int, int, float]] = []
        missing_route_records: list[dict[str, Any]] = []

        for origin_id, destination_id, demand in positive_od_pairs:
            route_indices = self.route_set.get_route_indices_for_od(origin_id, destination_id)
            if route_indices:
                assignable_od_pairs.append((origin_id, destination_id, demand))
                continue

            record = {
                "origin_id": int(origin_id),
                "destination_id": int(destination_id),
                "demand": float(demand),
                "reason": "no_routes_available_in_route_set",
            }
            missing_route_records.append(record)

        if missing_route_records and fail_on_missing_routes:
            raise ValueError(
                "RouteSet is missing routes for positive-demand OD pairs. "
                f"First missing records: {missing_route_records[:10]}."
            )

        return assignable_od_pairs, missing_route_records

    @staticmethod
    def _build_od_demands(positive_od_pairs: list[tuple[int, int, float]]) -> dict[tuple[int, int], float]:
        """Build a strict OD-demand dictionary and reject duplicate OD records."""
        od_demands: dict[tuple[int, int], float] = {}
        for origin_id, destination_id, demand in positive_od_pairs:
            od_pair = (int(origin_id), int(destination_id))
            if od_pair in od_demands:
                raise ValueError(f"Duplicate positive OD pair found after OD extraction: {od_pair}.")
            if not np.isfinite(float(demand)) or float(demand) <= 0.0:
                raise ValueError(f"Demand for OD {od_pair} must be positive and finite. Received {demand!r}.")
            od_demands[od_pair] = float(demand)
        return od_demands

    @staticmethod
    def _compute_costs(
        link_flows: np.ndarray,
        link_table: pd.DataFrame,
        cost_config: VDFConfig,
    ) -> np.ndarray:
        """Compute link costs using the configured VDF and explicit column mapping."""
        validate_vector("link_flows", link_flows, len(link_table))
        validate_non_negative_vector("link_flows", link_flows)

        vdf_class = cost_config.get_vdf_class()
        kwargs = cost_config.get_columns_kwargs()

        costs = vdf_class.evaluate_costs_numpy(link_flows, link_table, **kwargs)
        
        if not np.all(np.isfinite(costs)):
            raise ValueError("Computed costs contain NaN or infinite values.")
        if np.any(costs <= 0.0):
            raise ValueError("Computed costs must be strictly positive.")
        return costs.astype(float)

    def _compute_logit_auxiliary_route_flows(
        self,
        od_demands: Mapping[tuple[int, int], float],
        route_costs: np.ndarray,
        theta: float,
        fail_on_invalid_probability: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute Logit auxiliary route flows for all assignable OD pairs."""
        validate_vector("route_costs", route_costs, self.route_set.number_of_routes)
        if np.any(route_costs <= 0.0):
            raise ValueError("route_costs must be strictly positive for SUE Logit assignment.")

        auxiliary_route_flows = np.zeros(self.route_set.number_of_routes, dtype=float)
        od_probability_summary: list[dict[str, Any]] = []

        for raw_od_pair, raw_demand in od_demands.items():
            od_pair = (int(raw_od_pair[0]), int(raw_od_pair[1]))
            demand = float(raw_demand)
            route_indices = self.route_set.get_route_indices_for_od(*od_pair)
            if not route_indices:
                raise ValueError(f"RouteSet has no routes for assignable OD pair {od_pair}.")

            od_route_costs = route_costs[np.asarray(route_indices, dtype=int)]
            probabilities = self._stable_logit_probabilities(
                route_costs=od_route_costs,
                theta=theta,
                fail_on_invalid_probability=fail_on_invalid_probability,
            )
            auxiliary_route_flows[np.asarray(route_indices, dtype=int)] = demand * probabilities

            od_probability_summary.append(
                {
                    "origin_id": int(od_pair[0]),
                    "destination_id": int(od_pair[1]),
                    "number_of_routes": int(len(route_indices)),
                    "demand": demand,
                    "minimum_route_cost": float(np.min(od_route_costs)),
                    "maximum_route_cost": float(np.max(od_route_costs)),
                    "minimum_probability": float(np.min(probabilities)),
                    "maximum_probability": float(np.max(probabilities)),
                }
            )

        validate_vector("auxiliary_route_flows", auxiliary_route_flows, self.route_set.number_of_routes)
        validate_non_negative_vector("auxiliary_route_flows", auxiliary_route_flows)
        diagnostics = {
            "theta": float(theta),
            "assigned_od_pairs": int(len(od_demands)),
            "assigned_route_flow_total": float(np.sum(auxiliary_route_flows)),
            "od_probability_summary": od_probability_summary,
        }
        return auxiliary_route_flows, diagnostics

    @staticmethod
    def _stable_logit_probabilities(
        route_costs: np.ndarray,
        theta: float,
        fail_on_invalid_probability: bool,
    ) -> np.ndarray:
        """Compute numerically stable Logit probabilities from route costs."""
        validate_vector("route_costs", route_costs, len(route_costs))
        if len(route_costs) == 0:
            raise ValueError("route_costs cannot be empty.")
        if np.any(route_costs <= 0.0):
            raise ValueError("route_costs must be strictly positive for Logit probabilities.")
        if not np.isfinite(float(theta)) or float(theta) <= 0.0:
            raise ValueError(f"theta must be positive and finite. Received {theta!r}.")
        if not isinstance(fail_on_invalid_probability, bool):
            raise TypeError("fail_on_invalid_probability must be a bool.")

        utilities = -float(theta) * route_costs.astype(float)
        shifted_utilities = utilities - float(np.max(utilities))
        exp_values = np.exp(shifted_utilities)
        denominator = float(np.sum(exp_values))

        invalid = (
            not np.all(np.isfinite(exp_values))
            or not np.isfinite(denominator)
            or denominator <= 0.0
        )
        if invalid:
            if fail_on_invalid_probability:
                raise ValueError(
                    "Invalid Logit probability calculation. "
                    f"route_costs={route_costs}, theta={theta}, denominator={denominator}."
                )
            return np.full(len(route_costs), 1.0 / float(len(route_costs)), dtype=float)

        probabilities = exp_values / denominator
        if not np.all(np.isfinite(probabilities)) or np.any(probabilities < -1e-12):
            if fail_on_invalid_probability:
                raise ValueError(f"Logit probabilities are invalid: {probabilities}.")
            return np.full(len(route_costs), 1.0 / float(len(route_costs)), dtype=float)

        probability_sum = float(np.sum(probabilities))
        if not np.isclose(probability_sum, 1.0, rtol=1e-10, atol=1e-10):
            probabilities = probabilities / probability_sum
        return np.maximum(probabilities.astype(float), 0.0)

    @staticmethod
    def _compute_fixed_point_gap_metrics(
        current_link_flows: np.ndarray,
        auxiliary_link_flows: np.ndarray,
        min_flow_for_relative_gap: float,
    ) -> dict[str, float]:
        """Compute SUE fixed-point gap metrics in link space."""
        validate_vector("current_link_flows", current_link_flows, len(current_link_flows))
        validate_vector("auxiliary_link_flows", auxiliary_link_flows, len(current_link_flows))
        validate_non_negative_vector("current_link_flows", current_link_flows)
        validate_non_negative_vector("auxiliary_link_flows", auxiliary_link_flows)
        if not np.isfinite(float(min_flow_for_relative_gap)) or float(min_flow_for_relative_gap) <= 0.0:
            raise ValueError("min_flow_for_relative_gap must be positive and finite.")

        absolute_gap = np.abs(auxiliary_link_flows - current_link_flows)
        denominator = np.maximum(np.abs(current_link_flows), float(min_flow_for_relative_gap))
        relative_gap = absolute_gap / denominator
        return {
            "sue_fixed_point_gap_l1": float(np.linalg.norm(absolute_gap, ord=1)),
            "sue_fixed_point_max_absolute_gap": float(np.max(absolute_gap)) if len(absolute_gap) else 0.0,
            "sue_fixed_point_max_relative_gap": float(np.max(relative_gap)) if len(relative_gap) else 0.0,
        }

    @staticmethod
    def _has_converged(
        gap_metrics: Mapping[str, float],
        convergence: SUEConvergenceConfig,
        iteration: int,
    ) -> bool:
        """Return True when all configured SUE fixed-point thresholds are met."""
        if iteration < convergence.minimum_iterations:
            return False
        return (
            float(gap_metrics["sue_fixed_point_gap_l1"]) <= float(convergence.equilibrium_l1_threshold)
            and float(gap_metrics["sue_fixed_point_max_absolute_gap"]) <= float(convergence.max_absolute_gap_threshold)
            and float(gap_metrics["sue_fixed_point_max_relative_gap"]) <= float(convergence.max_relative_gap_threshold)
        )


__all__ = [
    "RouteBasedSUEPolicyConfig",
    "RouteBasedStochasticUserEquilibriumModel",
    "RouteBasedStochasticUserEquilibriumRuntimeConfig",
    "SUEConvergenceConfig",
    "SUELogitConfig",
]
