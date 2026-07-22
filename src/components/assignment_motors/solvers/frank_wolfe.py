"""Route-based Frank-Wolfe line-search solver.

This module implements Frank-Wolfe as a numerical solver for deterministic,
route-based User Equilibrium with separable BPR link costs.

Conceptual scope
----------------
- The behavioral model builds the auxiliary route-flow solution.
- This solver computes the Frank-Wolfe line-search step size.
- The returned state is always a route-flow vector.
- Link flows are used only to evaluate the Beckmann line-search derivative and
  to produce diagnostics.

Important methodological boundary
---------------------------------
This solver is intended for deterministic UE under separable monotone link-cost
functions. It should not be enabled for SUE unless a SUE-compatible objective is
formally implemented. Compatibility is enforced in assignment_configs.py and in
the composition layer, not here.

Technical debt
--------------
BPR evaluation is still implemented inside this solver because the project does
not yet expose a shared LinkCostFunction interface. A future refactor should
move BPR cost evaluation and the Beckmann integral/derivative logic to a VDF
module and inject that object into this solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from ..assignment_configs import FrankWolfeConfig
from src.components.vdf.config import VDFConfig
from ..base import (
    BaseRouteBasedSolver,
    RouteBasedSolverStepRequest,
    RouteBasedSolverStepResult,
    aggregate_route_flows_to_link_flows,
    validate_non_negative_vector,
    validate_vector,
)


@dataclass(frozen=True)
class FrankWolfeLineSearchDiagnostics:
    """Diagnostics produced by one route-based Frank-Wolfe update.

    Attributes
    ----------
    iteration:
        One-based assignment iteration.
    line_search_method:
        Explicit line-search method selected in assignment.yaml.
    step_size:
        Convex-combination weight used to move from current route flows toward
        auxiliary route flows.
    derivative_at_zero:
        Beckmann directional derivative at alpha=0.
    derivative_at_one:
        Beckmann directional derivative at alpha=1.
    bisection_iterations:
        Number of bisection iterations used by the line search.
    route_direction_l1_norm:
        L1 norm of auxiliary_route_flows - current_route_flows.
    link_direction_l1_norm:
        L1 norm of auxiliary_link_flows - current_link_flows.
    route_step_l1_norm:
        L1 norm of new_route_flows - current_route_flows.
    link_step_l1_norm:
        L1 norm of the link-flow change implied by new_route_flows.
    current_total_route_flow:
        Sum of current route flows before the update.
    auxiliary_total_route_flow:
        Sum of auxiliary route flows before the update.
    new_total_route_flow:
        Sum of route flows after the update.
    current_total_link_flow:
        Sum of current link flows before the update.
    auxiliary_total_link_flow:
        Sum of auxiliary link flows before the update.
    new_total_link_flow:
        Sum of link flows implied by the updated route flows.
    """

    iteration: int
    line_search_method: str
    step_size: float
    derivative_at_zero: float
    derivative_at_one: float
    bisection_iterations: int
    route_direction_l1_norm: float
    link_direction_l1_norm: float
    route_step_l1_norm: float
    link_step_l1_norm: float
    current_total_route_flow: float
    auxiliary_total_route_flow: float
    new_total_route_flow: float
    current_total_link_flow: float
    auxiliary_total_link_flow: float
    new_total_link_flow: float

    def as_metadata(self) -> Mapping[str, float | int | str]:
        """Return diagnostics as serializable solver metadata."""
        return {
            "solver": "frank_wolfe",
            "iteration": self.iteration,
            "line_search_method": self.line_search_method,
            "step_size": self.step_size,
            "derivative_at_zero": self.derivative_at_zero,
            "derivative_at_one": self.derivative_at_one,
            "bisection_iterations": self.bisection_iterations,
            "route_direction_l1_norm": self.route_direction_l1_norm,
            "link_direction_l1_norm": self.link_direction_l1_norm,
            "route_step_l1_norm": self.route_step_l1_norm,
            "link_step_l1_norm": self.link_step_l1_norm,
            "current_total_route_flow": self.current_total_route_flow,
            "auxiliary_total_route_flow": self.auxiliary_total_route_flow,
            "new_total_route_flow": self.new_total_route_flow,
            "current_total_link_flow": self.current_total_link_flow,
            "auxiliary_total_link_flow": self.auxiliary_total_link_flow,
            "new_total_link_flow": self.new_total_link_flow,
        }


class FrankWolfeLineSearchAlgorithm(BaseRouteBasedSolver):
    """Route-based Frank-Wolfe line-search solver for deterministic UE.

    The solver computes a scalar line-search step size and applies it in
    route-flow space:

        new_route_flows = current_route_flows
            + alpha * (auxiliary_route_flows - current_route_flows)

    The line search itself evaluates the Beckmann directional derivative in
    link-flow space because BPR costs are separable by link. This is consistent
    with a route-based architecture because link flows are derived from route
    flows, not stored as the primary solver state.
    """

    _CANONICAL_NAME = "frank_wolfe"

    def __init__(self, config: FrankWolfeConfig, cost_config: VDFConfig) -> None:
        """Create a Frank-Wolfe solver from explicit validated configs.

        Parameters
        ----------
        config:
            Strict Frank-Wolfe line-search configuration.
        cost_config:
            Explicit VDF configuration contract used by the internal VDF
            evaluator.
        """
        self._validate_constructor_configs(config=config, cost_config=cost_config)
        self.config = config
        self.cost_config = cost_config

    @property
    def name(self) -> str:
        """Return the canonical solver name used in configs and reports."""
        return self._CANONICAL_NAME

    def compute_route_flow_update(self, request: RouteBasedSolverStepRequest) -> RouteBasedSolverStepResult:
        """Compute one Frank-Wolfe route-flow update.

        Parameters
        ----------
        request:
            Fully validated route-based state produced by a deterministic UE
            behavior model.

        Returns
        -------
        RouteBasedSolverStepResult
            Updated feasible route-flow vector plus line-search diagnostics.
        """
        self._validate_step_request(request)
        self._validate_link_table_for_bpr(request.link_table)
        self._validate_current_cost_consistency(request)

        link_direction = request.auxiliary_link_flows - request.current_link_flows
        route_direction = request.auxiliary_route_flows - request.current_route_flows

        link_direction_l1_norm = float(np.linalg.norm(link_direction, ord=1))
        route_direction_l1_norm = float(np.linalg.norm(route_direction, ord=1))

        if link_direction_l1_norm <= self.config.tolerance:
            step_size = 0.0
            derivative_at_zero = 0.0
            derivative_at_one = 0.0
            bisection_iterations = 0
        else:
            derivative_at_zero = self._directional_derivative(
                alpha=0.0,
                current_flows=request.current_link_flows,
                direction=link_direction,
                link_table=request.link_table,
            )
            derivative_at_one = self._directional_derivative(
                alpha=1.0,
                current_flows=request.current_link_flows,
                direction=link_direction,
                link_table=request.link_table,
            )
            step_size, bisection_iterations = self._compute_bisection_step_size(
                derivative_at_zero=derivative_at_zero,
                derivative_at_one=derivative_at_one,
                current_flows=request.current_link_flows,
                direction=link_direction,
                link_table=request.link_table,
            )

        self._validate_step_size(step_size)

        new_route_flows = request.current_route_flows + float(step_size) * route_direction
        self._validate_new_route_flows(new_route_flows=new_route_flows, request=request)

        new_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=new_route_flows,
            route_set=request.route_set,
        )

        diagnostics = FrankWolfeLineSearchDiagnostics(
            iteration=request.iteration,
            line_search_method=self.config.line_search_method,
            step_size=float(step_size),
            derivative_at_zero=float(derivative_at_zero),
            derivative_at_one=float(derivative_at_one),
            bisection_iterations=int(bisection_iterations),
            route_direction_l1_norm=route_direction_l1_norm,
            link_direction_l1_norm=link_direction_l1_norm,
            route_step_l1_norm=float(np.linalg.norm(new_route_flows - request.current_route_flows, ord=1)),
            link_step_l1_norm=float(np.linalg.norm(new_link_flows - request.current_link_flows, ord=1)),
            current_total_route_flow=float(np.sum(request.current_route_flows)),
            auxiliary_total_route_flow=float(np.sum(request.auxiliary_route_flows)),
            new_total_route_flow=float(np.sum(new_route_flows)),
            current_total_link_flow=float(np.sum(request.current_link_flows)),
            auxiliary_total_link_flow=float(np.sum(request.auxiliary_link_flows)),
            new_total_link_flow=float(np.sum(new_link_flows)),
        )

        result = RouteBasedSolverStepResult(
            new_route_flows=new_route_flows,
            metadata=diagnostics.as_metadata(),
        )
        result.validate(route_set=request.route_set, od_demands=request.od_demands)
        return result

    @staticmethod
    def _validate_constructor_configs(config: FrankWolfeConfig, cost_config: VDFConfig) -> None:
        """Validate constructor configs without filling missing values."""
        if not isinstance(config, FrankWolfeConfig):
            raise TypeError("config must be a FrankWolfeConfig instance.")
        if not isinstance(cost_config, VDFConfig):
            raise TypeError("cost_config must be a VDFConfig instance.")
        config.validate()
        cost_config.validate()

    @staticmethod
    def _validate_step_request(request: RouteBasedSolverStepRequest) -> None:
        """Validate the incoming route-based update request."""
        if not isinstance(request, RouteBasedSolverStepRequest):
            raise TypeError("request must be a RouteBasedSolverStepRequest instance.")
        request.validate()

    def _validate_link_table_for_bpr(self, link_table: pd.DataFrame) -> None:
        """Validate that link_table contains all BPR columns needed by FW."""
        if not isinstance(link_table, pd.DataFrame):
            raise TypeError("link_table must be a pandas DataFrame.")
        if link_table.empty:
            raise ValueError("link_table cannot be empty.")

        required_columns = [
            self.cost_config.free_flow_time_col,
            self.cost_config.capacity_col,
            self.cost_config.alpha_col,
            self.cost_config.beta_col,
        ]
        if self.cost_config.toll_col is not None:
            required_columns.append(self.cost_config.toll_col)

        missing_columns = [column for column in required_columns if column not in link_table.columns]
        if missing_columns:
            raise ValueError(f"link_table is missing required BPR columns for Frank-Wolfe: {missing_columns}.")

        for column in required_columns:
            values = link_table[column].to_numpy(dtype=float)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"link_table column {column!r} contains NaN or infinite values.")

        if np.any(link_table[self.cost_config.free_flow_time_col].to_numpy(dtype=float) <= 0.0):
            raise ValueError(f"Column {self.cost_config.free_flow_time_col!r} must be strictly positive.")
        if np.any(link_table[self.cost_config.capacity_col].to_numpy(dtype=float) <= 0.0):
            raise ValueError(f"Column {self.cost_config.capacity_col!r} must be strictly positive.")
        if np.any(link_table[self.cost_config.alpha_col].to_numpy(dtype=float) < 0.0):
            raise ValueError(f"Column {self.cost_config.alpha_col!r} must be non-negative.")
        if np.any(link_table[self.cost_config.beta_col].to_numpy(dtype=float) < 0.0):
            raise ValueError(f"Column {self.cost_config.beta_col!r} must be non-negative.")

    def _validate_current_cost_consistency(self, request: RouteBasedSolverStepRequest) -> None:
        """Ensure request.current_link_costs matches this solver's BPR config.

        The Frank-Wolfe line search is valid only if the link costs used by the
        behavior model are generated by the same BPR columns and capacity
        scaling used here. A mismatch usually indicates wrong VDF columns,
        wrong capacity scaling, or broken link ordering.
        """
        recomputed_costs = self._compute_bpr_costs(
            link_flows=request.current_link_flows,
            link_table=request.link_table,
        )
        if not np.allclose(
            request.current_link_costs,
            recomputed_costs,
            rtol=self.config.tolerance,
            atol=self.config.tolerance,
        ):
            max_abs_difference = float(np.max(np.abs(request.current_link_costs - recomputed_costs)))
            raise ValueError(
                "request.current_link_costs is inconsistent with FrankWolfeLineSearchAlgorithm.cost_config. "
                "Check VDF columns, capacity scaling, and link ordering. "
                f"max_abs_difference={max_abs_difference}."
            )

    def _compute_bisection_step_size(
        self,
        *,
        derivative_at_zero: float,
        derivative_at_one: float,
        current_flows: np.ndarray,
        direction: np.ndarray,
        link_table: pd.DataFrame,
    ) -> tuple[float, int]:
        """Solve the one-dimensional Frank-Wolfe line search by bisection."""
        if not np.isfinite(derivative_at_zero):
            raise ValueError(f"derivative_at_zero must be finite. Received {derivative_at_zero}.")
        if not np.isfinite(derivative_at_one):
            raise ValueError(f"derivative_at_one must be finite. Received {derivative_at_one}.")

        # A non-negative derivative at zero means the proposed direction is not
        # a descent direction under the configured objective. Returning zero is
        # explicit and is reported in metadata.
        if derivative_at_zero >= -self.config.tolerance:
            return 0.0, 0

        # A non-positive derivative at one means the objective is decreasing
        # across the whole feasible segment, so the full FW step is optimal
        # within [0, 1].
        if derivative_at_one <= self.config.tolerance:
            return 1.0, 0

        low = 0.0
        high = 1.0
        iterations_used = 0

        for iteration in range(1, self.config.max_bisection_iterations + 1):
            iterations_used = iteration
            midpoint = 0.5 * (low + high)
            derivative_at_midpoint = self._directional_derivative(
                alpha=midpoint,
                current_flows=current_flows,
                direction=direction,
                link_table=link_table,
            )

            if abs(derivative_at_midpoint) <= self.config.tolerance:
                return float(midpoint), iterations_used
            if (high - low) <= self.config.tolerance:
                return float(midpoint), iterations_used

            if derivative_at_midpoint < 0.0:
                low = midpoint
            else:
                high = midpoint

        return float(0.5 * (low + high)), iterations_used

    def _directional_derivative(
        self,
        *,
        alpha: float,
        current_flows: np.ndarray,
        direction: np.ndarray,
        link_table: pd.DataFrame,
    ) -> float:
        """Compute d/dalpha Beckmann(x + alpha*d) for BPR link costs."""
        if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError(f"alpha must be finite and within [0, 1]. Received {alpha}.")

        trial_flows = current_flows + float(alpha) * direction
        self._validate_finite_vector("trial_flows", trial_flows, expected_length=len(link_table))
        if np.any(trial_flows < -1e-10):
            raise ValueError(
                "Frank-Wolfe line search generated materially negative trial flows. "
                "This indicates current and auxiliary link flows are not both feasible."
            )

        # Tiny negative values can occur from floating-point arithmetic along a
        # convex segment. Material negatives are rejected above.
        trial_flows = np.maximum(trial_flows, 0.0)
        trial_costs = self._compute_costs(link_flows=trial_flows, link_table=link_table)
        derivative = float(np.dot(trial_costs, direction))
        if not np.isfinite(derivative):
            raise ValueError("Frank-Wolfe directional derivative is NaN or infinite.")
        return derivative

    def _compute_costs(self, *, link_flows: np.ndarray, link_table: pd.DataFrame) -> np.ndarray:
        """Compute link costs using the configured VDF and explicit column mapping."""
        self._validate_finite_vector("link_flows", link_flows, expected_length=len(link_table))
        if np.any(link_flows < -1e-10):
            raise ValueError("link_flows contains materially negative values before VDF cost computation.")

        # The clip is only a numerical guard after rejecting material negatives.
        link_flows = np.maximum(link_flows, 0.0)

        vdf_class = self.cost_config.get_vdf_class()
        kwargs = self.cost_config.get_columns_kwargs()

        costs = vdf_class.evaluate_costs_numpy(link_flows, link_table, **kwargs)
        
        self._validate_finite_vector("costs", costs, expected_length=len(link_table))
        if np.any(costs <= 0.0):
            raise ValueError("Computed costs must be strictly positive.")
        return costs.astype(float)

    @staticmethod
    def _validate_finite_vector(name: str, value: np.ndarray, expected_length: int) -> None:
        """Validate a one-dimensional finite vector with an expected length."""
        validate_vector(name=name, value=value, expected_length=expected_length)

    @staticmethod
    def _validate_new_route_flows(new_route_flows: np.ndarray, request: RouteBasedSolverStepRequest) -> None:
        """Validate the raw updated route-flow vector before wrapping it."""
        validate_vector(
            name="new_route_flows",
            value=new_route_flows,
            expected_length=request.route_set.number_of_routes,
        )
        validate_non_negative_vector(name="new_route_flows", value=new_route_flows)

    @staticmethod
    def _validate_step_size(step_size: float) -> None:
        """Validate that the computed step size is a convex-combination weight."""
        if not np.isfinite(step_size):
            raise ValueError(f"Frank-Wolfe produced a non-finite step_size={step_size}.")
        if step_size < 0.0 or step_size > 1.0:
            raise ValueError(f"Frank-Wolfe step_size must be within [0, 1]. Received {step_size}.")


__all__ = [
    "FrankWolfeLineSearchAlgorithm",
    "FrankWolfeLineSearchDiagnostics",
]
