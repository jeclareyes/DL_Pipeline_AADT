"""Route-based Gradient Projection solver.

This module implements Gradient Projection as a full route-based numerical
solver. The solver updates route flows directly and preserves OD demand by
projecting each OD-specific route-flow vector onto its simplex constraint.

Important methodological scope
------------------------------
This implementation is intended for deterministic route-based UE in the
current architecture. It is not automatically enabled for SUE, because SUE
requires a stochastic fixed-point or entropy-aware formulation. That
compatibility decision is enforced in assignment_configs.py and the composition
layer, not inside this numerical solver.

Design principles
-----------------
1. No backward-compatibility aliases are exposed.
2. No hidden default parameters are used.
3. No loose **kwargs are accepted.
4. The solver updates route flows, not link flows.
5. Behavioral models provide the complete route-space state; this solver only
   applies the projected route-flow update.
6. The feasible set is enforced OD by OD:
      sum(route_flows for routes of OD) = OD demand
      route_flow >= 0
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from ..assignment_configs import GradientProjectionConfig
from ..base import (
    BaseRouteBasedSolver,
    RouteBasedSolverStepRequest,
    RouteBasedSolverStepResult,
    aggregate_route_flows_to_link_flows,
    compute_route_costs_from_link_costs,
    validate_non_negative_vector,
    validate_route_flow_demand_conservation,
    validate_vector,
)


@dataclass(frozen=True)
class GradientProjectionODStepDiagnostics:
    """Diagnostics for one OD-specific projected-gradient update.

    Attributes
    ----------
    origin_id:
        Origin ID of the OD pair.
    destination_id:
        Destination ID of the OD pair.
    number_of_routes:
        Number of candidate routes available for this OD pair.
    demand:
        Positive OD demand preserved by the simplex projection.
    raw_step_size:
        Step size before cost scaling and clipping.
    cost_scale:
        Cost normalizer used by the configured cost-scaling policy.
    effective_step_size:
        Step size after cost scaling and clipping.
    minimum_route_cost:
        Minimum route cost among this OD pair's candidate routes.
    maximum_route_cost:
        Maximum route cost among this OD pair's candidate routes.
    current_assigned_demand:
        Demand assigned by current route flows before projection.
    new_assigned_demand:
        Demand assigned by new route flows after projection.
    route_step_l1_norm:
        L1 norm of the route-flow change for this OD pair.
    """

    origin_id: int
    destination_id: int
    number_of_routes: int
    demand: float
    raw_step_size: float
    cost_scale: float
    effective_step_size: float
    minimum_route_cost: float
    maximum_route_cost: float
    current_assigned_demand: float
    new_assigned_demand: float
    route_step_l1_norm: float

    def to_metadata(self) -> dict[str, float | int]:
        """Convert OD-level diagnostics into serializable metadata."""
        return {
            "origin_id": self.origin_id,
            "destination_id": self.destination_id,
            "number_of_routes": self.number_of_routes,
            "demand": self.demand,
            "raw_step_size": self.raw_step_size,
            "cost_scale": self.cost_scale,
            "effective_step_size": self.effective_step_size,
            "minimum_route_cost": self.minimum_route_cost,
            "maximum_route_cost": self.maximum_route_cost,
            "current_assigned_demand": self.current_assigned_demand,
            "new_assigned_demand": self.new_assigned_demand,
            "route_step_l1_norm": self.route_step_l1_norm,
        }


@dataclass(frozen=True)
class GradientProjectionStepDiagnostics:
    """Diagnostics produced by one Gradient Projection solver update.

    Attributes
    ----------
    solver:
        Canonical solver name.
    iteration:
        One-based assignment iteration.
    step_rule:
        Configured base step-size rule.
    projection_method:
        Configured projection method. Currently only simplex is supported by
        GradientProjectionConfig.
    cost_scaling:
        Configured route-cost scaling policy.
    base_step_size:
        Base step-size coefficient from assignment.yaml.
    raw_step_size:
        Iteration-adjusted step size before OD-specific cost scaling.
    minimum_effective_step_size:
        Minimum effective OD step size used in this iteration.
    maximum_effective_step_size:
        Maximum effective OD step size used in this iteration.
    current_total_route_flow:
        Sum of route flows before the update.
    new_total_route_flow:
        Sum of route flows after the update.
    current_total_link_flow:
        Sum of link flows before the update.
    new_total_link_flow:
        Sum of link flows implied by the updated route flows.
    route_step_l1_norm:
        L1 norm of the route-flow update.
    link_step_l1_norm:
        L1 norm of the implied link-flow update.
    od_steps:
        OD-specific projection diagnostics.
    """

    solver: str
    iteration: int
    step_rule: str
    projection_method: str
    cost_scaling: str
    base_step_size: float
    raw_step_size: float
    minimum_effective_step_size: float
    maximum_effective_step_size: float
    current_total_route_flow: float
    new_total_route_flow: float
    current_total_link_flow: float
    new_total_link_flow: float
    route_step_l1_norm: float
    link_step_l1_norm: float
    od_steps: tuple[GradientProjectionODStepDiagnostics, ...]

    def to_metadata(self) -> dict[str, object]:
        """Convert diagnostics into serializable solver metadata."""
        return {
            "solver": self.solver,
            "iteration": self.iteration,
            "step_rule": self.step_rule,
            "projection_method": self.projection_method,
            "cost_scaling": self.cost_scaling,
            "base_step_size": self.base_step_size,
            "raw_step_size": self.raw_step_size,
            "minimum_effective_step_size": self.minimum_effective_step_size,
            "maximum_effective_step_size": self.maximum_effective_step_size,
            "current_total_route_flow": self.current_total_route_flow,
            "new_total_route_flow": self.new_total_route_flow,
            "current_total_link_flow": self.current_total_link_flow,
            "new_total_link_flow": self.new_total_link_flow,
            "route_step_l1_norm": self.route_step_l1_norm,
            "link_step_l1_norm": self.link_step_l1_norm,
            "od_steps": [item.to_metadata() for item in self.od_steps],
        }


class GradientProjectionAlgorithm(BaseRouteBasedSolver):
    """Route-based Gradient Projection solver.

    The solver performs one projected-gradient update in route-flow space. For
    each positive-demand OD pair, the solver takes a descent step using current
    route costs and projects the resulting vector onto the OD simplex. This
    preserves OD demand exactly while enforcing non-negative route flows.

    The behavior model remains responsible for:
    - computing link flows from current route flows;
    - computing VDF link costs;
    - computing route costs;
    - building the RouteBasedSolverStepRequest;
    - checking convergence.

    This class only computes the next feasible route-flow vector.
    """

    def __init__(self, config: GradientProjectionConfig) -> None:
        """Create a Gradient Projection solver from explicit configuration.

        Parameters
        ----------
        config:
            Strict Gradient Projection configuration created from
            assignment.yaml. The solver does not create or infer missing
            parameters.
        """
        if not isinstance(config, GradientProjectionConfig):
            raise TypeError("config must be a GradientProjectionConfig instance.")
        config.validate()
        self._config = config

    @property
    def name(self) -> str:
        """Return the canonical solver name used by assignment.yaml."""
        return "gradient_projection"

    @property
    def config(self) -> GradientProjectionConfig:
        """Return the immutable Gradient Projection configuration."""
        return self._config

    def compute_route_flow_update(
        self,
        request: RouteBasedSolverStepRequest,
    ) -> RouteBasedSolverStepResult:
        """Compute one projected-gradient route-flow update.

        Parameters
        ----------
        request:
            Fully validated route-based solver request.

        Returns
        -------
        RouteBasedSolverStepResult
            New feasible route-flow vector and solver diagnostics.
        """
        if not isinstance(request, RouteBasedSolverStepRequest):
            raise TypeError("request must be a RouteBasedSolverStepRequest instance.")
        request.validate()

        raw_step_size = self._compute_raw_step_size(iteration=request.iteration)
        self._validate_raw_step_size(raw_step_size=raw_step_size)

        new_route_flows, od_step_diagnostics = self._compute_projected_update(
            request=request,
            raw_step_size=raw_step_size,
        )
        self._validate_new_route_flows(new_route_flows=new_route_flows, request=request)

        new_link_flows = aggregate_route_flows_to_link_flows(
            route_flows=new_route_flows,
            route_set=request.route_set,
        )
        self._validate_implied_cost_inputs(request=request, new_link_flows=new_link_flows)

        effective_steps = [item.effective_step_size for item in od_step_diagnostics]
        if not effective_steps:
            raise ValueError("Gradient Projection produced no OD-level updates.")

        diagnostics = GradientProjectionStepDiagnostics(
            solver=self.name,
            iteration=request.iteration,
            step_rule=self.config.step_rule,
            projection_method=self.config.projection_method,
            cost_scaling=self.config.cost_scaling,
            base_step_size=float(self.config.base_step_size),
            raw_step_size=float(raw_step_size),
            minimum_effective_step_size=float(min(effective_steps)),
            maximum_effective_step_size=float(max(effective_steps)),
            current_total_route_flow=float(np.sum(request.current_route_flows)),
            new_total_route_flow=float(np.sum(new_route_flows)),
            current_total_link_flow=float(np.sum(request.current_link_flows)),
            new_total_link_flow=float(np.sum(new_link_flows)),
            route_step_l1_norm=float(np.linalg.norm(new_route_flows - request.current_route_flows, ord=1)),
            link_step_l1_norm=float(np.linalg.norm(new_link_flows - request.current_link_flows, ord=1)),
            od_steps=tuple(od_step_diagnostics),
        )

        result = RouteBasedSolverStepResult(
            new_route_flows=new_route_flows,
            metadata=diagnostics.to_metadata(),
        )
        result.validate(route_set=request.route_set, od_demands=request.od_demands)
        return result

    def _compute_raw_step_size(self, iteration: int) -> float:
        """Compute the iteration-level step size before cost scaling."""
        if not isinstance(iteration, int) or iteration < 1:
            raise ValueError(f"iteration must be an integer >= 1. Received {iteration!r}.")

        base_step_size = float(self.config.base_step_size)
        if self.config.step_rule == "constant":
            return base_step_size
        if self.config.step_rule == "1_over_n":
            return base_step_size / float(iteration)
        if self.config.step_rule == "sqrt":
            return base_step_size / float(np.sqrt(iteration))
        if self.config.step_rule == "scaled_sqrt":
            return base_step_size / float(np.sqrt(iteration))

        raise ValueError(
            "Unsupported Gradient Projection step_rule. This should have been rejected by "
            f"GradientProjectionConfig.validate(). Received {self.config.step_rule!r}."
        )

    def _compute_projected_update(
        self,
        request: RouteBasedSolverStepRequest,
        raw_step_size: float,
    ) -> tuple[np.ndarray, list[GradientProjectionODStepDiagnostics]]:
        """Apply projected-gradient updates OD by OD."""
        new_route_flows = np.zeros_like(request.current_route_flows, dtype=float)
        od_step_diagnostics: list[GradientProjectionODStepDiagnostics] = []

        for raw_od_pair, raw_demand in request.od_demands.items():
            origin_id = int(raw_od_pair[0])
            destination_id = int(raw_od_pair[1])
            demand = float(raw_demand)
            route_indices = request.route_set.get_route_indices_for_od(origin_id, destination_id)
            if not route_indices:
                raise ValueError(f"RouteSet has no routes for OD pair {(origin_id, destination_id)}.")

            current_od_route_flows = request.current_route_flows[route_indices]
            od_route_costs = request.current_route_costs[route_indices]
            validate_vector(
                name="current_od_route_flows",
                value=current_od_route_flows,
                expected_length=len(route_indices),
            )
            validate_vector(
                name="od_route_costs",
                value=od_route_costs,
                expected_length=len(route_indices),
            )
            validate_non_negative_vector(name="current_od_route_flows", value=current_od_route_flows)

            cost_scale = self._compute_cost_scale(route_costs=od_route_costs)
            effective_step_size = self._compute_effective_step_size(
                raw_step_size=raw_step_size,
                cost_scale=cost_scale,
            )
            tentative_route_flows = current_od_route_flows - effective_step_size * od_route_costs
            projected_route_flows = self._project_to_simplex(
                values=tentative_route_flows,
                simplex_sum=demand,
                demand_tolerance=float(self.config.demand_tolerance),
            )
            new_route_flows[route_indices] = projected_route_flows

            od_step_diagnostics.append(
                GradientProjectionODStepDiagnostics(
                    origin_id=origin_id,
                    destination_id=destination_id,
                    number_of_routes=len(route_indices),
                    demand=demand,
                    raw_step_size=float(raw_step_size),
                    cost_scale=float(cost_scale),
                    effective_step_size=float(effective_step_size),
                    minimum_route_cost=float(np.min(od_route_costs)),
                    maximum_route_cost=float(np.max(od_route_costs)),
                    current_assigned_demand=float(np.sum(current_od_route_flows)),
                    new_assigned_demand=float(np.sum(projected_route_flows)),
                    route_step_l1_norm=float(np.linalg.norm(projected_route_flows - current_od_route_flows, ord=1)),
                )
            )

        return new_route_flows, od_step_diagnostics

    def _compute_cost_scale(self, route_costs: np.ndarray) -> float:
        """Compute the OD-level cost normalizer for the configured policy."""
        validate_vector(name="route_costs", value=route_costs, expected_length=len(route_costs))
        if np.any(route_costs <= 0.0):
            raise ValueError("route_costs must be strictly positive for Gradient Projection.")

        if self.config.cost_scaling == "none":
            return 1.0
        if self.config.cost_scaling == "mean_abs_route_cost":
            scale = float(np.mean(np.abs(route_costs)))
            if not np.isfinite(scale) or scale <= 0.0:
                raise ValueError(f"Invalid mean_abs_route_cost scale={scale}.")
            return scale

        raise ValueError(
            "Unsupported Gradient Projection cost_scaling. This should have been rejected by "
            f"GradientProjectionConfig.validate(). Received {self.config.cost_scaling!r}."
        )

    def _compute_effective_step_size(self, raw_step_size: float, cost_scale: float) -> float:
        """Apply cost scaling and explicit clipping to the raw step size.

        The clipping bounds come from GradientProjectionConfig. This is not a
        silent correction of an invalid update; it is the configured numerical
        safeguard that keeps the projected-gradient step inside a controlled
        range before the OD-simplex projection is applied.
        """
        if not np.isfinite(raw_step_size) or raw_step_size < 0.0:
            raise ValueError(f"raw_step_size must be finite and non-negative. Received {raw_step_size}.")
        if not np.isfinite(cost_scale) or cost_scale <= 0.0:
            raise ValueError(f"cost_scale must be finite and positive. Received {cost_scale}.")

        scaled_step_size = float(raw_step_size) / float(cost_scale)
        clipped_step_size = min(
            max(scaled_step_size, float(self.config.minimum_step_size)),
            float(self.config.maximum_step_size),
        )
        if not np.isfinite(clipped_step_size):
            raise ValueError(f"Effective step size is not finite. Received {clipped_step_size}.")
        if clipped_step_size < 0.0 or clipped_step_size > 1.0:
            raise ValueError(f"Effective step size must be within [0, 1]. Received {clipped_step_size}.")
        return float(clipped_step_size)

    @staticmethod
    def _project_to_simplex(values: np.ndarray, simplex_sum: float, demand_tolerance: float) -> np.ndarray:
        """Project a vector onto the simplex {x >= 0, sum(x) = simplex_sum}.

        The projection is the Euclidean projection onto a non-negative simplex.
        It preserves the OD demand exactly up to a final numerical correction.
        The final ``np.maximum(..., 0.0)`` is only a floating-point cleanup after
        explicit validation has already rejected material negative values.
        """
        validate_vector(name="values", value=np.asarray(values, dtype=float), expected_length=len(values))
        if not np.isfinite(simplex_sum) or simplex_sum <= 0.0:
            raise ValueError(f"simplex_sum must be positive and finite. Received {simplex_sum}.")
        if not np.isfinite(demand_tolerance) or demand_tolerance <= 0.0:
            raise ValueError(f"demand_tolerance must be positive and finite. Received {demand_tolerance}.")

        values_array = np.asarray(values, dtype=float)
        if len(values_array) == 1:
            return np.array([float(simplex_sum)], dtype=float)

        sorted_values = np.sort(values_array)[::-1]
        cumulative_sum = np.cumsum(sorted_values)
        candidate_mask = sorted_values - (cumulative_sum - float(simplex_sum)) / (np.arange(len(values_array)) + 1) > 0.0

        if not np.any(candidate_mask):
            projected = np.full(len(values_array), float(simplex_sum) / float(len(values_array)), dtype=float)
        else:
            rho = int(np.nonzero(candidate_mask)[0][-1])
            theta = (float(cumulative_sum[rho]) - float(simplex_sum)) / float(rho + 1)
            projected = np.maximum(values_array - theta, 0.0)

        projected_sum = float(np.sum(projected))
        if projected_sum <= 0.0 or not np.isfinite(projected_sum):
            projected = np.full(len(values_array), float(simplex_sum) / float(len(values_array)), dtype=float)
            projected_sum = float(np.sum(projected))

        projected *= float(simplex_sum) / projected_sum
        if not np.all(np.isfinite(projected)):
            raise ValueError("Simplex projection produced NaN or infinite values.")
        if np.any(projected < -1e-10):
            raise ValueError("Simplex projection produced negative values.")
        if not np.isclose(float(np.sum(projected)), float(simplex_sum), rtol=1e-8, atol=demand_tolerance):
            raise ValueError(
                "Simplex projection failed to preserve demand. "
                f"Projected sum={float(np.sum(projected))}, demand={float(simplex_sum)}."
            )
        return np.maximum(projected.astype(float), 0.0)

    @staticmethod
    def _validate_raw_step_size(raw_step_size: float) -> None:
        """Validate the iteration-level raw step size."""
        if not np.isfinite(raw_step_size):
            raise ValueError(f"raw_step_size must be finite. Received {raw_step_size!r}.")
        if raw_step_size < 0.0:
            raise ValueError(f"raw_step_size must be non-negative. Received {raw_step_size}.")

    @staticmethod
    def _validate_new_route_flows(
        new_route_flows: np.ndarray,
        request: RouteBasedSolverStepRequest,
    ) -> None:
        """Validate feasibility and OD-demand conservation of the projected update."""
        validate_vector(
            name="new_route_flows",
            value=new_route_flows,
            expected_length=request.route_set.number_of_routes,
        )
        validate_non_negative_vector(name="new_route_flows", value=new_route_flows)
        validate_route_flow_demand_conservation(
            route_flows=new_route_flows,
            od_demands=request.od_demands,
            route_set=request.route_set,
            vector_name="new_route_flows",
        )

    @staticmethod
    def _validate_implied_cost_inputs(
        request: RouteBasedSolverStepRequest,
        new_link_flows: np.ndarray,
    ) -> None:
        """Validate route-link consistency after the projected route-flow update.

        The solver does not recompute link costs because VDF computation belongs
        to the behavior model or a dedicated VDF object. This check only ensures
        that the route set can map the new route solution back to link space and
        that the current cost vectors remain internally consistent.
        """
        validate_vector(
            name="new_link_flows",
            value=new_link_flows,
            expected_length=request.route_set.number_of_links,
        )
        validate_non_negative_vector(name="new_link_flows", value=new_link_flows)

        expected_current_route_costs = compute_route_costs_from_link_costs(
            link_costs=request.current_link_costs,
            route_set=request.route_set,
        )
        if not np.allclose(request.current_route_costs, expected_current_route_costs, rtol=1e-8, atol=1e-8):
            raise ValueError("current_route_costs is not consistent with current_link_costs and RouteSet.")


__all__ = [
    "GradientProjectionAlgorithm",
    "GradientProjectionODStepDiagnostics",
    "GradientProjectionStepDiagnostics",
]