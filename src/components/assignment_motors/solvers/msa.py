"""Route-based Method of Successive Averages solver.

This module implements MSA as a numerical route-flow update algorithm. It does
not represent a behavioral assignment model. Behavioral assumptions belong to
models such as deterministic User Equilibrium (UE) or Stochastic User
Equilibrium (SUE). MSA only decides how much of the auxiliary route-flow
solution should be mixed into the current route-flow solution at each
iteration.

Architectural rules
-------------------
1. MSA updates route flows, not link flows.
2. Link flows are used only for diagnostics and consistency checks.
3. No silent defaults are used; every parameter comes from MSAConfig.
4. The public API consumes and returns the strict contracts defined in base.py.
5. The solver validates both the input request and the output result before
   returning anything to the behavior model.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Mapping

import numpy as np

from ..assignment_configs import MSAConfig
from ..base import (
    BaseRouteBasedSolver,
    RouteBasedSolverStepRequest,
    RouteBasedSolverStepResult,
)


@dataclass(frozen=True)
class MSAStepDiagnostics:
    """Diagnostic information produced by one route-based MSA update.

    Attributes
    ----------
    iteration:
        One-based assignment iteration used to compute the step size.
    step_rule:
        Canonical MSA step-size rule selected in assignment.yaml.
    step_size:
        Convex-combination weight used in route-flow space.
    current_total_route_flow:
        Sum of the current route-flow vector before the update.
    auxiliary_total_route_flow:
        Sum of the auxiliary route-flow vector produced by the behavior model.
    new_total_route_flow:
        Sum of the updated route-flow vector after the MSA step.
    current_total_link_flow:
        Sum of the link-flow vector implied by current_route_flows.
    auxiliary_total_link_flow:
        Sum of the link-flow vector implied by auxiliary_route_flows.
    route_direction_l1_norm:
        L1 norm of the route-space direction
        auxiliary_route_flows - current_route_flows.
    link_direction_l1_norm:
        L1 norm of the implied link-space direction
        auxiliary_link_flows - current_link_flows.
    route_step_l1_norm:
        L1 norm of the route-flow update actually applied.
    """

    iteration: int
    step_rule: str
    step_size: float
    harmonic_step_size: float
    adaptive_step_size: float
    gap_value: float
    previous_gap_value: float
    gap_improved: bool
    current_total_route_flow: float
    auxiliary_total_route_flow: float
    new_total_route_flow: float
    current_total_link_flow: float
    auxiliary_total_link_flow: float
    route_direction_l1_norm: float
    link_direction_l1_norm: float
    route_step_l1_norm: float

    def as_metadata(self) -> Mapping[str, float | int | str]:
        """Return diagnostics as a simple metadata mapping.

        The project metadata layer currently uses dictionaries. This method
        keeps the solver internals typed while still returning a simple mapping
        through RouteBasedSolverStepResult.
        """
        return {
            "solver": "msa",
            "iteration": self.iteration,
            "step_rule": self.step_rule,
            "step_size": self.step_size,
            "harmonic_step_size": self.harmonic_step_size,
            "adaptive_step_size": self.adaptive_step_size,
            "gap_value": self.gap_value,
            "previous_gap_value": self.previous_gap_value,
            "gap_improved": self.gap_improved,
            "current_total_route_flow": self.current_total_route_flow,
            "auxiliary_total_route_flow": self.auxiliary_total_route_flow,
            "new_total_route_flow": self.new_total_route_flow,
            "current_total_link_flow": self.current_total_link_flow,
            "auxiliary_total_link_flow": self.auxiliary_total_link_flow,
            "route_direction_l1_norm": self.route_direction_l1_norm,
            "link_direction_l1_norm": self.link_direction_l1_norm,
            "route_step_l1_norm": self.route_step_l1_norm,
        }


class MSAEquilibriumAlgorithm(BaseRouteBasedSolver):
    """Method of Successive Averages route-flow solver.

    MSA computes a scalar step size and applies the convex update directly in
    route-flow space:

        new_route_flows = current_route_flows
            + step_size * (auxiliary_route_flows - current_route_flows)

    Because both current_route_flows and auxiliary_route_flows must conserve OD
    demand before they enter this solver, the convex update also conserves OD
    demand. The solver still validates the resulting RouteBasedSolverStepResult
    using the provided RouteSet and od_demands before returning it.
    """

    _CANONICAL_NAME = "msa"

    def __init__(self, config: MSAConfig) -> None:
        """Create an MSA solver from an explicit validated configuration.

        Parameters
        ----------
        config:
            Strict MSA solver configuration created by assignment_configs.py
            from assignment.yaml. No fallback step rule is allowed.
        """
        self._validate_constructor_config(config)
        self.config = config

        self._previous_gap: float | None = None
        self._adaptive_step_size: float | None = (
            None if config.initial_step_size is None else float(config.initial_step_size)
        )

        logging.info(f"Initialized MSAEquilibriumAlgorithm with config: {config.step_rule}")

    @property
    def name(self) -> str:
        """Return the canonical solver name used in configs and reports."""
        return self._CANONICAL_NAME

    def compute_route_flow_update(
        self,
        request: RouteBasedSolverStepRequest,
    ) -> RouteBasedSolverStepResult:
        """Compute one route-flow MSA update.

        Parameters
        ----------
        request:
            Fully validated route-based solver request produced by a behavior
            model. The request contains both route-flow vectors and their
            implied link-flow vectors. MSA updates the route-flow vector; link
            flows are only used for diagnostics and consistency validation.

        Returns
        -------
        RouteBasedSolverStepResult
            Updated feasible route-flow vector plus diagnostic metadata.
        """
        self._validate_step_request(request)

        step_size, step_metadata = self._compute_step_size(request=request)
        self._validate_step_size(step_size)

        route_direction = request.auxiliary_route_flows - request.current_route_flows
        link_direction = request.auxiliary_link_flows - request.current_link_flows
        new_route_flows = request.current_route_flows + float(step_size) * route_direction

        diagnostics = MSAStepDiagnostics(
            iteration=request.iteration,
            step_rule=self.config.step_rule,
            step_size=float(step_size),
            current_total_route_flow=float(np.sum(request.current_route_flows)),
            auxiliary_total_route_flow=float(np.sum(request.auxiliary_route_flows)),
            new_total_route_flow=float(np.sum(new_route_flows)),
            current_total_link_flow=float(np.sum(request.current_link_flows)),
            auxiliary_total_link_flow=float(np.sum(request.auxiliary_link_flows)),
            route_direction_l1_norm=float(np.linalg.norm(route_direction, ord=1)),
            link_direction_l1_norm=float(np.linalg.norm(link_direction, ord=1)),
            route_step_l1_norm=float(np.linalg.norm(new_route_flows - request.current_route_flows, ord=1)),
            harmonic_step_size=float(step_metadata["harmonic_step_size"]),
            adaptive_step_size=float(step_metadata["adaptive_step_size"]),
            gap_value=float(step_metadata["gap_value"]),
            previous_gap_value=float(step_metadata["previous_gap_value"]),
            gap_improved=bool(step_metadata["gap_improved"]),
        )

        result = RouteBasedSolverStepResult(
            new_route_flows=np.asarray(new_route_flows, dtype=float),
            metadata=diagnostics.as_metadata(),
        )
        result.validate(
            route_set=request.route_set,
            od_demands=request.od_demands,
        )
        return result

    @staticmethod
    def _validate_constructor_config(config: MSAConfig) -> None:
        """Validate constructor configuration without adding missing values."""
        if not isinstance(config, MSAConfig):
            raise TypeError("config must be an MSAConfig instance.")
        config.validate()

    @staticmethod
    def _validate_step_request(request: RouteBasedSolverStepRequest) -> None:
        """Validate the incoming update request before using it."""
        if not isinstance(request, RouteBasedSolverStepRequest):
            raise TypeError("request must be a RouteBasedSolverStepRequest instance.")
        request.validate()

    @staticmethod
    def _compute_rule_based_step_size(iteration: int, step_rule: str) -> float:
        """Compute the step size associated with the configured MSA rule.

        Supported rules
        ---------------
        - ``1_over_n``: ``step_size = 1 / iteration``.
        - ``1_over_n_plus_1``: ``step_size = 1 / (iteration + 1)``.

        The supported set mirrors MSAConfig validation. The explicit branch is
        still kept here so the solver fails safely if the config contract is
        extended without updating this implementation.
        """
        if not isinstance(iteration, int) or iteration < 1:
            raise ValueError(f"iteration must be an integer >= 1. Received {iteration!r}.")
        if step_rule == "1_over_n":
            return 1.0 / float(iteration)
        if step_rule == "1_over_n_plus_1":
            return 1.0 / float(iteration + 1)
        raise ValueError(f"Unsupported MSA step_rule={step_rule!r}.")

    @staticmethod
    def _validate_step_size(step_size: float) -> None:
        """Validate that the computed step size is a convex-combination weight."""
        if not np.isfinite(step_size):
            raise ValueError(f"MSA produced a non-finite step_size={step_size}.")
        if step_size < 0.0 or step_size > 1.0:
            raise ValueError(f"MSA step_size must be within [0, 1]. Received {step_size}.")


    def _compute_step_size(
        self,
        request: RouteBasedSolverStepRequest,
    ) -> tuple[float, dict[str, float | bool]]:
        """Compute the configured MSA step size.

        Supported policies
        ------------------
        - 1_over_n:
            Classical harmonic MSA step.
        - 1_over_n_plus_1:
            Slightly more conservative harmonic MSA step.
        - adaptive_gap:
            Uses only the adaptive gap-based step.
        - hybrid_adaptive_gap:
            Combines harmonic MSA and adaptive gap-based step.
        """
        harmonic_step = self._compute_rule_based_step_size(
            iteration=request.iteration,
            step_rule="1_over_n",
        )

        if self.config.step_rule in {"1_over_n", "1_over_n_plus_1"}:
            step_size = self._compute_rule_based_step_size(
                iteration=request.iteration,
                step_rule=self.config.step_rule,
            )
            return float(step_size), {
                "harmonic_step_size": float(step_size),
                "adaptive_step_size": float("nan"),
                "gap_value": float("nan"),
                "previous_gap_value": float("nan"),
                "gap_improved": False,
            }

        adaptive_step, adaptive_metadata = self._compute_adaptive_gap_step_size(
            request=request,
        )

        if self.config.step_rule == "adaptive_gap":
            return float(adaptive_step), {
                "harmonic_step_size": float(harmonic_step),
                "adaptive_step_size": float(adaptive_step),
                **adaptive_metadata,
            }

        if self.config.step_rule == "hybrid_adaptive_gap":
            hybrid_weight = self._required_config_float("hybrid_harmonic_weight")

            # The harmonic component prevents the adaptive rule from becoming too
            # aggressive, while the adaptive component prevents late-stage harmonic
            # steps from becoming unnecessarily tiny.
            candidate_step = max(
                float(harmonic_step) * float(hybrid_weight),
                float(adaptive_step),
            )

            maximum_step_size = self._required_config_float("maximum_step_size")
            minimum_step_size = self._required_config_float("minimum_step_size")
            step_size = min(
                max(candidate_step, minimum_step_size),
                maximum_step_size,
            )

            return float(step_size), {
                "harmonic_step_size": float(harmonic_step),
                "adaptive_step_size": float(adaptive_step),
                **adaptive_metadata,
            }

        raise ValueError(f"Unsupported MSA step_rule={self.config.step_rule!r}.")


    def _compute_adaptive_gap_step_size(
        self,
        request: RouteBasedSolverStepRequest,
    ) -> tuple[float, dict[str, float | bool]]:
        """Compute a gap-aware adaptive MSA step size.

        The behavior model must provide request.metadata['current_gap'].
        For SUE, this should be sue_fixed_point_gap_l1.
        """
        current_gap = self._read_current_gap(request=request)

        initial_step_size = self._required_config_float("initial_step_size")
        minimum_step_size = self._required_config_float("minimum_step_size")
        maximum_step_size = self._required_config_float("maximum_step_size")
        improvement_tolerance = self._required_config_float("improvement_tolerance")
        increase_factor = self._required_config_float("increase_factor")
        decrease_factor = self._required_config_float("decrease_factor")

        previous_gap = self._previous_gap

        if self._adaptive_step_size is None:
            self._adaptive_step_size = initial_step_size

        if previous_gap is None:
            gap_improved = True
            new_step = self._adaptive_step_size
        else:
            gap_improved = current_gap < previous_gap * (1.0 - improvement_tolerance)

            if gap_improved:
                new_step = self._adaptive_step_size * increase_factor
            else:
                new_step = self._adaptive_step_size * decrease_factor

        new_step = min(max(float(new_step), minimum_step_size), maximum_step_size)

        self._adaptive_step_size = float(new_step)
        self._previous_gap = float(current_gap)

        return float(new_step), {
            "gap_value": float(current_gap),
            "previous_gap_value": float("nan") if previous_gap is None else float(previous_gap),
            "gap_improved": bool(gap_improved),
        }


    @staticmethod
    def _read_current_gap(request: RouteBasedSolverStepRequest) -> float:
        """Read current gap from request metadata without silent fallback."""
        if "current_gap" not in request.metadata:
            raise ValueError(
                "MSA step_rule requires request.metadata['current_gap']. "
                "For SUE, pass sue_fixed_point_gap_l1 from sue_motor.py."
            )

        gap = float(request.metadata["current_gap"])
        if not np.isfinite(gap) or gap < 0.0:
            raise ValueError(
                "request.metadata['current_gap'] must be finite and non-negative. "
                f"Received {gap!r}."
            )
        return gap


    def _required_config_float(self, field_name: str) -> float:
        """Return a required adaptive MSA config field as float."""
        value = getattr(self.config, field_name)
        if value is None:
            raise ValueError(
                f"MSA step_rule={self.config.step_rule!r} requires config.{field_name}."
            )

        value_float = float(value)
        if not np.isfinite(value_float):
            raise ValueError(f"config.{field_name} must be finite. Received {value!r}.")
        return value_float

__all__ = [
    "MSAEquilibriumAlgorithm",
    "MSAStepDiagnostics",
]