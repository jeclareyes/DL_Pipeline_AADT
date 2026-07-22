"""Public composition surface for full route-based traffic assignment.

This package exposes the canonical objects used by the full route-based
assignment architecture. The module is intentionally a composition layer: it
wires validated configuration objects to concrete domain objects, solvers and
behavior models.

Architectural responsibilities
------------------------------
- assignment_configs.py validates assignment.yaml into explicit dataclasses and
  owns the behavior-model/solver compatibility policy.
- route_set.py prepares and validates fixed route alternatives.
- behavior models create auxiliary route-flow solutions.
- solvers update route-flow vectors.
- VDF objects will later own link-cost and integral calculations.

This module is a composition layer: it wires validated configs to concrete
objects. It should be the only place where high-level construction decisions
are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .assignment_configs import (
    AssignmentCommonConfig,
    AssignmentConfig,
    BehaviorModelName,
    BehaviorModelSelectionConfig,
    BehaviorModelSource,
    CapacityScalingConfig,
    CapacityScalingSource,
    FrankWolfeConfig,
    GradientProjectionConfig,
    MSAConfig,
    RouteBasedUESolveConfig,
    SUEConvergenceConfig,
    SUELogitConfig,
    SUEPolicyConfig,
    SUESolveConfig,
    SolverName,
    SolversConfig,
    allowed_solvers_for_behavior_model,
    build_assignment_config_from_mapping,
    load_assignment_config_from_yaml,
    validate_behavior_solver_compatibility,
)
from .base import (
    AssignmentResult,
    BaseRouteBasedAssignmentModel,
    BaseRouteBasedSolver,
    RouteBasedSolverStepRequest,
    RouteBasedSolverStepResult,
    aggregate_route_flows_to_link_flows,
    compute_route_costs_from_link_costs,
    validate_od_demands,
    validate_route_flow_demand_conservation,
    validate_vector,
)
from .assignment_models.sue_motor import (
    RouteBasedSUEPolicyConfig,
    RouteBasedStochasticUserEquilibriumModel,
    RouteBasedStochasticUserEquilibriumRuntimeConfig,
    SUEConvergenceConfig as SUEConvergenceRuntimeConfig,
    SUELogitConfig as SUELogitRuntimeConfig,
)
from .assignment_models.ue_motor import (
    RouteBasedUserEquilibriumModel,
    RouteBasedUserEquilibriumRuntimeConfig,
)
from .route_set import RouteInputFormat, RouteSet, RouteSetBuildConfig
from .solvers.frank_wolfe import FrankWolfeLineSearchAlgorithm
from .solvers.gradient_projection import GradientProjectionAlgorithm
from .solvers.msa import MSAEquilibriumAlgorithm
from src.components.vdf.config import VDFConfig


RouteBasedRuntimeConfig = (
    RouteBasedUserEquilibriumRuntimeConfig
    | RouteBasedStochasticUserEquilibriumRuntimeConfig
)
RouteBasedBehaviorModel = (
    RouteBasedUserEquilibriumModel
    | RouteBasedStochasticUserEquilibriumModel
)


@dataclass(frozen=True)
class AssignmentComposition:
    """Fully wired assignment objects for one route-based assignment run.

    Attributes
    ----------
    assignment_config:
        Validated top-level assignment configuration.
    behavior_model_name:
        Canonical behavior-model name selected after resolving the behavior
        source policy.
    route_set:
        Prepared route representation aligned with links_df.
    solver:
        Concrete route-based solver selected by assignment_config.solvers.
    behavior_model:
        Concrete route-based behavior model selected by behavior_model_name.
    runtime_config:
        Runtime config object expected by behavior_model.solve(...).
    """

    assignment_config: AssignmentConfig
    behavior_model_name: BehaviorModelName
    route_set: RouteSet
    solver: BaseRouteBasedSolver
    behavior_model: RouteBasedBehaviorModel
    runtime_config: RouteBasedRuntimeConfig

    def validate(self) -> None:
        """Validate that the composition is internally consistent."""
        if not isinstance(self.assignment_config, AssignmentConfig):
            raise TypeError("assignment_config must be an AssignmentConfig instance.")
        if not isinstance(self.behavior_model_name, BehaviorModelName):
            raise TypeError("behavior_model_name must be a BehaviorModelName value.")
        if not isinstance(self.route_set, RouteSet):
            raise TypeError("route_set must be a RouteSet instance.")
        if not isinstance(self.solver, BaseRouteBasedSolver):
            raise TypeError("solver must be a BaseRouteBasedSolver instance.")
        if not isinstance(self.behavior_model, BaseRouteBasedAssignmentModel):
            raise TypeError("behavior_model must be a BaseRouteBasedAssignmentModel instance.")
        if not isinstance(
            self.runtime_config,
            (RouteBasedUserEquilibriumRuntimeConfig, RouteBasedStochasticUserEquilibriumRuntimeConfig),
        ):
            raise TypeError("runtime_config has an unsupported type.")

        self.assignment_config.validate()
        self.route_set.require_same_link_order(self.behavior_model.links_df)

        if self.solver.name != self.assignment_config.solvers.active_solver.value:
            raise ValueError(
                "Injected solver name does not match assignment_config.solvers.active_solver. "
                f"solver.name={self.solver.name!r}, "
                f"active_solver={self.assignment_config.solvers.active_solver.value!r}."
            )

        validate_behavior_solver_compatibility(
            behavior_model=self.behavior_model_name,
            active_solver=self.assignment_config.solvers.active_solver,
            context="assignment_config"
        )


def build_route_set(
    links_df: pd.DataFrame,
    routes_by_od: Mapping[tuple[int, int], Sequence[Sequence[int]]],
    assignment_config: AssignmentConfig,
) -> RouteSet:
    """Build a RouteSet using the route-set section of AssignmentConfig.

    Parameters
    ----------
    links_df:
        Canonical directed link table. Its row order becomes the link-flow
        order used throughout assignment.
    routes_by_od:
        Raw route dictionary keyed by (origin_id, destination_id).
    assignment_config:
        Validated assignment configuration containing RouteSetBuildConfig.
    """
    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError("assignment_config must be an AssignmentConfig instance.")
    assignment_config.validate()
    return RouteSet.from_routes_by_od(
        links_df=links_df,
        routes_by_od=routes_by_od,
        config=assignment_config.route_set,
    )


def build_route_based_solver(assignment_config: AssignmentConfig) -> BaseRouteBasedSolver:
    """Build the active route-based solver from AssignmentConfig.

    The function accepts only canonical SolverName values already validated by
    assignment_configs.py. It does not accept aliases, misspellings, dictionaries,
    or optional fallback values.
    """
    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError("assignment_config must be an AssignmentConfig instance.")
    assignment_config.validate()

    active_solver = assignment_config.solvers.active_solver
    if active_solver == SolverName.MSA:
        return MSAEquilibriumAlgorithm(config=assignment_config.solvers.msa)
    if active_solver == SolverName.FRANK_WOLFE:
        return FrankWolfeLineSearchAlgorithm(
            config=assignment_config.solvers.frank_wolfe,
            cost_config=assignment_config.cost_function,
        )
    if active_solver == SolverName.GRADIENT_PROJECTION:
        return GradientProjectionAlgorithm(config=assignment_config.solvers.gradient_projection)

    raise ValueError(f"Unsupported route-based solver: {active_solver!r}.")


def resolve_behavior_model_name(
    assignment_config: AssignmentConfig,
    training_config: Mapping[str, Any],
) -> BehaviorModelName:
    """Resolve the active behavior model without duplicating compatibility rules.

    The behavior-model selection itself is delegated to
    BehaviorModelSelectionConfig. The model/solver compatibility check is
    delegated to assignment_configs.py, which is the single source of truth.
    """
    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError("assignment_config must be an AssignmentConfig instance.")
    if not isinstance(training_config, Mapping):
        raise TypeError("training_config must be a mapping.")

    assignment_config.validate()

    behavior_model_name = assignment_config.behavior_model.resolve(
        training_config=training_config)

    validate_behavior_solver_compatibility(
        behavior_model=behavior_model_name,
        active_solver=assignment_config.solvers.active_solver,
        context="assignment_config"
    )
    return behavior_model_name


def build_runtime_config(
    assignment_config: AssignmentConfig,
    behavior_model_name: BehaviorModelName,
    training_config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> RouteBasedRuntimeConfig:
    """Build the runtime config required by the selected behavior model.

    This function is the only place where assignment-level configuration is
    translated into behavior-model runtime objects. In particular, SUE theta is
    resolved here so that sue_motor.py receives a numeric theta and never reads
    artifacts directly.

    Parameters
    ----------
    assignment_config:
        Validated top-level assignment configuration.
    behavior_model_name:
        Resolved canonical behavior model.
    training_config:
        Training configuration mapping. It is required explicitly even when
        assignment.yaml does not read from it.
    artifacts:
        Artifact mapping used by SUE when theta_source='artifact'. It is
        required explicitly even when SUE uses an explicit theta.
    """
    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError("assignment_config must be an AssignmentConfig instance.")
    if not isinstance(behavior_model_name, BehaviorModelName):
        raise TypeError("behavior_model_name must be a BehaviorModelName value.")
    if not isinstance(training_config, Mapping):
        raise TypeError("training_config must be a mapping.")
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping.")

    assignment_config.validate()
    validate_behavior_solver_compatibility(
        behavior_model=behavior_model_name,
        active_solver=assignment_config.solvers.active_solver,
        context="build_runtime_config",
    )

    capacity_scaling_factor = assignment_config.common.capacity_scaling.resolve(
        training_config=training_config,
    )

    if behavior_model_name == BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM:
        if assignment_config.common.capacity_scaling.source == CapacityScalingSource.TRAINING_CONFIG:
            assignment_config.common.capacity_scaling.resolve(training_config=training_config)
            raise ValueError(
                "RouteBasedUserEquilibriumRuntimeConfig currently does not carry training_config, "
                "but common.capacity_scaling.source='training_config' requires it. "
                "Use source='explicit' for UE until the UE runtime config is extended."
            )
        capacity_scaling_factor = assignment_config.common.capacity_scaling.resolve(training_config=training_config)
        runtime_config = RouteBasedUserEquilibriumRuntimeConfig(
            max_iterations=assignment_config.common.max_iterations,
            capacity_scaling_factor=capacity_scaling_factor,
            cost_function=assignment_config.cost_function,
            convergence=assignment_config.route_based_user_equilibrium.convergence,
            initialization=assignment_config.route_based_user_equilibrium.initialization,
            policy=assignment_config.route_based_user_equilibrium.policy,
        )
        runtime_config.validate()
        return runtime_config

    if behavior_model_name == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM:
        sue_config = assignment_config.stochastic_user_equilibrium
        theta = sue_config.logit.resolve_theta(artifacts=artifacts)

        runtime_convergence = SUEConvergenceRuntimeConfig(
            equilibrium_l1_threshold=sue_config.convergence.equilibrium_l1_threshold,
            max_absolute_gap_threshold=sue_config.convergence.max_absolute_gap_threshold,
            max_relative_gap_threshold=sue_config.convergence.max_relative_gap_threshold,
            min_flow_for_relative_gap=sue_config.convergence.min_flow_for_relative_gap,
            minimum_iterations=sue_config.convergence.minimum_iterations,
        )
        runtime_logit = SUELogitRuntimeConfig(
            theta=theta,
            fail_on_invalid_probability=sue_config.logit.fail_on_invalid_probability,
        )
        runtime_policy = RouteBasedSUEPolicyConfig(
            fail_on_missing_routes=sue_config.policy.fail_on_missing_routes,
            fail_on_skipped_od_pairs=sue_config.policy.fail_on_skipped_od_pairs,
            intrazonal_policy=sue_config.policy.intrazonal_policy,
            expected_solver=sue_config.policy.expected_solver.value,
        )

        runtime_config = RouteBasedStochasticUserEquilibriumRuntimeConfig(
            max_iterations=assignment_config.common.max_iterations,
            capacity_scaling_factor=capacity_scaling_factor,
            cost_function=assignment_config.cost_function,
            convergence=runtime_convergence,
            logit=runtime_logit,
            policy=runtime_policy,
        )
        runtime_config.validate()
        return runtime_config

    raise ValueError(
        "Unsupported behavior model for this route-based composition layer. "
        f"Received behavior_model_name={behavior_model_name.value!r}. "
        f"Allowed route-based models are "
        f"{BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM.value!r} and "
        f"{BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value!r}."
    )


def build_behavior_model(
    links_df: pd.DataFrame,
    route_set: RouteSet,
    solver: BaseRouteBasedSolver,
    zone_id_to_idx: Mapping[int, int],
    behavior_model_name: BehaviorModelName,
) -> RouteBasedBehaviorModel:
    """Build the selected route-based behavior model from explicit collaborators.

    This function wires already-created collaborators. It does not create a
    RouteSet, select solvers, parse YAML, or read artifacts.
    """
    if not isinstance(route_set, RouteSet):
        raise TypeError("route_set must be a RouteSet instance.")
    if not isinstance(solver, BaseRouteBasedSolver):
        raise TypeError("solver must be a BaseRouteBasedSolver instance.")
    if not isinstance(behavior_model_name, BehaviorModelName):
        raise TypeError("behavior_model_name must be a BehaviorModelName value.")

    if behavior_model_name == BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM:
        return RouteBasedUserEquilibriumModel(
            links_df=links_df,
            route_set=route_set,
            solver=solver,
            zone_id_to_idx=zone_id_to_idx,
        )
    if behavior_model_name == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM:
        return RouteBasedStochasticUserEquilibriumModel(
            links_df=links_df,
            route_set=route_set,
            solver=solver,
            zone_id_to_idx=zone_id_to_idx,
        )

    raise ValueError(
        "Unsupported behavior model for route-based assignment composition. "
        f"Received {behavior_model_name.value!r}."
    )


def build_assignment_composition(
    links_df: pd.DataFrame,
    routes_by_od: Mapping[tuple[int, int], Sequence[Sequence[int]]],
    zone_id_to_idx: Mapping[int, int],
    assignment_config: AssignmentConfig,
    training_config: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> AssignmentComposition:
    """Build all collaborators needed for one full route-based assignment run.

    This is the highest-level factory in the package. It remains strict:
    callers must provide every required object explicitly, including training
    config and artifacts mappings even when the selected YAML options do not use
    them.
    """
    if not isinstance(assignment_config, AssignmentConfig):
        raise TypeError("assignment_config must be an AssignmentConfig instance.")
    if not isinstance(training_config, Mapping):
        raise TypeError("training_config must be a mapping.")
    if not isinstance(artifacts, Mapping):
        raise TypeError("artifacts must be a mapping.")

    assignment_config.validate()
    behavior_model_name = resolve_behavior_model_name(
        assignment_config=assignment_config,
        training_config=training_config,
    )
    route_set = build_route_set(
        links_df=links_df,
        routes_by_od=routes_by_od,
        assignment_config=assignment_config,
    )
    solver = build_route_based_solver(assignment_config=assignment_config)
    runtime_config = build_runtime_config(
        assignment_config=assignment_config,
        behavior_model_name=behavior_model_name,
        training_config=training_config,
        artifacts=artifacts,
    )
    behavior_model = build_behavior_model(
        links_df=links_df,
        route_set=route_set,
        solver=solver,
        zone_id_to_idx=zone_id_to_idx,
        behavior_model_name=behavior_model_name,
    )

    composition = AssignmentComposition(
        assignment_config=assignment_config,
        behavior_model_name=behavior_model_name,
        route_set=route_set,
        solver=solver,
        behavior_model=behavior_model,
        runtime_config=runtime_config,
    )
    composition.validate()
    return composition


__all__ = [
    "AssignmentCommonConfig",
    "AssignmentComposition",
    "AssignmentConfig",
    "AssignmentResult",
    "BaseRouteBasedAssignmentModel",
    "BaseRouteBasedSolver",
    "BehaviorModelName",
    "BehaviorModelSelectionConfig",
    "BehaviorModelSource",
    "CapacityScalingConfig",
    "CapacityScalingSource",
    "FrankWolfeConfig",
    "FrankWolfeLineSearchAlgorithm",
    "GradientProjectionAlgorithm",
    "GradientProjectionConfig",
    "MSAConfig",
    "MSAEquilibriumAlgorithm",
    "RouteBasedRuntimeConfig",
    "RouteBasedBehaviorModel",
    "RouteBasedSolverStepRequest",
    "RouteBasedSolverStepResult",
    "RouteBasedStochasticUserEquilibriumModel",
    "RouteBasedStochasticUserEquilibriumRuntimeConfig",
    "RouteBasedSUEPolicyConfig",
    "RouteBasedUESolveConfig",
    "RouteBasedUserEquilibriumModel",
    "RouteBasedUserEquilibriumRuntimeConfig",
    "RouteInputFormat",
    "RouteSet",
    "RouteSetBuildConfig",
    "SUEConvergenceConfig",
    "SUELogitConfig",
    "SUEPolicyConfig",
    "SUESolveConfig",
    "SolverName",
    "SolversConfig",
    "VDFConfig",
    "aggregate_route_flows_to_link_flows",
    "allowed_solvers_for_behavior_model",
    "build_assignment_composition",
    "build_assignment_config_from_mapping",
    "build_behavior_model",
    "build_route_based_solver",
    "build_route_set",
    "build_runtime_config",
    "compute_route_costs_from_link_costs",
    "load_assignment_config_from_yaml",
    "resolve_behavior_model_name",
    "validate_behavior_solver_compatibility",
    "validate_od_demands",
    "validate_route_flow_demand_conservation",
    "validate_vector",
]
