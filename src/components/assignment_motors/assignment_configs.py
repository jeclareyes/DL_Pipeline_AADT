"""Strict assignment configuration contracts and YAML resolver.

This module is the boundary between Hydra/OmegaConf YAML configuration and the
traffic-assignment domain objects used by behavior models and solvers.

Design principles
-----------------
1. No silent defaults: every field must be present in assignment.yaml.
2. No loose dictionaries inside models: YAML is converted into typed dataclasses.
3. No automatic typo correction: invalid names raise explicit errors.
4. No unknown keys: unexpected YAML keys raise explicit errors.
5. Clear conceptual separation:
   - behavioral assignment models: UE/SUE variants;
   - numerical solvers: MSA, Frank-Wolfe, Gradient Projection;
   - route-domain objects: RouteSet;
   - volume-delay functions: VDF configurations, currently BPR.

The concrete behavior models and solvers should receive the dataclasses defined
here instead of reading raw YAML/OmegaConf objects directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .route_set import RouteInputFormat, RouteSetBuildConfig
from src.components.vdf.config import VDFConfig


# =============================================================================
# Canonical names
# =============================================================================


class BehaviorModelName(str, Enum):
    """Canonical behavioral assignment model names supported by the project."""

    GRAPH_BASED_USER_EQUILIBRIUM = "graph_based_user_equilibrium"
    ROUTE_BASED_USER_EQUILIBRIUM = "route_based_user_equilibrium"
    STOCHASTIC_USER_EQUILIBRIUM = "stochastic_user_equilibrium"


class BehaviorModelSource(str, Enum):
    """Where the behavioral model selection comes from."""

    EXPLICIT = "explicit"
    TRAINING_CONFIG = "training_config"


class SolverName(str, Enum):
    """Canonical numerical solver names supported by the assignment layer."""

    MSA = "msa"
    FRANK_WOLFE = "frank_wolfe"
    GRADIENT_PROJECTION = "gradient_projection"


class CapacityScalingSource(str, Enum):
    """Source policy for the capacity scaling factor."""

    EXPLICIT = "explicit"
    TRAINING_CONFIG = "training_config"


# =============================================================================
# Central behavior-model / solver compatibility policy
# =============================================================================


def _allowed_solvers_for_behavior_model(behavior_model: BehaviorModelName) -> frozenset[SolverName]:
    """Return the canonical solver set allowed for one behavior model.

    This is the single source of truth for model-solver compatibility.
    Composition code should call this function instead of defining its own
    compatibility table.

    Current methodological policy
    -----------------------------
    - Route-based deterministic UE can use MSA, Frank-Wolfe, or Gradient
      Projection because all three update feasible route-flow vectors.
    - Route-based SUE is intentionally restricted to MSA for now. This Logit
      implementation is treated as a fixed-point process. Frank-Wolfe and
      Gradient Projection must not be enabled for SUE until a specific
      SUE-compatible objective/update formulation is implemented.
    - Graph-based UE is kept as an explicit legacy option. It is not part of
      the full route-based refactor, but the schema still accepts it to avoid
      breaking older YAML files immediately.
    """
    if not isinstance(behavior_model, BehaviorModelName):
        raise TypeError("behavior_model must be a BehaviorModelName value.")

    if behavior_model == BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM:
        return frozenset({
            SolverName.MSA,
            SolverName.FRANK_WOLFE,
            SolverName.GRADIENT_PROJECTION,
        })

    if behavior_model == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM:
        return frozenset({SolverName.MSA})

    if behavior_model == BehaviorModelName.GRAPH_BASED_USER_EQUILIBRIUM:
        return frozenset({SolverName.MSA, SolverName.FRANK_WOLFE})

    raise ValueError(f"Unsupported behavior_model={behavior_model!r}.")


def allowed_solvers_for_behavior_model(behavior_model: BehaviorModelName) -> frozenset[SolverName]:
    """Public wrapper around the canonical compatibility policy."""
    return _allowed_solvers_for_behavior_model(behavior_model)


def validate_behavior_solver_compatibility(
    *,
    behavior_model: BehaviorModelName,
    active_solver: SolverName,
    context: str,
) -> None:
    """Validate that a behavior model can be paired with an active solver."""
    if not isinstance(active_solver, SolverName):
        raise TypeError(f"{context}.active_solver must be a SolverName value.")

    allowed = _allowed_solvers_for_behavior_model(behavior_model)
    if active_solver not in allowed:
        raise ValueError(
            f"Selected behavior model and solver are incompatible in {context}. "
            f"behavior_model={behavior_model.value!r}, "
            f"active_solver={active_solver.value!r}, "
            f"allowed={sorted(solver.value for solver in allowed)}."
        )


# =============================================================================
# Shared assignment configs
# =============================================================================


@dataclass(frozen=True)
class BehaviorModelSelectionConfig:
    """Strict policy for selecting the behavioral assignment model.

    Attributes
    ----------
    source:
        Whether the behavior model is explicitly selected in assignment.yaml or
        read from the model training configuration.
    name:
        Canonical model name when source='explicit'. Must be None when
        source='training_config' to avoid conflicting sources of truth.
    """

    source: BehaviorModelSource
    name: BehaviorModelName | None

    def validate(self) -> None:
        """Validate behavior-model source policy."""
        if not isinstance(self.source, BehaviorModelSource):
            raise TypeError("behavior_model.source must be a BehaviorModelSource value.")
        if self.source == BehaviorModelSource.EXPLICIT and not isinstance(self.name, BehaviorModelName):
            raise ValueError("behavior_model.name must be provided when behavior_model.source='explicit'.")
        if self.source == BehaviorModelSource.TRAINING_CONFIG and self.name is not None:
            raise ValueError("behavior_model.name must be null when behavior_model.source='training_config'.")

    def resolve(self, training_config: Mapping[str, Any]) -> BehaviorModelName:
        """Resolve the canonical behavior model name."""
        self.validate()
        if self.source == BehaviorModelSource.EXPLICIT:
            if self.name is None:
                raise RuntimeError("Validated explicit behavior_model.name unexpectedly became None.")
            return self.name

        raise NotImplementedError("Resolving behavior model from training config is not implemented.")

@dataclass(frozen=True)
class CapacityScalingConfig:
    """Strict policy for resolving assignment capacity scaling.

    Attributes
    ----------
    source:
        Whether the factor is explicitly declared or read from the training
        artifact configuration.
    value:
        Positive finite value when source='explicit'. Must be None when
        source='training_config'.
    training_config_path:
        Dot-separated path to the field in the training config when
        source='training_config'. Must be None when source='explicit'.
    """

    source: CapacityScalingSource
    value: float | None
    training_config_path: str | None

    def validate(self) -> None:
        """Validate capacity-scaling source policy."""
        if not isinstance(self.source, CapacityScalingSource):
            raise TypeError("capacity_scaling.source must be a CapacityScalingSource value.")

        if self.source == CapacityScalingSource.EXPLICIT:
            if self.value is None:
                raise ValueError("capacity_scaling.value is required when source='explicit'.")
            _validate_positive_finite_float("capacity_scaling.value", self.value)
            if self.training_config_path is not None:
                raise ValueError("capacity_scaling.training_config_path must be null when source='explicit'.")
            return

        if self.source == CapacityScalingSource.TRAINING_CONFIG:
            if self.value is not None:
                raise ValueError("capacity_scaling.value must be null when source='training_config'.")
            _validate_non_empty_string("capacity_scaling.training_config_path", self.training_config_path)
            return

        raise ValueError(f"Unsupported capacity_scaling.source={self.source!r}.")

    def resolve(self, training_config: Mapping[str, Any]) -> float:
        """Resolve the numeric capacity factor using the explicit policy.

        This method does not use fallback values. If the requested path is
        absent in the training configuration, an error is raised.
        """
        self.validate()
        if self.source == CapacityScalingSource.EXPLICIT:
            if self.value is None:
                raise RuntimeError("Validated explicit capacity_scaling.value unexpectedly became None.")
            return float(self.value)

        if not isinstance(training_config, Mapping):
            raise TypeError("training_config must be a mapping when resolving capacity scaling from training_config.")
        if self.training_config_path is None:
            raise RuntimeError("Validated training_config_path unexpectedly became None.")

        raw_value = _read_dot_path(training_config, self.training_config_path)
        factor = _to_float("capacity_scaling value resolved from training_config", raw_value)
        _validate_positive_finite_float("capacity_scaling resolved value", factor)
        return factor


# BPRCostFunctionConfig and VolumeDelayFunctionConfig have been moved to src.components.vdf.config


@dataclass(frozen=True)
class AssignmentCommonConfig:
    """Common assignment controls shared by behavior models."""

    max_iterations: int
    capacity_scaling: CapacityScalingConfig

    def validate(self) -> None:
        """Validate common assignment controls."""
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError(f"common.max_iterations must be an integer >= 1. Received {self.max_iterations!r}.")
        if not isinstance(self.capacity_scaling, CapacityScalingConfig):
            raise TypeError("common.capacity_scaling must be a CapacityScalingConfig instance.")
        self.capacity_scaling.validate()


# =============================================================================
# Solver configs
# =============================================================================


@dataclass(frozen=True)
class MSAConfig:
    """Configuration for the Method of Successive Averages solver."""

    step_rule: str
    initial_step_size: float | None
    minimum_step_size: float | None
    maximum_step_size: float | None
    improvement_tolerance: float | None
    increase_factor: float | None
    decrease_factor: float | None
    hybrid_harmonic_weight: float | None

    def validate(self) -> None:
        """Validate supported MSA step-size rules."""
        _validate_choice(
            "solvers.msa.step_rule",
            self.step_rule,
            {
                "1_over_n",
                "1_over_n_plus_1",
                "adaptive_gap",
                "hybrid_adaptive_gap",
            },
        )

        if self.step_rule in {"1_over_n", "1_over_n_plus_1"}:
            return

        _validate_positive_finite_float("solvers.msa.initial_step_size", self.initial_step_size)
        _validate_positive_finite_float("solvers.msa.minimum_step_size", self.minimum_step_size)
        _validate_positive_finite_float("solvers.msa.maximum_step_size", self.maximum_step_size)
        _validate_non_negative_finite_float("solvers.msa.improvement_tolerance", self.improvement_tolerance)
        _validate_positive_finite_float("solvers.msa.increase_factor", self.increase_factor)
        _validate_positive_finite_float("solvers.msa.decrease_factor", self.decrease_factor)
        _validate_non_negative_finite_float("solvers.msa.hybrid_harmonic_weight", self.hybrid_harmonic_weight)

        if self.minimum_step_size > self.maximum_step_size:
            raise ValueError("solvers.msa.minimum_step_size cannot exceed maximum_step_size.")

        if self.maximum_step_size > 1.0:
            raise ValueError("solvers.msa.maximum_step_size must be <= 1.0.")

        if self.increase_factor <= 1.0:
            raise ValueError("solvers.msa.increase_factor must be > 1.0.")

        if self.decrease_factor >= 1.0:
            raise ValueError("solvers.msa.decrease_factor must be < 1.0.")

@dataclass(frozen=True)
class FrankWolfeConfig:
    """Configuration for the Frank-Wolfe line-search solver."""

    line_search_method: str
    tolerance: float
    max_bisection_iterations: int

    def validate(self) -> None:
        """Validate Frank-Wolfe line-search parameters."""
        _validate_choice("solvers.frank_wolfe.line_search_method", self.line_search_method, {"bisection"})
        _validate_positive_finite_float("solvers.frank_wolfe.tolerance", self.tolerance)
        if not isinstance(self.max_bisection_iterations, int) or self.max_bisection_iterations < 1:
            raise ValueError(
                "solvers.frank_wolfe.max_bisection_iterations must be an integer >= 1. "
                f"Received {self.max_bisection_iterations!r}."
            )


@dataclass(frozen=True)
class GradientProjectionConfig:
    """Configuration for a route-based Gradient Projection solver.

    This config is intentionally route-flow oriented. The Gradient Projection
    implementation should live in a solvers package and consume this object.
    """

    step_rule: str
    base_step_size: float
    projection_method: str
    cost_scaling: str
    minimum_step_size: float
    maximum_step_size: float
    demand_tolerance: float

    def validate(self) -> None:
        """Validate Gradient Projection route-flow controls."""
        _validate_choice("solvers.gradient_projection.step_rule", self.step_rule, {"constant", "1_over_n", "sqrt", "scaled_sqrt"})
        _validate_positive_finite_float("solvers.gradient_projection.base_step_size", self.base_step_size)
        _validate_choice("solvers.gradient_projection.projection_method", self.projection_method, {"simplex"})
        _validate_choice("solvers.gradient_projection.cost_scaling", self.cost_scaling, {"none", "mean_abs_route_cost"})
        _validate_non_negative_finite_float("solvers.gradient_projection.minimum_step_size", self.minimum_step_size)
        _validate_positive_finite_float("solvers.gradient_projection.maximum_step_size", self.maximum_step_size)
        _validate_positive_finite_float("solvers.gradient_projection.demand_tolerance", self.demand_tolerance)
        if self.minimum_step_size > self.maximum_step_size:
            raise ValueError("minimum_step_size cannot exceed maximum_step_size.")
        if self.maximum_step_size > 1.0:
            raise ValueError("maximum_step_size must be <= 1.0 for convex route-flow updates.")


@dataclass(frozen=True)
class SolversConfig:
    """Solver selection and per-solver parameter contracts."""

    active_solver: SolverName
    msa: MSAConfig
    frank_wolfe: FrankWolfeConfig
    gradient_projection: GradientProjectionConfig

    def validate(self) -> None:
        """Validate all solver sections, including inactive ones."""
        if not isinstance(self.active_solver, SolverName):
            raise TypeError("solvers.active_solver must be a SolverName value.")
        if not isinstance(self.msa, MSAConfig):
            raise TypeError("solvers.msa must be an MSAConfig instance.")
        if not isinstance(self.frank_wolfe, FrankWolfeConfig):
            raise TypeError("solvers.frank_wolfe must be a FrankWolfeConfig instance.")
        if not isinstance(self.gradient_projection, GradientProjectionConfig):
            raise TypeError("solvers.gradient_projection must be a GradientProjectionConfig instance.")
        self.msa.validate()
        self.frank_wolfe.validate()
        self.gradient_projection.validate()


# =============================================================================
# Behavior-model solve configs
# =============================================================================


@dataclass(frozen=True)
class GraphBasedUEConvergenceConfig:
    """Convergence controls for graph-based deterministic UE."""

    relative_gap_threshold: float
    minimum_iterations: int
    zero_total_cost_tolerance: float

    def validate(self) -> None:
        """Validate graph-based UE convergence parameters."""
        _validate_non_negative_finite_float("graph_based_user_equilibrium.convergence.relative_gap_threshold", self.relative_gap_threshold)
        _validate_minimum_iterations("graph_based_user_equilibrium.convergence.minimum_iterations", self.minimum_iterations)
        _validate_non_negative_finite_float("graph_based_user_equilibrium.convergence.zero_total_cost_tolerance", self.zero_total_cost_tolerance)


@dataclass(frozen=True)
class GraphBasedUEPolicyConfig:
    """Explicit edge-case policies for graph-based UE."""

    fail_on_skipped_od_pairs: bool
    intrazonal_policy: str
    graph_weight_attribute: str

    def validate(self) -> None:
        """Validate graph-based UE policies."""
        _validate_bool("graph_based_user_equilibrium.policy.fail_on_skipped_od_pairs", self.fail_on_skipped_od_pairs)
        _validate_choice("graph_based_user_equilibrium.policy.intrazonal_policy", self.intrazonal_policy, {"raise", "skip_and_report"})
        _validate_non_empty_string("graph_based_user_equilibrium.policy.graph_weight_attribute", self.graph_weight_attribute)


@dataclass(frozen=True)
class GraphBasedUESolveConfig:
    """Complete solve config for graph-based deterministic UE."""

    convergence: GraphBasedUEConvergenceConfig
    policy: GraphBasedUEPolicyConfig

    def validate(self) -> None:
        """Validate graph-based UE solve config."""
        if not isinstance(self.convergence, GraphBasedUEConvergenceConfig):
            raise TypeError("graph_based_user_equilibrium.convergence must be a GraphBasedUEConvergenceConfig instance.")
        if not isinstance(self.policy, GraphBasedUEPolicyConfig):
            raise TypeError("graph_based_user_equilibrium.policy must be a GraphBasedUEPolicyConfig instance.")
        self.convergence.validate()
        self.policy.validate()


@dataclass(frozen=True)
class RouteBasedUEConvergenceConfig:
    """Convergence controls for route-based deterministic UE."""

    relative_gap_threshold: float
    minimum_iterations: int
    zero_total_cost_tolerance: float

    def validate(self) -> None:
        """Validate route-based UE convergence parameters."""
        _validate_non_negative_finite_float("route_based_user_equilibrium.convergence.relative_gap_threshold", self.relative_gap_threshold)
        _validate_minimum_iterations("route_based_user_equilibrium.convergence.minimum_iterations", self.minimum_iterations)
        _validate_non_negative_finite_float("route_based_user_equilibrium.convergence.zero_total_cost_tolerance", self.zero_total_cost_tolerance)


@dataclass(frozen=True)
class RouteBasedUEInitializationConfig:
    """Initial route-flow policy for route-based UE."""

    policy: str

    def validate(self) -> None:
        """Validate route-flow initialization policy."""
        _validate_choice("route_based_user_equilibrium.initialization.policy", self.policy, {"free_flow_shortest", "uniform"})


@dataclass(frozen=True)
class RouteBasedUEPolicyConfig:
    """Explicit policies for route-based deterministic UE."""

    fail_on_missing_routes: bool
    fail_on_skipped_od_pairs: bool
    intrazonal_policy: str
    expected_solver: SolverName

    def validate(self) -> None:
        """Validate route-based UE policies."""
        _validate_bool("route_based_user_equilibrium.policy.fail_on_missing_routes", self.fail_on_missing_routes)
        _validate_bool("route_based_user_equilibrium.policy.fail_on_skipped_od_pairs", self.fail_on_skipped_od_pairs)
        _validate_choice(
            "route_based_user_equilibrium.policy.intrazonal_policy",
            self.intrazonal_policy,
            {"raise", "skip_and_report", "assign_if_routes_exist"},
        )
        if not isinstance(self.expected_solver, SolverName):
            raise TypeError("route_based_user_equilibrium.policy.expected_solver must be a SolverName value.")


@dataclass(frozen=True)
class RouteBasedUESolveConfig:
    """Complete solve config for route-based deterministic UE."""

    convergence: RouteBasedUEConvergenceConfig
    initialization: RouteBasedUEInitializationConfig
    policy: RouteBasedUEPolicyConfig

    def validate(self) -> None:
        """Validate route-based UE solve config."""
        if not isinstance(self.convergence, RouteBasedUEConvergenceConfig):
            raise TypeError("route_based_user_equilibrium.convergence must be a RouteBasedUEConvergenceConfig instance.")
        if not isinstance(self.initialization, RouteBasedUEInitializationConfig):
            raise TypeError("route_based_user_equilibrium.initialization must be a RouteBasedUEInitializationConfig instance.")
        if not isinstance(self.policy, RouteBasedUEPolicyConfig):
            raise TypeError("route_based_user_equilibrium.policy must be a RouteBasedUEPolicyConfig instance.")
        self.convergence.validate()
        self.initialization.validate()
        self.policy.validate()


@dataclass(frozen=True)
class SUEConvergenceConfig:
    """Convergence thresholds for route-based SUE fixed-point assignment."""

    equilibrium_l1_threshold: float
    max_absolute_gap_threshold: float
    max_relative_gap_threshold: float
    min_flow_for_relative_gap: float
    minimum_iterations: int

    def validate(self) -> None:
        """Validate SUE fixed-point convergence thresholds."""
        _validate_non_negative_finite_float("stochastic_user_equilibrium.convergence.equilibrium_l1_threshold", self.equilibrium_l1_threshold)
        _validate_non_negative_finite_float("stochastic_user_equilibrium.convergence.max_absolute_gap_threshold", self.max_absolute_gap_threshold)
        _validate_non_negative_finite_float("stochastic_user_equilibrium.convergence.max_relative_gap_threshold", self.max_relative_gap_threshold)
        _validate_positive_finite_float("stochastic_user_equilibrium.convergence.min_flow_for_relative_gap", self.min_flow_for_relative_gap)
        _validate_minimum_iterations("stochastic_user_equilibrium.convergence.minimum_iterations", self.minimum_iterations)


@dataclass(frozen=True)
class SUELogitConfig:
    """Logit route-choice configuration for SUE."""

    theta_source: str
    theta_value: float | None
    theta_artifact_key: str | None
    fail_on_invalid_probability: bool

    def validate(self) -> None:
        """Validate Logit theta source and probability policy."""
        _validate_choice("stochastic_user_equilibrium.logit.theta_source", self.theta_source, {"explicit", "artifact"})
        if self.theta_source == "explicit":
            if self.theta_value is None:
                raise ValueError("theta_value is required when theta_source='explicit'.")
            _validate_positive_finite_float("stochastic_user_equilibrium.logit.theta_value", self.theta_value)
            if self.theta_artifact_key is not None:
                raise ValueError("theta_artifact_key must be null when theta_source='explicit'.")
        if self.theta_source == "artifact":
            if self.theta_value is not None:
                raise ValueError("theta_value must be null when theta_source='artifact'.")
            _validate_non_empty_string("stochastic_user_equilibrium.logit.theta_artifact_key", self.theta_artifact_key)
        _validate_bool("stochastic_user_equilibrium.logit.fail_on_invalid_probability", self.fail_on_invalid_probability)

    def resolve_theta(self, artifacts: Mapping[str, Any]) -> float:
        """Resolve theta from explicit YAML or from model artifacts.

        No fallback is used. Missing artifact keys raise an error.
        """
        self.validate()
        if self.theta_source == "explicit":
            if self.theta_value is None:
                raise RuntimeError("Validated explicit theta unexpectedly became None.")
            return float(self.theta_value)

        if not isinstance(artifacts, Mapping):
            raise TypeError("artifacts must be a mapping when theta_source='artifact'.")
        if self.theta_artifact_key is None:
            raise RuntimeError("Validated theta_artifact_key unexpectedly became None.")
        if self.theta_artifact_key not in artifacts:
            raise KeyError(f"Theta artifact key {self.theta_artifact_key!r} is missing.")
        theta = _to_float("theta resolved from artifacts", _unwrap_scalar_artifact(artifacts[self.theta_artifact_key]))
        _validate_positive_finite_float("theta resolved from artifacts", theta)
        return theta


@dataclass(frozen=True)
class SUEPolicyConfig:
    """Explicit policies for route-based SUE."""

    fail_on_missing_routes: bool
    fail_on_skipped_od_pairs: bool
    intrazonal_policy: str
    expected_solver: SolverName

    def validate(self) -> None:
        """Validate route-based SUE policies."""
        _validate_bool("stochastic_user_equilibrium.policy.fail_on_missing_routes", self.fail_on_missing_routes)
        _validate_bool("stochastic_user_equilibrium.policy.fail_on_skipped_od_pairs", self.fail_on_skipped_od_pairs)
        _validate_choice(
            "stochastic_user_equilibrium.policy.intrazonal_policy",
            self.intrazonal_policy,
            {"raise", "skip_and_report", "assign_if_routes_exist"},
        )
        if not isinstance(self.expected_solver, SolverName):
            raise TypeError("stochastic_user_equilibrium.policy.expected_solver must be a SolverName value.")
        if self.expected_solver not in _allowed_solvers_for_behavior_model(BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM):
            raise ValueError(
                "stochastic_user_equilibrium.policy.expected_solver is not supported by the current SUE implementation. "
                "Only expected_solver='msa' is allowed until a SUE-compatible Frank-Wolfe or "
                "Gradient Projection formulation is implemented. "
                f"Received {self.expected_solver.value!r}."
            )


@dataclass(frozen=True)
class SUESolveConfig:
    """Complete solve config for route-based SUE."""

    convergence: SUEConvergenceConfig
    logit: SUELogitConfig
    policy: SUEPolicyConfig

    def validate(self) -> None:
        """Validate SUE solve config."""
        if not isinstance(self.convergence, SUEConvergenceConfig):
            raise TypeError("stochastic_user_equilibrium.convergence must be a SUEConvergenceConfig instance.")
        if not isinstance(self.logit, SUELogitConfig):
            raise TypeError("stochastic_user_equilibrium.logit must be a SUELogitConfig instance.")
        if not isinstance(self.policy, SUEPolicyConfig):
            raise TypeError("stochastic_user_equilibrium.policy must be a SUEPolicyConfig instance.")
        self.convergence.validate()
        self.logit.validate()
        self.policy.validate()


# =============================================================================
# Top-level config
# =============================================================================


@dataclass(frozen=True)
class AssignmentConfig:
    """Complete validated assignment configuration from assignment.yaml."""

    behavior_model: BehaviorModelSelectionConfig
    common: AssignmentCommonConfig
    cost_function: VDFConfig
    route_set: RouteSetBuildConfig
    solvers: SolversConfig
    graph_based_user_equilibrium: GraphBasedUESolveConfig
    route_based_user_equilibrium: RouteBasedUESolveConfig
    stochastic_user_equilibrium: SUESolveConfig

    def validate(self) -> None:
        """Validate every assignment configuration section."""
        if not isinstance(self.behavior_model, BehaviorModelSelectionConfig):
            raise TypeError("behavior_model must be a BehaviorModelSelectionConfig instance.")
        if not isinstance(self.common, AssignmentCommonConfig):
            raise TypeError("common must be an AssignmentCommonConfig instance.")
        if not isinstance(self.cost_function, VDFConfig):
            raise TypeError("assignment.cost_function must be a VDFConfig instance.")
        if not isinstance(self.route_set, RouteSetBuildConfig):
            raise TypeError("route_set must be a RouteSetBuildConfig instance.")
        if not isinstance(self.solvers, SolversConfig):
            raise TypeError("solvers must be a SolversConfig instance.")
        if not isinstance(self.graph_based_user_equilibrium, GraphBasedUESolveConfig):
            raise TypeError("graph_based_user_equilibrium must be a GraphBasedUESolveConfig instance.")
        if not isinstance(self.route_based_user_equilibrium, RouteBasedUESolveConfig):
            raise TypeError("route_based_user_equilibrium must be a RouteBasedUESolveConfig instance.")
        if not isinstance(self.stochastic_user_equilibrium, SUESolveConfig):
            raise TypeError("stochastic_user_equilibrium must be a SUESolveConfig instance.")

        self.behavior_model.validate()
        self.common.validate()
        self.cost_function.validate()
        self.route_set.__post_init__()
        self.solvers.validate()
        self.graph_based_user_equilibrium.validate()
        self.route_based_user_equilibrium.validate()
        self.stochastic_user_equilibrium.validate()
        self._validate_behavior_solver_compatibility()
        self._validate_active_policy_expected_solver()

    def _validate_behavior_solver_compatibility(self) -> None:
        """Validate compatibility when behavior model is explicitly selected."""
        if self.behavior_model.source != BehaviorModelSource.EXPLICIT:
            return
        if self.behavior_model.name is None:
            raise RuntimeError("Validated explicit behavior_model.name unexpectedly became None.")

        validate_behavior_solver_compatibility(
            behavior_model=self.behavior_model.name,
            active_solver=self.solvers.active_solver,
            context="assignment",
        )

    def _validate_active_policy_expected_solver(self) -> None:
        """Ensure the selected behavior policy agrees with solvers.active_solver.

        The YAML has two places that can express the active numerical solver:
        solvers.active_solver and the selected behavior model's policy.expected_solver.
        They must match for the active behavior model. This prevents ambiguous
        configurations such as UE declaring expected_solver='gradient_projection'
        while solvers.active_solver='msa'.
        """
        if self.behavior_model.source != BehaviorModelSource.EXPLICIT:
            return
        if self.behavior_model.name is None:
            raise RuntimeError("Validated explicit behavior_model.name unexpectedly became None.")

        expected_solver: SolverName | None = None
        expected_solver_context: str | None = None

        if self.behavior_model.name == BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM:
            expected_solver = self.route_based_user_equilibrium.policy.expected_solver
            expected_solver_context = "route_based_user_equilibrium.policy.expected_solver"
        elif self.behavior_model.name == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM:
            expected_solver = self.stochastic_user_equilibrium.policy.expected_solver
            expected_solver_context = "stochastic_user_equilibrium.policy.expected_solver"

        # Graph-based UE is retained as legacy and does not currently carry an
        # expected_solver field in its policy config. Its compatibility is still
        # enforced by _validate_behavior_solver_compatibility().
        if expected_solver is None:
            return

        if expected_solver != self.solvers.active_solver:
            raise ValueError(
                "Selected behavior model policy.expected_solver must match solvers.active_solver. "
                f"behavior_model={self.behavior_model.name.value!r}, "
                f"{expected_solver_context}={expected_solver.value!r}, "
                f"solvers.active_solver={self.solvers.active_solver.value!r}."
            )


# =============================================================================
# YAML / mapping resolver
# =============================================================================


def load_assignment_config_from_yaml(yaml_path: str | Path) -> AssignmentConfig:
    """Load and validate AssignmentConfig from an assignment.yaml file.

    Parameters
    ----------
    yaml_path:
        Path to the assignment YAML file. The file must contain exactly the
        schema expected by build_assignment_config_from_mapping.
    """
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Assignment YAML file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Assignment YAML path is not a file: {path}")

    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise ImportError("PyYAML is required to load assignment.yaml from disk.") from exc

    with path.open("r", encoding="utf-8") as file:
        raw_config = yaml.safe_load(file)
    if raw_config is None:
        raise ValueError(f"Assignment YAML file is empty: {path}")
    return build_assignment_config_from_mapping(raw_config)


def build_assignment_config_from_mapping(raw_config: Mapping[str, Any]) -> AssignmentConfig:
    """Build AssignmentConfig from a strict mapping.

    This function accepts plain dictionaries or OmegaConf containers already
    converted to primitive objects. It never fills missing values.
    """
    config = _to_plain_mapping(raw_config, context="assignment")
    _require_exact_keys(
        config,
        required={
            "behavior_model",
            "common",
            "cost_function",
            "route_set",
            "solvers",
            "graph_based_user_equilibrium",
            "route_based_user_equilibrium",
            "stochastic_user_equilibrium",
        },
        context="assignment",
    )

    assignment_config = AssignmentConfig(
        behavior_model=_build_behavior_model_selection(config["behavior_model"]),
        common=_build_common_config(config["common"]),
        cost_function=_build_vdf_config(config["cost_function"]),
        route_set=_build_route_set_config(config["route_set"]),
        solvers=_build_solvers_config(config["solvers"]),
        graph_based_user_equilibrium=_build_graph_based_ue_config(config["graph_based_user_equilibrium"]),
        route_based_user_equilibrium=_build_route_based_ue_config(config["route_based_user_equilibrium"]),
        stochastic_user_equilibrium=_build_sue_config(config["stochastic_user_equilibrium"]),
    )
    assignment_config.validate()
    return assignment_config


def _build_behavior_model_selection(raw_section: Any) -> BehaviorModelSelectionConfig:
    section = _to_plain_mapping(raw_section, context="assignment.behavior_model")
    _require_exact_keys(section, required={"source", "name"}, context="assignment.behavior_model")
    source = _parse_enum(BehaviorModelSource, section["source"], "assignment.behavior_model.source")
    name = None if section["name"] is None else _parse_enum(BehaviorModelName, section["name"], "assignment.behavior_model.name")
    result = BehaviorModelSelectionConfig(source=source, name=name)
    result.validate()
    return result


def _build_common_config(raw_section: Any) -> AssignmentCommonConfig:
    section = _to_plain_mapping(raw_section, context="assignment.common")
    _require_exact_keys(section, required={"max_iterations", "capacity_scaling"}, context="assignment.common")
    result = AssignmentCommonConfig(
        max_iterations=_to_int("assignment.common.max_iterations", section["max_iterations"]),
        capacity_scaling=_build_capacity_scaling_config(section["capacity_scaling"]),
    )
    result.validate()
    return result


def _build_capacity_scaling_config(raw_section: Any) -> CapacityScalingConfig:
    section = _to_plain_mapping(raw_section, context="assignment.common.capacity_scaling")
    _require_exact_keys(section, required={"source", "value", "training_config_path"}, context="assignment.common.capacity_scaling")
    result = CapacityScalingConfig(
        source=_parse_enum(CapacityScalingSource, section["source"], "assignment.common.capacity_scaling.source"),
        value=None if section["value"] is None else _to_float("assignment.common.capacity_scaling.value", section["value"]),
        training_config_path=None if section["training_config_path"] is None else _to_str("assignment.common.capacity_scaling.training_config_path", section["training_config_path"]),
    )
    result.validate()
    return result


def _build_vdf_config(raw_section: Any) -> VDFConfig:
    section = _to_plain_mapping(raw_section, context="assignment.cost_function")
    _require_exact_keys(
        section,
        required={
            "vdf_name",
            "free_flow_time_col",
            "capacity_col",
            "toll_col",
            "toll_required",
            "parameters_cols",
        },
        context="assignment.cost_function",
    )

    parameters_cols = _to_plain_mapping(
        section["parameters_cols"],
        context="assignment.cost_function.parameters_cols",
    )

    result = VDFConfig(
        vdf_name=_to_str("assignment.cost_function.vdf_name", section["vdf_name"]),
        free_flow_time_col=_to_str("assignment.cost_function.free_flow_time_col", section["free_flow_time_col"]),
        capacity_col=_to_str("assignment.cost_function.capacity_col", section["capacity_col"]),
        toll_col=None if section["toll_col"] is None else _to_str(
            "assignment.cost_function.toll_col",
            section["toll_col"],
        ),
        toll_required=_to_bool("assignment.cost_function.toll_required", section["toll_required"]),
        parameters_cols={
            _to_str(f"assignment.cost_function.parameters_cols.{k}", k): _to_str(
                f"assignment.cost_function.parameters_cols.{k}",
                v,
            )
            for k, v in parameters_cols.items()
        },
    )
    result.validate()
    return result


def _build_route_set_config(raw_section: Any) -> RouteSetBuildConfig:
    section = _to_plain_mapping(raw_section, context="assignment.route_set")
    _require_exact_keys(
        section,
        required={
            "route_input_format",
            "link_id_col",
            "init_node_col",
            "term_node_col",
            "route_cost_col",
            "fail_on_duplicate_routes",
            "fail_on_empty_route_set",
            "fail_on_missing_od_routes",
            "require_simple_node_routes",
            "require_unique_link_ids",
            "require_unique_directed_edges",
        },
        context="assignment.route_set",
    )
    result = RouteSetBuildConfig(
        route_input_format=_parse_enum(RouteInputFormat, section["route_input_format"], "assignment.route_set.route_input_format"),
        link_id_col=_to_str("assignment.route_set.link_id_col", section["link_id_col"]),
        init_node_col=_to_str("assignment.route_set.init_node_col", section["init_node_col"]),
        term_node_col=_to_str("assignment.route_set.term_node_col", section["term_node_col"]),
        route_cost_col=_to_str("assignment.route_set.route_cost_col", section["route_cost_col"]),
        fail_on_duplicate_routes=_to_bool("assignment.route_set.fail_on_duplicate_routes", section["fail_on_duplicate_routes"]),
        fail_on_empty_route_set=_to_bool("assignment.route_set.fail_on_empty_route_set", section["fail_on_empty_route_set"]),
        fail_on_missing_od_routes=_to_bool("assignment.route_set.fail_on_missing_od_routes", section["fail_on_missing_od_routes"]),
        require_simple_node_routes=_to_bool("assignment.route_set.require_simple_node_routes", section["require_simple_node_routes"]),
        require_unique_link_ids=_to_bool("assignment.route_set.require_unique_link_ids", section["require_unique_link_ids"]),
        require_unique_directed_edges=_to_bool("assignment.route_set.require_unique_directed_edges", section["require_unique_directed_edges"]),
    )
    result.__post_init__()
    return result


def _build_solvers_config(raw_section: Any) -> SolversConfig:
    section = _to_plain_mapping(raw_section, context="assignment.solvers")
    _require_exact_keys(section, required={"active_solver", "msa", "frank_wolfe", "gradient_projection"}, context="assignment.solvers")
    result = SolversConfig(
        active_solver=_parse_enum(SolverName, section["active_solver"], "assignment.solvers.active_solver"),
        msa=_build_msa_config(section["msa"]),
        frank_wolfe=_build_frank_wolfe_config(section["frank_wolfe"]),
        gradient_projection=_build_gradient_projection_config(section["gradient_projection"]),
    )
    result.validate()
    return result


def _build_msa_config(raw_section: Any) -> MSAConfig:
    section = _to_plain_mapping(raw_section, context="assignment.solvers.msa")
    _require_exact_keys(
        section,
        required={
            "step_rule",
            "initial_step_size",
            "minimum_step_size",
            "maximum_step_size",
            "improvement_tolerance",
            "increase_factor",
            "decrease_factor",
            "hybrid_harmonic_weight",
        },
        context="assignment.solvers.msa",
    )

    result = MSAConfig(
        step_rule=_to_str("assignment.solvers.msa.step_rule", section["step_rule"]),
        initial_step_size=None if section["initial_step_size"] is None else _to_float("assignment.solvers.msa.initial_step_size", section["initial_step_size"]),
        minimum_step_size=None if section["minimum_step_size"] is None else _to_float("assignment.solvers.msa.minimum_step_size", section["minimum_step_size"]),
        maximum_step_size=None if section["maximum_step_size"] is None else _to_float("assignment.solvers.msa.maximum_step_size", section["maximum_step_size"]),
        improvement_tolerance=None if section["improvement_tolerance"] is None else _to_float("assignment.solvers.msa.improvement_tolerance", section["improvement_tolerance"]),
        increase_factor=None if section["increase_factor"] is None else _to_float("assignment.solvers.msa.increase_factor", section["increase_factor"]),
        decrease_factor=None if section["decrease_factor"] is None else _to_float("assignment.solvers.msa.decrease_factor", section["decrease_factor"]),
        hybrid_harmonic_weight=None if section["hybrid_harmonic_weight"] is None else _to_float("assignment.solvers.msa.hybrid_harmonic_weight", section["hybrid_harmonic_weight"]),
    )
    result.validate()
    return result


def _build_frank_wolfe_config(raw_section: Any) -> FrankWolfeConfig:
    section = _to_plain_mapping(raw_section, context="assignment.solvers.frank_wolfe")
    _require_exact_keys(section, required={"line_search_method", "tolerance", "max_bisection_iterations"}, context="assignment.solvers.frank_wolfe")
    result = FrankWolfeConfig(
        line_search_method=_to_str("assignment.solvers.frank_wolfe.line_search_method", section["line_search_method"]),
        tolerance=_to_float("assignment.solvers.frank_wolfe.tolerance", section["tolerance"]),
        max_bisection_iterations=_to_int("assignment.solvers.frank_wolfe.max_bisection_iterations", section["max_bisection_iterations"]),
    )
    result.validate()
    return result


def _build_gradient_projection_config(raw_section: Any) -> GradientProjectionConfig:
    section = _to_plain_mapping(raw_section, context="assignment.solvers.gradient_projection")
    _require_exact_keys(
        section,
        required={
            "step_rule",
            "base_step_size",
            "projection_method",
            "cost_scaling",
            "minimum_step_size",
            "maximum_step_size",
            "demand_tolerance",
        },
        context="assignment.solvers.gradient_projection",
    )
    result = GradientProjectionConfig(
        step_rule=_to_str("assignment.solvers.gradient_projection.step_rule", section["step_rule"]),
        base_step_size=_to_float("assignment.solvers.gradient_projection.base_step_size", section["base_step_size"]),
        projection_method=_to_str("assignment.solvers.gradient_projection.projection_method", section["projection_method"]),
        cost_scaling=_to_str("assignment.solvers.gradient_projection.cost_scaling", section["cost_scaling"]),
        minimum_step_size=_to_float("assignment.solvers.gradient_projection.minimum_step_size", section["minimum_step_size"]),
        maximum_step_size=_to_float("assignment.solvers.gradient_projection.maximum_step_size", section["maximum_step_size"]),
        demand_tolerance=_to_float("assignment.solvers.gradient_projection.demand_tolerance", section["demand_tolerance"]),
    )
    result.validate()
    return result


def _build_graph_based_ue_config(raw_section: Any) -> GraphBasedUESolveConfig:
    section = _to_plain_mapping(raw_section, context="assignment.graph_based_user_equilibrium")
    _require_exact_keys(section, required={"convergence", "policy"}, context="assignment.graph_based_user_equilibrium")
    result = GraphBasedUESolveConfig(
        convergence=_build_graph_based_ue_convergence(section["convergence"]),
        policy=_build_graph_based_ue_policy(section["policy"]),
    )
    result.validate()
    return result


def _build_graph_based_ue_convergence(raw_section: Any) -> GraphBasedUEConvergenceConfig:
    section = _to_plain_mapping(raw_section, context="assignment.graph_based_user_equilibrium.convergence")
    _require_exact_keys(section, required={"relative_gap_threshold", "minimum_iterations", "zero_total_cost_tolerance"}, context="assignment.graph_based_user_equilibrium.convergence")
    result = GraphBasedUEConvergenceConfig(
        relative_gap_threshold=_to_float("assignment.graph_based_user_equilibrium.convergence.relative_gap_threshold", section["relative_gap_threshold"]),
        minimum_iterations=_to_int("assignment.graph_based_user_equilibrium.convergence.minimum_iterations", section["minimum_iterations"]),
        zero_total_cost_tolerance=_to_float("assignment.graph_based_user_equilibrium.convergence.zero_total_cost_tolerance", section["zero_total_cost_tolerance"]),
    )
    result.validate()
    return result


def _build_graph_based_ue_policy(raw_section: Any) -> GraphBasedUEPolicyConfig:
    section = _to_plain_mapping(raw_section, context="assignment.graph_based_user_equilibrium.policy")
    _require_exact_keys(section, required={"fail_on_skipped_od_pairs", "intrazonal_policy", "graph_weight_attribute"}, context="assignment.graph_based_user_equilibrium.policy")
    result = GraphBasedUEPolicyConfig(
        fail_on_skipped_od_pairs=_to_bool("assignment.graph_based_user_equilibrium.policy.fail_on_skipped_od_pairs", section["fail_on_skipped_od_pairs"]),
        intrazonal_policy=_to_str("assignment.graph_based_user_equilibrium.policy.intrazonal_policy", section["intrazonal_policy"]),
        graph_weight_attribute=_to_str("assignment.graph_based_user_equilibrium.policy.graph_weight_attribute", section["graph_weight_attribute"]),
    )
    result.validate()
    return result


def _build_route_based_ue_config(raw_section: Any) -> RouteBasedUESolveConfig:
    section = _to_plain_mapping(raw_section, context="assignment.route_based_user_equilibrium")
    _require_exact_keys(section, required={"convergence", "initialization", "policy"}, context="assignment.route_based_user_equilibrium")
    result = RouteBasedUESolveConfig(
        convergence=_build_route_based_ue_convergence(section["convergence"]),
        initialization=_build_route_based_ue_initialization(section["initialization"]),
        policy=_build_route_based_ue_policy(section["policy"]),
    )
    result.validate()
    return result


def _build_route_based_ue_convergence(raw_section: Any) -> RouteBasedUEConvergenceConfig:
    section = _to_plain_mapping(raw_section, context="assignment.route_based_user_equilibrium.convergence")
    _require_exact_keys(section, required={"relative_gap_threshold", "minimum_iterations", "zero_total_cost_tolerance"}, context="assignment.route_based_user_equilibrium.convergence")
    result = RouteBasedUEConvergenceConfig(
        relative_gap_threshold=_to_float("assignment.route_based_user_equilibrium.convergence.relative_gap_threshold", section["relative_gap_threshold"]),
        minimum_iterations=_to_int("assignment.route_based_user_equilibrium.convergence.minimum_iterations", section["minimum_iterations"]),
        zero_total_cost_tolerance=_to_float("assignment.route_based_user_equilibrium.convergence.zero_total_cost_tolerance", section["zero_total_cost_tolerance"]),
    )
    result.validate()
    return result


def _build_route_based_ue_initialization(raw_section: Any) -> RouteBasedUEInitializationConfig:
    section = _to_plain_mapping(raw_section, context="assignment.route_based_user_equilibrium.initialization")
    _require_exact_keys(section, required={"policy"}, context="assignment.route_based_user_equilibrium.initialization")
    result = RouteBasedUEInitializationConfig(
        policy=_to_str("assignment.route_based_user_equilibrium.initialization.policy", section["policy"]),
    )
    result.validate()
    return result


def _build_route_based_ue_policy(raw_section: Any) -> RouteBasedUEPolicyConfig:
    section = _to_plain_mapping(raw_section, context="assignment.route_based_user_equilibrium.policy")
    _require_exact_keys(section, required={"fail_on_missing_routes", "fail_on_skipped_od_pairs", "intrazonal_policy", "expected_solver"}, context="assignment.route_based_user_equilibrium.policy")
    result = RouteBasedUEPolicyConfig(
        fail_on_missing_routes=_to_bool("assignment.route_based_user_equilibrium.policy.fail_on_missing_routes", section["fail_on_missing_routes"]),
        fail_on_skipped_od_pairs=_to_bool("assignment.route_based_user_equilibrium.policy.fail_on_skipped_od_pairs", section["fail_on_skipped_od_pairs"]),
        intrazonal_policy=_to_str("assignment.route_based_user_equilibrium.policy.intrazonal_policy", section["intrazonal_policy"]),
        expected_solver=_parse_enum(SolverName, section["expected_solver"], "assignment.route_based_user_equilibrium.policy.expected_solver"),
    )
    result.validate()
    return result


def _build_sue_config(raw_section: Any) -> SUESolveConfig:
    section = _to_plain_mapping(raw_section, context="assignment.stochastic_user_equilibrium")
    _require_exact_keys(section, required={"convergence", "logit", "policy"}, context="assignment.stochastic_user_equilibrium")
    result = SUESolveConfig(
        convergence=_build_sue_convergence(section["convergence"]),
        logit=_build_sue_logit(section["logit"]),
        policy=_build_sue_policy(section["policy"]),
    )
    result.validate()
    return result


def _build_sue_convergence(raw_section: Any) -> SUEConvergenceConfig:
    section = _to_plain_mapping(raw_section, context="assignment.stochastic_user_equilibrium.convergence")
    _require_exact_keys(
        section,
        required={
            "equilibrium_l1_threshold",
            "max_absolute_gap_threshold",
            "max_relative_gap_threshold",
            "min_flow_for_relative_gap",
            "minimum_iterations",
        },
        context="assignment.stochastic_user_equilibrium.convergence",
    )
    result = SUEConvergenceConfig(
        equilibrium_l1_threshold=_to_float("assignment.stochastic_user_equilibrium.convergence.equilibrium_l1_threshold", section["equilibrium_l1_threshold"]),
        max_absolute_gap_threshold=_to_float("assignment.stochastic_user_equilibrium.convergence.max_absolute_gap_threshold", section["max_absolute_gap_threshold"]),
        max_relative_gap_threshold=_to_float("assignment.stochastic_user_equilibrium.convergence.max_relative_gap_threshold", section["max_relative_gap_threshold"]),
        min_flow_for_relative_gap=_to_float("assignment.stochastic_user_equilibrium.convergence.min_flow_for_relative_gap", section["min_flow_for_relative_gap"]),
        minimum_iterations=_to_int("assignment.stochastic_user_equilibrium.convergence.minimum_iterations", section["minimum_iterations"]),
    )
    result.validate()
    return result


def _build_sue_logit(raw_section: Any) -> SUELogitConfig:
    section = _to_plain_mapping(raw_section, context="assignment.stochastic_user_equilibrium.logit")
    _require_exact_keys(section, required={"theta_source", "theta_value", "theta_artifact_key", "fail_on_invalid_probability"}, context="assignment.stochastic_user_equilibrium.logit")
    result = SUELogitConfig(
        theta_source=_to_str("assignment.stochastic_user_equilibrium.logit.theta_source", section["theta_source"]),
        theta_value=None if section["theta_value"] is None else _to_float("assignment.stochastic_user_equilibrium.logit.theta_value", section["theta_value"]),
        theta_artifact_key=None if section["theta_artifact_key"] is None else _to_str("assignment.stochastic_user_equilibrium.logit.theta_artifact_key", section["theta_artifact_key"]),
        fail_on_invalid_probability=_to_bool("assignment.stochastic_user_equilibrium.logit.fail_on_invalid_probability", section["fail_on_invalid_probability"]),
    )
    result.validate()
    return result


def _build_sue_policy(raw_section: Any) -> SUEPolicyConfig:
    section = _to_plain_mapping(raw_section, context="assignment.stochastic_user_equilibrium.policy")
    _require_exact_keys(section, required={"fail_on_missing_routes", "fail_on_skipped_od_pairs", "intrazonal_policy", "expected_solver"}, context="assignment.stochastic_user_equilibrium.policy")
    result = SUEPolicyConfig(
        fail_on_missing_routes=_to_bool("assignment.stochastic_user_equilibrium.policy.fail_on_missing_routes", section["fail_on_missing_routes"]),
        fail_on_skipped_od_pairs=_to_bool("assignment.stochastic_user_equilibrium.policy.fail_on_skipped_od_pairs", section["fail_on_skipped_od_pairs"]),
        intrazonal_policy=_to_str("assignment.stochastic_user_equilibrium.policy.intrazonal_policy", section["intrazonal_policy"]),
        expected_solver=_parse_enum(SolverName, section["expected_solver"], "assignment.stochastic_user_equilibrium.policy.expected_solver"),
    )
    result.validate()
    return result


# =============================================================================
# Utility helpers
# =============================================================================


def _to_plain_mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Convert plain dict or OmegaConf DictConfig into a regular mapping."""
    if isinstance(value, Mapping):
        return value
    try:
        from omegaconf import OmegaConf  # type: ignore
    except ImportError:
        OmegaConf = None  # type: ignore
    if OmegaConf is not None and OmegaConf.is_config(value):
        converted = OmegaConf.to_container(value, resolve=True)
        if isinstance(converted, Mapping):
            return converted
    raise TypeError(f"{context} must be a mapping-like object. Received {type(value).__name__}.")


def _require_exact_keys(section: Mapping[str, Any], required: set[str], context: str) -> None:
    """Validate that a config section has exactly the required keys."""
    actual = set(section.keys())
    missing = sorted(required.difference(actual))
    extra = sorted(actual.difference(required))
    if missing or extra:
        message_parts = []
        if missing:
            message_parts.append(f"missing keys={missing}")
        if extra:
            message_parts.append(f"unknown keys={extra}")
        raise ValueError(f"Invalid keys in {context}: {', '.join(message_parts)}.")


def _parse_enum(enum_type: type[Enum], raw_value: Any, context: str) -> Any:
    """Parse a YAML string into a strict Enum value without aliases."""
    if not isinstance(raw_value, str):
        raise TypeError(f"{context} must be a string. Received {raw_value!r}.")
    try:
        return enum_type(raw_value)
    except ValueError as exc:
        allowed = sorted(item.value for item in enum_type)
        raise ValueError(f"Invalid value for {context}: {raw_value!r}. Allowed values: {allowed}.") from exc


def _validate_choice(context: str, value: str, allowed: set[str]) -> None:
    """Validate a strict string choice without aliases."""
    _validate_non_empty_string(context, value)
    if value not in allowed:
        raise ValueError(f"Invalid value for {context}: {value!r}. Allowed values: {sorted(allowed)}.")


def _validate_non_empty_string(context: str, value: Any) -> None:
    """Validate a non-empty string field."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string. Received {value!r}.")


def _validate_bool(context: str, value: Any) -> None:
    """Validate an exact boolean field."""
    if not isinstance(value, bool):
        raise TypeError(f"{context} must be a bool. Received {value!r}.")


def _validate_positive_finite_float(context: str, value: float) -> None:
    """Validate a positive finite float."""
    if not np.isfinite(float(value)) or float(value) <= 0.0:
        raise ValueError(f"{context} must be finite and > 0. Received {value!r}.")


def _validate_non_negative_finite_float(context: str, value: float) -> None:
    """Validate a non-negative finite float."""
    if not np.isfinite(float(value)) or float(value) < 0.0:
        raise ValueError(f"{context} must be finite and >= 0. Received {value!r}.")


def _validate_minimum_iterations(context: str, value: int) -> None:
    """Validate an integer minimum-iteration count."""
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{context} must be an integer >= 1. Received {value!r}.")


def _to_str(context: str, value: Any) -> str:
    """Convert and validate a string config value."""
    if not isinstance(value, str):
        raise TypeError(f"{context} must be a string. Received {value!r}.")
    _validate_non_empty_string(context, value)
    return value


def _to_bool(context: str, value: Any) -> bool:
    """Convert and validate a boolean config value."""
    _validate_bool(context, value)
    return bool(value)


def _to_int(context: str, value: Any) -> int:
    """Convert and validate an integer config value without accepting bools."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{context} must be an integer. Received {value!r}.")
    return int(value)


def _to_float(context: str, value: Any) -> float:
    import numbers
    """Convert and validate a numeric config value without accepting bools."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{context} must be numeric. Received {value!r}.")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{context} must be finite. Received {value!r}.")
    return result


def _read_dot_path(container: Mapping[str, Any], dot_path: str) -> Any:
    """Read a dot-separated path from a nested mapping without defaults."""
    current: Any = container
    for token in dot_path.split("."):
        if not isinstance(current, Mapping):
            raise KeyError(f"Cannot read path {dot_path!r}: parent of {token!r} is not a mapping.")
        if token not in current:
            raise KeyError(f"Cannot read path {dot_path!r}: missing key {token!r}.")
        current = current[token]
    return current


def _unwrap_scalar_artifact(value: Any) -> Any:
    """Extract a scalar-like value from common tensor/array artifacts."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise ValueError(f"Expected a scalar artifact, received shape={np.asarray(value).shape}.")
    return array[0]


__all__ = [
    "AssignmentCommonConfig",
    "AssignmentConfig",
    "BehaviorModelName",
    "BehaviorModelSelectionConfig",
    "BehaviorModelSource",
    "CapacityScalingConfig",
    "CapacityScalingSource",
    "FrankWolfeConfig",
    "GraphBasedUEConvergenceConfig",
    "GraphBasedUEPolicyConfig",
    "GraphBasedUESolveConfig",
    "GradientProjectionConfig",
    "MSAConfig",
    "RouteBasedUEConvergenceConfig",
    "RouteBasedUEInitializationConfig",
    "RouteBasedUEPolicyConfig",
    "RouteBasedUESolveConfig",
    "SolverName",
    "SolversConfig",
    "SUEConvergenceConfig",
    "SUELogitConfig",
    "SUEPolicyConfig",
    "SUESolveConfig",
    "VDFConfig",
    "allowed_solvers_for_behavior_model",
    "build_assignment_config_from_mapping",
    "load_assignment_config_from_yaml",
    "validate_behavior_solver_compatibility",
]
