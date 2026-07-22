"""Strict configuration schemas for reusable artifacts and asset pipelines.

This module is the boundary between YAML configuration and the artifact
materialization layer. It intentionally rejects unknown keys and avoids
implicit defaults that could hide configuration drift.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

try:  # Optional dependency during type checking or lightweight imports.
    from omegaconf import DictConfig, ListConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover - fallback for environments without OmegaConf.
    DictConfig = None  # type: ignore
    ListConfig = None  # type: ignore
    OmegaConf = None  # type: ignore


class DatasetNature(str, Enum):
    """High-level dataset nature used by the asset pipeline."""

    SYNTHETIC = "synthetic"
    REAL = "real"


@dataclass(frozen=True)
class DatasetAvailabilityConfig:
    """Capabilities exposed by a dataset.

    The asset pipeline should reason primarily about these capabilities rather
    than about the dataset being synthetic or real.
    """

    has_complete_od_ground_truth: bool
    has_observed_link_flows: bool
    has_ground_truth_assignment: bool

    def __post_init__(self) -> None:
        for field_name in (
            "has_complete_od_ground_truth",
            "has_observed_link_flows",
            "has_ground_truth_assignment",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a bool.")


@dataclass(frozen=True)
class DatasetProfileConfig:
    """Minimal dataset profile extracted from the dataset YAML."""

    nature: DatasetNature
    data_availability: DatasetAvailabilityConfig

    def __post_init__(self) -> None:
        if not isinstance(self.nature, DatasetNature):
            raise TypeError("nature must be a DatasetNature value.")
        if not isinstance(self.data_availability, DatasetAvailabilityConfig):
            raise TypeError("data_availability must be a DatasetAvailabilityConfig value.")


@dataclass(frozen=True)
class AssetPolicyConfig:
    """Asset materialization policy."""

    on_missing: str = "fail"
    on_stale: str = "fail"
    overwrite_existing: bool = False
    require_exact_fingerprint: bool = True

    def __post_init__(self) -> None:
        _validate_choice("on_missing", self.on_missing, {"fail", "build"})
        _validate_choice("on_stale", self.on_stale, {"fail", "rebuild", "use_existing"})
        _validate_bool("overwrite_existing", self.overwrite_existing)
        _validate_bool("require_exact_fingerprint", self.require_exact_fingerprint)


@dataclass(frozen=True)
class RouteSetRequirementConfig:
    """Experiment-level requirement for a route-set asset."""

    spec_id: str
    k_active: int

    def __post_init__(self) -> None:
        _validate_non_empty_string("spec_id", self.spec_id)
        _validate_positive_int("k_active", self.k_active)


@dataclass(frozen=True)
class AssignmentSetRequirementConfig:
    """Experiment-level requirement for an assignment-set asset."""

    spec_id: str

    def __post_init__(self) -> None:
        _validate_non_empty_string("spec_id", self.spec_id)


@dataclass(frozen=True)
class AssetRequirementsConfig:
    """Declared asset requirements for an experiment."""

    route_set: RouteSetRequirementConfig | None = None
    assignment_set: AssignmentSetRequirementConfig | None = None


@dataclass(frozen=True)
class AssetsConfig:
    """Top-level asset policy and requirements."""

    policy: AssetPolicyConfig = field(default_factory=AssetPolicyConfig)
    requirements: AssetRequirementsConfig = field(default_factory=AssetRequirementsConfig)


@dataclass(frozen=True)
class RouteSetBuilderConfig:
    """Recipe used to construct a route-set asset."""

    engine: str
    weight: str
    k_generate: int

    def __post_init__(self) -> None:
        _validate_non_empty_string("engine", self.engine)
        _validate_non_empty_string("weight", self.weight)
        _validate_positive_int("k_generate", self.k_generate)


@dataclass(frozen=True)
class RouteSetConstraintsConfig:
    """Route-generation constraints."""

    allow_duplicates: bool
    allow_loops: bool
    allow_auto_routes: bool

    def __post_init__(self) -> None:
        _validate_bool("allow_duplicates", self.allow_duplicates)
        _validate_bool("allow_loops", self.allow_loops)
        _validate_bool("allow_auto_routes", self.allow_auto_routes)


@dataclass(frozen=True)
class RouteSetConnectorsConfig:
    """Link types that count as connectors during route generation."""

    connector_link_types: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.connector_link_types, tuple):
            raise TypeError("connector_link_types must be a tuple of integers.")
        if not self.connector_link_types:
            raise ValueError("connector_link_types cannot be empty.")
        for value in self.connector_link_types:
            if not isinstance(value, int):
                raise TypeError("connector_link_types must contain integers only.")


@dataclass(frozen=True)
class RouteSetOrderingConfig:
    """Route ordering recipe for the route bank."""

    route_rank_policy: str
    cost_field: str

    def __post_init__(self) -> None:
        _validate_non_empty_string("route_rank_policy", self.route_rank_policy)
        _validate_non_empty_string("cost_field", self.cost_field)


@dataclass(frozen=True)
class RouteSetCompatibilityConfig:
    """Compatibility requirements for a route-set asset."""

    requires_network_fingerprint: bool
    requires_od_space_fingerprint: bool

    def __post_init__(self) -> None:
        _validate_bool("requires_network_fingerprint", self.requires_network_fingerprint)
        _validate_bool("requires_od_space_fingerprint", self.requires_od_space_fingerprint)


@dataclass(frozen=True)
class RouteSetStorageConfig:
    """Physical storage policy for route-set assets."""

    format: str
    directory: str

    def __post_init__(self) -> None:
        _validate_non_empty_string("format", self.format)
        _validate_non_empty_string("directory", self.directory)


@dataclass(frozen=True)
class RouteSetSpecConfig:
    """Strict route-set specification loaded from YAML."""

    id: str
    asset_type: str
    builder: RouteSetBuilderConfig
    constraints: RouteSetConstraintsConfig
    connectors: RouteSetConnectorsConfig
    ordering: RouteSetOrderingConfig
    compatibility: RouteSetCompatibilityConfig
    storage: RouteSetStorageConfig

    def __post_init__(self) -> None:
        _validate_non_empty_string("id", self.id)
        _validate_non_empty_string("asset_type", self.asset_type)
        if self.asset_type != "route_set":
            raise ValueError("route_set specs must declare asset_type='route_set'.")
        _validate_dataclass("builder", self.builder, RouteSetBuilderConfig)
        _validate_dataclass("constraints", self.constraints, RouteSetConstraintsConfig)
        _validate_dataclass("connectors", self.connectors, RouteSetConnectorsConfig)
        _validate_dataclass("ordering", self.ordering, RouteSetOrderingConfig)
        _validate_dataclass("compatibility", self.compatibility, RouteSetCompatibilityConfig)
        _validate_dataclass("storage", self.storage, RouteSetStorageConfig)


@dataclass(frozen=True)
class AssignmentSetBehaviorModelConfig:
    """Behavior model recipe for assignment-set assets."""

    name: str
    theta: float

    def __post_init__(self) -> None:
        _validate_non_empty_string("name", self.name)
        _validate_positive_finite_float("theta", self.theta)


@dataclass(frozen=True)
class AssignmentSetSolverConfig:
    """Solver recipe for assignment-set assets."""

    name: str
    max_iterations: int
    convergence_gap: float

    def __post_init__(self) -> None:
        _validate_non_empty_string("name", self.name)
        _validate_positive_int("max_iterations", self.max_iterations)
        _validate_positive_finite_float("convergence_gap", self.convergence_gap)


@dataclass(frozen=True)
class AssignmentSetVdfConfig:
    """VDF recipe for assignment-set assets."""

    name: str

    def __post_init__(self) -> None:
        _validate_non_empty_string("name", self.name)


@dataclass(frozen=True)
class AssignmentSetSpecConfig:
    """Strict assignment-set specification loaded from YAML."""

    id: str
    asset_type: str
    requires: RouteSetRequirementConfig
    behavior_model: AssignmentSetBehaviorModelConfig
    solver: AssignmentSetSolverConfig
    vdf: AssignmentSetVdfConfig

    def __post_init__(self) -> None:
        _validate_non_empty_string("id", self.id)
        _validate_non_empty_string("asset_type", self.asset_type)
        if self.asset_type != "assignment_set":
            raise ValueError("assignment_set specs must declare asset_type='assignment_set'.")
        _validate_dataclass("requires", self.requires, RouteSetRequirementConfig)
        _validate_dataclass("behavior_model", self.behavior_model, AssignmentSetBehaviorModelConfig)
        _validate_dataclass("solver", self.solver, AssignmentSetSolverConfig)
        _validate_dataclass("vdf", self.vdf, AssignmentSetVdfConfig)


def load_dataset_profile(data: Mapping[str, Any]) -> DatasetProfileConfig:
    """Parse the minimal dataset profile required by the asset pipeline."""

    mapping = _to_plain_mapping(data, context="dataset profile")
    if "dataset" in mapping and isinstance(mapping["dataset"], Mapping):
        mapping = _to_plain_mapping(mapping["dataset"], context="dataset profile.dataset")
    _require_required_keys(
        mapping,
        required={"nature", "data_availability"},
        context="dataset profile",
    )

    nature = _parse_enum(DatasetNature, mapping["nature"], "dataset profile.nature")
    availability = load_dataset_availability(mapping["data_availability"])
    return DatasetProfileConfig(nature=nature, data_availability=availability)


def load_dataset_availability(data: Mapping[str, Any]) -> DatasetAvailabilityConfig:
    """Parse dataset availability flags from YAML."""

    mapping = _to_plain_mapping(data, context="dataset availability")
    _require_exact_keys(
        mapping,
        required={
            "has_complete_od_ground_truth",
            "has_observed_link_flows",
            "has_ground_truth_assignment",
        },
        context="dataset availability",
    )
    return DatasetAvailabilityConfig(
        has_complete_od_ground_truth=_to_bool(
            "dataset availability.has_complete_od_ground_truth",
            mapping["has_complete_od_ground_truth"],
        ),
        has_observed_link_flows=_to_bool(
            "dataset availability.has_observed_link_flows",
            mapping["has_observed_link_flows"],
        ),
        has_ground_truth_assignment=_to_bool(
            "dataset availability.has_ground_truth_assignment",
            mapping["has_ground_truth_assignment"],
        ),
    )


def load_assets_config(data: Mapping[str, Any]) -> AssetsConfig:
    """Parse experiment asset requirements."""

    mapping = _to_plain_mapping(data, context="assets")
    _require_required_keys(mapping, required={"policy", "requirements"}, context="assets")

    policy = _load_asset_policy(mapping["policy"])
    requirements = _load_asset_requirements(mapping["requirements"])
    return AssetsConfig(policy=policy, requirements=requirements)


def load_route_set_spec(data: Mapping[str, Any]) -> RouteSetSpecConfig:
    """Parse a route-set spec from YAML."""

    mapping = _to_plain_mapping(data, context="route_set spec")
    _require_exact_keys(
        mapping,
        required={
            "id",
            "asset_type",
            "builder",
            "constraints",
            "connectors",
            "ordering",
            "compatibility",
            "storage",
        },
        context="route_set spec",
    )

    return RouteSetSpecConfig(
        id=_to_str("route_set spec.id", mapping["id"]),
        asset_type=_to_str("route_set spec.asset_type", mapping["asset_type"]),
        builder=_load_route_set_builder(mapping["builder"]),
        constraints=_load_route_set_constraints(mapping["constraints"]),
        connectors=_load_route_set_connectors(mapping["connectors"]),
        ordering=_load_route_set_ordering(mapping["ordering"]),
        compatibility=_load_route_set_compatibility(mapping["compatibility"]),
        storage=_load_route_set_storage(mapping["storage"]),
    )


def load_assignment_set_spec(data: Mapping[str, Any]) -> AssignmentSetSpecConfig:
    """Parse an assignment-set spec from YAML."""

    mapping = _to_plain_mapping(data, context="assignment_set spec")
    _require_exact_keys(
        mapping,
        required={"id", "asset_type", "requires", "behavior_model", "solver", "vdf"},
        context="assignment_set spec",
    )

    return AssignmentSetSpecConfig(
        id=_to_str("assignment_set spec.id", mapping["id"]),
        asset_type=_to_str("assignment_set spec.asset_type", mapping["asset_type"]),
        requires=_load_route_set_requirement(mapping["requires"]),
        behavior_model=_load_assignment_behavior_model(mapping["behavior_model"]),
        solver=_load_assignment_solver(mapping["solver"]),
        vdf=_load_assignment_vdf(mapping["vdf"]),
    )


def _load_asset_policy(data: Mapping[str, Any]) -> AssetPolicyConfig:
    mapping = _to_plain_mapping(data, context="assets.policy")
    _require_exact_keys(
        mapping,
        required={"on_missing", "on_stale", "overwrite_existing", "require_exact_fingerprint"},
        context="assets.policy",
    )
    return AssetPolicyConfig(
        on_missing=_to_str("assets.policy.on_missing", mapping["on_missing"]),
        on_stale=_to_str("assets.policy.on_stale", mapping["on_stale"]),
        overwrite_existing=_to_bool("assets.policy.overwrite_existing", mapping["overwrite_existing"]),
        require_exact_fingerprint=_to_bool(
            "assets.policy.require_exact_fingerprint",
            mapping["require_exact_fingerprint"],
        ),
    )


def _load_asset_requirements(data: Mapping[str, Any]) -> AssetRequirementsConfig:
    mapping = _to_plain_mapping(data, context="assets.requirements")
    _require_allowed_keys(mapping, allowed={"route_set", "assignment_set"}, context="assets.requirements")

    route_set = None
    if "route_set" in mapping and mapping["route_set"] is not None:
        route_set = _load_route_set_requirement(mapping["route_set"])

    assignment_set = None
    if "assignment_set" in mapping and mapping["assignment_set"] is not None:
        assignment_set = _load_assignment_set_requirement(mapping["assignment_set"])

    return AssetRequirementsConfig(route_set=route_set, assignment_set=assignment_set)


def _load_route_set_requirement(data: Mapping[str, Any]) -> RouteSetRequirementConfig:
    mapping = _to_plain_mapping(data, context="assets.requirements.route_set")
    _require_exact_keys(mapping, required={"spec_id", "k_active"}, context="assets.requirements.route_set")
    return RouteSetRequirementConfig(
        spec_id=_to_str("assets.requirements.route_set.spec_id", mapping["spec_id"]),
        k_active=_to_positive_int("assets.requirements.route_set.k_active", mapping["k_active"]),
    )


def _load_assignment_set_requirement(data: Mapping[str, Any]) -> AssignmentSetRequirementConfig:
    mapping = _to_plain_mapping(data, context="assets.requirements.assignment_set")
    _require_exact_keys(mapping, required={"spec_id"}, context="assets.requirements.assignment_set")
    return AssignmentSetRequirementConfig(
        spec_id=_to_str("assets.requirements.assignment_set.spec_id", mapping["spec_id"]),
    )


def _load_route_set_builder(data: Mapping[str, Any]) -> RouteSetBuilderConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.builder")
    _require_exact_keys(mapping, required={"engine", "weight", "k_generate"}, context="route_set spec.builder")
    return RouteSetBuilderConfig(
        engine=_to_str("route_set spec.builder.engine", mapping["engine"]),
        weight=_to_str("route_set spec.builder.weight", mapping["weight"]),
        k_generate=_to_positive_int("route_set spec.builder.k_generate", mapping["k_generate"]),
    )


def _load_route_set_constraints(data: Mapping[str, Any]) -> RouteSetConstraintsConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.constraints")
    _require_exact_keys(
        mapping,
        required={"allow_duplicates", "allow_loops", "allow_auto_routes"},
        context="route_set spec.constraints",
    )
    return RouteSetConstraintsConfig(
        allow_duplicates=_to_bool("route_set spec.constraints.allow_duplicates", mapping["allow_duplicates"]),
        allow_loops=_to_bool("route_set spec.constraints.allow_loops", mapping["allow_loops"]),
        allow_auto_routes=_to_bool("route_set spec.constraints.allow_auto_routes", mapping["allow_auto_routes"]),
    )


def _load_route_set_connectors(data: Mapping[str, Any]) -> RouteSetConnectorsConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.connectors")
    _require_exact_keys(mapping, required={"connector_link_types"}, context="route_set spec.connectors")
    connector_values = mapping["connector_link_types"]
    sequence_types: tuple[type[Any], ...] = (list, tuple)
    if ListConfig is not None:
        sequence_types = sequence_types + (ListConfig,)
    if not isinstance(connector_values, sequence_types):
        raise TypeError("route_set spec.connectors.connector_link_types must be a list of integers.")
    values = tuple(int(value) for value in connector_values)
    if not values:
        raise ValueError("route_set spec.connectors.connector_link_types cannot be empty.")
    return RouteSetConnectorsConfig(connector_link_types=values)


def _load_route_set_ordering(data: Mapping[str, Any]) -> RouteSetOrderingConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.ordering")
    _require_exact_keys(
        mapping,
        required={"route_rank_policy", "cost_field"},
        context="route_set spec.ordering",
    )
    return RouteSetOrderingConfig(
        route_rank_policy=_to_str("route_set spec.ordering.route_rank_policy", mapping["route_rank_policy"]),
        cost_field=_to_str("route_set spec.ordering.cost_field", mapping["cost_field"]),
    )


def _load_route_set_compatibility(data: Mapping[str, Any]) -> RouteSetCompatibilityConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.compatibility")
    _require_exact_keys(
        mapping,
        required={"requires_network_fingerprint", "requires_od_space_fingerprint"},
        context="route_set spec.compatibility",
    )
    return RouteSetCompatibilityConfig(
        requires_network_fingerprint=_to_bool(
            "route_set spec.compatibility.requires_network_fingerprint",
            mapping["requires_network_fingerprint"],
        ),
        requires_od_space_fingerprint=_to_bool(
            "route_set spec.compatibility.requires_od_space_fingerprint",
            mapping["requires_od_space_fingerprint"],
        ),
    )


def _load_route_set_storage(data: Mapping[str, Any]) -> RouteSetStorageConfig:
    mapping = _to_plain_mapping(data, context="route_set spec.storage")
    _require_exact_keys(
        mapping,
        required={"format", "directory"},
        context="route_set spec.storage",
    )
    return RouteSetStorageConfig(
        format=_to_str("route_set spec.storage.format", mapping["format"]),
        directory=_to_str("route_set spec.storage.directory", mapping["directory"]),
    )


def _load_assignment_behavior_model(data: Mapping[str, Any]) -> AssignmentSetBehaviorModelConfig:
    mapping = _to_plain_mapping(data, context="assignment_set spec.behavior_model")
    _require_exact_keys(
        mapping,
        required={"name", "theta"},
        context="assignment_set spec.behavior_model",
    )
    return AssignmentSetBehaviorModelConfig(
        name=_to_str("assignment_set spec.behavior_model.name", mapping["name"]),
        theta=_to_positive_finite_float(
            "assignment_set spec.behavior_model.theta",
            mapping["theta"],
        ),
    )


def _load_assignment_solver(data: Mapping[str, Any]) -> AssignmentSetSolverConfig:
    mapping = _to_plain_mapping(data, context="assignment_set spec.solver")
    _require_exact_keys(
        mapping,
        required={"name", "max_iterations", "convergence_gap"},
        context="assignment_set spec.solver",
    )
    return AssignmentSetSolverConfig(
        name=_to_str("assignment_set spec.solver.name", mapping["name"]),
        max_iterations=_to_positive_int(
            "assignment_set spec.solver.max_iterations",
            mapping["max_iterations"],
        ),
        convergence_gap=_to_positive_finite_float(
            "assignment_set spec.solver.convergence_gap",
            mapping["convergence_gap"],
        ),
    )


def _load_assignment_vdf(data: Mapping[str, Any]) -> AssignmentSetVdfConfig:
    mapping = _to_plain_mapping(data, context="assignment_set spec.vdf")
    _require_exact_keys(mapping, required={"name"}, context="assignment_set spec.vdf")
    return AssignmentSetVdfConfig(name=_to_str("assignment_set spec.vdf.name", mapping["name"]))


def _validate_dataclass(name: str, value: Any, expected_type: type[Any]) -> None:
    if not isinstance(value, expected_type):
        raise TypeError(f"{name} must be a {expected_type.__name__} instance.")


def _to_plain_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if OmegaConf is not None and OmegaConf.is_config(value):  # type: ignore[attr-defined]
        value = OmegaConf.to_container(value, resolve=True)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"{context} must be a mapping.")


def _require_exact_keys(mapping: Mapping[str, Any], *, required: set[str], context: str) -> None:
    missing = required - set(mapping.keys())
    unknown = set(mapping.keys()) - required
    if missing:
        raise KeyError(f"{context} is missing required keys: {sorted(missing)}")
    if unknown:
        raise KeyError(f"{context} contains unknown keys: {sorted(unknown)}")


def _require_required_keys(mapping: Mapping[str, Any], *, required: set[str], context: str) -> None:
    missing = required - set(mapping.keys())
    if missing:
        raise KeyError(f"{context} is missing required keys: {sorted(missing)}")


def _require_allowed_keys(mapping: Mapping[str, Any], *, allowed: set[str], context: str) -> None:
    unknown = set(mapping.keys()) - allowed
    if unknown:
        raise KeyError(f"{context} contains unknown keys: {sorted(unknown)}")


def _validate_choice(name: str, value: str, allowed: set[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    if value not in allowed:
        raise ValueError(f"{name} must be one of {sorted(allowed)}. Received {value!r}.")


def _validate_bool(name: str, value: Any) -> None:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")


def _validate_non_empty_string(name: str, value: Any) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")


def _validate_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")


def _validate_positive_finite_float(name: str, value: Any) -> None:
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be a positive finite float.")


def _to_str(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def _to_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool.")
    return value


def _to_positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return int(value)


def _to_positive_finite_float(name: str, value: Any) -> float:
    number = float(value)
    if not (math.isfinite(number) and number > 0.0):
        raise ValueError(f"{name} must be a positive finite float.")
    return number


def _parse_enum(enum_type: type[Enum], value: Any, context: str) -> Enum:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} must be a non-empty string.")
    try:
        return enum_type(value)
    except ValueError as exc:
        allowed = [member.value for member in enum_type]
        raise ValueError(f"{context} must be one of {allowed}. Received {value!r}.") from exc
