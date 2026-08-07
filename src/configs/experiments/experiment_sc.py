from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# =============================================================================
# Experiment Structured Configs
#
# Schema that backs the experiment overlay files under configs/experiments/*.yaml
# for everything consumed by src/components/artifacts/asset_pipeline.py:
#   - experiment section (name / description);
#   - data_selection (volume_year / weight_column);
#   - assets (policy + requirements.route_set / requirements.assignment_set).
#
# Training-related sections (training, model, vdf, sampling, data_handling) are
# intentionally out of scope for now.
# =============================================================================


@dataclass
class ExperimentSectionConfig:
    name: str = MISSING
    description: Optional[str] = None


@dataclass
class ExperimentDataSelectionConfig:
    volume_year: Optional[str] = None
    weight_column: Optional[str] = None
    description: Optional[str] = None


@dataclass
class ExperimentRouteSetRequirementConfig:
    spec_id: str = MISSING
    k_active: int = MISSING
    weight_column: Optional[str] = None


@dataclass
class ExperimentAssignmentSetRequirementConfig:
    spec_id: str = MISSING


@dataclass
class ExperimentAssetRequirementsConfig:
    route_set: Optional[ExperimentRouteSetRequirementConfig] = None
    assignment_set: Optional[ExperimentAssignmentSetRequirementConfig] = None


@dataclass
class ExperimentAssetPolicyConfig:
    on_missing: str = "fail"
    on_stale: str = "fail"
    overwrite_existing: bool = False
    require_exact_fingerprint: bool = True


@dataclass
class ExperimentAssetsConfig:
    policy: ExperimentAssetPolicyConfig = field(default_factory=ExperimentAssetPolicyConfig)
    requirements: ExperimentAssetRequirementsConfig = field(
        default_factory=ExperimentAssetRequirementsConfig
    )


@dataclass
class ExperimentConfig:
    experiment: ExperimentSectionConfig = field(default_factory=ExperimentSectionConfig)
    data_selection: ExperimentDataSelectionConfig = field(
        default_factory=ExperimentDataSelectionConfig
    )
    assets: ExperimentAssetsConfig = field(default_factory=ExperimentAssetsConfig)


cs = ConfigStore.instance()
cs.store(group="experiments", name="experiment", node=ExperimentConfig)
