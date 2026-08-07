from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Optional
from omegaconf import MISSING
from enum import Enum


# =============================================================================
# Dataset Structured Configs
#
# Schema that backs the YAML files under configs/dataset/*.yaml
# (e.g. Synthetic05.yaml, Linkoping.yaml).
# =============================================================================


class DatasetNature(Enum):
    real = "real"
    synthetic = "synthetic"


@dataclass
class DatasetAvailabilityConfig:
    has_full_od_ground_truth: bool = MISSING
    has_ground_truth_link_flows: bool = MISSING
    has_observed_link_flows: bool = MISSING


@dataclass
class NodeClassificationRuleConfig:
    class_values: List[str] = MISSING
    type_values: List[str] = MISSING
    include_in_od: bool = MISSING


@dataclass
class NodeClassificationNormalizationConfig:
    case_sensitive: bool = False
    strip_whitespace: bool = True


@dataclass
class NodeClassificationConfig:
    rules: List[NodeClassificationRuleConfig] = MISSING
    normalization: NodeClassificationNormalizationConfig = field(
        default_factory=NodeClassificationNormalizationConfig
    )


@dataclass
class DatasetNetworkConfig:
    weight_column: str = MISSING


@dataclass
class DatasetFlowColumnsConfig:
    traffic_counts: List[str] = field(default_factory=list)
    reference_assignment: Optional[str] = None
    estimated_flows: Optional[str] = None


@dataclass
class TntpFilesConfig:
    nodes: str = MISSING
    network: str = MISSING
    flows: str = MISSING
    trips: str = MISSING
    routes: str = MISSING


@dataclass
class DatasetManifestsConfig:
    creation: Optional[str] = MISSING
    base: Optional[str] = MISSING


@dataclass
class DatasetArtifactsConfig:
    creation: Optional[str] = MISSING
    base: Optional[str] = MISSING


@dataclass
class DatasetAssetsConfig:
    assignment_banks: str = MISSING
    route_banks: str = MISSING


@dataclass
class PathsConfig:
    raw_dir: str = MISSING
    processed_dir: str = MISSING
    tntp_files: TntpFilesConfig = field(default_factory=TntpFilesConfig)
    manifests: DatasetManifestsConfig = field(default_factory=DatasetManifestsConfig)
    artifacts: DatasetArtifactsConfig = field(default_factory=DatasetArtifactsConfig)
    assets: DatasetAssetsConfig = field(default_factory=DatasetAssetsConfig)


@dataclass
class DatasetConfig:
    name: str
    nature: DatasetNature = MISSING

    data_availability: DatasetAvailabilityConfig = field(
        default_factory=DatasetAvailabilityConfig
    )
    node_classification: NodeClassificationConfig = field(
        default_factory=NodeClassificationConfig
    )
    network: DatasetNetworkConfig = field(default_factory=DatasetNetworkConfig)
    flow_columns: DatasetFlowColumnsConfig = field(
        default_factory=DatasetFlowColumnsConfig
    )
    paths: PathsConfig = field(default_factory=PathsConfig)


cs = ConfigStore.instance()
cs.store(group="dataset", name="dataset_config", node=DatasetConfig)
