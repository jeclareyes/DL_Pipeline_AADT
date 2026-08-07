from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from typing import List, Optional


# =============================================================================
# Data Processing Structured Configs
#
# Schema that backs configs/data_ingestion/data_processing/data_processing.yaml,
# consumed at runtime as cfg.data_handling.data_processing.
# =============================================================================


@dataclass
class NodeReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True
    required_columns: List[str] = field(
        default_factory=lambda: ["node_id", "x", "y", "type", "class"]
    )


@dataclass
class NetworkOptionalDefaultsConfig:
    lanes: int = 1
    b: float = 0.15
    power: float = 4.0
    speed: Optional[float] = None
    vdf: Optional[float] = None
    toll: float = 0.0
    link_type: int = 0


@dataclass
class NetworkReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True
    required_columns: List[str] = field(
        default_factory=lambda: [
            "init_node",
            "term_node",
            "capacity_per_lane",
            "total_capacity",
            "effective_capacity",
            "length",
            "free_flow_time",
        ]
    )
    apply_default_optional_columns: bool = False
    optional_defaults: NetworkOptionalDefaultsConfig = field(
        default_factory=NetworkOptionalDefaultsConfig
    )


@dataclass
class FlowReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True


@dataclass
class TripReaderConfig:
    strict: bool = True
    aggregation: str = "average_daily"
    multiday_od: bool = True
    matrix_format: str = "csr"
    include_zero_flows: bool = False
    use_zone_id_mapping: bool = True
    infer_zone_ids_from_origin_lines: bool = True


@dataclass
class RouteReaderConfig:
    strict: bool = True
    allow_empty_routes: bool = True
    preserve_empty_od_pairs: bool = True
    compact_format_uses_zone_ids: bool = False
    max_routes_per_od: int = 10


@dataclass
class ReadersConfig:
    nodes: NodeReaderConfig = field(default_factory=NodeReaderConfig)
    network: NetworkReaderConfig = field(default_factory=NetworkReaderConfig)
    flows: FlowReaderConfig = field(default_factory=FlowReaderConfig)
    trips: TripReaderConfig = field(default_factory=TripReaderConfig)
    routes: RouteReaderConfig = field(default_factory=RouteReaderConfig)


@dataclass
class LinkTableBuilderConfig:
    strict: bool = True
    preserve_extra_flow_columns: bool = True
    aggregate_duplicate_flows: bool = True
    duplicate_flow_aggregation: str = "mean"


@dataclass
class GraphBuilderConfig:
    strict: bool = True
    dataset_weight_column: str = "free_flow_time"
    preserve_extra_attributes: bool = True


@dataclass
class OdIndexingConfig:
    include_intrazonal_pairs: bool = True


@dataclass
class BuildersConfig:
    link_table: LinkTableBuilderConfig = field(default_factory=LinkTableBuilderConfig)
    graph: GraphBuilderConfig = field(default_factory=GraphBuilderConfig)
    od_indexing: OdIndexingConfig = field(default_factory=OdIndexingConfig)


@dataclass
class ValidationConfig:
    enabled: bool = True
    strict: bool = True
    check_route_graph_compatibility: bool = True
    max_reported_items: int = 20


@dataclass
class ArtifactConfig:
    save_joblib: bool = True
    save_manifest: bool = True


@dataclass
class DataProcessingConfig:
    description: str = ""
    readers: ReadersConfig = field(default_factory=ReadersConfig)
    builders: BuildersConfig = field(default_factory=BuildersConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)


cs = ConfigStore.instance()
cs.store(group="data_ingestion/data_processing", name="data_processing", node=DataProcessingConfig)
