from dataclasses import dataclass, field
from typing import Optional, List, Dict, Union, Any

# -----------------------------------------------------------------------------
# Dataclasses for Data Processing Configuration
# -----------------------------------------------------------------------------

@dataclass
class InputRoutesConfig:
    general_route: str
    node_route: str
    network_route: str
    trips_route: str
    routes_route: str
    flow_route: str

@dataclass
class OutputRoutesConfig:
    processed_route: str
    artifact_filename: str
    manifest_filename: str
    overwrite_existing: bool = True
    save_manifest: bool = True

@dataclass
class NodeReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True
    required_columns: List[str] = field(default_factory=lambda: ["node_id", "x", "y", "type"])

@dataclass
class NetworkReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True
    required_columns: List[str] = field(default_factory=lambda: [
        "init_node", "term_node", "capacity_per_lane", "total_capacity",
        "effective_capacity", "length", "free_flow_time"
    ])
    apply_default_optional_columns: bool = False
    optional_defaults: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FlowReaderConfig:
    strict: bool = True
    preserve_extra_columns: bool = True

@dataclass
class TripReaderConfig:
    strict: bool = True
    aggregation: str = "average_daily"
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
    max_routes_per_od: Optional[int] = None

@dataclass
class ReadersConfig:
    nodes: NodeReaderConfig = field(default_factory=NodeReaderConfig)
    network: NetworkReaderConfig = field(default_factory=NetworkReaderConfig)
    flows: FlowReaderConfig = field(default_factory=FlowReaderConfig)
    trips: TripReaderConfig = field(default_factory=TripReaderConfig)
    routes: RouteReaderConfig = field(default_factory=RouteReaderConfig)

@dataclass
class LinkTableConfig:
    strict: bool = True
    preserve_extra_flow_columns: bool = True
    aggregate_duplicate_flows: bool = True
    duplicate_flow_aggregation: str = "mean"

@dataclass
class GraphBuilderConfig:
    strict: bool = True
    add_missing_link_nodes: bool = False
    preserve_extra_attributes: bool = True
    weight_column: str = "free_flow_time"
    store_edge_endpoints_as_attributes: bool = True
    edge_order_source: str = "link_df_row_order"
    node_order_source: str = "node_df_row_order"

@dataclass
class OdIndexingConfig:
    use_real_zone_ids: bool = True
    zone_id_source_priority: List[str] = field(default_factory=lambda: ["node_class", "node_type", "origin_lines"])
    zone_class_values: List[str] = field(default_factory=lambda: ["zone", "zones", "taz"])
    exclude_intrazonal_pairs: bool = False
    store_zone_id_to_idx: bool = True
    store_idx_to_zone_id: bool = True

@dataclass
class CompressionConfig:
    enabled: bool = False
    method: Optional[str] = None
    level: Optional[int] = None

@dataclass
class ArtifactConfig:
    artifact_type: str = "base_artifact"
    artifact_version: str = "1.0"
    save_joblib: bool = True
    save_manifest: bool = True
    include_raw_layer: bool = True
    include_processed_layer: bool = True
    include_model_ready_layer: bool = False
    include_reader_metadata: bool = True
    include_builder_metadata: bool = True
    include_environment_metadata: bool = True
    include_config_snapshot: bool = True
    compression: CompressionConfig = field(default_factory=CompressionConfig)

@dataclass
class LoggingConfig:
    level: str = "INFO"
    print_summary: bool = True
    log_reader_summaries: bool = True
    log_builder_summaries: bool = True
    save_diagnostics: bool = True
    diagnostics_filename: str = "data_processing_diagnostics.json"

@dataclass
class DebugConfig:
    enabled: bool = False
    verbose: bool = False
    dry_run: bool = False
    sample_mode: bool = False

@dataclass
class DataProcessingConfig:
    description: str = ""
    input_routes: InputRoutesConfig = field(default_factory=InputRoutesConfig)
    output_routes: OutputRoutesConfig = field(default_factory=OutputRoutesConfig)
    readers: ReadersConfig = field(default_factory=ReadersConfig)
    link_table: LinkTableConfig = field(default_factory=LinkTableConfig)
    graph: GraphBuilderConfig = field(default_factory=GraphBuilderConfig)
    od_indexing: OdIndexingConfig = field(default_factory=OdIndexingConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    debug: DebugConfig = field(default_factory=DebugConfig)
