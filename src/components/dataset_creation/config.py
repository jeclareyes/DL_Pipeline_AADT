"""
This module defines the structural contract for the scenario creation configuration.
All configurations are strongly typed using OmegaConf structured dataclasses
to enforce validation at system boundaries and guarantee scientific reproducibility.

According to the Project Constitution:
- Configuration is the single source of truth.
- Unknown or mismatched keys will cause explicit execution failures.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

# ==============================================================================
# 1. PATHING CONFIGURATION (INFRASTRUCTURE LAYER)
# ==============================================================================

@dataclass
class TNTPFileNamesConfig:
    """Defines the rigid naming convention for standard TNTP network files."""
    network: str = MISSING
    nodes: str = MISSING
    trips: str = MISSING
    routes: str = MISSING
    flows: str = MISSING
    info: str = MISSING

@dataclass
class FolderNamesConfig:
    """Defines the baseline partition names within the data storage ecosystem."""
    raw: str = MISSING
    info: str = MISSING
    
@dataclass
class ExportDirs:
    """Absolute or resolved runtime directories managed by the orchestrator."""
    generated_dataset: str = MISSING
    info_dataset: str = MISSING

@dataclass
class ExportFilePaths:
    """Target filepaths for serialization of networks, demands, and telemetry."""
    nodes: str = MISSING
    network: str = MISSING
    trips: str = MISSING
    routes: str = MISSING
    flows: str = MISSING
    info: str = MISSING

@dataclass
class PathsConfig:
    """Centralizes path resolutions, utilizing anchors and dynamic environment routing."""
    root_dir: str = MISSING
    folder_names: FolderNamesConfig = field(default_factory=FolderNamesConfig)
    file_names: TNTPFileNamesConfig = field(default_factory=TNTPFileNamesConfig)
    export_dirs: ExportDirs = field(default_factory=ExportDirs)
    export_filepaths: ExportFilePaths = field(default_factory=ExportFilePaths)

# ==============================================================================
# 2. TNTP SCIENTIFIC PARAMETERS (DOMAIN LAYER)
# ==============================================================================

@dataclass
class NodeQuantityConfig:
    """Enforces explicit numbers of objects to generate per Node category."""
    TAZ: int = MISSING
    AUX: int = MISSING
    INTERSECTION: int = MISSING

@dataclass
class NodeCoordinatesConfig:
    """Spatial bounding box constraints for geographical grid deployment."""
    x_min: float = MISSING
    x_max: float = MISSING
    y_min: float = MISSING
    y_max: float = MISSING

@dataclass
class NodeTypesConfig:
    """Categorized topology division between demand zones and non-zone nodes."""
    Zones: List[str] = MISSING
    NonZones: List[str] = MISSING

@dataclass
class NodeTabularConfig:
    """Defines structural boundaries and required schemas for dataframes."""
    Columns: List[str] = MISSING

@dataclass
class NodeParametersConfig:
    """Assembles node properties, counts, limits, and topological classifications."""
    Quantity: NodeQuantityConfig = field(default_factory=NodeQuantityConfig)
    Coordinates: NodeCoordinatesConfig = field(default_factory=NodeCoordinatesConfig)
    Types: NodeTypesConfig = field(default_factory=NodeTypesConfig)
    Tabular: NodeTabularConfig = field(default_factory=NodeTabularConfig)

# ==============================================================================
# 3. DEMAND PARAMETERS
# ==============================================================================

@dataclass
class DemandMagnitudeConfig:
    """Limits bounds for the generated Origin-Destination demand matrix elements."""
    min_trips_od: int = MISSING
    max_trips_od: int = MISSING

@dataclass
class DemandTabularConfig:
    """Validation column checklist for the raw demand tables."""
    Columns: List[str] = MISSING

@dataclass
class DemandParametersConfig:
    """Orchestrates temporal profiles, day intervals, and volume magnitude limits."""
    Num_Days: int = MISSING
    Start_Date: str = MISSING
    Time_Interval: str = MISSING
    Accept_IntraZonal_Demand: bool = MISSING
    Magnitude: DemandMagnitudeConfig = field(default_factory=DemandMagnitudeConfig)
    Profile: Dict[str, float] = field(default_factory=dict)
    Tabular: DemandTabularConfig = field(default_factory=DemandTabularConfig)

# ==============================================================================
# 4. NETWORK & TOPOLOGY CATALOGUES
# ==============================================================================

@dataclass
class NetworkTabularConfig:
    """Column requirements mapping input data features to output assignment states."""
    Assignment_Columns: List[str] = MISSING
    Base_Columns: List[str] = MISSING
    Columns: List[str] = MISSING

@dataclass
class LinkCatalogueItem:
    """Catalogue entry for explicit behavioral parameters of link classifications."""
    description: str = MISSING
    is_connector: bool = MISSING
    capacity: int = MISSING
    lanes: int = MISSING
    toll: int = MISSING
    allowed_vdfs: List[int] = field(default_factory=list)

@dataclass
class VDFCatalogueItem:
    """Volume Delay Function cost profile entry parameters."""
    description: str = MISSING
    b: float = MISSING
    power: float = MISSING
    speed_limit: float = MISSING

@dataclass
class NetworkCataloguesConfig:
    """Encapsulates structured dictionaries for deterministic infrastructure settings."""
    Network: Dict[int, LinkCatalogueItem] = field(default_factory=dict)
    VDF: Dict[int, VDFCatalogueItem] = field(default_factory=dict)

@dataclass
class NetworkParametersConfig:
    """Combines link/VDF catalogs with administrative data tabular contracts."""
    Tabular: NetworkTabularConfig = field(default_factory=NetworkTabularConfig)
    Catalogues: NetworkCataloguesConfig = field(default_factory=NetworkCataloguesConfig)

# ==============================================================================
# 5. ALGORITHMIC & SOLVER CONTROL PARAMETERS
# ==============================================================================

@dataclass
class RouteParametersConfig:
    """Routing constraints mapping limits for shortest paths algorithm calculations."""
    K_paths: int = MISSING
    Weight: str = MISSING
    allow_duplicates: bool = MISSING
    allow_loops: bool = MISSING
    allow_auto_routes: bool = MISSING

@dataclass
class AssignmentTabularConfig:
    """Enforces mathematical columns for output flow vectors."""
    Columns: List[str] = MISSING

@dataclass
class ConvergenceThresholdsConfig:
    """Mathematical convergence criteria metrics for macro traffic assignment."""
    equilibrium_l1_threshold: float = MISSING
    max_absolute_gap_threshold: float = MISSING
    max_relative_gap_threshold: float = MISSING
    min_flow_for_relative_gap: float = MISSING

@dataclass
class SUEParamsConfig:
    """Stochastic User Equilibrium logit dispersion parameters."""
    theta: float = MISSING
    step_size: float = MISSING

@dataclass
class MSAParamsConfig:
    """Method of Successive Averages dampening constraints step rules."""
    msa_step_rule: str = MISSING

@dataclass
class BalanceToleranceItem:
    """Absolute and relative thresholds for scientific network integrity checks."""
    absolute_tolerance: float = MISSING
    relative_tolerance: float = MISSING

@dataclass
class AssignmentControlConfig:
    """Limits conservation of flow errors across demographic nodes."""
    DemandBalance: BalanceToleranceItem = field(default_factory=BalanceToleranceItem)
    ZoneNodeBalance: BalanceToleranceItem = field(default_factory=BalanceToleranceItem)
    NonZoneNodeBalance: BalanceToleranceItem = field(default_factory=BalanceToleranceItem)

@dataclass
class AssignmentParametersConfig:
    """Assembles adjustment factors, paradigms, methods, and solver metrics."""
    Capacity_Adjustment_Factors: Dict[str, float] = field(default_factory=dict)
    Paradigm: str = MISSING
    Method: str = MISSING
    Max_Iterations: int = MISSING
    ConvergenceParameters: ConvergenceThresholdsConfig = field(default_factory=ConvergenceThresholdsConfig)
    SUE_Parameters: SUEParamsConfig = field(default_factory=SUEParamsConfig)
    MSA: MSAParamsConfig = field(default_factory=MSAParamsConfig)
    Tabular: AssignmentTabularConfig = field(default_factory=AssignmentTabularConfig)
    AssignmentControl: AssignmentControlConfig = field(default_factory=AssignmentControlConfig)

# ==============================================================================
# 6. VISUALIZATION & MISCELLANEOUS
# ==============================================================================

@dataclass
class NumDecimalsConfig:
    """Controls numeric precision rendering throughout data generation exports."""
    coordinates: int = MISSING
    demand: int = MISSING
    flows: int = MISSING

@dataclass
class VisualElementStyle:
    """Stylistic rendering variables for matplotlib/GIS graph representations."""
    Size: int = MISSING
    Color: str = MISSING
    Show_Label: bool = MISSING
    Marker: str = MISSING
    Label: str = MISSING

@dataclass
class VisualEdgeStyle:
    """Stylistic configuration boundaries for spatial edges/links."""
    Width: float = MISSING
    Color: str = MISSING
    Show_Label: bool = MISSING
    Label_Size: int = MISSING
    Label_Color: str = MISSING


@dataclass
class VDFCurvesVisualConfig:
    """Stylistic configuration boundaries for VDF curve visualizations."""
    Show: bool = MISSING
    Color: str = MISSING
    Width: float = MISSING
    Show_Label: bool = MISSING
    Label_Size: int = MISSING
    Label_Color: str = MISSING

@dataclass
class NodesVisualConfig:
    """Encapsulates canvas styling parameters specific to physical vertices."""
    Label_Size: int = MISSING
    Label_Color: str = MISSING
    TAZ: VisualElementStyle = field(default_factory=VisualElementStyle)
    AUX: VisualElementStyle = field(default_factory=VisualElementStyle)
    INTERSECTIONS: VisualElementStyle = field(default_factory=VisualElementStyle)

@dataclass
class VisualizationParametersConfig:
    """Centralizes visualization graph properties for nodes, edges, and cost paths."""
    Nodes: NodesVisualConfig = field(default_factory=NodesVisualConfig)
    Edges: VisualEdgeStyle = field(default_factory=VisualEdgeStyle)
    VDF_Curves: VDFCurvesVisualConfig = field(default_factory=VDFCurvesVisualConfig)

# ==============================================================================
# 7. ROOT COMPOSITION SCHEMA (ORCHESTRATOR)
# ==============================================================================

@dataclass
class DatasetConfig:
    """
    The top-level unified system configuration schema.
    Acts as the concrete class definition matching the composed config root.
    """
    name: str = MISSING
    dataset_to_create: str = MISSING
    description: str = MISSING
    seed: int = MISSING
    overwrite_existing: bool = MISSING

    # Infrastructure Layout
    paths: PathsConfig = field(default_factory=PathsConfig)

    # Scientific transport parameters
    NodeParameters: NodeParametersConfig = field(default_factory=NodeParametersConfig)
    DemandParameters: DemandParametersConfig = field(default_factory=DemandParametersConfig)
    NetworkParameters: NetworkParametersConfig = field(default_factory=NetworkParametersConfig)
    RouteParameters: RouteParametersConfig = field(default_factory=RouteParametersConfig)
    AssignmentParameters: AssignmentParametersConfig = field(default_factory=AssignmentParametersConfig)

    # Miscellaneous configs
    VisualizationParameters: VisualizationParametersConfig = field(default_factory=VisualizationParametersConfig)
    num_decimals: NumDecimalsConfig = field(default_factory=NumDecimalsConfig)


cs = ConfigStore.instance()
cs.store(name="base_config", node=DatasetConfig)
