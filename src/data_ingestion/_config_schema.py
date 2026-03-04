# python
import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Union
from omegaconf import MISSING

@dataclass
class InputRoutes:
    general_route: str = MISSING
    flow_route: str = MISSING
    network_route: str = MISSING
    node_route: str = MISSING
    routes_route: Optional[str] = None
    trips_route: Optional[str] = None

#%%
@dataclass
class OutputRoutes:
    root: str = MISSING
    processed_route: str = MISSING
    routing_cache_route: str = MISSING


#%%
@dataclass
class DataProcessingConfig:
    dataset: str = MISSING
    random_seed: int = MISSING
    multiday_od: bool = MISSING
    volume_year: Optional[Union[int, str, bool]] = None
    input_routes: InputRoutes = dataclasses.field(default_factory=InputRoutes)
    output_routes: OutputRoutes = dataclasses.field(default_factory=OutputRoutes)
    route_calculation: Dict[str, Any] = field(default_factory=dict)
