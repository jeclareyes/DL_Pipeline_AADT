from .base_engine import RouteEngine
from .networkx_engine import NetworkXEngine
from .rustworkx_engine import RustworkXEngine
from .rustworkx_optimized_engine import RustworkXOptimizedEngine

def get_route_engine(engine_name: str, graph) -> RouteEngine:
    name_lower = engine_name.lower()
    if name_lower == "networkx":
        return NetworkXEngine(graph)
    elif name_lower == "rustworkx":
        return RustworkXEngine(graph)
    elif name_lower in ("rustworkx_optimized", "rx_optimized"):
        return RustworkXOptimizedEngine(graph)
    else:
        raise ValueError(f"Unknown route engine: {engine_name}")

