from abc import ABC, abstractmethod
import networkx as nx
from typing import List, Tuple, Dict, Set, Optional

Route = List[int]

class RouteEngine(ABC):
    """
    Abstract base class for route calculation engines.
    """
    
    @abstractmethod
    def __init__(self, graph: nx.DiGraph):
        """
        Initialize the engine with a NetworkX graph.
        Engines that use a different backend (like RustworkX) should perform
        the conversion from the NetworkX graph during initialization.
        """
        pass

    @abstractmethod
    def get_k_routes(
        self,
        origin_id: int,
        destination_id: int,
        k: int,
        weight: str,
        constraints: Dict[str, bool],
        connector_link_types: Set[int]
    ) -> List[Route]:
        """
        Calculates the K shortest paths from origin to destination.
        
        Args:
            origin_id: The origin node ID.
            destination_id: The destination node ID.
            k: The maximum number of routes to return.
            weight: The edge attribute to use as weight.
            constraints: Dictionary with boolean constraints:
                - allow_duplicates: If True, duplicate paths are allowed.
                - allow_loops: If True, loops within a path are allowed.
                - allow_auto_routes: If True, paths where origin == destination are allowed.
            connector_link_types: Set of link types considered as connectors.
            
        Returns:
            A list of routes, where each route is a list of node IDs.
        """
        pass
