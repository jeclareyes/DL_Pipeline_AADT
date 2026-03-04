import torch
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any


class RouteModelAdapter:
    """
    Adapter for route-based traffic assignment models (Cyclic, Static/Ultra, SPSA).

    This adapter serves as a translation layer between raw transportation network data 
    (NetworkX graphs and route dictionaries) and PyTorch tensor representations required 
    by traffic assignment models. It handles the complex transformation of sparse route 
    structures into memory-efficient tensor formats suitable for GPU computation.

    Key Responsibilities:
    - Convert NetworkX DiGraph to structured PyTorch tensors (link attributes)
    - Transform K-shortest path routes into sparse incidence matrices
    - Ensure consistent edge indexing across all tensor representations
    - Map categorical link attributes to continuous indices for embeddings

    Attributes:
        device (str): PyTorch device for tensor allocation ('cpu' or 'cuda')

    Example Usage:
        >>> adapter = RouteModelAdapter(device='cuda')
        >>> graph = nx.DiGraph()  # Loaded transportation network
        >>> routes = {...}  # K-shortest paths dictionary
        >>> model_inputs = adapter.transform(graph, routes)
        >>> model = CyclicModel(**model_inputs)
    """

    def __init__(self, device: str = "cpu"):
        """
        Initialize the adapter with target computation device.

        Args:
            device (str): Target device for tensor allocation. Options: 'cpu', 'cuda', 
                         'cuda:0', etc. Default is 'cpu'.
        """
        self.device = device

    def transform(self, graph: nx.DiGraph, routes_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Main transformation pipeline: converts raw network data to model-ready tensors.

        This orchestrator function coordinates the entire data transformation process:
        1. Establishes canonical edge ordering (critical for index consistency)
        2. Extracts physical network properties (free-flow times, capacities)
        3. Builds sparse route-link incidence matrices (supporting both legacy 3D and 
           optimized 2D topological formats).
        4. Packages all components into a unified dictionary

        Args:
            graph (nx.DiGraph): NetworkX directed graph representing the transportation 
                               network. Each edge must contain attributes:
                               - 'free_flow_time': Link travel time at zero flow
                               - 'capacity': Maximum flow (vehicles/hour)
                               - 'length': Physical length (meters or km)
                               - 'lanes': Number of lanes (integer)
                               - 'speed': Speed limit
                               - 'link_type': Categorical road type (highway, arterial, etc.)

            routes_data (Dict[str, Any]): Route information in one of two formats:
                                         Format A (Raw Dict): {(origin, dest): [[node_path1], [node_path2], ...]}
                                         Format B (Processed): {'routes': np.ndarray, 'od_pairs': list}

        Returns:
            Dict[str, torch.Tensor]: Complete model input dictionary containing:
                - 'num_links' (int): Total number of links in network
                - 'num_od_pairs' (int): Number of origin-destination pairs

                - 'route_masks' (torch.sparse.FloatTensor): 3D legacy incidence [OD, K, Links]
                - 'delta_matrix' (torch.sparse.FloatTensor): 2D transpose incidence [Links, OD*K]
                - 'route_validity_mask' (torch.BoolTensor): 2D valid routes boolean [OD, K]
                
                - 'od_pair_indices' (torch.LongTensor): [OD, 2] node index pairs
                - 't0' (torch.FloatTensor): [Links] free-flow travel times
                - 'capacity' (torch.FloatTensor): [Links] link capacities
                - 'length' (torch.FloatTensor): [Links] physical lengths
                - 'lanes' (torch.LongTensor): [Links] lane counts
                - 'speed' (torch.FloatTensor): [Links] speed limits
                - 'link_group' (torch.LongTensor): [Links] categorical indices
                - 'num_link_groups' (int): Number of unique link types

        Raises:
            KeyError: If required edge attributes are missing from graph
            ValueError: If routes_data format is unrecognized
        """
        # 1. Define canonical edge ordering (crucial for index alignment across tensors)
        edge_list = list(graph.edges())
        edge_to_idx = {e: i for i, e in enumerate(edge_list)}

        # 2. Extract physical network properties from graph
        physics_tensors = self._extract_physics(graph, edge_list)

        # 3. Build sparse route-link incidence masks
        # TODO: to deprecate the old 3D route_masks and replace with the new 2D delta_matrix and validity_mask in the model.
        # masks, od_indices = self._build_sparse_route_masks(routes_data, graph, edge_to_idx)

        masks, delta_matrix, validity_mask, od_indices = self._build_sparse_route_masks(
            routes_data, graph, edge_to_idx
        )

        # 4. Assemble final model input package
        model_inputs = {
            'num_links': len(edge_list),
            'num_od_pairs': masks.shape[0],
            'route_masks': masks.to(self.device),
            'od_pair_indices': od_indices.to(self.device),
            'delta_matrix': delta_matrix.to(self.device), # NEW: Transpose incidence for efficient link-based computations
            'route_validity_mask': validity_mask.to(self.device), # NEW: 2D boolean mask indicating valid routes per OD pair
            **physics_tensors  # Unpacks t0, capacity, link_group, etc.
        }

        return model_inputs

    def _extract_physics(self, graph: nx.DiGraph, edge_list: List[Tuple]) -> Dict[str, torch.Tensor]:
        """
        Extract physical and operational attributes from network edges.

        This method processes each edge in canonical order to extract link properties
        required for traffic flow modeling (BPR functions, capacity constraints, etc.).
        It also handles categorical link types by converting them to continuous indices
        suitable for embedding layers.

        Args:
            graph (nx.DiGraph): Transportation network graph with edge attributes
            edge_list (List[Tuple]): Ordered list of edges (defines canonical indexing)

        Returns:
            Dict[str, torch.Tensor]: Dictionary of link property tensors:
                - 't0': Free-flow travel times (minutes or seconds)
                - 'capacity': Maximum flow capacity (vehicles/hour)
                - 'length': Physical link length
                - 'lanes': Number of lanes per link
                - 'speed': Speed limit or free-flow speed
                - 'link_group': Mapped categorical indices [0, num_groups-1]
                - 'num_link_groups': Total number of unique link types

        Notes:
            - Missing attributes are filled with safe defaults (t0=1.0, capacity=1000.0)
            - Link types are remapped to contiguous indices [0, N-1] for efficiency
            - Original link_type values can be arbitrary integers (1, 5, 99, etc.)
        """
        # Temporary lists for attribute accumulation
        t0_list, cap_list, len_list, lanes_list, speed_list, type_list = [], [], [], [], [], []

        for u, v in edge_list:
            data = graph[u][v]
            t0_list.append(data.get('free_flow_time', 1.0))
            cap_list.append(data.get('capacity', 1000.0))
            len_list.append(data.get('length', 1.0))
            lanes_list.append(data.get('lanes', 1))
            speed_list.append(data.get('speed', 15.0))
            type_list.append(data.get('link_type', 0))

        # Process Link Groups: Map arbitrary integers to contiguous indices
        # Example: [1, 5, 1, 99, 5] -> [0, 1, 0, 2, 1]
        link_types_arr = np.array(type_list, dtype=np.int32)
        unique_types = np.unique(link_types_arr)
        type_map = {t: i for i, t in enumerate(unique_types)}
        link_group_indices = [type_map[t] for t in link_types_arr]

        return {
            't0': torch.FloatTensor(t0_list).to(self.device),
            'capacity': torch.FloatTensor(cap_list).to(self.device),
            'length': torch.FloatTensor(len_list).to(self.device),
            'lanes': torch.LongTensor(lanes_list).to(self.device),
            'speed': torch.FloatTensor(speed_list).to(self.device),
            'link_group': torch.LongTensor(link_group_indices).to(self.device),
            'num_link_groups': len(unique_types)
        }

    def _build_sparse_route_masks(self, routes_data: Dict, graph: nx.DiGraph, edge_to_idx: Dict) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Construct sparse topological tensors from K-shortest path data.
        
        CRITICAL ARCHITECTURAL WARNING:
        This method generates TWO distinct sparse incidence matrices to satisfy 
        different hardware and mathematical constraints across models. DO NOT delete 
        either matrix, as they serve different generations of traffic models.
        
        1. 'route_masks' (3D Sparse) -> [num_od, k_limit, num_links]
           - Legacy tensor for CGAME_DataDriven model.
           - Represents the conceptual hierarchy: OD -> Routes -> Links.
           - Cannot be efficiently multiplied using torch.sparse.mm() due to 3D shape.
           
        2. 'delta_matrix' (2D Sparse Transposed) -> [num_links, num_od * k_limit]
           - Required by Topo-CGAME (CRAME_DataDriven) physics-informed architecture.
           - Flattens the [OD, K] dimensions into a single continuous route index (r).
           - Pragmatic design: allows ultra-fast $x = \Delta f$ projection on GPUs 
             using standard 2D sparse matrix multiplication without OOM errors.
             
        3. 'route_validity_mask' (2D Dense Boolean) -> [num_od, k_limit]
           - Required by Topo-CGAME (CRAME_DataDriven) Attention Mechanism.
           - Maps which routes actually exist (True) vs padded empty slots (False).
           - Used to inject negative infinity before Softmax to prevent flow leakage.

        Args:
            routes_data (Dict): Route information format.
            graph (nx.DiGraph): Network graph (used for node mapping).
            edge_to_idx (Dict[Tuple, int]): Maps edge tuples to canonical link indices.

        Returns:
            Tuple containing (legacy_3d_mask, delta_2d_matrix, route_validity_mask, od_pair_indices)
        """
        raw_routes = routes_data if 'routes' not in routes_data else self._convert_tensor_to_dict(routes_data, graph)

        k_limit = 10
        num_od = len(raw_routes)
        num_links = len(edge_to_idx)
        
        # --- 1. Initialization for Legacy 3D Tensor ---
        indices_od_3d, indices_k_3d, indices_link_3d, values_3d = [], [], [], []
        
        # --- 2. Initialization for New 2D Delta Tensor ---
        indices_link_2d, indices_r_2d, values_2d = [], [], []
        
        # --- 3. Initialization for Route Validity Mask ---
        validity_mask = torch.zeros((num_od, k_limit), dtype=torch.bool, device=self.device)

        od_pair_list = []
        od_idx = 0

        for (u, v), paths in raw_routes.items():
            od_pair_list.append([u, v])

            for k, path in enumerate(paths):
                if k >= k_limit:
                    break
                
                # Flag this route slot as valid (physically exists)
                validity_mask[od_idx, k] = True
                
                # Flattened route index 'r' for the 2D Delta matrix
                r_idx = (od_idx * k_limit) + k

                for i in range(len(path) - 1):
                    u_node, v_node = path[i], path[i + 1]
                    if (u_node, v_node) in edge_to_idx:
                        l_idx = edge_to_idx[(u_node, v_node)]

                        # Populate Legacy 3D coords
                        indices_od_3d.append(od_idx)
                        indices_k_3d.append(k)
                        indices_link_3d.append(l_idx)
                        values_3d.append(1.0)
                        
                        # Populate New 2D coords [Link, Route]
                        indices_link_2d.append(l_idx)
                        indices_r_2d.append(r_idx)
                        values_2d.append(1.0)

            od_idx += 1

        if not indices_od_3d:
            empty_tensor = torch.empty(0, device=self.device)
            return empty_tensor, empty_tensor, validity_mask, empty_tensor

        # --- Assembly: Legacy 3D Sparse Tensor ---
        i_tensor_3d = torch.LongTensor([indices_od_3d, indices_k_3d, indices_link_3d])
        v_tensor_3d = torch.FloatTensor(values_3d)
        legacy_mask = torch.sparse_coo_tensor(
            i_tensor_3d, v_tensor_3d,
            size=(num_od, k_limit, num_links),
            device=self.device
        ).coalesce()

        # --- Assembly: New 2D Delta Sparse Tensor ---
        i_tensor_2d = torch.LongTensor([indices_link_2d, indices_r_2d])
        v_tensor_2d = torch.FloatTensor(values_2d)
        total_routes_r = num_od * k_limit
        delta_matrix = torch.sparse_coo_tensor(
            i_tensor_2d, v_tensor_2d,
            size=(num_links, total_routes_r),
            device=self.device
        ).coalesce()

        # --- Assembly: OD Indices ---
        node_list = sorted(list(graph.nodes()))
        node_map = {n: i for i, n in enumerate(node_list)}
        od_indices_num = [[node_map.get(u, -1), node_map.get(v, -1)] for u, v in od_pair_list]
        od_indices_tensor = torch.LongTensor(od_indices_num).to(self.device)

        return legacy_mask, delta_matrix, validity_mask, od_indices_tensor

    def _build_sparse_route_masks_deprecated(self, routes_data: Dict, graph: nx.DiGraph, edge_to_idx: Dict) -> Tuple[
        torch.Tensor, torch.Tensor]:
        """
        Construct sparse route-link incidence tensor from K-shortest path data.

        This is the most memory-critical component of the adapter. For real-world networks,
        a dense [OD × K × Links] tensor would require gigabytes of RAM (e.g., 1000 OD pairs × 
        10 routes × 5000 links × 4 bytes = 200 MB, but typically much larger). Sparse COO 
        format reduces this to only storing non-zero entries (route-link memberships).

        The function converts node sequences (routes) into binary incidence indicators:
        - Value 1.0 at [od_idx, route_k, link_l] means route k of OD pair od_idx uses link l
        - All other entries are implicitly 0

        Args:
            routes_data (Dict): Route information, expected as:
                               {(origin_node, dest_node): [[node1, node2, ...], [node1, node3, ...], ...]}
            graph (nx.DiGraph): Network graph (used for node mapping)
            edge_to_idx (Dict[Tuple, int]): Maps edge tuples (u,v) to canonical link indices

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - route_masks: Sparse COO tensor [num_od, k_limit, num_links] with 1.0 for 
                              route-link incidences
                - od_pair_indices: Dense tensor [num_od, 2] mapping OD indices to node pairs

        Implementation Details:
            - Routes are truncated at k_limit (default 10) to control memory
            - Node paths are converted to edge sequences: [n1,n2,n3] -> [(n1,n2), (n2,n3)]
            - Missing edges in graph are silently skipped (defensive programming)
            - OD pairs are assigned sequential indices [0, num_od-1]

        Notes:
            - This logic is ported from the legacy cyclic_model_data_ingestion.py
            - Assumes routes_data uses actual node IDs, not pre-mapped indices
        """
        # Handle raw dictionary format: {(u,v): [[node_path1], [node_path2], ...]}
        raw_routes = routes_data if 'routes' not in routes_data else self._convert_tensor_to_dict(routes_data, graph)

        # COO sparse tensor components
        indices_od = []
        indices_k = []
        indices_link = []
        values = []

        od_pair_list = []  # Metadata: which index corresponds to which OD pair

        od_idx = 0
        k_limit = 10  # Maximum routes per OD pair (configurable if passed to __init__)

        for (u, v), paths in raw_routes.items():
            od_pair_list.append([u, v])

            for k, path in enumerate(paths):
                if k >= k_limit:
                    break

                # Convert node sequence to edge sequence
                for i in range(len(path) - 1):
                    u_node, v_node = path[i], path[i + 1]
                    if (u_node, v_node) in edge_to_idx:
                        l_idx = edge_to_idx[(u_node, v_node)]

                        # Record sparse entry: OD od_idx, route k, link l_idx has value 1.0
                        indices_od.append(od_idx)
                        indices_k.append(k)
                        indices_link.append(l_idx)
                        values.append(1.0)

            od_idx += 1

        # Handle edge case: no valid routes found
        if not indices_od:
            return torch.empty(0).to(self.device), torch.empty(0).to(self.device)

        # Construct sparse COO tensor
        i_tensor = torch.LongTensor([indices_od, indices_k, indices_link])
        v_tensor = torch.FloatTensor(values)

        num_od = od_idx
        num_links = len(edge_to_idx)

        sparse_mask = torch.sparse_coo_tensor(
            i_tensor, v_tensor,
            size=(num_od, k_limit, num_links),
            device=self.device
        ).coalesce()  # Merge duplicate entries and sort indices

        # Map OD pair node IDs to integer indices for tensor operations
        node_list = sorted(list(graph.nodes()))
        node_map = {n: i for i, n in enumerate(node_list)}

        od_indices_num = []
        for u, v in od_pair_list:
            od_indices_num.append([node_map.get(u, -1), node_map.get(v, -1)])

        return sparse_mask, torch.LongTensor(od_indices_num)

    def _convert_tensor_to_dict(self, routes_data: Dict, graph: nx.DiGraph) -> Dict:
        """
        Helper method to convert tensor-based route format to dictionary format.

        This is a compatibility layer for legacy route data stored as numpy arrays
        instead of raw dictionaries. Enables reuse of the main sparse building logic.

        Args:
            routes_data (Dict): Routes in processed format {'routes': np.ndarray, 'od_pairs': list}
            graph (nx.DiGraph): Network graph for node mapping

        Returns:
            Dict: Routes in raw dictionary format {(u,v): [[path1], [path2], ...]}

        Note:
            Currently returns empty dict (stub implementation). Implement if your pipeline
            requires loading pre-processed tensor-based route caches.
        """
        # TODO: Implement if tensor-based route format is used in pipeline
        return {}
