import logging
import pickle
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path
from scipy import sparse
from typing import Dict, Any, Union, Optional

logger = logging.getLogger(__name__)


class TrafficArtifactLoader:
    """
    Pure I/O handler for traffic network data artifacts (processed data).

    This loader acts as the "Kitchen Assistant" in the data pipeline - its sole 
    responsibility is locating and reading preprocessed artifacts from disk. It 
    handles file format variations (.pkl, .parquet, .npz) and naming inconsistencies 
    across different network datasets without any knowledge of PyTorch, model structures, 
    or tensor operations.

    Design Principles:
    - **Path Robustness**: Uses glob patterns to find files despite name variations
      (e.g., 'Linköping_graph.pkl' vs 'SiouxFalls_graph.pkl')
    - **Format Agnostic**: Returns standard Python data structures (pandas.DataFrame,
      networkx.DiGraph, scipy.sparse matrices)
    - **Defensive Loading**: Provides informative errors with available files when
      artifacts are missing
    - **No ML Dependencies**: Deliberately avoids PyTorch/TensorFlow to maintain
      clean separation of concerns

    Attributes:
        base_path (Path): Root directory containing processed data artifacts

    Typical Directory Structure:
        data/processed/Linköping/
        ├── Linköping_graph.pkl           # NetworkX graph with network topology
        ├── Linköping_od_matrix.npz       # Sparse OD demand matrix
        ├── Linköping_link_data.parquet   # Link attributes DataFrame
        └── routing_cache/
            └── Linköping_kshortest_paths.pkl  # K-shortest path routes

    Example Usage:
        >>> loader = TrafficArtifactLoader("data/processed/Linköping")
        >>> artifacts = loader.load_all()
        >>> graph = artifacts['graph']  # networkx.DiGraph
        >>> od_matrix = artifacts['od_matrix']  # scipy.sparse.csr_matrix
    """

    def __init__(self, data_path: Union[str, Path]):
        """
        Initialize loader with base path to processed data directory.

        Args:
            data_path (Union[str, Path]): Path to directory containing processed
                                          traffic network artifacts. Can be absolute
                                          or relative to current working directory.

        Raises:
            FileNotFoundError: If neither the specified path nor the fallback path
                              'data/processed/<data_path>' exists.

        Notes:
            - Automatically attempts fallback to 'data/processed/<data_path>' if
              initial path doesn't exist (handles execution from different roots)
            - Logs absolute resolved path for debugging multi-environment setups
        """
        self.base_path = Path(data_path)

        if not self.base_path.exists():
            # Fallback strategy for execution from different project roots
            alt_path = Path("data/processed") / data_path
            if alt_path.exists():
                self.base_path = alt_path
            else:
                raise FileNotFoundError(f"Data directory does not exist: {self.base_path}")

        logger.info(f"ArtifactLoader initialized at: {self.base_path.absolute()}")

    def load_all(self) -> Dict[str, Any]:
        """
        Load all standard artifacts required for traffic assignment models.

        This convenience method orchestrates loading of the four core artifacts needed
        by most route-based traffic models. It calls individual read methods in sequence
        and packages results into a unified dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing all loaded artifacts:
                - 'graph' (nx.DiGraph): Network topology with link attributes
                - 'od_matrix' (sparse.csr_matrix): Origin-destination demand matrix
                - 'link_data' (pd.DataFrame): Tabular link properties
                - 'routes_data' (Dict): K-shortest path route information

        Raises:
            FileNotFoundError: If any required artifact file is missing
            Exception: If file loading or parsing fails for any artifact

        Example:
            >>> loader = TrafficArtifactLoader("data/processed/SiouxFalls")
            >>> artifacts = loader.load_all()
            >>> print(f"Network has {len(artifacts['graph'].edges)} links")
        """
        return {
            "graph": self.read_graph(),
            "od_matrix": self.read_od_matrix(),
            "link_data": self.read_link_data(),
            "routes_data": self.read_routes()
        }

    def read_graph(self) -> nx.DiGraph:
        """
        Locate and load NetworkX graph from pickle file.

        Searches for any file matching pattern '*_graph.pkl' in the base directory.
        The graph should contain network topology (nodes and directed edges) with
        edge attributes required for traffic modeling (free_flow_time, capacity,
        length, lanes, speed, link_type).

        Returns:
            nx.DiGraph: Directed graph representing the transportation network.
                       Nodes are typically integer IDs, edges contain dict attributes.

        Raises:
            FileNotFoundError: If no file matching '*_graph.pkl' exists
            pickle.UnpicklingError: If file is corrupted or not a valid pickle
            Exception: For other I/O or parsing errors

        Example Output:
            Graph with 24 nodes, 76 edges (typical for Sioux Falls network)

        Notes:
            - Uses glob pattern to handle dataset-specific prefixes automatically
            - Logs node and edge counts for verification
        """
        fpath = self._find_file("*_graph.pkl", "Graph")

        try:
            with open(fpath, 'rb') as f:
                graph = pickle.load(f)
            logger.info(f"Graph loaded: {len(graph.nodes)} nodes, {len(graph.edges)} links.")
            return graph
        except Exception as e:
            logger.error(f"Error loading graph from {fpath}: {e}")
            raise

    def read_od_matrix(self) -> sparse.csr_matrix:
        """
        Locate and load origin-destination demand matrix from compressed numpy file.

        Reconstructs a scipy sparse CSR matrix from .npz archive containing the
        standard CSR components (data, indices, indptr, shape). The matrix represents
        travel demand between node pairs, typically in vehicles per hour or trips per day.

        Returns:
            sparse.csr_matrix: Sparse matrix of shape [num_nodes, num_nodes] where
                              entry [i,j] represents demand from origin i to destination j.
                              Most entries are zero (hence sparse storage).

        Raises:
            FileNotFoundError: If no file matching '*_od_matrix.npz' exists
            KeyError: If .npz file missing required keys (data, indices, indptr, shape)
            Exception: For other I/O or reconstruction errors

        Notes:
            - CSR (Compressed Sparse Row) format is efficient for matrix-vector operations
            - Logs matrix shape and number of non-zero OD pairs for verification
            - 'nnz' (number of non-zeros) indicates how many OD pairs have demand > 0
        """
        fpath = self._find_file("*_od_matrix.npz", "OD Matrix")

        try:
            data = np.load(fpath)
            # Reconstruct CSR sparse matrix from saved components
            matrix = sparse.csr_matrix(
                (data['data'], data['indices'], data['indptr']),
                shape=tuple(data['shape'])
            )
            logger.info(f"OD Matrix loaded. Shape: {matrix.shape}. Non-zeros: {matrix.nnz}")
            return matrix
        except Exception as e:
            logger.error(f"Error loading OD matrix from {fpath}: {e}")
            raise

    def read_link_data(self) -> pd.DataFrame:
        """
        Locate and load link attributes table from Parquet file.

        Loads a tabular representation of link properties stored in efficient columnar
        Parquet format. This typically duplicates information in the graph but provides
        easier access for analytics, validation, and DataFrame-based operations.

        Returns:
            pd.DataFrame: Table with one row per link, columns typically include:
                         - 'link_id' or ('from_node', 'to_node'): Link identifier
                         - 'free_flow_time': Travel time at zero congestion
                         - 'capacity': Maximum flow (vehicles/hour)
                         - 'length': Physical length
                         - 'lanes': Number of lanes
                         - 'speed': Speed limit
                         - 'link_type': Road classification

        Raises:
            FileNotFoundError: If no file matching '*_link_data.parquet' exists
            Exception: For Parquet parsing or I/O errors

        Notes:
            - Parquet format provides ~10x compression vs CSV and faster loading
            - DataFrame indexing may vary (integer index vs. multi-index on node pairs)
        """
        fpath = self._find_file("*_link_data.parquet", "Link DataFrame")

        try:
            df = pd.read_parquet(fpath)
            logger.info(f"Link DataFrame loaded. {len(df)} rows.")
            return df
        except Exception as e:
            logger.error(f"Error loading Link Data from {fpath}: {e}")
            raise

    def read_routes(self) -> Dict[str, Any]:
        """
        Locate and load K-shortest path route data with multi-location search strategy.

        Routes are critical for route-choice models but may be stored in various locations
        depending on preprocessing workflow. This method implements a cascading search
        strategy to maximize compatibility across different project structures.

        Search Strategy:
        1. Check 'routing_cache/' subdirectory (common for cached computations)
        2. Check base directory root
        3. Fallback to generic 'routes.pkl' filename

        Returns:
            Dict[str, Any]: Route data structure, typically in one of two formats:
                           Format A (Raw Dict): 
                               {(origin, dest): [[node_path1], [node_path2], ...], ...}
                           Format B (Processed):
                               {'routes': np.ndarray, 'od_pairs': list, 'metadata': dict}

        Raises:
            FileNotFoundError: If no route file found in any search location, with
                              detailed message listing attempted patterns and available files

        Notes:
            - Uses pattern '*kshortest_paths.pkl' to handle dataset-specific prefixes
            - Logs the specific file used (useful for debugging route source)
            - Route computation is expensive, so caching in subdirectory is common practice

        Example Structure:
            {(1, 20): [[1, 2, 5, 20],      # Shortest path
                       [1, 3, 12, 20],     # 2nd shortest
                       [1, 2, 6, 11, 20]], # 3rd shortest
             (1, 21): [...],
             ...}
        """
        # Search strategy 1: routing_cache/ subdirectory (common structure)
        cache_dir = self.base_path / "routing_cache"
        target_pattern = "*kshortest_paths.pkl"

        found_path = None

        # Strategy 1: Search in routing_cache/ subdirectory
        if cache_dir.exists():
            matches = list(cache_dir.glob(target_pattern))
            if matches:
                found_path = matches[0]

        # Strategy 2: Search in base directory root
        if not found_path:
            matches = list(self.base_path.glob(target_pattern))
            if matches:
                found_path = matches[0]

        # Strategy 3: Fallback to generic filename
        if not found_path:
            fallback = self.base_path / "routes.pkl"
            if fallback.exists():
                found_path = fallback

        if not found_path:
            raise FileNotFoundError(
                f"No route file ({target_pattern}) found in {self.base_path} or subdirectories."
            )

        logger.info(f"Loading routes from: {found_path.name}")
        with open(found_path, 'rb') as f:
            routes = pickle.load(f)

        return routes

    def _find_file(self, pattern: str, description: str) -> Path:
        """
        Robust helper for locating files using glob patterns with informative errors.

        This utility method encapsulates the file search logic used by all artifact
        readers. It automatically handles naming variations (e.g., 'Linköping' vs 
        'Linkoping', 'SiouxFalls' vs 'sioux_falls') by using wildcard patterns instead
        of hardcoded filenames.

        Args:
            pattern (str): Glob pattern to match (e.g., '*_graph.pkl', '*od_matrix.npz')
            description (str): Human-readable artifact name for error messages

        Returns:
            Path: Absolute path to the first matching file

        Raises:
            FileNotFoundError: If no files match the pattern, with detailed message
                              listing the pattern, search directory, and available files

        Behavior:
            - If multiple matches exist, uses the first and logs a warning
            - Could be extended to select most recent file by modification time
            - Lists all available files in directory when pattern fails (debugging aid)

        Example:
            >>> path = self._find_file("*_graph.pkl", "Network Graph")
            >>> # Returns: Path('/data/processed/Linköping/Linköping_graph.pkl')
        """
        matches = list(self.base_path.glob(pattern))

        if not matches:
            # Debug assistance: show what files are actually available
            available = [p.name for p in self.base_path.glob("*")]
            raise FileNotFoundError(
                f"No file found for '{description}' with pattern '{pattern}' in {self.base_path}.\n"
                f"Available files: {available}"
            )

        # Handle multiple matches: use first, but warn user
        if len(matches) > 1:
            logger.warning(
                f"Multiple files found for {pattern}: {[m.name for m in matches]}. "
                f"Using {matches[0].name}"
            )

        return matches[0]
