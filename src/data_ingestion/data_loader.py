"""
DataLoader - movimientos de carga separados desde DataManager

Esta clase realiza exclusivamente las tareas de lectura de archivos (red, flujos,
matriz OD, coordenadas de nodos). Está diseñada para trabajar junto a
`DataManager` (que mantiene la resolución de rutas y metadatos) y actualizar
los atributos del manager directamente para compatibilidad con el código
existente.
"""

from typing import Optional
import pandas as pd
from scipy import sparse
from pathlib import Path
import sys

# Try relative imports (normal package usage). If the module is executed directly
# (e.g. python data_loader.py), the relative imports fail with "attempted relative
# import with no known parent package". In that case, add 'src' to sys.path and
# import using the package path.
try:
    from .processing_modules.network_loader import TNTPNetworkLoader
    from .processing_modules.trips_reader import FlowReader
    from .processing_modules.od_matrix_generator import ODMatrixGenerator, MultidayODMatrixGenerator
    from .processing_modules.node_loader import TNTPNodeLoader
except Exception:
    # Add the 'src' directory (parent of this package) to sys.path so the package
    # can be imported when running this file as a script.
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from data_ingestion.processing_modules.network_loader import TNTPNetworkLoader
    from data_ingestion.processing_modules.trips_reader import FlowReader
    from data_ingestion.processing_modules.od_matrix_generator import ODMatrixGenerator, MultidayODMatrixGenerator
    from data_ingestion.processing_modules.node_loader import TNTPNodeLoader


class DataLoader:
    """Carga red, flujos, matriz OD y coordenadas y actualiza un DataManager.

    Diseño: el `DataLoader` opera sobre un objeto `manager` (instancia de
    `DataManager`) que ya resolvió las rutas (manager.network_path, manager.flow_path, ...).
    El loader actualiza `manager.network_df`, `manager.flow_df`, `manager.od_matrix`
    y `manager.node_coords_df` así como `manager.metadata`.
    """

    def __init__(self, multiday: bool = False):
        self.multiday = multiday

    def load_network(self, manager):
        """
        Loads the TNTP network data, performs column normalization, and standardizes
        the 'from_node' and 'to_node' columns into canonical names and types.
        """
        # 1. Initial Data Loading
        loader = TNTPNetworkLoader(manager.network_path)
        network_df, net_metadata = loader.load()
        manager.network_df = network_df
        manager.metadata['network'] = net_metadata
        df = manager.network_df  # Use an alias for conciseness

        # --- Auxiliary Functions for Column Identification ---
        # Map all column names to lowercase for robust searching
        cols_map = {c.lower(): c for c in df.columns}

        def find_col_name(*candidates):
            """Finds the actual column name given a list of possible lowercase candidates."""
            for c in candidates:
                # Check if the lowercase candidate exists in the map
                if c.lower() in cols_map:
                    # Return the original (cased) column name
                    return cols_map[c.lower()]
            return None

        # 2. Normalization of Optional Columns (lanes, VDF)

        # 2a. Normalize 'lanes' (number of lanes)
        lanes_col = find_col_name('lanes')
        if lanes_col:
            # Convert to numeric, setting errors (non-numeric data) to NaN ('coerce').
            lanes = pd.to_numeric(df[lanes_col], errors='coerce')
            # Fill NaN values (non-numeric or missing) with 1 (default number of lanes).
            # Use 'Int64' to store integers while correctly handling potential NaNs.
            df['lanes'] = lanes.fillna(1).astype('Int64')

        # 2b. Normalize 'VDF' (Volume Delay Function identifier/parameter)
        vdf_col = find_col_name('VDF')
        if vdf_col:
            # Convert VDF to numeric, allowing errors to become NaN, which is usually acceptable
            # for missing or malformed VDF parameters.
            df['VDF'] = pd.to_numeric(df[vdf_col], errors='coerce')

        # 3. Standardization of Node Columns

        # Identify potential columns for initial node (inode) and terminal node (jnode)
        inode_col = find_col_name('init_node', 'from_node', 'from', 'u')
        jnode_col = find_col_name('term_node', 'to_node', 'to', 'v')

        if inode_col and jnode_col:
            # Attempt conversion to numeric 'Int64' type for canonical columns.
            df['from_node'] = pd.to_numeric(df[inode_col], errors='coerce').astype('Int64')
            df['to_node'] = pd.to_numeric(df[jnode_col], errors='coerce').astype('Int64')

            # Fallback check: If the original data contained non-numeric IDs (resulting in NaNs
            # during numeric conversion) and wasn't already a string type (object),
            # convert them explicitly to string type (str/object) as a final attempt.
            if df['from_node'].isna().any() and df[inode_col].dtype != 'object':
                df['from_node'] = df[inode_col].astype(str)
                df['to_node'] = df[jnode_col].astype(str)

        # 4. Last Resort: Use the first two available columns if canonical names were not found.
        # This executes only if the main 'if' block above failed AND the columns still don't exist.
        elif 'from_node' not in df.columns or 'to_node' not in df.columns:
            possible = list(df.columns)
            if len(possible) >= 2:
                # Assume the first two columns in the file are the 'from' and 'to' nodes.
                df['from_node'] = df[possible[0]]
                df['to_node'] = df[possible[1]]

        return manager.network_df

    def load_flow(self, manager, year=None):
        """
        Loads the flow data, standardizes node columns, and selects the appropriate
        'volume' column based on the provided year parameter.

        :param manager: The manager object containing the flow_path.
        :param year: The specific year (e.g., 2023) to select the volume column from.
        """
        # 1. Initial Data Loading
        reader = FlowReader(manager.flow_path)
        manager.flow_df = reader.load()
        df = manager.flow_df  # Alias for conciseness

        # --- Auxiliary Functions and Column Identification ---
        # Map all column names to lowercase for robust searching
        colmap = {c.lower(): c for c in df.columns}

        # Candidate names for node columns
        from_candidates = ['from', 'from_node', 'init_node', 'origin']
        to_candidates = ['to', 'to_node', 'term_node', 'destination']

        def _find_in_df(cands):
            """Finds the actual column name given a list of possible lowercase candidates."""
            for c in cands:
                if c in colmap:
                    return colmap[c]
            return None

        # 2. Standardization of Node Columns (from_node, to_node)
        fcol = _find_in_df(from_candidates)
        tcol = _find_in_df(to_candidates)

        if fcol and tcol:
            # Rename identified columns to canonical names
            try:
                df.rename(columns={fcol: 'from_node', tcol: 'to_node'}, inplace=True)
            except Exception:
                pass

        # 3. Normalization and Selection of Volume Column

        # Identify all columns containing "volume"
        all_vol_cols = [c for c in df.columns if 'volume' in c.lower()]

        # 3a. Volume selection logic
        selected_vol_col = None

        if len(all_vol_cols) == 1:
            # Case 1: If only one volume column exists, select it regardless of the year.
            selected_vol_col = all_vol_cols[0]

        elif len(all_vol_cols) > 1 and year is not None:
            # Case 2: If multiple volume columns exist and a year is provided, find the match.
            target_suffix = f'_{year}'

            # Look for a column name that ends with the target suffix (case-insensitive search is implicit via the list)
            for c in all_vol_cols:
                if c.endswith(target_suffix):
                    selected_vol_col = c
                    break

        elif len(all_vol_cols) > 1 and year is None:
            # Case 3: Multiple volume columns, but no year provided. Use a default/fallback (e.g., the last one alphabetically or stop here).
            # For simplicity, we choose the last one in the list (often the most recent, depending on file structure).
            # A better implementation might raise an error or select the latest year available.
            selected_vol_col = all_vol_cols[-1]

        # 3b. Create the canonical 'volume' column from the selected source
        if selected_vol_col:
            # Rename the selected column to the canonical 'volume' name.
            if selected_vol_col != 'volume':
                df.rename(columns={selected_vol_col: 'volume'}, inplace=True)

        else:
            # If no volume column was found after all attempts, ensure 'volume' column exists and is zeroed out.
            if 'volume' not in df.columns:
                df['volume'] = 0.0

        # 4. Final Type Conversions for Canonical Columns

        # Convert 'from_node'
        if 'from_node' in df.columns:
            try:
                # Attempt conversion to pandas nullable integer type
                df['from_node'] = pd.to_numeric(df['from_node'], errors='coerce').astype('Int64')
            except Exception:
                # Fallback to string type if numeric conversion fails
                df['from_node'] = df['from_node'].astype(str)

        # Convert 'to_node'
        if 'to_node' in df.columns:
            try:
                df['to_node'] = pd.to_numeric(df['to_node'], errors='coerce').astype('Int64')
            except Exception:
                df['to_node'] = df['to_node'].astype(str)

        # Convert 'volume' (Ensuring it is numeric and filling NaNs with 0.0)
        if 'volume' in df.columns:
            try:
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce').fillna(0.0)
            except Exception:
                # If all numeric conversion fails, do nothing (column remains as is, likely an issue with the data)
                pass

        # 5. Calculation and Assignment of Statistics
        try:
            # Attempt to get statistics directly from the reader object
            stats = reader.get_statistics()
        except Exception:
            # Manual fallback calculation of statistics
            stats = {
                'total_links': len(df),
                'links_with_flow': int((df['volume'] > 0).sum()) if 'volume' in df.columns else 0,
                'links_no_flow': int((df['volume'] == 0).sum()) if 'volume' in df.columns else 0,
                'total_volume': float(df['volume'].sum()) if 'volume' in df.columns else 0.0,
                'avg_volume': float(df['volume'].mean()) if 'volume' in df.columns else 0.0,
                'max_volume': float(df['volume'].max()) if 'volume' in df.columns else 0.0,
            }

        manager.metadata['flow'] = stats
        return manager.flow_df

    def load_node_features(self, manager):
        if manager.node_path is None:
            return pd.DataFrame()
        loader = TNTPNodeLoader(manager.node_path)
        manager.node_coords_df, node_metadata = loader.load()
        manager.metadata['node_coordinates'] = node_metadata
        return manager.node_coords_df

    def load_od_matrix(self, manager) -> sparse.spmatrix:
        if self.multiday:
            loader = MultidayODMatrixGenerator(manager.od_path)
        else:
            loader = ODMatrixGenerator(manager.od_path)

        od_dataframe, od_metadata = loader.load()
        manager.od_dataframe = od_dataframe
        # Convert to sparse matrix for od_matrix attribute
        manager.od_matrix = loader.to_sparse_matrix()
        manager.metadata['od_matrix'] = od_metadata
        return manager.od_matrix

    def load_all(self, manager):
        """Convenience: ejecuta todas las cargas en orden y actualiza manager."""
        self.load_network(manager)
        self.load_flow(manager)
        self.load_od_matrix(manager)
        self.load_node_features(manager)
        # Expandir matriz OD con nodos auxiliares (demanda NaN)
        manager.add_aux_od_matrix()
        return manager.network_df, manager.flow_df, manager.od_matrix, manager.node_coords_df
