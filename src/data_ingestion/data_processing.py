"""
⚠️ DEPRECATED: This module is deprecated. Use data_pipeline.py instead.

This file will be removed in a future release.
Use data_pipeline.py which includes support for route_source and num_routes parameters.

Data Processing - renombrado desde data_manager.

Contiene la implementación completa de DataManager (conservé el nombre de clase
`DataManager` para compatibilidad) pero el módulo se llama `data_processing`.
Además el CLI principal lee `data_processing_args.yaml` en el mismo directorio
para valores por defecto.
"""
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Tuple, Dict
from scipy import sparse
import sys
import numpy as np
import argparse
try:
    import yaml
except Exception:
    yaml = None

# Import strategy: if this module is imported as part of the package (normal use),
# __package__ will be set and we should use relative imports. If it's executed
# directly (for debugging/CLI), __package__ is None/'' and relative imports
# will raise "attempted relative import"; in that case add 'src' to sys.path and
# import the modules by absolute package name.
if __package__ in (None, ''):
    # Running as script: add project src dir to sys.path and use absolute imports
    src_dir = str(Path(__file__).resolve().parents[1])
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from data_ingestion.data_loader import DataLoader
    from data_ingestion.data_saver import DataSaver
else:
    # Running as package import: use relative imports
    from .data_loader import DataLoader
    from .data_saver import DataSaver

# low-level loaders are delegated to DataLoader/DataSaver now

import networkx as nx

# Root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuración de datasets disponibles
AVAILABLE_NETWORKS = {
    'Barcelona': {
        'net': 'Barcelona_net.tntp',
        'flow': 'Barcelona_flow.tntp',
        'trips': 'Barcelona_trips.tntp',
        'node': None,  # Barcelona no tiene coordenadas
        'folder': 'data/raw'
    },
    'SiouxFalls': {
        'net': 'SiouxFalls_net.tntp',
        'flow': 'SiouxFalls_flow.tntp',
        'trips': 'SiouxFalls_trips.tntp',
        'node': 'SiouxFalls_node.tntp',
        'folder': 'data/raw/SiouxFalls'
    },
    'Linköping': {
        'net': 'Linköping_net.tntp',
        'flow': 'Linköping_flow.tntp',
        'trips': 'Linköping_trips.tntp',
        'node': 'Linköping_node.tntp',
        'folder': 'data/interim/Linköping'
    }
}


class DataManager:
    """
    Gestor centralizado de datos para Barcelona GNN (generalizado para múltiples redes).

    Mantengo el nombre de la clase `DataManager` para compatibilidad con código
    existente; el módulo ahora se llama `data_processing`.
    """

    def __init__(self,
                 network_name: str = 'Barcelona',
                 network_path: Optional[Union[Path, str]] = None,
                 flow_path: Optional[Union[Path, str]] = None,
                 od_path: Optional[Union[Path, str]] = None,
                 node_path: Optional[Union[Path, str]] = None,
                 data_root: Optional[Union[Path, str]] = None,
                 multiday: bool = False):
        self.network_name = network_name
        self.multiday = multiday

        # metadata container (initialize early)
        self.metadata = {
            'network_name': network_name,
            'multiday': multiday
        }

        # Determine canonical network key (use AVAILABLE_NETWORKS canonical key)
        import unicodedata
        import difflib

        def _norm_key(s: str) -> str:
            nf = unicodedata.normalize('NFKD', s)
            return ''.join(c for c in nf if not unicodedata.combining(c)).casefold()

        # 1) try exact match
        canonical_name = network_name if network_name in AVAILABLE_NETWORKS else None

        # 2) try normalized exact match
        if canonical_name is None:
            req_norm = _norm_key(network_name)
            for key in AVAILABLE_NETWORKS.keys():
                if _norm_key(key) == req_norm:
                    canonical_name = key
                    break

        # keep original network_name for logging; we'll use canonical_name for outputs when present
        self.network_name = network_name
        self.canonical_name = canonical_name

        # Resolve input data directory. Prefer explicit data_root, otherwise try data/external/<case> then data/raw/<case>.
        if data_root is not None:
            data_root = Path(data_root)
        else:
            data_root = PROJECT_ROOT / 'data'

        candidate_external = PROJECT_ROOT / 'data' / 'external' / network_name
        candidate_raw = PROJECT_ROOT / 'data' / 'raw' / network_name

        # Helper to find a folder under a parent that matches the network_name
        import unicodedata

        def _norm(s: str) -> str:
            nf = unicodedata.normalize('NFKD', s)
            return ''.join(c for c in nf if not unicodedata.combining(c)).casefold()

        def find_matching_subfolder(parent: Path, target_name: str) -> Optional[Path]:
            try:
                if not parent.exists():
                    return None
                target_norm = _norm(target_name)
                for child in parent.iterdir():
                    if not child.is_dir():
                        continue
                    if _norm(child.name) == target_norm:
                        return child
            except Exception:
                return None
            return None

        # Before deciding base_folder, attempt to locate a matching folder on disk
        # under common data parents (interim, raw, external) using a normalized
        # name comparison so we accept 'Linkping' for 'Linköping'. If found, use it.
        base_folder = None
        for parent in [PROJECT_ROOT / 'data' / 'interim', PROJECT_ROOT / 'data' / 'raw', PROJECT_ROOT / 'data' / 'external']:
            match = find_matching_subfolder(parent, network_name)
            if match is not None:
                base_folder = match
                break

        # 3) If we found a base_folder and still lack a canonical_name, try to derive a canonical
        # key from the on-disk folder name (normalized), or via a fuzzy match.
        if base_folder is not None and self.canonical_name is None:
            try:
                base_on_disk = base_folder.name
                base_norm = _norm_key(base_on_disk)
                for key in AVAILABLE_NETWORKS.keys():
                    if _norm_key(key) == base_norm:
                        self.canonical_name = key
                        break
                if self.canonical_name is None:
                    candidates = difflib.get_close_matches(base_on_disk, list(AVAILABLE_NETWORKS.keys()), n=1, cutoff=0.6)
                    if candidates:
                        self.canonical_name = candidates[0]
            except Exception:
                pass

        # Determine expected filenames from AVAILABLE_NETWORKS if present (needed to check folders)
        config = AVAILABLE_NETWORKS.get(network_name, {})
        net_name = config.get('net', f"{network_name}_net.tntp")
        flow_name = config.get('flow', f"{network_name}_flow.tntp")
        trips_name = config.get('trips', f"{network_name}_trips.tntp")
        node_name = config.get('node', None)

        # collect candidate folders in priority order but prefer configured folder if it contains files
        config_folder = None
        if config is not None and 'folder' in config:
            config_folder = PROJECT_ROOT / config['folder']

        # helper to check if folder contains expected files
        def folder_has_expected(folder: Path, net_name: str, flow_name: str, trips_name: str):
            try:
                if not folder.exists():
                    return False
                if (folder / net_name).exists() or (folder / flow_name).exists() or (folder / trips_name).exists():
                    return True
                # also check for any *_net.tntp files
                for p in folder.glob('*_net.tntp'):
                    return True
            except Exception:
                return False
            return False

        # Prefer config_folder if it contains expected files
        if config_folder is not None and folder_has_expected(config_folder, net_name, flow_name, trips_name):
            base_folder = config_folder
        elif candidate_external.exists() and folder_has_expected(candidate_external, net_name, flow_name, trips_name):
            base_folder = candidate_external
        elif candidate_raw.exists() and folder_has_expected(candidate_raw, net_name, flow_name, trips_name):
            base_folder = candidate_raw
        else:
            # If none contain expected files, prefer external if exists, otherwise config_folder, otherwise raw, otherwise data_root
            if candidate_external.exists():
                base_folder = candidate_external
            elif config_folder is not None and config_folder.exists():
                base_folder = config_folder
            elif candidate_raw.exists():
                base_folder = candidate_raw
            else:
                base_folder = data_root

        # Rutas por defecto según lo que encontremos en base_folder
        def _resolve_file(explicit_path, *possible_names):
            if explicit_path is not None:
                return Path(explicit_path)
            for name in possible_names:
                p = Path(base_folder) / name
                if p.exists():
                    return p
            # try direct filename pattern: <network_name>_net.tntp etc.
            for suffix in ['net.tntp', 'flow.tntp', 'trips.tntp', 'node.tntp']:
                p = Path(base_folder) / f"{network_name}_{suffix}"
                if p.exists():
                    return p
            # fallback to first possible name under base_folder (may not exist)
            return Path(base_folder) / possible_names[0]

        self.network_path = _resolve_file(network_path, net_name)
        self.flow_path = _resolve_file(flow_path, flow_name)
        self.od_path = _resolve_file(od_path, trips_name)
        self.node_path = _resolve_file(node_path, node_name) if node_name else (Path(node_path) if node_path else None)

        # Processed outputs directory for this case -- prefer canonical key if available
        processed_case_name = self.canonical_name if getattr(self, 'canonical_name', None) else network_name
        self.processed_dir = PROJECT_ROOT / 'data' / 'processed' / processed_case_name
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # routing_cache folder inside processed/<case>/routing_cache
        self.routing_cache_dir = self.processed_dir / 'routing_cache'
        self.routing_cache_dir.mkdir(parents=True, exist_ok=True)

        # expose chosen base folder in metadata
        self.metadata['data_source_folder'] = str(base_folder)

        # keep paths as Path objects
        self.network_path = Path(self.network_path)
        self.flow_path = Path(self.flow_path)
        self.od_path = Path(self.od_path)
        self.node_path = Path(self.node_path) if self.node_path else None

        # Contenedores de datos
        self.network_df = None
        self.flow_df = None
        self.od_matrix = None
        self.od_dataframe = None
        self.node_coords_df = None
        self.unified_df = None
        self.graph = None

    def load_network(self) -> pd.DataFrame:
        """Delega la carga de red a DataLoader para mantener separacion de responsabilidades."""
        loader = DataLoader(multiday=self.multiday)
        df = loader.load_network(self)
        print(f"   ✓ Red cargada: {df.shape[0]} enlaces, {df.shape[1]} atributos")
        return df

    def load_flow(self) -> pd.DataFrame:
        """Delega la carga de flujos a DataLoader."""
        loader = DataLoader(multiday=self.multiday)
        df = loader.load_flow(self)
        print(f"   ✓ Flujos cargados: {df.shape[0]} enlaces")
        return df

    def load_od_matrix(self) -> sparse.spmatrix:
        """Delega la carga de la matriz OD a DataLoader."""
        loader = DataLoader(multiday=self.multiday)
        od = loader.load_od_matrix(self)
        try:
            print(f"   ✓ Matriz OD cargada: {od.shape[0]} filas, {od.shape[1]} columnas")
        except Exception:
            pass
        return od

    def load_node_coordinates(self) -> pd.DataFrame:
        """Delega la carga de coordenadas de nodos a DataLoader."""
        loader = DataLoader(multiday=self.multiday)
        df = loader.load_node_features(self)
        if df is not None and not df.empty:
            print(f"   ✓ Coordenadas de nodos cargadas: {df.shape[0]} nodos")
        else:
            print("   ⚠️ No se proporcionó archivo de nodos o está vacío. Saltando carga de coordenadas.")
        return df

    def add_aux_od_matrix(self):
        """Expande la matriz OD para incluir nodos auxiliares con demanda desconocida (NaN).

        Los nodos auxiliares (type='aux') no tienen demanda conocida, por lo que se añaden
        a la matriz OD con valores NaN. Esto incluye:
        - Pares aux-aux
        - Pares aux-taz
        - Pares taz-aux

        La matriz expandida será aproximadamente 108x108 (69 TAZ + ~39 aux nodes).
        """
        import numpy as np
        from scipy import sparse as sp

        if self.node_coords_df is None or self.node_coords_df.empty:
            print("   ⚠️  No hay información de nodos. No se puede expandir la matriz OD.")
            return

        if self.od_matrix is None:
            print("   ⚠️  No hay matriz OD cargada. No se puede expandir.")
            return

        # Identificar nodos TAZ (los que ya están en la matriz OD actual)
        # Asumimos que la matriz OD actual es NxN donde N = número de nodos TAZ
        node_df = self.node_coords_df.copy()

        # Identificar columna de node ID
        node_col = None
        for candidate in ['node', 'node_id', 'Node', 'NODE']:
            if candidate in node_df.columns:
                node_col = candidate
                break
        if node_col is None:
            node_col = node_df.columns[0]

        # Identificar columna de type
        type_col = None
        for candidate in ['type', 'Type', 'TYPE', 'node_type']:
            if candidate in node_df.columns:
                type_col = candidate
                break

        if type_col is None:
            print("   ⚠️  No se encontró columna 'type' en node_coords_df. No se puede identificar nodos auxiliares.")
            return

        # Extraer nodos TAZ y auxiliares
        taz_nodes = node_df[node_df[type_col] == 'taz'][node_col].astype(str).tolist()
        aux_nodes = node_df[node_df[type_col] == 'aux'][node_col].astype(str).tolist()

        taz_nodes = sorted(taz_nodes, key=lambda x: int(x) if x.isdigit() else x)
        aux_nodes = sorted(aux_nodes, key=lambda x: int(x) if x.isdigit() else x)

        n_taz = len(taz_nodes)
        n_aux = len(aux_nodes)

        if n_aux == 0:
            print(f"   ℹ️  No se encontraron nodos auxiliares. Matriz OD permanece {n_taz}x{n_taz}")
            return

        print(f"   🔄 Expandiendo matriz OD: {n_taz}x{n_taz} → {n_taz + n_aux}x{n_taz + n_aux}")
        print(f"      - Nodos TAZ (demanda conocida): {n_taz}")
        print(f"      - Nodos auxiliares (demanda NaN): {n_aux}")

        # Crear matriz expandida
        n_total = n_taz + n_aux
        all_nodes = taz_nodes + aux_nodes

        # Convert current OD matrix to dense if it's sparse
        if sp.issparse(self.od_matrix):
            od_dense = self.od_matrix.toarray()
        else:
            od_dense = np.array(self.od_matrix)

        # Verificar dimensiones de la matriz actual
        if od_dense.shape[0] != n_taz or od_dense.shape[1] != n_taz:
            print(f"   ⚠️  Advertencia: Matriz OD actual es {od_dense.shape}, esperado {n_taz}x{n_taz}")
            # Ajustar si es necesario
            if od_dense.shape[0] < n_taz:
                # Pad con ceros si es más pequeña
                pad_size = n_taz - od_dense.shape[0]
                od_dense = np.pad(od_dense, ((0, pad_size), (0, pad_size)), mode='constant', constant_values=0)
            elif od_dense.shape[0] > n_taz:
                # Truncar si es más grande
                od_dense = od_dense[:n_taz, :n_taz]

        # Crear nueva matriz expandida con NaN
        expanded_od = np.full((n_total, n_total), np.nan, dtype=np.float32)

        # Copiar valores conocidos (TAZ-TAZ) en la esquina superior izquierda
        expanded_od[:n_taz, :n_taz] = od_dense

        # Las demás entradas quedan como NaN (aux-aux, aux-taz, taz-aux)

        # Actualizar la matriz OD en el manager
        self.od_matrix = expanded_od

        # Actualizar od_dataframe si existe para reflejar la nueva estructura
        if self.od_dataframe is not None:
            # Crear nuevo dataframe con todas las combinaciones
            od_pairs = []
            for i, origin in enumerate(all_nodes):
                for j, dest in enumerate(all_nodes):
                    demand = expanded_od[i, j]
                    od_pairs.append({
                        'origin': origin,
                        'destination': dest,
                        'demand': demand,
                        'origin_idx': i,
                        'dest_idx': j
                    })

            self.od_dataframe = pd.DataFrame(od_pairs)
            print(f"   ✓ Matriz OD expandida: {expanded_od.shape[0]}x{expanded_od.shape[1]}")
            print(f"      - Pares con demanda conocida: {np.sum(~np.isnan(expanded_od))}")
            print(f"      - Pares con demanda NaN: {np.sum(np.isnan(expanded_od))}")
        else:
            print(f"   ✓ Matriz OD expandida: {expanded_od.shape[0]}x{expanded_od.shape[1]}")

        # Guardar el mapeo de índices a IDs de nodos en metadata
        self.metadata['od_node_mapping'] = all_nodes
        self.metadata['n_taz_nodes'] = n_taz
        self.metadata['n_aux_nodes'] = n_aux

    def merge_network_flow(self) -> pd.DataFrame:
        """Combina los datos de la red y los flujos en un único DataFrame unificado.

        This function is robust to network DataFrame using 'init_node'/'term_node'
        column names (common in TNTP loaders) or already using 'from_node'/'to_node'.
        It performs a LEFT merge keeping all flow rows.
        """
        if self.network_df is None or self.flow_df is None:
            raise RuntimeError("Debe cargar la red y los flujos antes de fusionarlos.")

        # Work on a copy of network to avoid mutating original
        network_renamed = self.network_df.copy()

        # If network uses TNTP standard names, rename to from_node/to_node
        if 'init_node' in network_renamed.columns and 'term_node' in network_renamed.columns:
            network_renamed.rename(columns={'init_node': 'from_node', 'term_node': 'to_node'}, inplace=True)

        # Drop any duplicated column labels to avoid merge errors (do after rename)
        if network_renamed.columns.duplicated().any():
            network_renamed = network_renamed.loc[:, ~network_renamed.columns.duplicated()]

        # Ensure both frames have the join columns
        if 'from_node' not in network_renamed.columns or 'to_node' not in network_renamed.columns:
            raise RuntimeError("La tabla de red no contiene columnas 'from_node'/'to_node' ni 'init_node'/'term_node'.")
        if 'from_node' not in self.flow_df.columns or 'to_node' not in self.flow_df.columns:
            raise RuntimeError("La tabla de flujos no contiene columnas 'from_node'/'to_node'.")

        # Ensure flow_df has no duplicated columns either
        if self.flow_df.columns.duplicated().any():
            self.flow_df = self.flow_df.loc[:, ~self.flow_df.columns.duplicated()]

        # Normalize types to string for a safe merge
        def _to_str_col(df, col):
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col]).astype(int).astype(str)
            except Exception:
                df[col] = df[col].astype(str)

        _to_str_col(network_renamed, 'from_node')
        _to_str_col(network_renamed, 'to_node')
        _to_str_col(self.flow_df, 'from_node')
        _to_str_col(self.flow_df, 'to_node')

        # Perform LEFT merge to keep all flows
        merged_df = pd.merge(
            self.flow_df,
            network_renamed,
            on=['from_node', 'to_node'],
            how='left',
            indicator=True
        )

        # Verificar si hay columnas duplicadas después de la fusión
        duplicated_columns = merged_df.columns[merged_df.columns.duplicated()].tolist()
        if duplicated_columns:
            print(f"Advertencia: Se encontraron columnas duplicadas en la fusión: {duplicated_columns}")

        # Normalize lanes and VDF if present
        if 'lanes' in merged_df.columns:
            try:
                merged_df['lanes'] = pd.to_numeric(merged_df['lanes'], errors='coerce').fillna(1).astype(int)
            except Exception:
                merged_df['lanes'] = merged_df['lanes'].astype(int, errors='ignore')

        if 'VDF' in merged_df.columns:
            merged_df['VDF'] = pd.to_numeric(merged_df['VDF'], errors='coerce')

        # Reorder columns: place 'lanes' after 'capacity' and 'VDF' after 'speed' when present
        cols = list(merged_df.columns)
        def move_after(lst, col_to_move, after_col):
            if col_to_move in lst and after_col in lst:
                lst.remove(col_to_move)
                idx = lst.index(after_col)
                lst.insert(idx+1, col_to_move)
            return lst

        cols = move_after(cols, 'lanes', 'capacity')
        cols = move_after(cols, 'VDF', 'speed')
        merged_df = merged_df[cols]

        # Actualizar los metadatos para reflejar la fusión
        self.metadata['merged_columns'] = list(merged_df.columns)

        # Store unified_df
        self.unified_df = merged_df

        return merged_df

    def _calculate_unified_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        # ... (sin cambios)
        pass

    def build_graph(self) -> nx.Graph:
        """Construye el grafo de NetworkX a partir del DataFrame unificado."""
        if self.unified_df is None:
            raise RuntimeError("Debe cargar y fusionar la red y los flujos antes de construir el grafo.")

        # Ensure canonical join columns exist
        if 'from_node' not in self.unified_df.columns or 'to_node' not in self.unified_df.columns:
            raise RuntimeError("unified_df no contiene columnas 'from_node'/'to_node' para construir el grafo")

        df = self.unified_df.copy()
        df['from_node'] = df['from_node'].astype(str)
        df['to_node'] = df['to_node'].astype(str)

        # Análisis de depuración para identificar discrepancias en el número de links
        print(f"Debug: DataFrame has {len(df)} rows.")
        null_from = df['from_node'].isnull().sum()
        null_to = df['to_node'].isnull().sum()
        print(f"Debug: Null 'from_node': {null_from}, Null 'to_node': {null_to}")
        self_loops = (df['from_node'] == df['to_node']).sum()
        print(f"Debug: Self-loops (from_node == to_node): {self_loops}")
        duplicates = df.duplicated(subset=['from_node', 'to_node']).sum()
        print(f"Debug: Duplicate edges (same from_node, to_node): {duplicates}")
        empty_from = (df['from_node'] == '').sum()
        empty_to = (df['to_node'] == '').sum()
        print(f"Debug: Empty 'from_node': {empty_from}, Empty 'to_node': {empty_to}")

        if duplicates > 0:
            dup_df = df[df.duplicated(subset=['from_node', 'to_node'], keep=False)]
            print("Duplicate edges:")
            print(dup_df.to_string())

            # TODO: Implement better deduplication logic later
            # For now, prefer rows where link_type != 99
            if 'link_type' in df.columns:
                # Sort so that link_type != 99 comes first (assuming 99 is the highest value)
                df = df.sort_values(by='link_type', ascending=True)
                # Drop duplicates, keeping the first (which will be non-99 if available)
                df = df.drop_duplicates(subset=['from_node', 'to_node'], keep='first')
                print(f"After deduplication (preferring link_type != 99): {len(df)} rows.")
            else:
                print("Warning: 'link_type' column not found, skipping deduplication preference.")

        # Update the unified_df to match the deduplicated dataframe
        self.unified_df = df.copy()

        # Sync flow masks with the deduplicated dataframe
        self._sync_flow_masks()

        # Crear el grafo a partir del DataFrame
        graph = nx.from_pandas_edgelist(df, 'from_node', 'to_node', edge_attr=True, create_using=nx.DiGraph())

        # Agregar atributos de nodos si están disponibles
        if self.node_coords_df is not None and not self.node_coords_df.empty:
            # node loader may produce column 'node' or 'node_id'
            node_col = None
            for candidate in ['node', 'node_id', 'Node', 'NODE']:
                if candidate in self.node_coords_df.columns:
                    node_col = candidate
                    break

            if node_col is None:
                # fallback: use first column
                node_col = list(self.node_coords_df.columns)[0]

            for _, row in self.node_coords_df.iterrows():
                nid = row[node_col]
                try:
                    node_id = str(int(nid))
                except Exception:
                    node_id = str(nid)

                if graph.has_node(node_id):
                    graph.nodes[node_id]['x'] = row.get('x', None)
                    graph.nodes[node_id]['y'] = row.get('y', None)
                    graph.nodes[node_id]['pos'] = (row.get('x', None), row.get('y', None))
                    graph.nodes[node_id]['type'] = row.get('type', None)
                else:
                    # Node exists in node file but not in network - add it to graph
                    graph.add_node(node_id)
                    graph.nodes[node_id]['x'] = row.get('x', None)
                    graph.nodes[node_id]['y'] = row.get('y', None)
                    graph.nodes[node_id]['pos'] = (row.get('x', None), row.get('y', None))
                    graph.nodes[node_id]['type'] = row.get('type', None)

        # Verificar que el número de links en el grafo coincida con el DataFrame
        num_links_df = len(self.unified_df)
        num_links_graph = graph.number_of_edges()
        if num_links_df != num_links_graph:
            print(f"Warning: DataFrame has {num_links_df} links, but graph has {num_links_graph} edges.")
        else:
            print(f"Verification: Graph has {num_links_graph} edges, matching DataFrame.")

        return graph

    def process_routes(self, graph, route_source='file', num_routes=10,
                       route_algorithm: str = 'yen_astar',
                       execution_mode: str = 'sequential',
                       weight: str = 'free_flow_time',
                       force_recompute: bool = False,
                       use_route_cache: bool = True):
        """Procesa rutas utilizando el grafo cargado delegando en `route_calculator`.

        Esta implementación construye los pares OD a partir de los nodos TAZ y AUX
        del grafo, delega el cálculo de rutas en el orquestador central definido en
        `notebooks/route_calculator.py` y normaliza la salida en forma de tensor
        con padding.

        Args:
            graph: Grafo de NetworkX con la red ya construida.
            route_source (str):
                - 'file'  -> intentar cargar rutas precomputadas desde {case}_routes.tntp.
                - 'compute' -> calcular rutas a partir del grafo.
            num_routes (int): Número de rutas alternativas a calcular por par OD
                (solo si route_source='compute').
            route_algorithm (str): Algoritmo de rutas a usar cuando `route_source='compute'`.
                Valores típicos: 'yen_astar', 'yen_dijkstra', 'ksp_dijkstra', 'dijkstra', 'astar'.
            execution_mode (str): 'sequential' o 'parallel' para el cálculo cuando
                `route_source='compute'`.
            weight (str): Nombre del atributo de arista a utilizar como peso
                (por defecto 'free_flow_time').
            force_recompute (bool): Si True, fuerza el recálculo de rutas incluso si
                existen rutas cacheadas.
            use_route_cache (bool): Si True, utiliza rutas cacheadas si están disponibles.

        Returns:
            dict: con las claves
                - 'routes': numpy.ndarray de shape [num_od_pairs, num_routes, max_route_length]
                - 'od_pairs': lista de tuplas (origin, destination)
                - 'num_routes': número de rutas por par (k efectivo)
                - 'max_route_length': longitud máxima de cualquier ruta en nodos

        Notas:
            - Se utiliza un archivo de caché `kshortest_paths.pkl` dentro de
              `self.routing_cache_dir` para evitar recomputar rutas si ya existen.
            - Si `route_source='file'` pero el archivo de rutas no existe o no se
              puede leer, se hace fallback automático a `route_source='compute'`.
        """
        import pickle
        import numpy as np
        import ast

        # Importación perezosa de utilidades de route_calculator. Se asume que el
        # archivo `notebooks/route_calculator.py` está disponible y contiene las
        # funciones `compute_routes_for_od_pairs` y `create_routes_tensor_optimized`.
        try:
            # Añadimos la ruta de notebooks al sys.path sólo si es necesario
            from pathlib import Path as _Path
            import sys as _sys

            notebooks_dir = _Path(__file__).resolve().parents[2] / 'notebooks'
            if str(notebooks_dir) not in _sys.path:
                _sys.path.insert(0, str(notebooks_dir))

            import route_calculator as _rc
        except Exception as e:  # pragma: no cover - fallo poco probable
            print(f"   ⚠️  No se pudo importar route_calculator: {e}")
            _rc = None

        cache_file = self.routing_cache_dir / 'kshortest_paths.pkl'

        # Determine effective force_recompute by combining the function parameter
        # with any YAML setting (supporting both 'force_recompute' and 'force_precompute').
        effective_force = bool(force_recompute)
        try:
            yaml_path = Path(__file__).resolve().parent / 'data_pipeline_args.yaml'
            if yaml_path.exists():
                ycfg = _read_yaml_defaults(yaml_path)
                if isinstance(ycfg, dict):
                    if ycfg.get('force_recompute'):
                        effective_force = True
                    if ycfg.get('force_precompute'):
                        effective_force = True
        except Exception:
            pass

        # Debug: show effective flags
        try:
            print(f"   ℹ️  Flags -> force_recompute(param)={force_recompute}, effective_force={effective_force}, use_route_cache={use_route_cache}, route_source_requested={route_source}")
        except Exception:
            pass

        # Global cache check: if a cache file exists and the user allows using it
        # and they did not request force recompute, return cached results immediately.
        if use_route_cache and (not effective_force) and cache_file.exists():
            try:
                print(f"   ℹ️  Rutas cacheadas encontradas, cargando desde cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                return cached_data
            except Exception:
                # If cache loading fails for any reason, continue to recompute
                print(f"   ⚠️  Error leyendo cache de rutas, se procederá a recomputar.")

        # Extraer nodos TAZ (Traffic Analysis Zones) y nodos auxiliares desde el grafo
        taz_nodes = []
        aux_nodes = []
        for node_id, node_data in graph.nodes(data=True):
            node_type = node_data.get('type') or node_data.get('Type')
            if node_type == 'taz':
                taz_nodes.append(node_id)
            elif node_type == 'aux':
                aux_nodes.append(node_id)

        od_nodes = taz_nodes + aux_nodes

        # Si no hay nodos etiquetados como TAZ/AUX, usamos todos los nodos como OD
        if not od_nodes:
            print(f"   ⚠️  No se encontraron nodos TAZ ni aux en el grafo. Usando todos los nodos como OD.")
            od_nodes = list(graph.nodes())
            taz_nodes = od_nodes
            aux_nodes = []

        # Ordenar para una enumeración consistente
        def _sort_key(x):
            s = str(x)
            return (0, int(s)) if s.isdigit() else (1, s)

        taz_nodes = sorted(taz_nodes, key=_sort_key)
        aux_nodes = sorted(aux_nodes, key=_sort_key)
        od_nodes = sorted(od_nodes, key=_sort_key)

        print(f"   📍 Encontrados {len(taz_nodes)} nodos TAZ + {len(aux_nodes)} nodos aux = {len(od_nodes)} nodos OD totales")

        # Generar todos los pares OD (n×n incluyendo la diagonal)
        od_pairs = []
        for origin in od_nodes:
            for dest in od_nodes:
                od_pairs.append((origin, dest))

        print(f"   📊 Procesando {len(od_pairs)} pares OD ({len(od_nodes)}×{len(od_nodes)} - {len(od_nodes)} diagonales)")

        # 1) Caso route_source = 'file': intentar leer rutas precomputadas
        if route_source == 'file':
            # If the effective_force flag is set (param or YAML), skip loading from file
            if effective_force:
                print("   ⚠️  effective_force=True -> ignorando archivo de rutas y forzando cálculo")
                route_source = 'compute'
            else:
                all_routes = self._load_routes_from_tntp(num_routes)
                if all_routes is None:
                    print(f"   ⚠️  No se pudo cargar rutas desde archivo, cambiando a cálculo compute...")
                    route_source = 'compute'
                else:
                    # Convertimos las rutas de archivo al tensor esperado y devolvemos
                    # directamente sin usar route_calculator.
                    max_route_length = 0
                    for od_routes in all_routes:
                        for route in od_routes:
                            max_route_length = max(max_route_length, len(route))

                    print(f"   📏 Longitud máxima de ruta (archivo): {max_route_length} nodos")
                    print(f"   🔢 Padding rutas a longitud uniforme (archivo)...")

                    padded_routes = []
                    for od_routes in all_routes:
                        padded_od = []
                        for route in od_routes:
                            padded = route + [-1] * (max_route_length - len(route))
                            padded_od.append(padded)
                        while len(padded_od) < num_routes:
                            padded_od.append([-1] * max_route_length)
                        padded_routes.append(padded_od)

                    routes_array = np.array(padded_routes, dtype=np.int32)

                    cache_data = {
                        'routes': routes_array,
                        'od_pairs': od_pairs,
                        'num_routes': num_routes,
                        'max_route_length': max_route_length,
                    }

                    # Guardar en caché para futuras ejecuciones (si se permite)
                    try:
                        with open(cache_file, 'wb') as f:
                            pickle.dump(cache_data, f)
                    except Exception:
                        pass

                    print(f"   💾 Rutas (archivo) guardadas en cache: {cache_file}")
                    return cache_data

        # 2) Caso compute: delegar en route_calculator si está disponible
        if route_source == 'compute':
            if _rc is None:
                # Si no pudimos importar route_calculator, usamos el antiguo A* como fallback
                print("   ⚠️  route_calculator no disponible, usando implementación interna A* como fallback.")
                all_routes = self._compute_routes_astar(graph, od_pairs, num_routes)

                # Encontrar longitud máxima y hacer padding (mismo patrón que antes)
                max_route_length = 0
                for od_routes in all_routes:
                    for route in od_routes:
                        max_route_length = max(max_route_length, len(route))

                print(f"   📏 Longitud máxima de ruta: {max_route_length} nodos")
                print(f"   🔢 Padding rutas a longitud uniforme...")

                padded_routes = []
                for od_routes in all_routes:
                    padded_od = []
                    for route in od_routes:
                        padded = route + [-1] * (max_route_length - len(route))
                        padded_od.append(padded)
                    while len(padded_od) < num_routes:
                        padded_od.append([-1] * max_route_length)
                    padded_routes.append(padded_od)

                routes_array = np.array(padded_routes, dtype=np.int32)

                cache_data = {
                    'routes': routes_array,
                    'od_pairs': od_pairs,
                    'num_routes': num_routes,
                    'max_route_length': max_route_length,
                }

                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)

                print(f"   💾 Rutas guardadas en cache (fallback A*): {cache_file}")
                return cache_data

            # Delegar en el orquestador central con configuración flexible
            print(f"   🔄 Calculando rutas vía route_calculator (algoritmo={route_algorithm}, modo={execution_mode}, k={num_routes})...")
            raw_routes = _rc.compute_routes_for_od_pairs(
                graph,
                od_pairs,
                algorithm=route_algorithm,
                k=num_routes,
                weight=weight,
                execution_mode=execution_mode,
                show_progress=True,
            )

            # Convertir lista de rutas en tensor con padding uniforme
            routes_array = _rc.create_routes_tensor_optimized(raw_routes, num_routes)

            max_route_length = routes_array.shape[2] if routes_array.ndim == 3 else 0
            print(f"   ✅ Rutas procesadas (route_calculator): shape {routes_array.shape}")

            cache_data = {
                'routes': routes_array,
                'od_pairs': od_pairs,
                'num_routes': num_routes,
                'max_route_length': max_route_length,
            }

            # Guardar en caché
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)

            print(f"   💾 Rutas guardadas en cache: {cache_file}")
            return cache_data

        # Si route_source no es ni 'file' ni 'compute', devolvemos estructura vacía
        print(f"   ⚠️  route_source='{route_source}' no reconocido, devolviendo resultados vacíos.")
        empty = np.zeros((0, num_routes, 0), dtype=np.int32)
        return {
            'routes': empty,
            'od_pairs': [],
            'num_routes': num_routes,
            'max_route_length': 0,
        }

    def _load_routes_from_tntp(self, num_routes):
        """Carga rutas desde archivo {case}_routes.tntp en formato TNTP.

        El archivo se busca utilizando primero el nombre canónico del caso
        (`self.canonical_name`) si está disponible, y en su defecto `self.network_name`.
        Cada línea del archivo debe contener una representación textual de una lista
        de rutas para un par OD (se parsea usando ``ast.literal_eval``).

        Args:
            num_routes (int): Número máximo de rutas a considerar por par OD.

        Returns:
            list[list[list[int]]] | None: Lista de rutas por OD, o ``None`` si el
            archivo no se encuentra o no se puede leer.
        """
        import ast

        # Try to find routes file using canonical name if available
        case_name = self.canonical_name if self.canonical_name else self.network_name

        # Try to find routes file
        routes_file = None
        possible_paths = [
            self.network_path.parent / f"{case_name}_routes.tntp",
            self.network_path.parent / f"{self.network_name}_routes.tntp",
            PROJECT_ROOT / 'data' / 'interim' / f"{case_name}_routes.tntp",
            PROJECT_ROOT / 'data' / 'interim' / f"{self.network_name}_routes.tntp",
        ]

        for path in possible_paths:
            if path.exists():
                routes_file = path
                break

        if routes_file is None:
            print(f"   ⚠️  No se encontró archivo de rutas {case_name}_routes.tntp")
            return None

        print(f"   📂 Cargando rutas desde: {routes_file}")

        try:
            all_routes = []
            with open(routes_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    # Skip comments and empty lines
                    if not line or line.startswith('#'):
                        continue

                    # Parse the list of routes using ast.literal_eval (safe Python eval)
                    try:
                        od_routes = ast.literal_eval(line)
                        if not isinstance(od_routes, list):
                            print(f"   ⚠️  Línea {i+1} no es una lista válida")
                            continue
                        all_routes.append(od_routes)
                    except Exception as e:
                        print(f"   ⚠️  Error parseando línea {i+1}: {e}")
                        continue

            print(f"   ✅ Cargadas {len(all_routes)} rutas desde archivo")
            return all_routes

        except Exception as e:
            print(f"   ❌ Error cargando archivo de rutas: {e}")
            return None

    def _compute_routes_astar(self, graph, od_pairs, num_routes):
        """Calcula rutas usando A* con heurística euclidiana.

        Esta función implementa la lógica legacy utilizada antes de delegar en
        `route_calculator`. Se conserva como fallback para entornos donde no es
        posible importar dicho módulo.
        """
        import numpy as np

        print(f"   🔄 Calculando {num_routes}-shortest paths para {len(od_pairs)} pares OD (A* legacy)...")

        # Prepare heuristic function for A* using Euclidean distance
        def euclidean_heuristic(node1, node2):
            """Calculate Euclidean distance between two nodes using their coordinates.

            Si alguna coordenada falta o no es válida, se devuelve 0 para mantener
            una heurística admisible y evitar errores.
            """
            try:
                n1_data = graph.nodes[node1]
                n2_data = graph.nodes[node2]
                x1, y1 = n1_data.get('x', 0), n1_data.get('y', 0)
                x2, y2 = n2_data.get('x', 0), n2_data.get('y', 0)

                if x1 is None or y1 is None or x2 is None or y2 is None:
                    return 0.0

                return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            except Exception:
                return 0.0

        all_routes = []

        for i, (origin, dest) in enumerate(od_pairs):
            if i % 500 == 0 and i > 0:
                print(f"      Procesando par {i}/{len(od_pairs)}...")

            origin_str = str(origin)
            dest_str = str(dest)

            # Check if nodes exist in graph
            if origin_str not in graph or dest_str not in graph:
                # No path possible, store empty routes
                all_routes.append([])
                continue

            try:
                # Use A* for first path, then k-shortest for alternatives
                paths = list(self._k_shortest_paths_astar(graph, origin_str, dest_str, num_routes,
                                                          weight='free_flow_time',
                                                          heuristic=euclidean_heuristic))

                # Store as list of node sequences (convert to int)
                route_list = []
                for path in paths:
                    route = []
                    for node in path:
                        try:
                            route.append(int(node))
                        except (ValueError, TypeError):
                            route.append(hash(node) % (2**31))  # Use hash for non-int nodes
                    route_list.append(route)

                all_routes.append(route_list)
            except Exception:
                # No path found or error
                all_routes.append([])

        # Guardar rutas en disco para reutilización futura
        self._save_routes_to_tntp(all_routes)

        return all_routes

    def _save_routes_to_tntp(self, all_routes):
        """Guarda rutas en formato TNTP {case}_routes.tntp.

        Cada línea contiene la representación textual (estilo lista de Python) de
        todas las rutas asociadas a un par OD. Este formato es el mismo que
        consume :meth:`_load_routes_from_tntp` para facilitar la reutilización.
        """
        case_name = self.canonical_name if self.canonical_name else self.network_name
        output_file = self.processed_dir / f"{case_name}_routes.tntp"

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Routes (one per line)\n")
                for od_routes in all_routes:
                    # Write as Python list format
                    f.write(str(od_routes) + "\n")

            print(f"   💾 Rutas guardadas en formato TNTP: {output_file}")
        except Exception as e:
            print(f"   ⚠️  Error guardando rutas en formato TNTP: {e}")

    def _k_shortest_paths_astar(self, graph, source, target, k, weight='weight', heuristic=None):
        """Generador de k-shortest paths usando A* para el primer camino y
        luego `shortest_simple_paths` como aproximación tipo Yen.

        Se mantiene esta implementación como parte del camino legacy de
        cálculo de rutas cuando no se usa `route_calculator`.
        """
        import networkx as nx
        # First path using A* with heuristic if available
        try:
            if heuristic:
                # A* for first shortest path with heuristic
                first_path = nx.astar_path(graph, source, target, heuristic=heuristic, weight=weight)
                yield first_path

                # Then use Yen's algorithm for k-1 alternative paths
                count = 1
                for path in nx.shortest_simple_paths(graph, source, target, weight=weight):
                    # Skip the first one (already yielded from A*)
                    if count == 1:
                        count += 1
                        continue
                    if count >= k:
                        break
                    yield path
                    count += 1
            else:
                # Fallback to simple shortest paths without heuristic
                for i, path in enumerate(nx.shortest_simple_paths(graph, source, target, weight=weight)):
                    if i >= k:
                        break
                    yield path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return

    def process_all(self, save_graph_pickle: bool = False, save_graph_image: bool = False, graph_pickle_path: Optional[Union[str, Path]] = None, graph_image_path: Optional[Union[str, Path]] = None, graph_format: str = 'png', od_format: str = 'sparse'):
        """Ejecuta todo el procesamiento: loading (via DataLoader), fusión y guardado (via DataSaver).

        Returns unified dataframe.
        """
        # Load datasets via DataLoader
        loader = DataLoader(multiday=self.multiday)
        loader.load_all(self)

        # Merge network and flows
        self.unified_df = self.merge_network_flow()

        # Build graph
        self.graph = self.build_graph()

        # Save outputs via DataSaver
        saver = DataSaver(self.processed_dir)
        if save_graph_pickle:
            saver.save_graph_pickle(self.graph, graph_pickle_path)
        if save_graph_image:
            saver.save_graph_image(self.graph, graph_image_path, fmt=graph_format)

        return self.unified_df

    def load_and_merge(self, od_format: str = 'sparse') -> Tuple[pd.DataFrame, sparse.spmatrix]:
         """Convenience method expected by older code: carga red, flujos, OD y devuelve (unified_df, od_matrix).

         od_format: passed to load_od_matrix if applicable (ignored here; load_od_matrix returns sparse by default)
         """
         # Execute loads
         self.load_network()
         self.load_flow()
         self.load_od_matrix()

         # Merge
         unified = self.merge_network_flow()

         return unified, self.od_matrix

    def _sync_flow_masks(self):
        """Sincroniza las máscaras de flujos con unified_df después de deduplicación."""
        if self.unified_df is None:
            return

        year = self.config['data']['volume_year']
        train_split = self.config['data']['train_split']
        volume_col = f'Volume_{year}'

        if volume_col not in self.unified_df.columns:
            raise ValueError(f"Columna {volume_col} no encontrada en unified_df")

        flows = self.unified_df[volume_col].values
        valid_mask = ~np.isnan(flows)
        num_valid = valid_mask.sum()

        print(f"   📊 Sincronizando máscaras: {num_valid}/{len(flows)} enlaces válidos")

        # Split train/test solo en enlaces con observaciones
        np.random.seed(self.config['data']['random_seed'])

        valid_indices = np.where(valid_mask)[0]
        np.random.shuffle(valid_indices)
        split_idx = int(len(valid_indices) * train_split)

        train_indices = valid_indices[:split_idx]
        test_indices = valid_indices[split_idx:]

        # Crear máscaras
        self.train_flow_mask = np.zeros(len(self.unified_df), dtype=np.float32)
        self.test_flow_mask = np.zeros(len(self.unified_df), dtype=np.float32)

        self.train_flow_mask[train_indices] = 1.0
        self.test_flow_mask[test_indices] = 1.0

        print(f"      ✓ Train: {len(train_indices)} enlaces")
        print(f"      ✓ Test: {len(test_indices)} enlaces")


def _read_yaml_defaults(yaml_path: Path) -> Dict:
    """Lee valores por defecto desde un archivo YAML sencillo.

    Esta función se usa principalmente por el CLI legacy de este módulo para
    cargar parámetros básicos (como `case`, `source`, etc.) desde un archivo
    YAML de configuración. Se intenta usar PyYAML si está disponible; en caso
    contrario se aplica un pequeño parser manual sobre la primera sección del
    archivo.
    """
    # Try using PyYAML if available
    if yaml is not None:
        try:
            with open(yaml_path, 'r', encoding='utf-8') as f:
                # Use safe_load_all to handle multiple documents, take first valid dict
                for doc in yaml.safe_load_all(f):
                    if isinstance(doc, dict):
                        return doc
                return {}
        except Exception:
            # Si hay algún problema, continuamos con el parser ligero
            pass

    # Fallback: lightweight parser para key: value simples en el primer documento
    try:
        text = yaml_path.read_text(encoding='utf-8')
        lines = text.splitlines()
        res: Dict = {}
        in_block = False

        for raw in lines:
            line = raw.strip()
            if not line:
                continue

            # Stop at document separator (solo usamos el primer documento)
            if line == '---':
                break

            # Toggle triple-quoted block markers (skip lines inside them)
            if line.startswith('"""') or line.startswith("'''"):
                in_block = not in_block
                continue
            if in_block:
                continue

            # Skip full-line comments (YAML '#')
            if line.startswith('#') or line.startswith('//'):
                continue

            # Parse simple key: value pairs
            if ':' in line:
                try:
                    key, val = line.split(':', 1)
                    key = key.strip()
                    val = val.strip()

                    # Remove inline comments (anything after # that's not in quotes)
                    if '#' in val:
                        if not (val.startswith('"') or val.startswith("'")):
                            val = val.split('#')[0].strip()

                    # Remove surrounding quotes
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]

                    low = val.lower()
                    if low in ('true', 'false'):
                        parsed_val = (low == 'true')
                    else:
                        parsed_val = val
                        try:
                            if '.' in val:
                                parsed_val = float(val)
                            else:
                                parsed_val = int(val)
                        except Exception:
                            parsed_val = val

                    res[key] = parsed_val
                except Exception:
                    continue
        return res
    except Exception:
        return {}

