"""
Data Ingestion para Cyclic Model - Linköping Traffic Assignment

Este módulo maneja la carga y preparación de datos de Linköping para el CyclicODModel.
Adaptado para funcionar con Hydra de manera robusta.

Autor: Sistema de Acoplamiento Linköping
Fecha: Noviembre 2025
"""

import numpy as np
import pandas as pd
import pickle
import torch
import networkx as nx
from pathlib import Path
from scipy import sparse
from typing import Dict, Tuple, List, Optional
import logging
from omegaconf import DictConfig, OmegaConf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LinkopingDataLoader:
    """
    Cargador de datos para Linköping Traffic Assignment.
    Integrado con Hydra.
    """

    def __init__(self, cfg: DictConfig):
        """
        Inicializa el cargador de datos usando la configuración de Hydra.
        """
        self.cfg = cfg

        # --- 1. RESOLUCIÓN ROBUSTA DE CONFIGURACIÓN ---
        # Detectamos si 'data' está en la raíz (Global) o anidado en 'dataset'
        if 'data' in self.cfg:
            self.data_cfg = self.cfg.data
            self.network_cfg = self.cfg.get('network', {})
            self.sampling_cfg = self.cfg.get('sampling', {})
            logger.info("Configuración cargada desde namespace GLOBAL (Correcto).")
        elif 'dataset' in self.cfg and 'data' in self.cfg.dataset:
            self.data_cfg = self.cfg.dataset.data
            self.network_cfg = self.cfg.dataset.get('network', {})
            self.sampling_cfg = self.cfg.dataset.get('sampling', {})
            logger.warning("Configuración encontrada bajo 'dataset'. Se recomienda usar '# @package _global_' en el YAML.")
        else:
            # Fallback crítico para evitar crash inmediato, lanzará error descriptivo luego
            logger.error("No se encontró el bloque 'data' en la configuración.")
            self.data_cfg = DictConfig({'volume_year': 2022}) # Dummy para evitar crash en init

        # Alias para compatibilidad
        self.config = self.cfg

        # --- 2. RESOLVER PATH BASE ---
        path_str = self.data_cfg.get('base_path', None)

        if not path_str and hasattr(self.cfg, 'paths'):
             path_str = f"{self.cfg.paths.data_processed}/Linköping"

        if not path_str:
            path_str = "data/processed/Linköping"
            logger.warning(f"'base_path' no definido. Usando default: {path_str}")

        config_base = Path(path_str)

        # Lógica Smart Path Finding (manejo de encoding Windows)
        if config_base.exists():
            self.base_path = config_base
            logger.info(f"Usando path: {self.base_path}")
        else:
            # Búsqueda manual insensible a encoding
            processed_dir = Path("data/processed")
            if not processed_dir.exists():
                processed_dir = Path("../../data/processed") # Intento relativo

            found = False
            if processed_dir.exists():
                for item in processed_dir.iterdir():
                    if item.is_dir() and 'link' in item.name.lower():
                        self.base_path = item
                        logger.info(f"Directorio autodetectado: {self.base_path}")
                        found = True
                        break

            if not found:
                self.base_path = config_base
                logger.warning(f"No se encontró el directorio físico. Se usará: {self.base_path}")

        # Inicialización de variables
        self.graph: Optional[nx.DiGraph] = None
        self.link_data: Optional[pd.DataFrame] = None
        self.od_matrix: Optional[sparse.csr_matrix] = None
        self.routes_data: Optional[Dict] = None
        self.edge_list: Optional[List[Tuple]] = None
        self.network_params: Optional[Dict] = None

        logger.info(f" LinkopingDataLoader inicializado")

        # Acceso seguro a volume_year
        vol_year = self.data_cfg.get('volume_year', 'Unknown')
        logger.info(f"   Volume year: {vol_year}")

    def load_all(self) -> Tuple[nx.DiGraph, sparse.csr_matrix, pd.DataFrame, Dict]:
        """Carga todos los datos necesarios."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Cargando datos de Linköping")
        logger.info(f"{'='*80}")

        self.graph = self._load_graph()
        self.od_matrix = self._load_od_matrix()
        self.link_data = self._load_link_data()
        self.routes_data = self._load_routes()

        self._validate_data()
        return self.graph, self.od_matrix, self.link_data, self.routes_data

    def _load_graph(self) -> nx.DiGraph:
        """Carga el grafo de red."""
        graph_file = self.data_cfg.get('graph_file', 'Linköping_graph.pkl')
        return self._smart_load(graph_file, "Grafo", pickle_load=True)

    def _load_od_matrix(self) -> sparse.csr_matrix:
        """Carga la matriz OD sparse."""
        od_file = self.data_cfg.get('od_matrix_file', 'Linköping_od_matrix.npz')

        file_path = self._resolve_file_path(od_file)
        logger.info(f"Cargando matriz OD: {file_path}")

        data = np.load(file_path)
        od_matrix = sparse.csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )
        logger.info(f"Shape: {od_matrix.shape}")
        return od_matrix

    def _load_link_data(self) -> pd.DataFrame:
        """Carga datos de enlaces."""
        link_file = self.data_cfg.get('link_data_file', 'Linköping_link_data.parquet')
        file_path = self._resolve_file_path(link_file)

        logger.info(f"Cargando datos de enlaces: {file_path}")
        link_data = pd.read_parquet(file_path)

        # Verificar columnas
        volume_cols = [c for c in link_data.columns if 'Volume' in c]
        logger.info(f"Enlaces: {len(link_data)}. Años: {volume_cols}")
        return link_data

    def _load_routes(self) -> Dict:
        """Carga rutas precalculadas."""
        routes_file = self.data_cfg.get('routing_cache_file', 'routing_cache/Linköping_kshortest_paths.pkl')

        # Intentar ruta configurada
        routes_path = self.base_path / routes_file

        # Fallbacks inteligentes
        if not routes_path.exists():
            fallbacks = [
                self.base_path / 'routing_cache' / 'kshortest_paths.pkl',
                self.base_path / 'kshortest_paths.pkl',
                self.base_path / f"{self.data_cfg.get('dataset', 'Linköping')}_kshortest_paths.pkl"
            ]
            for p in fallbacks:
                if p.exists():
                    routes_path = p
                    break

        logger.info(f"Cargando rutas: {routes_path}")
        if not routes_path.exists():
            raise FileNotFoundError(f"No se encontró archivo de rutas en: {routes_path}")

        with open(routes_path, 'rb') as f:
            raw_data = pickle.load(f)

        # Conversión de formato si es necesario (Dictionary -> Tensors)
        first_key = next(iter(raw_data))
        if isinstance(first_key, tuple):
            logger.info("Formato crudo detectado (diccionario). Convirtiendo a tensores...")
            return self._convert_routes_dict_to_tensor(raw_data)

        return raw_data

    # --- HELPER METHODS ---

    def _resolve_file_path(self, filename: str) -> Path:
        """Busca un archivo manejando caracteres especiales (ö, ä, etc.)"""
        direct_path = self.base_path / filename
        if direct_path.exists():
            return direct_path

        # Búsqueda con comodines para caracteres especiales
        safe_pattern = filename.replace('ö', '*').replace('ä', '*').replace('å', '*')
        matches = list(self.base_path.glob(safe_pattern))
        if matches:
            return matches[0]

        # Búsqueda genérica por extensión
        ext = Path(filename).suffix
        matches = list(self.base_path.glob(f"*{ext}"))
        if matches:
            logger.warning(f"Archivo exacto no encontrado. Usando alternativa: {matches[0].name}")
            return matches[0]

        raise FileNotFoundError(f"No se encontró el archivo {filename} en {self.base_path}")

    def _smart_load(self, filename: str, desc: str, pickle_load: bool = False):
        path = self._resolve_file_path(filename)
        logger.info(f"Cargando {desc}: {path}")
        if pickle_load:
            with open(path, 'rb') as f:
                return pickle.load(f)
        return path

    def _convert_routes_dict_to_tensor(self, raw_data: Dict) -> Dict:
        """Convierte diccionario de rutas a tensores."""
        node_list = sorted(list(self.graph.nodes()))
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        k_paths = self.network_cfg.get('k_paths', 10)
        num_od = len(raw_data)

        # Calcular longitud máxima
        max_len = 0
        for paths in raw_data.values():
            for p in paths:
                max_len = max(max_len, len(p))

        routes_tensor = np.full((num_od, k_paths, max_len), -1, dtype=np.int32)
        od_pairs_array = np.zeros((num_od, 2), dtype=object)

        for i, ((u, v), paths) in enumerate(raw_data.items()):
            od_pairs_array[i] = [u, v]
            for k, path in enumerate(paths):
                if k >= k_paths: break
                try:
                    path_indices = [node_to_idx[n] for n in path]
                    routes_tensor[i, k, :len(path_indices)] = path_indices
                except KeyError:
                    pass # Nodo no encontrado

        return {'routes': routes_tensor, 'od_pairs': od_pairs_array}

    def _validate_data(self):
        """Validación básica de consistencia."""
        if self.graph and self.link_data is not None:
            if self.graph.number_of_edges() != len(self.link_data):
                logger.warning(f"Mismatch enlaces: Grafo={self.graph.number_of_edges()}, DF={len(self.link_data)}")
            else:
                logger.info("Consistencia de enlaces: OK")

    def prepare_observed_flows(self, year: Optional[int] = None, train_split: Optional[float] = None):
        if year is None: year = self.data_cfg.get('volume_year', 2022)
        if train_split is None: train_split = self.data_cfg.get('train_split', 0.8)

        volume_col = f'Volume_{year}'
        if volume_col not in self.link_data.columns:
            raise ValueError(f"Columna {volume_col} no encontrada.")

        flows = self.link_data[volume_col].values
        all_flows = np.nan_to_num(flows, nan=0.0)

        # Crear máscaras
        valid_indices = np.where(~np.isnan(flows))[0]
        np.random.seed(self.data_cfg.get('random_seed', 42))
        np.random.shuffle(valid_indices)

        split_idx = int(len(valid_indices) * train_split)
        train_mask = np.zeros_like(flows, dtype=np.float32)
        test_mask = np.zeros_like(flows, dtype=np.float32)

        train_mask[valid_indices[:split_idx]] = 1.0
        test_mask[valid_indices[split_idx:]] = 1.0

        logger.info(f" Train split: {len(valid_indices[:split_idx])} obs.")
        return all_flows, train_mask, test_mask

    def prepare_od_demand_vector(self):
        od_dense = self.od_matrix.toarray().flatten()
        od_mask = (~np.isnan(od_dense)).astype(np.float32)
        od_vector = np.nan_to_num(od_dense, nan=0.0)
        logger.info(f"Demandas OD conocidas: {od_mask.sum()}")
        return od_vector, od_mask

    def prepare_network_parameters(self) -> Dict:
        """Prepara tensores de red para el modelo."""
        self.edge_list = list(self.graph.edges())
        num_links = len(self.edge_list)

        t0 = np.array([self.graph[u][v].get('free_flow_time', 1.0) for u, v in self.edge_list], dtype=np.float32)
        capacity = np.array([self.graph[u][v].get('capacity', 1000.0) for u, v in self.edge_list], dtype=np.float32)
        link_type = np.array([self.graph[u][v].get('link_type', 0) for u, v in self.edge_list], dtype=np.int32)

        # Link Groups
        unique_types = np.unique(link_type)
        type_map = {t: i for i, t in enumerate(unique_types)}
        link_group = np.array([type_map[t] for t in link_type])

        # Route Masks
        # route_masks YA ES un torch.sparse_coo_tensor
        route_masks, od_pair_indices = self._build_route_masks()

        self.network_params = {
            'num_links': num_links,
            'num_od_pairs': route_masks.shape[0],
            't0': torch.FloatTensor(t0),
            'capacity': torch.FloatTensor(capacity),

            # --- CORRECCIÓN AQUÍ ---
            # No usar torch.FloatTensor(route_masks), usarlo directo:
            'route_masks': route_masks,
            # -----------------------

            'od_pair_indices': torch.LongTensor(od_pair_indices),
            'num_link_groups': len(unique_types),
            'link_group': torch.LongTensor(link_group)
        }
        logger.info("Parámetros de red preparados.")
        return self.network_params

    def _build_route_masks(self) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Construye route_masks como un Tensor Esparso (Sparse COO).
        Ahorra ~99% de memoria comparado con la versión densa.
        """
        logger.info("Construyendo máscaras de ruta (Modo Esparso)...")

        routes = self.routes_data['routes']
        od_pairs = self.routes_data['od_pairs']
        num_od, k_paths = routes.shape[:2]
        num_links = len(self.edge_list)

        # Mapeos rápidos
        edge_to_idx = {e: i for i, e in enumerate(self.edge_list)}
        node_list = sorted(list(self.graph.nodes()))
        node_to_idx = {n: i for i, n in enumerate(node_list)}

        # Listas para construir el formato COO (Coordinate List)
        # Indices: [dim_0, dim_1, dim_2] -> [od_idx, path_idx, link_idx]
        indices_od = []
        indices_k = []
        indices_link = []
        values = []

        for i in range(num_od):
            for k in range(k_paths):
                path = routes[i, k]
                path = path[path != -1]  # Quitar padding
                if len(path) < 2: continue

                for j in range(len(path) - 1):
                    u_idx, v_idx = path[j], path[j + 1]
                    # Recuperar nodos reales para buscar en edge_to_idx
                    # Nota: Si tus rutas ya tienen índices de nodo correctos, esto es rápido
                    u, v = node_list[u_idx], node_list[v_idx]

                    if (u, v) in edge_to_idx:
                        l_idx = edge_to_idx[(u, v)]

                        # Guardamos la coordenada del 1.0
                        indices_od.append(i)
                        indices_k.append(k)
                        indices_link.append(l_idx)
                        values.append(1.0)

        # Crear el Tensor Esparso de PyTorch
        if not indices_od:
            logger.warning("¡No se encontraron rutas válidas para la máscara!")
            # Retornar tensor vacío seguro
            return torch.sparse_coo_tensor(
                indices=torch.empty((3, 0), dtype=torch.long),
                values=torch.empty(0),
                size=(num_od, k_paths, num_links)
            ), np.array([])

        # Construir índices [3, N_non_zeros]
        i_tensor = torch.LongTensor([indices_od, indices_k, indices_link])
        v_tensor = torch.FloatTensor(values)

        sparse_route_masks = torch.sparse_coo_tensor(
            i_tensor,
            v_tensor,
            size=(num_od, k_paths, num_links)
        ).coalesce()  # Importante: coalesce ordena y optimiza la estructura

        logger.info(
            f"Máscara Esparsa creada. Densidad: {sparse_route_masks._nnz() / (num_od * k_paths * num_links):.6f}")

        # Preparar OD indices (igual que antes)
        od_indices = []
        for u, v in od_pairs:
            u_idx = node_to_idx.get(u, -1)
            v_idx = node_to_idx.get(v, -1)
            od_indices.append([u_idx, v_idx])

        return sparse_route_masks, np.array(od_indices, dtype=np.int64)

    def get_summary(self) -> Dict:
        return {'network': self.data_cfg.get('dataset', 'Unknown')}