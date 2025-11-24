"""
Data Ingestion para Cyclic Model - Linköping Traffic Assignment

Este módulo maneja la carga y preparación de datos de Linköping para el CyclicODModel.
Adaptado para funcionar con Hydra.

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

# Import sampling utilities
# from src.train.sampling import create_partial_data_masks

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

        Args:
            cfg: Configuración completa de Hydra (DictConfig)
        """
        self.cfg = cfg
        # Alias para mantener compatibilidad con métodos existentes que usan self.config['data']...
        self.config = cfg

        # 1. Resolver el path base desde la configuración
        # Intentamos obtener la ruta desde 'data.base_path' (común en Linköping.yaml)
        # o desde 'paths.data_processed' (si usas estructura global de paths)
        if hasattr(self.cfg, 'data') and hasattr(self.cfg.data, 'base_path'):
            path_str = self.cfg.data.base_path
        elif hasattr(self.cfg, 'paths') and hasattr(self.cfg.paths, 'data_processed'):
            path_str = self.cfg.paths.data_processed
        else:
            # Fallback por defecto
            path_str = "data/processed/Linkoping"
            logger.warning(f"   ⚠️ No se encontró 'base_path' en config. Usando default: {path_str}")

        config_base = Path(path_str)

        # 2. Lógica de recuperación de rutas (Smart Path Finding)
        # Mantenemos tu lógica original para manejar problemas de encoding/rutas en Windows
        if config_base.exists():
            self.base_path = config_base
            logger.info(f"   ✓ Usando path del config: {self.base_path}")
        else:
            # Buscar el directorio directamente usando listdir para evitar problemas de encoding
            # Intentamos buscar en la raiz del proyecto o relativo al cwd
            processed_dir = Path("data/processed")
            if not processed_dir.exists():
                # Si estamos corriendo desde src/train, quizás data está dos niveles arriba
                processed_dir = Path("../../data/processed")

            linkoping_dir = None
            if processed_dir.exists():
                for item in processed_dir.iterdir():
                    # Buscamos carpetas que parezcan ser de Linkoping
                    if item.is_dir() and ('link' in item.name.lower() or 'Link' in item.name):
                        # Verificar que tenga archivos del proyecto para confirmar
                        if list(item.glob('*_graph.pkl')):
                            linkoping_dir = item
                            break

            if linkoping_dir:
                self.base_path = linkoping_dir
                if 'link' in linkoping_dir.name.lower() and linkoping_dir.name != path_str.split('/')[-1]:
                    logger.info(f"   ⚠️ Path ajustado por encoding/ubicación: {self.base_path}")
            else:
                # Fallback final al path del config (aunque no exista, para que el error sea claro después)
                self.base_path = config_base
                logger.warning(f"   ⚠️ No se pudo autodetectar el directorio. Usando config: {self.base_path}")

        # Datos cargados (Inicialización)
        self.graph: Optional[nx.DiGraph] = None
        self.link_data: Optional[pd.DataFrame] = None
        self.od_matrix: Optional[sparse.csr_matrix] = None
        self.routes_data: Optional[Dict] = None
        self.edge_list: Optional[List[Tuple]] = None

        # Parámetros procesados
        self.network_params: Optional[Dict] = None

        logger.info(f"📁 LinkopingDataLoader inicializado")
        logger.info(f"   Base path: {self.base_path}")

        # Acceso seguro a volume_year usando OmegaConf (maneja puntos como diccionarios)
        vol_year = self.config.data.get('volume_year', 'Unknown')
        logger.info(f"   Volume year: {vol_year}")

    def load_all(self) -> Tuple[nx.DiGraph, sparse.csr_matrix, pd.DataFrame, Dict]:
        """
        Carga todos los datos necesarios.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 Cargando datos de Linköping")
        logger.info(f"{'='*80}")

        self.graph = self._load_graph()
        self.od_matrix = self._load_od_matrix()
        self.link_data = self._load_link_data()
        self.routes_data = self._load_routes()

        self._validate_data()

        return self.graph, self.od_matrix, self.link_data, self.routes_data

    def _load_graph(self) -> nx.DiGraph:
        """Carga el grafo de red."""
        # Acceso estilo objeto con Hydra: self.config.data.graph_file
        graph_file = self.config.data.graph_file
        graph_path = self.base_path / graph_file

        # Si el archivo no existe, buscar con glob (encoding issues)
        if not graph_path.exists():
            pattern = graph_file.replace('ö', '*').replace('ä', '*').replace('å', '*')
            matches = list(self.base_path.glob(pattern))
            if matches:
                graph_path = matches[0]
            else:
                # Intentar buscar cualquier archivo _graph.pkl
                matches = list(self.base_path.glob('*_graph.pkl'))
                if matches:
                    graph_path = matches[0]
                    logger.warning(f"   ⚠️ Usando archivo alternativo: {graph_path.name}")

        logger.info(f"   📍 Cargando grafo: {graph_path}")

        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        logger.info(f"      ✓ Nodos: {graph.number_of_nodes()}")
        logger.info(f"      ✓ Enlaces: {graph.number_of_edges()}")

        return graph

    def _load_od_matrix(self) -> sparse.csr_matrix:
        """Carga la matriz OD sparse."""
        od_file = self.config.data.od_matrix_file
        od_path = self.base_path / od_file

        # Si el archivo no existe, buscar con glob (encoding issues)
        if not od_path.exists():
            pattern = od_file.replace('ö', '*').replace('ä', '*').replace('å', '*')
            matches = list(self.base_path.glob(pattern))
            if matches:
                od_path = matches[0]
            else:
                matches = list(self.base_path.glob('*_od_matrix.npz'))
                if matches:
                    od_path = matches[0]
                    logger.warning(f"   ⚠️ Usando archivo alternativo: {od_path.name}")

        logger.info(f"   📍 Cargando matriz OD: {od_path}")

        data = np.load(od_path)
        od_matrix = sparse.csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )

        logger.info(f"      ✓ Shape: {od_matrix.shape}")
        logger.info(f"      ✓ Non-zero elements: {od_matrix.nnz}")
        # logger.info(f"      ✓ Sparsity: {od_matrix.nnz / (od_matrix.shape[0] * od_matrix.shape[1]):.2%}")

        # Contar NaNs
        num_nans = np.isnan(od_matrix.data).sum()
        logger.info(f"      ✓ NaN values: {num_nans}")

        return od_matrix

    def _load_link_data(self) -> pd.DataFrame:
        """Carga datos de enlaces."""
        link_file = self.config.data.link_data_file
        link_path = self.base_path / link_file

        # Si el archivo no existe, buscar con glob (encoding issues)
        if not link_path.exists():
            pattern = link_file.replace('ö', '*').replace('ä', '*').replace('å', '*')
            matches = list(self.base_path.glob(pattern))
            if matches:
                link_path = matches[0]
            else:
                matches = list(self.base_path.glob('*_link_data.parquet'))
                if matches:
                    link_path = matches[0]
                    logger.warning(f"   ⚠️ Usando archivo alternativo: {link_path.name}")

        logger.info(f"   📍 Cargando datos de enlaces: {link_path}")

        link_data = pd.read_parquet(link_path)

        logger.info(f"      ✓ Enlaces: {len(link_data)}")
        # logger.info(f"      ✓ Columnas: {list(link_data.columns)}")

        # Verificar columnas de volumen
        volume_cols = [c for c in link_data.columns if 'Volume' in c]
        logger.info(f"      ✓ Años disponibles: {volume_cols}")

        return link_data

    def _load_routes(self) -> Dict:
        """Carga rutas precalculadas."""
        routes_file = self.config.data.routing_cache_file
        routes_path = self.base_path / routes_file

        # Verificar si existe
        if not routes_path.exists():
            logger.warning(f"   ⚠️ Archivo de rutas no encontrado: {routes_path}")
            logger.warning(f"   ⚠️ Buscando alternativas...")
            # Buscar en routing_cache directamente
            alt_path = self.base_path / 'routing_cache' / 'kshortest_paths.pkl'
            if alt_path.exists():
                routes_path = alt_path

        logger.info(f"   📍 Cargando rutas: {routes_path}")

        with open(routes_path, 'rb') as f:
            routes_data = pickle.load(f)

        logger.info(f"      ✓ Routes shape: {routes_data['routes'].shape}")
        # logger.info(f"      ✓ Max route length: {routes_data['max_route_length']}")
        # logger.info(f"      ✓ Num routes per OD: {routes_data['num_routes']}")

        return routes_data

    def _validate_data(self):
        """Valida consistencia de los datos."""
        logger.info(f"\n   🔍 Validando consistencia de datos...")

        # 1. Número de enlaces
        num_edges_graph = self.graph.number_of_edges()
        num_edges_df = len(self.link_data)

        if num_edges_graph != num_edges_df:
            logger.warning(f"      ⚠️ Mismatch en enlaces: Grafo={num_edges_graph}, DataFrame={num_edges_df}")
        else:
            logger.info(f"      ✓ Enlaces consistentes: {num_edges_graph}")

        # 2. Matriz OD y rutas
        num_od_pairs = self.od_matrix.shape[0] * self.od_matrix.shape[1]
        num_routes = self.routes_data['routes'].shape[0]

        if num_od_pairs != num_routes:
            logger.warning(f"      ⚠️ Mismatch OD: Matriz={num_od_pairs}, Rutas={num_routes}")
        else:
            logger.info(f"      ✓ Pares OD consistentes: {num_od_pairs}")

        # 3. Atributos de enlaces
        sample_edge = list(self.graph.edges())[0]
        edge_attrs = list(self.graph.edges[sample_edge].keys())
        required_attrs = ['capacity', 'free_flow_time', 'length', 'link_type']

        missing_attrs = [attr for attr in required_attrs if attr not in edge_attrs]
        if missing_attrs:
            logger.warning(f"      ⚠️ Atributos faltantes: {missing_attrs}")
        else:
            logger.info(f"      ✓ Todos los atributos requeridos presentes")

    def prepare_observed_flows(self, year: Optional[int] = None,
                                train_split: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepara flujos observados con split train/test.
        """
        if year is None:
            year = self.config.data.volume_year
        if train_split is None:
            train_split = self.config.data.train_split

        volume_col = f'Volume_{year}'

        if volume_col not in self.link_data.columns:
            raise ValueError(f"Columna {volume_col} no encontrada. Disponibles: {self.link_data.columns.tolist()}")

        logger.info(f"\n   📊 Preparando flujos observados (año {year})...")

        # Obtener flujos del año seleccionado
        flows = self.link_data[volume_col].values

        # Identificar enlaces con observaciones válidas
        valid_mask = ~np.isnan(flows)
        # num_valid = valid_mask.sum()

        # Rellenar NaNs con 0 para compatibilidad
        all_flows = np.nan_to_num(flows, nan=0.0)

        # Split train/test solo en enlaces con observaciones
        np.random.seed(self.config.data.random_seed)

        # Índices de enlaces válidos
        valid_indices = np.where(valid_mask)[0]

        # Shuffle y split
        np.random.shuffle(valid_indices)
        split_idx = int(len(valid_indices) * train_split)

        train_indices = valid_indices[:split_idx]
        test_indices = valid_indices[split_idx:]

        # Crear máscaras
        train_mask = np.zeros(len(flows), dtype=np.float32)
        test_mask = np.zeros(len(flows), dtype=np.float32)

        train_mask[train_indices] = 1.0
        test_mask[test_indices] = 1.0

        logger.info(f"      ✓ Train: {len(train_indices)} enlaces")
        logger.info(f"      ✓ Test: {len(test_indices)} enlaces")

        return all_flows, train_mask, test_mask

    def prepare_od_demand_vector(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convierte matriz OD sparse a vector denso.
        """
        logger.info(f"\n   📊 Preparando vector de demandas OD...")

        # Convertir a denso
        od_dense = self.od_matrix.toarray()

        # Aplanar a vector
        od_vector = od_dense.flatten()

        # Contar válidos vs NaNs
        num_valid = (~np.isnan(od_vector)).sum()
        # num_nan = np.isnan(od_vector).sum()

        logger.info(f"      ✓ Demandas conocidas: {num_valid}/{len(od_vector)}")

        # Crear máscara de OD conocidas (no NaN)
        od_mask = (~np.isnan(od_vector)).astype(np.float32)

        # Reemplazar NaNs con 0
        od_vector = np.nan_to_num(od_vector, nan=0.0)

        return od_vector, od_mask

    def prepare_network_parameters(self) -> Dict:
        """
        Extrae y prepara todos los parámetros de red para CyclicODModel.
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"⚙️ Preparando parámetros de red")
        logger.info(f"{'='*80}")

        # Crear lista ordenada de enlaces
        self.edge_list = list(self.graph.edges())
        num_links = len(self.edge_list)

        # 1. Extraer atributos de enlaces
        logger.info(f"   📊 Extrayendo atributos de enlaces...")

        t0 = np.zeros(num_links, dtype=np.float32)
        capacity = np.zeros(num_links, dtype=np.float32)
        link_type = np.zeros(num_links, dtype=np.int32)

        for i, (u, v) in enumerate(self.edge_list):
            edge_data = self.graph[u][v]
            t0[i] = edge_data.get('free_flow_time', 1.0)
            capacity[i] = edge_data.get('capacity', 1000.0)
            link_type[i] = edge_data.get('link_type', 0)

        # Map link_type IDs to indices 0 to num_link_groups-1
        unique_link_types = np.unique(link_type)
        link_type_to_index = {lt: i for i, lt in enumerate(unique_link_types)}
        link_group = np.array([link_type_to_index[lt] for lt in link_type])

        # 2. Preparar máscaras de rutas
        logger.info(f"   📊 Preparando máscaras de rutas...")

        route_masks, od_pair_indices = self._build_route_masks()
        num_od_pairs = route_masks.shape[0]

        # 3. Grupos de enlaces (link types)
        num_link_groups = len(unique_link_types)

        # 4. Consolidar parámetros
        self.network_params = {
            'num_links': num_links,
            'num_od_pairs': num_od_pairs,
            't0': torch.FloatTensor(t0),
            'capacity': torch.FloatTensor(capacity),
            'route_masks': torch.FloatTensor(route_masks),
            'od_pair_indices': torch.LongTensor(od_pair_indices),
            'num_link_groups': num_link_groups,
            'link_group': torch.LongTensor(link_group)
        }

        logger.info(f"\n   ✅ Parámetros de red preparados")
        logger.info(f"      - Enlaces: {num_links}")
        logger.info(f"      - Pares OD: {num_od_pairs}")

        return self.network_params

    def _build_route_masks(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Construye máscaras de rutas desde las rutas precalculadas.
        """
        routes = self.routes_data['routes']  # [num_od_pairs, k_paths, max_route_length]
        od_pairs = self.routes_data['od_pairs']  # [num_od_pairs, 2]

        num_od_pairs = routes.shape[0]
        k_paths = routes.shape[1]
        num_links = len(self.edge_list)

        # Crear mapeo de aristas a índices
        edge_to_idx = {edge: i for i, edge in enumerate(self.edge_list)}

        # Crear lista ordenada de nodos del grafo (para mapear índices a IDs)
        node_list = sorted(list(self.graph.nodes()))

        # Inicializar máscaras
        route_masks = np.zeros((num_od_pairs, k_paths, num_links), dtype=np.float32)

        logger.info(f"      Construyendo máscaras de rutas...")

        edges_mapped = 0
        edges_not_found = 0

        # Para cada par OD y cada ruta
        for od_idx in range(num_od_pairs):
            for k in range(k_paths):
                route = routes[od_idx, k]
                route = route[route > 0] # Filtrar padding

                if len(route) < 2:
                    continue

                for i in range(len(route) - 1):
                    node_idx_from = int(route[i])
                    node_idx_to = int(route[i + 1])

                    if node_idx_from >= len(node_list) or node_idx_to >= len(node_list):
                        continue

                    node_from = node_list[node_idx_from]
                    node_to = node_list[node_idx_to]
                    edge = (node_from, node_to)

                    if edge in edge_to_idx:
                        link_idx = edge_to_idx[edge]
                        route_masks[od_idx, k, link_idx] = 1.0
                        edges_mapped += 1
                    else:
                        edges_not_found += 1

        # OD pair indices
        if isinstance(od_pairs, list):
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            od_pair_indices = []
            for origin_id, dest_id in od_pairs:
                if origin_id in node_to_idx and dest_id in node_to_idx:
                    od_pair_indices.append([node_to_idx[origin_id], node_to_idx[dest_id]])
                else:
                    od_pair_indices.append([-1, -1])
            od_pair_indices = np.array(od_pair_indices, dtype=np.int64)
        else:
            od_pair_indices = od_pairs.astype(np.int64)

        return route_masks, od_pair_indices

    def create_sampling_masks(self,
                             train_flow_mask: np.ndarray,
                             od_mask: np.ndarray,
                             flow_rate: Optional[float] = None,
                             od_rate: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Crea máscaras de muestreo adicionales usando el motor de sampling.
        """
        # Acceso vía Hydra
        if flow_rate is None:
            flow_rate = self.config.sampling.flow_rate
        if od_rate is None:
            od_rate = self.config.sampling.od_rate

        strategy = self.config.sampling.get('strategy', 'random')
        sampling_basis = self.config.sampling.get('sampling_basis', 'link_wise_based')
        random_seed = self.config.data.random_seed

        # Import local para evitar ciclos
        from src.components.sampling.sampling import create_partial_data_masks

        sampled_flow_mask, sampled_od_mask = create_partial_data_masks(
            train_flow_mask=train_flow_mask,
            od_mask=od_mask,
            flow_rate=flow_rate,
            od_rate=od_rate,
            random_seed=random_seed,
            strategy=strategy,
            sampling_basis=sampling_basis,
            graph=self.graph
        )

        return sampled_flow_mask, sampled_od_mask

    def get_summary(self) -> Dict:
        """Retorna resumen de datos cargados."""
        return {
            'network_name': self.config.data.network_name,
            'num_nodes': self.graph.number_of_nodes() if self.graph else 0,
            'num_links': self.graph.number_of_edges() if self.graph else 0,
            'num_od_pairs': self.od_matrix.shape[0] * self.od_matrix.shape[1] if self.od_matrix is not None else 0,
            'volume_year': self.config.data.volume_year,
            'k_paths': self.config.network.k_paths,
            'cost_function': self.config.network.cost_function
        }