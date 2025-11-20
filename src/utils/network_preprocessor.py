"""
Network Preprocessor - Extractor de parámetros de red para Cyclic Model.

Este módulo extrae todos los parámetros necesarios del grafo y matriz OD
para inicializar el modelo CyclicODModel.

Funciones principales:
1. Extracción de atributos de enlaces (t0, capacity, link_group)
2. Generación de route_masks usando k-shortest paths
3. Asignación de rutas a pares OD (od_pair_indices)
4. Agrupación de enlaces por tipo

Uso:
    from utils.network_preprocessor import NetworkPreprocessor

    preprocessor = NetworkPreprocessor(
        graph=graph,
        od_matrix=od_matrix,
        network_name='SiouxFalls',
        k_paths=10
    )

    params = preprocessor.extract_all_parameters()

    model = CyclicODModel(**params)
"""
import sys
from pathlib import Path
import torch
import numpy as np
import networkx as nx
from scipy import sparse
from typing import Dict, Tuple, List
import pickle
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.routing_cache import k_shortest_paths


class NetworkPreprocessor:
    """
    Extractor de parámetros de red para el modelo Cyclic.

    Attributes:
        graph: Grafo NetworkX de la red vial
        od_matrix: Matriz OD sparse
        network_name: Nombre de la red (para cache de rutas)
        k_paths: Número de rutas alternativas por par OD
        weight_attr: Atributo de peso para rutas (default: 'free_flow_time')
    """

    def __init__(self,
                 graph: nx.DiGraph,
                 od_matrix: sparse.spmatrix,
                 network_name: str = 'SiouxFalls',
                 k_paths: int = 10,
                 weight_attr: str = 'free_flow_time'):
        """
        Inicializa el preprocessor.

        Args:
            graph: Grafo NetworkX con atributos de enlaces
            od_matrix: Matriz OD sparse (n_zones x n_zones)
            network_name: Nombre de la red para cache
            k_paths: Número de rutas más cortas a generar por par OD
            weight_attr: Atributo para calcular rutas más cortas
        """
        self.graph = graph
        self.od_matrix = od_matrix
        self.network_name = network_name
        self.k_paths = k_paths
        self.weight_attr = weight_attr

        # Cache de rutas
        project_root = Path(__file__).resolve().parents[2]
        cache_dir = project_root / 'data' / 'processed' / 'routing_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_path = cache_dir / f'kshortest_paths_{network_name}.pkl'

        # Crear índice de enlaces
        self.edge_list = list(graph.edges())
        self.edge_to_idx = {edge: idx for idx, edge in enumerate(self.edge_list)}

        print(f"\n{'='*80}")
        print(f"🔧 NetworkPreprocessor inicializado")
        print(f"{'='*80}")
        print(f"   Red: {network_name}")
        print(f"   Enlaces: {len(self.edge_list)}")
        print(f"   Pares OD: {od_matrix.nnz}")
        print(f"   K rutas por OD: {k_paths}")
        print(f"   Peso de rutas: {weight_attr}")

    def extract_t0(self) -> torch.Tensor:
        """
        Extrae tiempos de flujo libre por enlace.

        Returns:
            Tensor [num_links] con free_flow_time de cada enlace
        """
        print("\n📊 Extrayendo t0 (free_flow_time)...")
        t0_list = []

        for u, v in self.edge_list:
            t0 = self.graph[u][v].get('free_flow_time',
                  self.graph[u][v].get('length', 1.0))

            # Validación
            if t0 <= 0:
                t0 = 1.0

            t0_list.append(t0)

        t0 = torch.tensor(t0_list, dtype=torch.float32)

        print(f"   ✓ t0 extraído: shape={t0.shape}")
        print(f"   ✓ Rango: [{t0.min():.2f}, {t0.max():.2f}]")
        print(f"   ✓ Media: {t0.mean():.2f}")

        return t0

    def extract_capacity(self) -> torch.Tensor:
        """
        Extrae capacidades por enlace.

        Returns:
            Tensor [num_links] con capacity de cada enlace
        """
        print("\n📊 Extrayendo capacity...")
        capacity_list = []

        for u, v in self.edge_list:
            capacity = self.graph[u][v].get('capacity', 1000.0)

            # Validación: capacidad mínima
            if capacity <= 0:
                capacity = 100.0

            capacity_list.append(capacity)

        capacity = torch.tensor(capacity_list, dtype=torch.float32)

        print(f"   ✓ Capacity extraído: shape={capacity.shape}")
        print(f"   ✓ Rango: [{capacity.min():.2f}, {capacity.max():.2f}]")
        print(f"   ✓ Media: {capacity.mean():.2f}")

        return capacity

    def extract_link_groups(self) -> Tuple[torch.Tensor, int]:
        """
        Agrupa enlaces por tipo (link_type).

        Returns:
            Tuple con:
                - link_group: Tensor [num_links] con grupo de cada enlace
                - num_link_groups: Número total de grupos
        """
        print("\n📊 Agrupando enlaces por tipo...")

        link_types = []
        for u, v in self.edge_list:
            link_type = self.graph[u][v].get('link_type', 0)
            link_types.append(link_type)

        link_types_array = np.array(link_types)
        unique_types = np.unique(link_types_array)
        num_link_groups = len(unique_types)

        # Crear mapeo de tipo original a índice 0, 1, 2, ...
        type_to_group = {original_type: idx for idx, original_type in enumerate(unique_types)}
        link_group_list = [type_to_group[lt] for lt in link_types]

        link_group = torch.tensor(link_group_list, dtype=torch.long)

        print(f"   ✓ Grupos creados: {num_link_groups}")
        print(f"   ✓ Tipos únicos: {unique_types}")
        print(f"   ✓ Distribución por grupo:")
        for group_idx, original_type in enumerate(unique_types):
            count = (link_group == group_idx).sum().item()
            print(f"      - Grupo {group_idx} (tipo {original_type}): {count} enlaces")

        return link_group, num_link_groups

    def _load_route_cache(self) -> Dict:
        """Carga cache de rutas si existe."""
        if self.cache_path.exists():
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"   ✓ Cache de rutas cargado: {len(cache)} pares OD")
                return cache
            except Exception as e:
                print(f"   ⚠ Error cargando cache: {e}")
                return {}
        return {}

    def _save_route_cache(self, cache: Dict):
        """Guarda cache de rutas."""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"   ✓ Cache guardado: {self.cache_path}")
        except Exception as e:
            print(f"   ⚠ Error guardando cache: {e}")

    def generate_routes_and_masks(self) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Genera rutas usando k-shortest paths y crea route_masks.

        Returns:
            Tuple con:
                - route_masks: Tensor [num_routes, num_links] binario
                - od_pair_indices: Tensor [num_routes] con índice de par OD por ruta
                - num_od_pairs: Número de pares OD con demanda
        """
        print("\n🛣️  Generando rutas con k-shortest paths...")

        # Cargar cache si existe
        route_cache = self._load_route_cache()

        # Obtener pares OD con demanda no-cero
        od_coo = self.od_matrix.tocoo()
        od_pairs = list(zip(od_coo.row, od_coo.col))
        num_od_pairs = len(od_pairs)

        print(f"   ✓ Pares OD con demanda: {num_od_pairs}")
        print(f"   ✓ Calculando {self.k_paths} rutas por par OD...")

        all_routes = []
        all_od_indices = []
        cache_hits = 0
        cache_misses = 0

        # Generar rutas para cada par OD
        for od_idx, (origin, dest) in enumerate(tqdm(od_pairs, desc="   Generando rutas")):
            # Ajustar índices (OD usa índices 0-based, nodos pueden ser 1-based)
            # Asumimos que od_matrix usa índices que corresponden a nodos
            # Si la red tiene nodos 1-N, origin/dest son 0-based, necesitamos +1
            # Verificar primero si los nodos existen

            # En la mayoría de datasets TNTP, los nodos de zona van de 1 a n_zones
            origin_node = origin + 1  # Convertir de índice 0-based a nodo 1-based
            dest_node = dest + 1

            # Verificar que los nodos existen
            if origin_node not in self.graph or dest_node not in self.graph:
                # Si no existen, probar sin ajuste
                origin_node = origin
                dest_node = dest

                if origin_node not in self.graph or dest_node not in self.graph:
                    print(f"   ⚠ Par OD ({origin}, {dest}) tiene nodos no encontrados, omitiendo")
                    continue

            cache_key = (origin_node, dest_node)

            # Intentar obtener del cache
            if cache_key in route_cache:
                paths = route_cache[cache_key]
                cache_hits += 1
            else:
                # Calcular rutas
                paths = k_shortest_paths(
                    self.graph,
                    origin_node,
                    dest_node,
                    k=self.k_paths,
                    weight=self.weight_attr
                )
                route_cache[cache_key] = paths
                cache_misses += 1

            # Convertir rutas a máscaras
            for path in paths:
                # Crear máscara binaria
                mask = np.zeros(len(self.edge_list), dtype=np.float32)

                # Marcar enlaces usados en la ruta
                for i in range(len(path) - 1):
                    edge = (path[i], path[i+1])
                    if edge in self.edge_to_idx:
                        mask[self.edge_to_idx[edge]] = 1.0

                all_routes.append(mask)
                all_od_indices.append(od_idx)

        # Guardar cache actualizado
        if cache_misses > 0:
            print(f"\n   💾 Guardando cache ({cache_misses} nuevos pares calculados)...")
            self._save_route_cache(route_cache)

        print(f"\n   ✓ Rutas generadas: {len(all_routes)}")
        print(f"   ✓ Cache hits: {cache_hits}, misses: {cache_misses}")
        print(f"   ✓ Promedio de rutas por OD: {len(all_routes)/num_od_pairs:.2f}")

        # Convertir a tensors
        route_masks = torch.tensor(np.array(all_routes), dtype=torch.float32)
        od_pair_indices = torch.tensor(all_od_indices, dtype=torch.long)

        print(f"   ✓ route_masks shape: {route_masks.shape}")
        print(f"   ✓ od_pair_indices shape: {od_pair_indices.shape}")

        return route_masks, od_pair_indices, num_od_pairs

    def extract_all_parameters(self) -> Dict:
        """
        Extrae todos los parámetros necesarios para CyclicODModel.

        Returns:
            Dict con todos los parámetros:
                - num_links: int
                - num_od_pairs: int
                - t0: Tensor [num_links]
                - capacity: Tensor [num_links]
                - route_masks: Tensor [num_routes, num_links]
                - od_pair_indices: Tensor [num_routes]
                - num_link_groups: int
                - link_group: Tensor [num_links]
        """
        print(f"\n{'='*80}")
        print("🚀 Extrayendo todos los parámetros de red")
        print(f"{'='*80}")

        # 1. Atributos de enlaces
        t0 = self.extract_t0()
        capacity = self.extract_capacity()
        link_group, num_link_groups = self.extract_link_groups()

        # 2. Rutas y máscaras
        route_masks, od_pair_indices, num_od_pairs = self.generate_routes_and_masks()

        # 3. Resumen
        num_links = len(self.edge_list)

        print(f"\n{'='*80}")
        print("✅ Parámetros extraídos exitosamente")
        print(f"{'='*80}")
        print(f"   num_links: {num_links}")
        print(f"   num_od_pairs: {num_od_pairs}")
        print(f"   num_routes: {len(route_masks)}")
        print(f"   num_link_groups: {num_link_groups}")
        print(f"\n   Shapes:")
        print(f"   - t0: {t0.shape}")
        print(f"   - capacity: {capacity.shape}")
        print(f"   - route_masks: {route_masks.shape}")
        print(f"   - od_pair_indices: {od_pair_indices.shape}")
        print(f"   - link_group: {link_group.shape}")

        return {
            'num_links': num_links,
            'num_od_pairs': num_od_pairs,
            't0': t0,
            'capacity': capacity,
            'route_masks': route_masks,
            'od_pair_indices': od_pair_indices,
            'num_link_groups': num_link_groups,
            'link_group': link_group
        }


def prepare_od_demand_vector(od_matrix: sparse.spmatrix) -> torch.Tensor:
    """
    Convierte matriz OD sparse a vector denso para el modelo.

    Args:
        od_matrix: Matriz OD sparse [n_zones, n_zones]

    Returns:
        Tensor [num_od_pairs] con demandas de pares OD no-cero
    """
    od_coo = od_matrix.tocoo()
    demands = od_coo.data

    return torch.tensor(demands, dtype=torch.float32)


def prepare_observed_flows(graph: nx.DiGraph, edge_list: List[Tuple[int, int]]) -> torch.Tensor:
    """
    Extrae flujos observados del grafo (atributo 'volume').

    Args:
        graph: Grafo con atributo 'volume' en enlaces
        edge_list: Lista de enlaces en orden

    Returns:
        Tensor [num_links] con flujos observados
    """
    flows = []

    for u, v in edge_list:
        volume = graph[u][v].get('volume', 0.0)
        flows.append(volume)

    return torch.tensor(flows, dtype=torch.float32)

