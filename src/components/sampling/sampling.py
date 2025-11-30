"""
Generalized Sampling Module for Traffic Assignment Models

This module provides sampling utilities for creating partial data masks
for flows and OD demands. It supports advanced spatial sampling (LHS)
and topological redundancy handling using NetworkX.

Author: Traffic Assignment System
Date: November 2025
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Tuple, Optional, List, Union
import logging
from scipy.stats import qmc
from scipy.spatial import cKDTree
import yaml
from pathlib import Path
import sys

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.components.models.Cyclic_Model.cyclic_model_data_ingestion import LinkopingDataLoader

logger = logging.getLogger(__name__)

# =============================================================================
# 1. Link Grouping Logic (Handling Redundancy)
# =============================================================================

def _group_consecutive_links(
    valid_indices: np.ndarray,
    internal_df: pd.DataFrame
) -> List[List[int]]:
    """
    Agrupa índices de links que representan una única 'lectura' de tráfico.

    Lógica:
    Se considera que dos links pertenecen al mismo grupo si:
    1. Son consecutivos (Target del Link A == Source del Link B).
    2. Tienen el mismo valor de flujo (Volume_YYYY).

    Args:
        valid_indices: Array de índices activos (mask > 0).
        internal_df: DataFrame interno con columnas ['source', 'target', 'flow', 'original_index'].

    Returns:
        Lista de listas de índices agrupados.
    """
    # Filtramos solo los links válidos para el análisis
    subset = internal_df.iloc[valid_indices].copy()

    # Si no hay datos de flujo, no podemos agrupar fiablemente
    if subset['flow'].isnull().all():
        logger.warning("   WARNING: No se detectaron datos de flujo (Volume_YYYY) para agrupar. Usando link-wise.")
        return [[idx] for idx in valid_indices]

    # Diccionario de adyacencia optimizado para búsqueda rápida:
    # Clave: Nodo Source -> Valor: (Nodo Target, Flow, Index)
    # Nota: Asumimos que en una red válida, un nodo source solo sale a un link en el subset lineal.
    # Si hubiera bifurcaciones, la lógica de "mismo flujo" rompería la cadena naturalmente.
    adj_map = {}
    for _, row in subset.iterrows():
        adj_map[row['source']] = (row['target'], row['flow'], int(row['original_index']))

    visited = set()
    groups = []

    # Ordenamos los índices para determinismo
    sorted_indices = np.sort(valid_indices)

    for idx in sorted_indices:
        # Obtenemos la fila correspondiente del subset original usando el índice global
        # (Es más rápido buscar en el df original si está indexado, pero aquí usamos la lógica iterativa)
        row = internal_df.iloc[idx]
        source_node = row['source']

        # Si este link ya fue procesado como parte de una cadena, saltar
        if idx in visited:
            continue

        # Iniciar nueva cadena
        current_chain = [idx]
        visited.add(idx)

        # Rastrear hacia adelante (Downstream)
        # Buscamos si el target de este link es el source de otro con el MISMO flujo
        curr_target = row['target']
        curr_flow = row['flow']

        while True:
            if curr_target in adj_map:
                next_target, next_flow, next_idx = adj_map[curr_target]

                # CRITERIO CLAVE: Conectado Y Flujo Idéntico
                # Usamos np.isclose para evitar errores de punto flotante
                if next_idx not in visited and np.isclose(curr_flow, next_flow):
                    current_chain.append(next_idx)
                    visited.add(next_idx)
                    curr_target = next_target # Avanzar pivote
                else:
                    break # Se rompe la cadena (diferente flujo o ya visitado)
            else:
                break # Fin de la línea topológica

        groups.append(current_chain)

    return groups


def _get_observation_groups(
    mask: np.ndarray,
    basis: str,
    internal_df: pd.DataFrame
) -> List[List[int]]:
    """
    Determina las unidades de muestreo.
    """
    valid_indices = np.where(mask > 0)[0]

    if basis == "link_wise_based":
        return [[idx] for idx in valid_indices]

    elif basis == "traffic_counts_based":
        return _group_consecutive_links(valid_indices, internal_df)

    else:
        raise ValueError(f"Basis '{basis}' no reconocido.")


# =============================================================================
# 2. Sampling Strategies
# =============================================================================

def _strategy_random(
    groups: List[List[int]],
    n_samples: int,
    seed: int
) -> List[List[int]]:
    """Estrategia Aleatoria Simple sobre los grupos."""
    rng = np.random.default_rng(seed)
    selected_group_indices = rng.choice(len(groups), size=n_samples, replace=False)
    return [groups[i] for i in selected_group_indices]


def _strategy_spatial_lhs(
    groups: List[List[int]],
    n_samples: int,
    seed: int,
    internal_df: pd.DataFrame
) -> List[List[int]]:
    """
    Estrategia LHS (Latin Hypercube Sampling) Espacial + Tipo.
    """
    # 1. Calcular features representativos para cada GRUPO
    group_features = []

    for g_indices in groups:
        # Extraer sub-dataframe de los links en este grupo
        sub = internal_df.iloc[g_indices]

        # Promedio de coordenadas (centroide del grupo)
        avg_x = (sub['start_x'] + sub['end_x']).mean() / 2
        avg_y = (sub['start_y'] + sub['end_y']).mean() / 2

        # Tipo de link (Usamos la moda o el primero)
        l_type = sub['link_type'].iloc[0]

        group_features.append([avg_x, avg_y, l_type])

    data_matrix = np.array(group_features)

    # 2. Normalización Min-Max [0, 1]
    min_vals = data_matrix.min(axis=0)
    max_vals = data_matrix.max(axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1.0 # Evitar div/0

    norm_matrix = (data_matrix - min_vals) / range_vals

    # 3. KDTree y LHS
    tree = cKDTree(norm_matrix)
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    ideal_points = sampler.random(n=n_samples)

    # 4. Matching (Vecino más cercano)
    selected_indices_set = set()
    dists, neighbors = tree.query(ideal_points, k=5) # Buscamos los 5 más cercanos

    final_selection_indices = []

    for i in range(n_samples):
        found = False
        # Iterar sobre los k vecinos candidatos para este punto ideal
        # neighbors[i] es una lista de índices de 'groups'
        for candidate_idx in neighbors[i]:
            if candidate_idx not in selected_indices_set:
                selected_indices_set.add(candidate_idx)
                final_selection_indices.append(candidate_idx)
                found = True
                break

        # Si todos los candidatos estaban cogidos, podríamos forzar selección
        # o simplemente no añadir (reduciendo la muestra ligeramente).
        # Aquí elegimos ser estrictos con 'replace=False'.

    return [groups[i] for i in final_selection_indices]


# =============================================================================
# 3. Modular Core Functions
# =============================================================================

def _sample_flows(
    train_flow_mask: np.ndarray,
    rate: float,
    basis: str,
    strategy: str,
    random_seed: int,
    internal_df: pd.DataFrame
) -> np.ndarray:
    """Orquesta el muestreo de flujos."""

    # 1. Definir universo (Grupos)
    groups = _get_observation_groups(train_flow_mask, basis, internal_df)
    num_groups = len(groups)
    num_samples = int(num_groups * rate)

    if num_samples == 0:
        logger.warning("   WARNING: Flow rate resulted in 0 samples.")
        return np.zeros_like(train_flow_mask)

    logger.info(f"      Sampling Logic: {basis} | Strategy: {strategy}")
    logger.info(f"      Population: {num_groups} groups -> Target Sample: {num_samples}")

    # 2. Seleccionar
    if strategy == "random":
        selected_groups = _strategy_random(groups, num_samples, random_seed)
    elif strategy == "spatial_lhs":
        selected_groups = _strategy_spatial_lhs(groups, num_samples, random_seed, internal_df)
    else:
        logger.warning(f"Strategy '{strategy}' not found. Defaulting to random.")
        selected_groups = _strategy_random(groups, num_samples, random_seed)

    # 3. Reconstruir máscara
    sampled_mask = np.zeros_like(train_flow_mask)
    for group in selected_groups:
        for link_idx in group:
            sampled_mask[link_idx] = 1.0

    return sampled_mask


def _sample_od(od_mask: np.ndarray, rate: float, random_seed: int) -> np.ndarray:
    """Orquesta el muestreo de OD (Aleatorio simple por ahora)."""
    known_indices = np.where(od_mask > 0)[0]
    num_known = len(known_indices)
    num_sampled = int(num_known * rate)

    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(known_indices, size=num_sampled, replace=False)

    sampled_mask = np.zeros_like(od_mask)
    sampled_mask[sampled_indices] = 1.0

    return sampled_mask


# =============================================================================
# 4. Helper: Graph to Internal Data
# =============================================================================

def _extract_graph_data(graph: nx.DiGraph, volume_year: Optional[int]) -> pd.DataFrame:
    """
    Convierte los datos relevantes del grafo a un DataFrame ligero para procesamiento.
    Preserva estrictamente el orden de list(graph.edges()).
    """
    data = []
    vol_key = f"Volume_{volume_year}" if volume_year else None

    # Iteramos sobre edges en el orden estándar de NetworkX
    for i, (u, v, attrs) in enumerate(graph.edges(data=True)):

        # Coordenadas de nodos (default a 0.0 si no existen)
        # Usamos graph.nodes[node] para acceder a atributos del nodo
        u_node = graph.nodes[u]
        v_node = graph.nodes[v]

        row = {
            'original_index': i,
            'source': u,
            'target': v,
            'start_x': u_node.get('x', 0.0),
            'start_y': u_node.get('y', 0.0),
            'end_x': v_node.get('x', 0.0),
            'end_y': v_node.get('y', 0.0),
            'link_type': attrs.get('link_type', 0), # Default 0 si no hay tipo
            'flow': attrs.get(vol_key, np.nan) if vol_key else np.nan
        }
        data.append(row)

    return pd.DataFrame(data)


# =============================================================================
# 5. Public API
# =============================================================================

def create_partial_data_masks(
    train_flow_mask: np.ndarray,
    od_mask: np.ndarray,
    flow_rate: float,
    od_rate: float,
    graph: Union[nx.Graph, nx.DiGraph],
    volume_year: Optional[int] = None,
    random_seed: int = 42,
    strategy: str = "random",
    sampling_basis: str = "link_wise_based"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Genera máscaras de muestreo parcial para flujos y ODs.

    Args:
        train_flow_mask: Máscara binaria alineada con list(graph.edges()).
        od_mask: Máscara binaria de ODs conocidos.
        flow_rate: Ratio de flujos a mantener (0.0 - 1.0).
        od_rate: Ratio de ODs a mantener (0.0 - 1.0).
        graph: Grafo NetworkX con atributos topológicos y de flujo.
               - Nodos deben tener 'x', 'y'.
               - Links deben tener 'link_type' y 'Volume_YYYY'.
        volume_year: Año para buscar la columna de flujo (ej. 2022 -> Volume_2022).
                     Necesario si sampling_basis='traffic_counts_based'.
        strategy: 'random' o 'spatial_lhs'.
        sampling_basis:
            - 'link_wise_based': Muestrea links individuales.
            - 'traffic_counts_based': Agrupa links consecutivos con mismo flujo.
    """
    logger.info(f"Creando máscaras de muestreo (Year: {volume_year})")

    # 1. Extraer datos del grafo a estructura tabular interna
    # Esto desacopla la lógica compleja de sampling de la estructura de grafo
    internal_df = _extract_graph_data(graph, volume_year)

    # Validación rápida
    if len(internal_df) != len(train_flow_mask):
        raise ValueError(
            f"Mismatch: El grafo tiene {len(internal_df)} links pero "
            f"train_flow_mask tiene longitud {len(train_flow_mask)}."
        )

    # 2. Sampling de Flujos (Modular)
    sampled_flow_mask = _sample_flows(
        train_flow_mask=train_flow_mask,
        rate=flow_rate,
        basis=sampling_basis,
        strategy=strategy,
        random_seed=random_seed,
        internal_df=internal_df
    )

    # 3. Sampling de OD (Modular)
    sampled_od_mask = _sample_od(
        od_mask=od_mask,
        rate=od_rate,
        random_seed=random_seed
    )

    return sampled_flow_mask, sampled_od_mask


# =============================================================================
# 6. Standalone Execution
# =============================================================================

def main():
    """Ejecuta el muestreo de datos parciales usando configuración de Linköping.yaml."""

    # Configurar logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ruta del proyecto (ajustar si es necesario)
    global project_root
    project_root = Path(__file__).resolve().parent.parent

    # Cargar configuración
    config_path = Path('C:/Users/jecla/Documents/Barcelona_GNN/configs/Linköping.yaml')
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print("="*80)
    print("EJECUCIÓN STANDALONE: Sampling Module")
    print("="*80)
    print(f"   Config: {config_path}")
    print(f"   Network: {config['data']['network_name']}")
    print(f"   Volume Year: {config['data']['volume_year']}")
    print(f"   Flow Rate: {config['sampling']['flow_rate']}")
    print(f"   OD Rate: {config['sampling']['od_rate']}")
    print(f"   Strategy: {config['sampling']['strategy']}")
    print(f"   Sampling Basis: {config['sampling']['sampling_basis']}")

    # Cargar datos usando LinkopingDataLoader
    print("\n📁 Cargando datos...")
    loader = LinkopingDataLoader(str(config_path))
    graph, od_matrix, link_data, routes_data = loader.load_all()

    # Preparar máscaras base
    all_flows, train_flow_mask, test_flow_mask = loader.prepare_observed_flows()
    od_vector, od_mask = loader.prepare_od_demand_vector()

    print(f"   ✓ Grafo cargado: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} enlaces")
    print(f"   ✓ Flujos de entrenamiento: {int(train_flow_mask.sum())}/{len(train_flow_mask)}")
    print(f"   ✓ ODs conocidos: {int(od_mask.sum())}/{len(od_mask)}")

    # Ejecutar muestreo
    print("\n📊 Ejecutando muestreo...")
    sampled_flow_mask, sampled_od_mask = create_partial_data_masks(
        train_flow_mask=train_flow_mask,
        od_mask=od_mask,
        flow_rate=config['sampling']['flow_rate'],
        od_rate=config['sampling']['od_rate'],
        graph=graph,
        volume_year=config['data']['volume_year'],
        random_seed=config['data']['random_seed'],
        strategy=config['sampling']['strategy'],
        sampling_basis=config['sampling']['sampling_basis']
    )

    # Estadísticas
    n_flows_sampled = int(sampled_flow_mask.sum())
    n_flows_train = int(train_flow_mask.sum())
    n_od_sampled = int(sampled_od_mask.sum())
    n_od_known = int(od_mask.sum())

    print("📈 Resultados del muestreo:")
    print(f"      Flujos: {n_flows_sampled}/{n_flows_train} ({n_flows_sampled/n_flows_train*100:.1f}%)")
    print(f"      OD: {n_od_sampled}/{n_od_known} ({n_od_sampled/n_od_known*100:.1f}%)")

    # Guardar máscaras
    output_dir = project_root / 'outputs' / 'masks'
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / 'sampled_flow_mask.npy', sampled_flow_mask)
    np.save(output_dir / 'sampled_od_mask.npy', sampled_od_mask)

    print(f"\n💾 Máscaras guardadas en: {output_dir}")
    print(f"   - sampled_flow_mask.npy")
    print(f"   - sampled_od_mask.npy")

    print("\n✅ Muestreo completado exitosamente!")


if __name__ == "__main__":
    main()
