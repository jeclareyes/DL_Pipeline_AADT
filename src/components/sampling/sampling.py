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


logger = logging.getLogger(__name__)


# =============================================================================
# 1. Data Extraction & Grouping Helpers
# =============================================================================

def extract_graph_data(graph: nx.DiGraph, volume_year: Optional[int]) -> pd.DataFrame:
    """
    Convierte los datos relevantes del grafo a un DataFrame interno.
    """
    data = []
    vol_key = f"Volume_{volume_year}" if volume_year else None

    for i, (u, v, attrs) in enumerate(graph.edges(data=True)):
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
            'link_type': attrs.get('link_type', 0),
            'flow': attrs.get(vol_key, np.nan) if vol_key else np.nan
        }
        data.append(row)

    return pd.DataFrame(data)


def group_consecutive_links(valid_indices: np.ndarray, internal_df: pd.DataFrame) -> List[List[int]]:
    """
    Agrupa índices de links topológicamente consecutivos con el mismo flujo.
    """
    subset = internal_df.iloc[valid_indices].copy()

    if subset['flow'].isnull().all():
        logger.warning("   WARNING: No flow data for grouping. Fallback to link-wise.")
        return [[idx] for idx in valid_indices]

    # Mapa de adyacencia optimizado
    adj_map = {}
    for _, row in subset.iterrows():
        adj_map[row['source']] = (row['target'], row['flow'], int(row['original_index']))

    visited = set()
    groups = []
    sorted_indices = np.sort(valid_indices)

    for idx in sorted_indices:
        if idx in visited:
            continue

        row = internal_df.iloc[idx]
        current_chain = [idx]
        visited.add(idx)

        curr_target = row['target']
        curr_flow = row['flow']

        # Rastrear downstream
        while True:
            if curr_target in adj_map:
                next_target, next_flow, next_idx = adj_map[curr_target]
                if next_idx not in visited and np.isclose(curr_flow, next_flow):
                    current_chain.append(next_idx)
                    visited.add(next_idx)
                    curr_target = next_target
                else:
                    break
            else:
                break
        groups.append(current_chain)

    return groups


def get_observation_groups(mask: np.ndarray, basis: str, internal_df: pd.DataFrame) -> List[List[int]]:
    """Define las unidades de muestreo (Links individuales o cadenas)."""
    valid_indices = np.where(mask > 0)[0]

    if basis == "link_wise_based":
        return [[idx] for idx in valid_indices]
    elif basis == "traffic_counts_based":
        return group_consecutive_links(valid_indices, internal_df)
    else:
        raise ValueError(f"Basis '{basis}' no reconocido.")


# =============================================================================
# 2. Mask Construction Helpers
# =============================================================================

def build_mask_from_groups(base_mask: np.ndarray, selected_groups: List[List[int]]) -> np.ndarray:
    """Reconstruye la máscara binaria a partir de los grupos seleccionados."""
    new_mask = np.zeros_like(base_mask)
    for group in selected_groups:
        for link_idx in group:
            new_mask[link_idx] = 1.0
    return new_mask


def sample_od_simple(od_mask: np.ndarray, rate: float, random_seed: int) -> np.ndarray:
    """
    Helper genérico para OD sampling aleatorio.
    (Las estrategias pueden usarlo o implementar el suyo propio).
    """
    known_indices = np.where(od_mask > 0)[0]
    num_known = len(known_indices)
    num_sampled = int(num_known * rate)

    rng = np.random.default_rng(random_seed)
    sampled_indices = rng.choice(known_indices, size=num_sampled, replace=False)

    sampled_mask = np.zeros_like(od_mask)
    sampled_mask[sampled_indices] = 1.0
    return sampled_mask


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
