"""
Script de diagnóstico para verificar:
1. Parámetros BPR (b y power) en los datos
2. Pares OD sin conectividad
3. Uso de valores por defecto vs valores reales
"""
import sys
from pathlib import Path
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager


def diagnose_bpr_parameters():
    """Verifica los parámetros BPR en los datos."""
    print("="*80)
    print("🔍 DIAGNÓSTICO DE PARÁMETROS BPR")
    print("="*80)

    # Cargar datos
    manager = DataManager()
    network_df = manager.load_network()

    print("\n1️⃣ COLUMNAS DISPONIBLES EN NETWORK:")
    print(network_df.columns.tolist())

    # Verificar parámetros BPR
    print("\n2️⃣ PARÁMETROS BPR EN EL DATASET:")

    if 'b' in network_df.columns:
        print("\n📊 Parámetro 'b' (coeficiente de congestión):")
        print(network_df['b'].describe())
        print(f"\nValores únicos de 'b': {network_df['b'].unique()}")
        print(f"Enlaces con b > 0: {(network_df['b'] > 0).sum()} de {len(network_df)}")
    else:
        print("❌ Columna 'b' NO ENCONTRADA en el dataset")

    if 'power' in network_df.columns:
        print("\n📊 Parámetro 'power' (exponente BPR):")
        print(network_df['power'].describe())
        print(f"\nValores únicos de 'power': {network_df['power'].unique()}")
        print(f"Enlaces con power > 0: {(network_df['power'] > 0).sum()} de {len(network_df)}")
    else:
        print("❌ Columna 'power' NO ENCONTRADA en el dataset")

    # Verificar otros parámetros necesarios
    print("\n3️⃣ OTROS PARÁMETROS NECESARIOS PARA BPR:")

    for param in ['free_flow_time', 'capacity', 'length']:
        if param in network_df.columns:
            print(f"\n✓ {param}:")
            print(f"   Min: {network_df[param].min():.2f}")
            print(f"   Max: {network_df[param].max():.2f}")
            print(f"   Mean: {network_df[param].mean():.2f}")
            print(f"   Valores <= 0: {(network_df[param] <= 0).sum()}")
        else:
            print(f"❌ {param} NO ENCONTRADO")

    return network_df


def diagnose_connectivity():
    """Diagnostica problemas de conectividad en la red."""
    print("\n" + "="*80)
    print("🔍 DIAGNÓSTICO DE CONECTIVIDAD DE LA RED")
    print("="*80)

    # Cargar grafo
    manager = DataManager()
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')
    graph = manager.build_graph()

    print(f"\n1️⃣ INFORMACIÓN DEL GRAFO:")
    print(f"   Nodos: {graph.number_of_nodes()}")
    print(f"   Enlaces: {graph.number_of_edges()}")
    print(f"   Es conexo: {nx.is_strongly_connected(graph)}")

    # Encontrar componentes conexas
    if not nx.is_strongly_connected(graph):
        print("\n⚠️ El grafo NO es fuertemente conexo")
        components = list(nx.strongly_connected_components(graph))
        print(f"   Número de componentes fuertemente conexas: {len(components)}")

        # Mostrar tamaños de componentes
        component_sizes = sorted([len(c) for c in components], reverse=True)
        print(f"   Tamaños de componentes: {component_sizes[:10]}")  # Top 10

        # Componente principal
        largest_component = max(components, key=len)
        print(f"\n   Componente principal: {len(largest_component)} nodos")
        print(f"   Componentes pequeños: {len(components) - 1}")
    else:
        print("✓ El grafo es fuertemente conexo")

    # Verificar conectividad débil
    weakly_connected = nx.is_weakly_connected(graph)
    print(f"\n   Es débilmente conexo: {weakly_connected}")

    # Analizar pares OD
    print("\n2️⃣ ANÁLISIS DE PARES OD:")
    od_coo = od_matrix.tocoo()
    total_od_pairs = len(od_coo.data)
    print(f"   Total de pares OD con demanda: {total_od_pairs}")

    # Verificar cuántos pares OD están en el grafo
    od_pairs_in_graph = 0
    od_pairs_not_in_graph = []
    od_pairs_no_path = []

    print("\n   Verificando rutas para pares OD...")

    for i, (row, col, demand) in enumerate(zip(od_coo.row, od_coo.col, od_coo.data)):
        origin = row + 1  # Convertir de 0-index a 1-index
        destination = col + 1

        # Verificar si los nodos existen
        if origin not in graph or destination not in graph:
            od_pairs_not_in_graph.append((origin, destination, demand))
            continue

        # Verificar si hay ruta
        try:
            path = nx.shortest_path(graph, origin, destination, weight='length')
            od_pairs_in_graph += 1
        except nx.NetworkXNoPath:
            od_pairs_no_path.append((origin, destination, demand))
        except nx.NodeNotFound:
            od_pairs_not_in_graph.append((origin, destination, demand))

        # Mostrar progreso cada 1000 pares
        if (i + 1) % 1000 == 0:
            print(f"      Procesados: {i+1}/{total_od_pairs}")

    print(f"\n3️⃣ RESULTADOS:")
    print(f"   ✓ Pares OD con ruta válida: {od_pairs_in_graph}")
    print(f"   ⚠️ Pares OD sin ruta (sin conectividad): {len(od_pairs_no_path)}")
    print(f"   ❌ Pares OD con nodos no encontrados: {len(od_pairs_not_in_graph)}")

    # Mostrar ejemplos de pares sin ruta
    if od_pairs_no_path:
        print(f"\n   📋 Ejemplos de pares SIN RUTA (primeros 10):")
        for origin, destination, demand in od_pairs_no_path[:10]:
            # Verificar si los nodos están en componentes diferentes
            if not nx.is_strongly_connected(graph):
                in_same_component = False
                for comp in nx.strongly_connected_components(graph):
                    if origin in comp and destination in comp:
                        in_same_component = True
                        break
                component_info = "misma componente" if in_same_component else "componentes diferentes"
            else:
                component_info = "grafo conexo"

            print(f"      {origin} → {destination} (demanda: {demand:.2f}) [{component_info}]")

    # Guardar reporte
    report_path = Path("data/processed/connectivity_report.txt")
    with open(report_path, 'w') as f:
        f.write(f"REPORTE DE CONECTIVIDAD\n")
        f.write(f"="*80 + "\n\n")
        f.write(f"Total pares OD: {total_od_pairs}\n")
        f.write(f"Pares con ruta: {od_pairs_in_graph}\n")
        f.write(f"Pares sin ruta: {len(od_pairs_no_path)}\n")
        f.write(f"Pares con nodos no encontrados: {len(od_pairs_not_in_graph)}\n\n")

        if od_pairs_no_path:
            f.write(f"PARES SIN RUTA:\n")
            for origin, destination, demand in od_pairs_no_path:
                f.write(f"{origin} → {destination} (demanda: {demand:.2f})\n")

        if od_pairs_not_in_graph:
            f.write(f"\nPARES CON NODOS NO ENCONTRADOS:\n")
            for origin, destination, demand in od_pairs_not_in_graph:
                f.write(f"{origin} → {destination} (demanda: {demand:.2f})\n")

    print(f"\n📁 Reporte guardado en: {report_path}")

    return {
        'total_pairs': total_od_pairs,
        'pairs_with_path': od_pairs_in_graph,
        'pairs_no_path': len(od_pairs_no_path),
        'pairs_not_in_graph': len(od_pairs_not_in_graph)
    }


if __name__ == '__main__':
    # Ejecutar diagnósticos
    print("\n🔬 INICIANDO DIAGNÓSTICOS...\n")

    # 1. Verificar parámetros BPR
    network_df = diagnose_bpr_parameters()

    # 2. Verificar conectividad
    connectivity_stats = diagnose_connectivity()

    # Resumen final
    print("\n" + "="*80)
    print("📊 RESUMEN FINAL")
    print("="*80)

    has_b = 'b' in network_df.columns
    has_power = 'power' in network_df.columns

    if has_b and has_power:
        print("✅ Parámetros BPR (b, power) DISPONIBLES en los datos")
    else:
        print("⚠️ Parámetros BPR NO completamente disponibles:")
        if not has_b:
            print("   ❌ Falta 'b' - se usará valor por defecto (0.15)")
        if not has_power:
            print("   ❌ Falta 'power' - se usará valor por defecto (4.0)")

    print(f"\n📍 Conectividad:")
    print(f"   Total pares OD: {connectivity_stats['total_pairs']}")
    print(f"   Con ruta: {connectivity_stats['pairs_with_path']} ({connectivity_stats['pairs_with_path']/connectivity_stats['total_pairs']*100:.1f}%)")
    print(f"   Sin ruta: {connectivity_stats['pairs_no_path']} ({connectivity_stats['pairs_no_path']/connectivity_stats['total_pairs']*100:.1f}%)")

    if connectivity_stats['pairs_no_path'] > 0:
        print(f"\n⚠️ PROBLEMA IDENTIFICADO: {connectivity_stats['pairs_no_path']} pares OD sin conectividad")
        print("   POSIBLES SOLUCIONES:")
        print("   1. Verificar si la red tiene componentes desconectadas")
        print("   2. Revisar si los nodos OD están en la componente principal")
        print("   3. Considerar añadir enlaces para conectar componentes aisladas")
        print("   4. Filtrar pares OD que no tienen conectividad")

    print("\n" + "="*80)
