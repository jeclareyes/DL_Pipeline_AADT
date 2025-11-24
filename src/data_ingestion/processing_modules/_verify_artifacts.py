"""
Script para verificar que los artefactos generados por data_pipeline.py sean correctos.
"""
import pickle
import pandas as pd
import networkx as nx
from pathlib import Path
from scipy import sparse

# Directorio de Linköping
processed_dir = Path('data/processed/Linköping')

print("="*80)
print("VERIFICACIÓN DE ARTEFACTOS GENERADOS PARA LINKÖPING")
print("="*80)

# 1. Verificar grafo
print("\n1. VERIFICANDO GRAFO...")
graph_file = processed_dir / 'Linköping_graph.pkl'
if graph_file.exists():
    with open(graph_file, 'rb') as f:
        graph = pickle.load(f)
    print(f"   ✓ Grafo cargado: {type(graph)}")
    print(f"   - Nodos: {graph.number_of_nodes()}")
    print(f"   - Aristas: {graph.number_of_edges()}")

    # Contar nodos TAZ
    taz_nodes = [n for n, d in graph.nodes(data=True) if d.get('type') == 'taz']
    print(f"   - Nodos TAZ: {len(taz_nodes)}")

    # Verificar atributos de aristas
    if graph.number_of_edges() > 0:
        sample_edge = list(graph.edges(data=True))[0]
        print(f"   - Atributos de arista (muestra): {list(sample_edge[2].keys())[:10]}")
else:
    print(f"   ✗ No se encontró {graph_file}")

# 2. Verificar dataframe de enlaces
print("\n2. VERIFICANDO DATAFRAME DE ENLACES...")
link_file = processed_dir / 'Linköping_link_data.parquet'
if link_file.exists():
    df = pd.read_parquet(link_file)
    print(f"   ✓ DataFrame cargado: {df.shape}")
    print(f"   - Columnas: {list(df.columns)}")
    print(f"   - Muestra:")
    print(df.head(3).to_string(index=False))
else:
    print(f"   ✗ No se encontró {link_file}")

# 3. Verificar matriz OD
print("\n3. VERIFICANDO MATRIZ OD...")
od_file = processed_dir / 'Linköping_od_matrix.npz'
if od_file.exists():
    od_matrix = sparse.load_npz(od_file)
    print(f"   ✓ Matriz OD cargada: {od_matrix.shape}")
    print(f"   - Formato: {type(od_matrix)}")
    print(f"   - Entradas no-cero: {od_matrix.nnz}")
    print(f"   - Flujo total: {od_matrix.sum():.2f}")
else:
    print(f"   ✗ No se encontró {od_file}")

# 4. Verificar OD dataframe
print("\n4. VERIFICANDO OD DATAFRAME...")
od_df_file = processed_dir / 'Linköping_od_dataframe.parquet'
if od_df_file.exists():
    od_df = pd.read_parquet(od_df_file)
    print(f"   ✓ OD DataFrame cargado: {od_df.shape}")
    print(f"   - Columnas: {list(od_df.columns)}")
    print(f"   - Pares OD únicos: {len(od_df)}")
    print(f"   - Flujo total: {od_df['flow'].sum():.2f}")
else:
    print(f"   ✗ No se encontró {od_df_file}")

# 5. Verificar rutas (k-shortest paths)
print("\n5. VERIFICANDO RUTAS (K-SHORTEST PATHS)...")
routes_file = processed_dir / 'routing_cache' / 'kshortest_paths.pkl'
if routes_file.exists():
    with open(routes_file, 'rb') as f:
        routes_data = pickle.load(f)
    print(f"   ✓ Rutas cargadas")
    print(f"   - Shape del tensor: {routes_data['routes'].shape}")
    print(f"   - Número de pares OD: {len(routes_data['od_pairs'])}")
    print(f"   - k (rutas por par): {routes_data['k']}")
    print(f"   - Longitud máxima de ruta: {routes_data['max_route_length']}")

    # Verificar algunas rutas de ejemplo
    routes_array = routes_data['routes']
    non_empty = 0
    for i in range(min(10, len(routes_array))):
        if routes_array[i][0][0] != -1:  # Primera ruta no vacía
            non_empty += 1
    print(f"   - Rutas no vacías (de las primeras 10): {non_empty}/10")
else:
    print(f"   ✗ No se encontró {routes_file}")

print("\n" + "="*80)
print("VERIFICACIÓN COMPLETADA")
print("="*80)

