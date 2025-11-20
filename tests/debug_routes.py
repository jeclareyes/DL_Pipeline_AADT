"""
Debug script para entender el problema de mapeo de rutas.
"""
import pickle
import networkx as nx
import numpy as np

# Cargar datos
graph = pickle.load(open('../data/processed/Linköping/Linköping_graph.pkl', 'rb'))
routes_data = pickle.load(open('../data/processed/Linköping/routing_cache/kshortest_paths.pkl', 'rb'))

print("=" * 80)
print("DEBUG: Mapeo de rutas")
print("=" * 80)

# 1. Nodos del grafo
graph_nodes = list(graph.nodes())
print(f"\n1. NODOS DEL GRAFO:")
print(f"   Total: {len(graph_nodes)}")
print(f"   Tipo: {type(graph_nodes[0])}")
print(f"   Primeros 5: {graph_nodes[:5]}")

# 2. Aristas del grafo
graph_edges = list(graph.edges())
print(f"\n2. ARISTAS DEL GRAFO:")
print(f"   Total: {len(graph_edges)}")
print(f"   Primera arista: {graph_edges[0]}")
print(f"   Tipo nodos: {type(graph_edges[0][0])}, {type(graph_edges[0][1])}")

# 3. Rutas
routes = routes_data['routes']
print(f"\n3. RUTAS:")
print(f"   Shape: {routes.shape}")
print(f"   Dtype: {routes.dtype}")

# Obtener primera ruta válida
first_valid_route = None
for od_idx in range(routes.shape[0]):
    for k in range(routes.shape[1]):
        route = routes[od_idx, k]
        route = route[route > 0]
        if len(route) >= 2:
            first_valid_route = route
            print(f"   Primera ruta válida (OD {od_idx}, path {k}): {route[:5]}...")
            break
    if first_valid_route is not None:
        break

# 4. Intentar mapear algunos nodos
if first_valid_route is not None:
    print(f"\n4. INTENTO DE MAPEO:")
    for i in range(min(3, len(first_valid_route) - 1)):
        node_from_int = int(first_valid_route[i])
        node_to_int = int(first_valid_route[i + 1])
        node_from_str = str(node_from_int)
        node_to_str = str(node_to_int)
        
        print(f"\n   Arista {i}:")
        print(f"      Nodos (int): ({node_from_int}, {node_to_int})")
        print(f"      Nodos (str): ('{node_from_str}', '{node_to_str}')")
        print(f"      ¿Nodo from en grafo? {node_from_str in graph_nodes}")
        print(f"      ¿Nodo to en grafo? {node_to_str in graph_nodes}")
        print(f"      ¿Arista en grafo? {(node_from_str, node_to_str) in graph_edges}")

# 5. Verificar od_pairs
od_pairs = routes_data.get('od_pairs', None)
print(f"\n5. OD PAIRS:")
if od_pairs is not None:
    print(f"   Tipo: {type(od_pairs)}")
    if isinstance(od_pairs, np.ndarray):
        print(f"   Shape: {od_pairs.shape}")
        print(f"   Dtype: {od_pairs.dtype}")
        print(f"   Primeros 3: {od_pairs[:3]}")
    else:
        print(f"   Length: {len(od_pairs)}")
        print(f"   Primeros 3: {od_pairs[:3]}")
else:
    print(f"   No hay od_pairs en routes_data")

# 6. Buscar coincidencias parciales
print(f"\n6. ANÁLISIS DE COINCIDENCIAS:")
if first_valid_route is not None:
    sample_nodes = [int(first_valid_route[i]) for i in range(min(5, len(first_valid_route)))]
    sample_nodes_str = [str(n) for n in sample_nodes]
    
    print(f"   Nodos de muestra: {sample_nodes}")
    print(f"   Nodos de muestra (str): {sample_nodes_str}")
    
    # Buscar nodos similares en el grafo
    for node_str in sample_nodes_str:
        similar = [n for n in graph_nodes if node_str in str(n) or str(n) in node_str]
        print(f"   Nodos similares a '{node_str}': {similar[:3] if len(similar) > 3 else similar}")

print("\n" + "=" * 80)

