"""
Script principal para Traffic Assignment con CONGESTIÓN.

Este módulo implementa el problema de User Equilibrium (UE) con función de costo
BPR (Bureau of Public Roads) que considera la congestión.

Soporta múltiples datasets: Barcelona, SiouxFalls, etc.

Función BPR:
    t(x) = t0 * [1 + b * (x/c)^power]

    donde:
    - t(x): tiempo de viaje con flujo x
    - t0: tiempo de flujo libre (free_flow_time)
    - b: parámetro de congestión (típicamente 0.15)
    - power: exponente (típicamente 4)
    - x: flujo actual en el enlace
    - c: capacidad del enlace

El algoritmo Frank-Wolfe iterará hasta convergencia, actualizando los costos
en cada iteración según el flujo asignado.

Uso:
    # Barcelona (por defecto)
    python src/models/traffic_assignment_congestion.py

    # SiouxFalls
    python src/models/traffic_assignment_congestion.py --network SiouxFalls
"""
import sys
from pathlib import Path
from typing import Dict
import networkx as nx
from scipy import sparse
import argparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager
from models.frank_wolfe_congestion import FrankWolfeAssignmentCongestion


def solve_traffic_assignment_with_congestion(
        graph: nx.DiGraph,
        od_matrix: sparse.spmatrix,
        network_name: str = 'Barcelona',
        algorithm: str = 'frank_wolfe',
        cost_function: str = 'bpr',
        solution_attr: str = 'solution_congestion',
        max_iterations: int = 100,
        convergence_threshold: float = 0.01,
        use_cache: bool = True,
        k_paths: int = 10,
        save_comparison: bool = True,
        comparison_path: str = None) -> Dict:
    """
    Resuelve el problema de asignación de tráfico CON CONGESTIÓN.

    Args:
        graph: Grafo NetworkX con la red de transporte
        od_matrix: Matriz origen-destino (sparse)
        network_name: Nombre de la red para rutas de salida
        algorithm: Algoritmo a usar ('frank_wolfe')
        cost_function: Función de costo ('bpr' para congestión)
        solution_attr: Atributo donde guardar la solución
        max_iterations: Número máximo de iteraciones
        convergence_threshold: Umbral de convergencia (gap relativo)
        use_cache: Usar cache de rutas k-shortest paths
        k_paths: Número de rutas alternativas a calcular
        save_comparison: Guardar comparación con flujos observados
        comparison_path: Ruta del archivo CSV de comparación

    Returns:
        Dict con estadísticas de la solución y comparación
    """
    # Ruta por defecto según red
    if comparison_path is None:
        comparison_path = f'data/processed/{network_name.lower()}_traffic_assignment_congestion_comparison.csv'

    print("\n" + "="*70)
    print(f"🚦 ASIGNACIÓN DE TRÁFICO CON CONGESTIÓN - User Equilibrium")
    print(f"   Red: {network_name}")
    print("="*70)
    print(f"   Algoritmo: {algorithm.upper()}")
    print(f"   Función de costo: {cost_function.upper()} (congestión)")
    print(f"   Solución guardada en: '{solution_attr}'")

    # Crear instancia del algoritmo según la opción
    if algorithm.lower() == 'frank_wolfe':
        assignment_model = FrankWolfeAssignmentCongestion(
            graph=graph,
            od_matrix=od_matrix,
            cost_function=cost_function,
            solution_attr=solution_attr,
            use_cache=use_cache,
            k_paths=k_paths
        )
    else:
        raise ValueError(f"Algoritmo '{algorithm}' no implementado. Use 'frank_wolfe'")

    # Resolver
    solution_stats = assignment_model.solve(
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        verbose=True
    )

    # Guardar comparación con flujos observados
    if save_comparison:
        print(f"\n📊 Generando comparación con flujos observados...")
        comparison_path = Path(comparison_path)
        comparison_df = assignment_model.compare_with_observed(observed_attr='volume')
        saved_path = assignment_model.save_comparison(comparison_path, observed_attr='volume')

        # Mostrar métricas principales
        print(f"\n📈 MÉTRICAS DE COMPARACIÓN:")
        print(f"   - MAE (Mean Absolute Error): {comparison_df.attrs['MAE']:.2f}")
        print(f"   - RMSE (Root Mean Squared Error): {comparison_df.attrs['RMSE']:.2f}")
        print(f"   - MAPE (Mean Absolute Percentage Error): {comparison_df.attrs['MAPE']:.2f}%")
        print(f"   - Total observado: {comparison_df.attrs['total_observed']:.2f}")
        print(f"   - Total predicho: {comparison_df.attrs['total_predicted']:.2f}")

        solution_stats['comparison'] = {
            'MAE': comparison_df.attrs['MAE'],
            'RMSE': comparison_df.attrs['RMSE'],
            'MAPE': comparison_df.attrs['MAPE'],
            'total_observed': comparison_df.attrs['total_observed'],
            'total_predicted': comparison_df.attrs['total_predicted'],
            'comparison_file': str(saved_path)
        }

    print("\n" + "="*70)
    print("✅ ASIGNACIÓN DE TRÁFICO CON CONGESTIÓN COMPLETADA")
    print("="*70)

    return solution_stats


def main(network_name: str = 'Barcelona',
         max_iterations: int = 1000,
         convergence_threshold: float = 0.01,
         use_cache: bool = True,
         k_paths: int = 50):
    """
    Función principal para ejecutar el Traffic Assignment CON CONGESTIÓN.

    Ejecuta el pipeline:
    1. Carga de datos (red, flujos, matriz OD)
    2. Construcción del grafo
    3. Asignación de tráfico con Frank-Wolfe + BPR
    4. Comparación con flujos observados
    5. Guardado de resultados

    Args:
        network_name: Nombre de la red ('Barcelona', 'SiouxFalls', etc.)
        max_iterations: Número máximo de iteraciones
        convergence_threshold: Umbral de convergencia
        use_cache: Usar cache de rutas
        k_paths: Número de rutas alternativas
    """
    print("\n" + "="*80)
    print(f"🚦 TRAFFIC ASSIGNMENT CON CONGESTIÓN - {network_name}")
    print("="*80)

    # 1. Inicializar DataManager y cargar datos
    print("\n📁 PASO 1: Cargando datos...")
    manager = DataManager(network_name=network_name)

    # Cargar y combinar datos
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')

    # Construir grafo
    graph = manager.build_graph()

    # Verificar que tenemos los atributos necesarios para BPR
    print("\n🔍 Verificando atributos BPR en el grafo...")
    sample_edge = list(graph.edges(data=True))[0]
    required_attrs = ['free_flow_time', 'capacity', 'b', 'power']

    missing_attrs = [attr for attr in required_attrs if attr not in sample_edge[2]]
    if missing_attrs:
        print(f"   ⚠ WARNING: Faltan atributos BPR: {missing_attrs}")
    else:
        print(f"   ✓ Todos los atributos BPR presentes: {required_attrs}")

    # Mostrar estadísticas de valores BPR
    print(f"\n📊 Estadísticas de parámetros BPR:")
    print(f"   b (media): {unified_df['b'].mean():.6e}")
    print(f"   power (media): {unified_df['power'].mean():.2f}")
    print(f"   capacity (media): {unified_df['capacity'].mean():.2f}")

    # Guardar grafo
    graph_path = f'data/processed/{network_name.lower()}_network_with_congestion.gpickle'
    manager.save_graph_pickle(
        output_path=graph_path,
        format='pickle'
    )

    # 2. Ejecutar Traffic Assignment CON CONGESTIÓN
    print("\n🚗 PASO 2: Ejecutando Traffic Assignment con Congestión (BPR)...")
    assignment_results = solve_traffic_assignment_with_congestion(
        graph=graph,
        od_matrix=od_matrix,
        network_name=network_name,
        algorithm='frank_wolfe',
        cost_function='bpr',
        solution_attr='solution_congestion',
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        use_cache=use_cache,
        k_paths=k_paths,
        save_comparison=True
    )

    # 3. Mostrar resultados
    print("\n" + "="*80)
    print(f"📊 RESULTADOS FINALES - {network_name}")
    print("="*80)

    print(f"\n🔄 Convergencia:")
    print(f"   - Convergió: {'Sí' if assignment_results['converged'] else 'No'}")
    print(f"   - Iteraciones: {assignment_results['iterations']}")
    print(f"   - Gap final: {assignment_results['final_gap']:.6f}")
    print(f"   - Costo total (con congestión): {assignment_results['total_cost']:.2f}")

    if 'cache_stats' in assignment_results and assignment_results['cache_stats']:
        print(f"\n💾 Estadísticas de Cache:")
        cache = assignment_results['cache_stats']
        print(f"   - Hits: {cache['hits']}")
        print(f"   - Misses: {cache['misses']}")
        print(f"   - Hit rate: {cache['hit_rate']*100:.1f}%")
        print(f"   - Pares cacheados: {cache['total_pairs_cached']}")

    if 'comparison' in assignment_results:
        print(f"\n📈 Comparación con Flujos Observados:")
        comp = assignment_results['comparison']
        print(f"   - MAE: {comp['MAE']:.2f}")
        print(f"   - RMSE: {comp['RMSE']:.2f}")
        print(f"   - MAPE: {comp['MAPE']:.2f}%")
        print(f"   - Total observado: {comp['total_observed']:.2f}")
        print(f"   - Total predicho: {comp['total_predicted']:.2f}")
        print(f"   - Balance error: {abs(comp['total_observed'] - comp['total_predicted']):.2f}")
        print(f"\n   📁 Archivo de comparación: {comp['comparison_file']}")

    # Mostrar algunas aristas con alto nivel de congestión
    print(f"\n🚨 Top 10 Enlaces más Congestionados:")
    edges_with_congestion = []
    for u, v, data in graph.edges(data=True):
        flow = data.get('solution_congestion', 0)
        capacity = data.get('capacity', 1)
        vc_ratio = flow / capacity if capacity > 0 else 0
        edges_with_congestion.append((u, v, flow, capacity, vc_ratio))

    edges_with_congestion.sort(key=lambda x: x[4], reverse=True)
    for i, (u, v, flow, cap, vc) in enumerate(edges_with_congestion[:10], 1):
        print(f"   {i}. Enlace {u}→{v}: V/C={vc:.2f} (flujo={flow:.0f}, cap={cap:.0f})")

    print("\n" + "="*80)
    print(f"✅ TRAFFIC ASSIGNMENT CON CONGESTIÓN COMPLETADO - {network_name}")
    print("="*80)

    return assignment_results


if __name__ == '__main__':
    """
    Ejecutar Traffic Assignment con Congestión desde línea de comandos:
    
        # Barcelona (por defecto)
        python src/models/traffic_assignment_congestion.py
        
        # SiouxFalls
        python src/models/traffic_assignment_congestion.py --network SiouxFalls
        
        # Con parámetros personalizados
        python src/models/traffic_assignment_congestion.py --network SiouxFalls --max-iter 500 --threshold 0.001
    """
    # Parsear argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Traffic Assignment con Congestión (User Equilibrium)')
    parser.add_argument('--network', type=str, default='Barcelona',
                        choices=['Barcelona', 'SiouxFalls'],
                        help='Nombre de la red a procesar (default: Barcelona)')
    parser.add_argument('--max-iter', type=int, default=1000,
                        help='Número máximo de iteraciones (default: 1000)')
    parser.add_argument('--threshold', type=float, default=0.01,
                        help='Umbral de convergencia (default: 0.01)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Desactivar cache de rutas')
    parser.add_argument('--k-paths', type=int, default=50,
                        help='Número de rutas alternativas (default: 50)')

    args = parser.parse_args()

    try:
        results = main(
            network_name=args.network,
            max_iterations=args.max_iter,
            convergence_threshold=args.threshold,
            use_cache=not args.no_cache,
            k_paths=args.k_paths
        )
    except Exception as e:
        print(f"\n❌ Error durante el Traffic Assignment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
