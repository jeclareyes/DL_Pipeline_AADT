"""
Script principal para ejecutar Traffic Assignment.

Este módulo contiene la lógica principal para resolver el problema de asignación
de tráfico (Traffic Assignment) usando diferentes algoritmos.

Uso:
    python src/models/traffic_assignment.py
"""
import sys
from pathlib import Path
from typing import Dict
import networkx as nx
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager
from models.frank_wolfe import FrankWolfeAssignment


def solve_traffic_assignment(graph: nx.DiGraph,
                            od_matrix: sparse.spmatrix,
                            algorithm: str = 'frank_wolfe',
                            cost_attr: str = 'length',
                            solution_attr: str = 'solution_nocongestion',
                            max_iterations: int = 100,
                            convergence_threshold: float = 0.01,
                            use_cache: bool = True,
                            k_paths: int = 50,
                            save_comparison: bool = True,
                            comparison_path: str = 'outputs/tables/traffic_assignment_comparison.csv') -> Dict:
    """
    Resuelve el problema de asignación de tráfico (Traffic Assignment).

    Args:
        graph: Grafo NetworkX con la red de transporte
        od_matrix: Matriz origen-destino (sparse)
        algorithm: Algoritmo a usar ('frank_wolfe', 'msa', etc.)
        cost_attr: Atributo del grafo a usar como costo ('length', 'free_flow_time')
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
    print("\n" + "="*70)
    print("🚦 ASIGNACIÓN DE TRÁFICO - User Equilibrium")
    print("="*70)
    print(f"   Algoritmo: {algorithm.upper()}")
    print(f"   Costo base: {cost_attr}")
    print(f"   Solución guardada en: '{solution_attr}'")

    # Crear instancia del algoritmo según la opción
    if algorithm.lower() == 'frank_wolfe':
        assignment_model = FrankWolfeAssignment(
            graph=graph,
            od_matrix=od_matrix,
            cost_attr=cost_attr,
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
    print("✅ ASIGNACIÓN DE TRÁFICO COMPLETADA")
    print("="*70)

    return solution_stats


def main():
    """
    Función principal para ejecutar el Traffic Assignment completo.

    Ejecuta el pipeline:
    1. Carga de datos (red, flujos, matriz OD)
    2. Construcción del grafo
    3. Asignación de tráfico con Frank-Wolfe
    4. Comparación con flujos observados
    5. Guardado de resultados
    """
    print("\n" + "="*80)
    print("🚦 TRAFFIC ASSIGNMENT - Frank-Wolfe")
    print("="*80)

    # 1. Inicializar DataManager y cargar datos
    print("\n📁 PASO 1: Cargando datos...")
    manager = DataManager()

    # Cargar y combinar datos
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')

    # Construir grafo
    graph = manager.build_graph()

    # Guardar grafo
    manager.save_graph_pickle(
        output_path='data/processed/barcelona_network_with_assignment.gpickle',
        format='pickle'
    )

    # 2. Ejecutar Traffic Assignment
    print("\n🚗 PASO 2: Ejecutando Traffic Assignment...")
    assignment_results = solve_traffic_assignment(
        graph=graph,
        od_matrix=od_matrix,
        algorithm='frank_wolfe',
        cost_attr='length',  # Usar longitud como costo
        solution_attr='solution_nocongestion',  # Guardar solución aquí
        max_iterations=100,
        convergence_threshold=0.01,
        use_cache=True,
        k_paths=50,
        save_comparison=True,
        comparison_path='/outputs/tables/traffic_assignment_comparison.csv'
    )

    # 3. Mostrar resultados
    print("\n" + "="*80)
    print("📊 RESULTADOS FINALES")
    print("="*80)

    print(f"\n🔄 Convergencia:")
    print(f"   - Convergió: {'Sí' if assignment_results['converged'] else 'No'}")
    print(f"   - Iteraciones: {assignment_results['iterations']}")
    print(f"   - Gap final: {assignment_results['final_gap']:.6f}")
    print(f"   - Costo total: {assignment_results['total_cost']:.2f}")

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

    print("\n" + "="*80)
    print("✅ TRAFFIC ASSIGNMENT COMPLETADO EXITOSAMENTE")
    print("="*80)

    return assignment_results


if __name__ == '__main__':
    """
    Ejecutar Traffic Assignment desde línea de comandos:
    
        python src/models/traffic_assignment.py
    """
    try:
        results = main()
    except Exception as e:
        print(f"\n❌ Error durante el Traffic Assignment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
