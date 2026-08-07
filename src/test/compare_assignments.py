"""
Script comparativo: Traffic Assignment SIN vs CON Congestión.

Este script ejecuta ambas modalidades y compara los resultados:
1. Sin congestión (costo fijo - distancia)
2. Con congestión (costo dinámico - BPR)

Muestra las diferencias en:
- Iteraciones necesarias
- Tiempo de convergencia
- Costo total del sistema
- Distribución de flujos
- Enlaces más congestionados

Uso:
    python src/models/compare_assignments.py
"""
import sys
from pathlib import Path
import time
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_handling import DataManager
from models.traffic_assignment import solve_traffic_assignment
from models.traffic_assignment_congestion import solve_traffic_assignment_with_congestion


def compare_assignments():
    """
    Ejecuta ambas modalidades de Traffic Assignment y compara resultados.
    """
    print("\n" + "="*80)
    print("🔬 COMPARACIÓN: Traffic Assignment SIN vs CON Congestión")
    print("="*80)

    # 1. Cargar datos (común para ambas modalidades)
    print("\n📁 PASO 1: Cargando datos...")
    manager = DataManager()
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')
    graph = manager.build_graph()

    print(f"\n📊 Datos cargados:")
    print(f"   - Nodos: {graph.number_of_nodes()}")
    print(f"   - Enlaces: {graph.number_of_edges()}")
    print(f"   - Pares OD: {od_matrix.nnz}")
    print(f"   - Demanda total: {od_matrix.sum():.2f} viajes")

    # ============================================================================
    # 2. MODALIDAD 1: SIN CONGESTIÓN
    # ============================================================================
    print("\n" + "="*80)
    print("🚗 MODALIDAD 1: SIN CONGESTIÓN (Costo Fijo)")
    print("="*80)

    start_time_no_cong = time.time()

    results_no_cong = solve_traffic_assignment(
        graph=graph,
        od_matrix=od_matrix,
        algorithm='frank_wolfe',
        cost_attr='length',
        solution_attr='solution_nocongestion',
        max_iterations=100,
        convergence_threshold=0.01,
        use_cache=True,
        k_paths=10,
        save_comparison=True,
        comparison_path='../components/models/data/processed/comparison_nocongestion.csv'
    )

    time_no_cong = time.time() - start_time_no_cong

    # ============================================================================
    # 3. MODALIDAD 2: CON CONGESTIÓN (BPR)
    # ============================================================================
    print("\n" + "="*80)
    print("🚦 MODALIDAD 2: CON CONGESTIÓN (Costo Dinámico - BPR)")
    print("="*80)

    start_time_cong = time.time()

    results_cong = solve_traffic_assignment_with_congestion(
        graph=graph,
        od_matrix=od_matrix,
        algorithm='frank_wolfe',
        cost_function='bpr',
        solution_attr='solution_congestion',
        max_iterations=100,
        convergence_threshold=0.01,
        use_cache=False,  # Menos efectivo con costos dinámicos
        k_paths=10,
        save_comparison=True,
        comparison_path='../components/models/data/processed/comparison_congestion.csv'
    )

    time_cong = time.time() - start_time_cong

    # ============================================================================
    # 4. COMPARACIÓN DE RESULTADOS
    # ============================================================================
    print("\n" + "="*80)
    print("📊 COMPARACIÓN DE RESULTADOS")
    print("="*80)

    # Tabla comparativa de métricas principales
    print("\n┌─────────────────────────────────┬──────────────────┬──────────────────┐")
    print("│ Métrica                         │ Sin Congestión   │ Con Congestión   │")
    print("├─────────────────────────────────┼──────────────────┼──────────────────┤")
    print(f"│ Iteraciones                     │ {results_no_cong['iterations']:>16} │ {results_cong['iterations']:>16} │")
    print(f"│ Gap final                       │ {results_no_cong['final_gap']:>16.6f} │ {results_cong['final_gap']:>16.6f} │")
    print(f"│ Convergió                       │ {'Sí':>16} │ {'Sí' if results_cong['converged'] else 'No':>16} │")
    print(f"│ Costo total sistema             │ {results_no_cong['total_cost']:>16.2f} │ {results_cong['total_cost']:>16.2f} │")
    print(f"│ Tiempo ejecución (seg)          │ {time_no_cong:>16.2f} │ {time_cong:>16.2f} │")
    print("└─────────────────────────────────┴──────────────────┴──────────────────┘")

    # Análisis del incremento de costo por congestión
    cost_increase = results_cong['total_cost'] - results_no_cong['total_cost']
    cost_increase_pct = (cost_increase / results_no_cong['total_cost']) * 100

    print(f"\n💡 Análisis:")
    print(f"   - Incremento de costo por congestión: {cost_increase:,.2f} ({cost_increase_pct:.1f}%)")
    print(f"   - Tiempo adicional por iteraciones: {time_cong - time_no_cong:.2f} segundos")
    print(f"   - Factor de ralentización: {time_cong / time_no_cong:.1f}x")

    # ============================================================================
    # 5. COMPARACIÓN DE FLUJOS POR ENLACE
    # ============================================================================
    print("\n" + "="*80)
    print("🔍 COMPARACIÓN DE FLUJOS POR ENLACE")
    print("="*80)

    flow_comparison = []
    for u, v, data in graph.edges(data=True):
        flow_no_cong = data.get('solution_nocongestion', 0)
        flow_cong = data.get('solution_congestion', 0)
        capacity = data.get('capacity', 1)

        flow_comparison.append({
            'from': u,
            'to': v,
            'flow_nocongestion': flow_no_cong,
            'flow_congestion': flow_cong,
            'difference': flow_cong - flow_no_cong,
            'capacity': capacity,
            'vc_nocongestion': flow_no_cong / capacity if capacity > 0 else 0,
            'vc_congestion': flow_cong / capacity if capacity > 0 else 0
        })

    df_flows = pd.DataFrame(flow_comparison)

    print(f"\n📊 Estadísticas de flujos:")
    print(f"   - Total flujo sin congestión: {df_flows['flow_nocongestion'].sum():,.2f}")
    print(f"   - Total flujo con congestión: {df_flows['flow_congestion'].sum():,.2f}")
    print(f"   - Diferencia promedio: {df_flows['difference'].mean():,.2f}")
    print(f"   - Diferencia máxima: {df_flows['difference'].abs().max():,.2f}")

    # Enlaces con mayor diferencia
    print(f"\n🔄 Top 10 Enlaces con Mayor Redistribución de Flujo:")
    df_flows_sorted = df_flows.sort_values('difference', key=abs, ascending=False)
    for i, row in df_flows_sorted.head(10).iterrows():
        print(f"   {i+1}. Enlace {row['from']}→{row['to']}: "
              f"Δ={row['difference']:+.0f} "
              f"(sin={row['flow_nocongestion']:.0f}, con={row['flow_congestion']:.0f})")

    # ============================================================================
    # 6. ANÁLISIS DE CONGESTIÓN
    # ============================================================================
    print("\n" + "="*80)
    print("🚨 ANÁLISIS DE CONGESTIÓN (Solo modalidad CON congestión)")
    print("="*80)

    # Clasificar enlaces por nivel de congestión
    congestion_levels = {
        'A (V/C < 0.6)': 0,
        'B (0.6 ≤ V/C < 0.7)': 0,
        'C (0.7 ≤ V/C < 0.8)': 0,
        'D (0.8 ≤ V/C < 0.9)': 0,
        'E (0.9 ≤ V/C < 1.0)': 0,
        'F (V/C ≥ 1.0)': 0
    }

    for _, row in df_flows.iterrows():
        vc = row['vc_congestion']
        if vc < 0.6:
            congestion_levels['A (V/C < 0.6)'] += 1
        elif vc < 0.7:
            congestion_levels['B (0.6 ≤ V/C < 0.7)'] += 1
        elif vc < 0.8:
            congestion_levels['C (0.7 ≤ V/C < 0.8)'] += 1
        elif vc < 0.9:
            congestion_levels['D (0.8 ≤ V/C < 0.9)'] += 1
        elif vc < 1.0:
            congestion_levels['E (0.9 ≤ V/C < 1.0)'] += 1
        else:
            congestion_levels['F (V/C ≥ 1.0)'] += 1

    print(f"\n📊 Distribución de Nivel de Servicio (LOS):")
    for level, count in congestion_levels.items():
        pct = (count / len(df_flows)) * 100
        bar = '█' * int(pct / 2)
        print(f"   {level:25} : {count:>5} enlaces ({pct:>5.1f}%) {bar}")

    # Top enlaces congestionados
    print(f"\n🚨 Top 10 Enlaces Más Congestionados:")
    df_congested = df_flows.sort_values('vc_congestion', ascending=False)
    for i, row in df_congested.head(10).iterrows():
        print(f"   {i+1}. Enlace {row['from']}→{row['to']}: "
              f"V/C={row['vc_congestion']:.2f} "
              f"(flujo={row['flow_congestion']:.0f}, cap={row['capacity']:.0f})")

    # ============================================================================
    # 7. GUARDAR COMPARACIÓN
    # ============================================================================
    print("\n" + "="*80)
    print("💾 GUARDANDO RESULTADOS")
    print("="*80)

    # Guardar comparación de flujos
    output_path = Path('../components/models/data/processed/flow_comparison_nocong_vs_cong.csv')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_flows.to_csv(output_path, index=False)
    print(f"   ✓ Comparación de flujos guardada: {output_path}")

    # Guardar grafo con ambas soluciones
    from data_handling import save_graph
    graph_path = save_graph(graph, 'data/processed/barcelona_network_both_solutions.gpickle')
    print(f"   ✓ Grafo con ambas soluciones guardado: {graph_path}")

    # Resumen en JSON
    import json
    summary = {
        'execution_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'no_congestion': {
            'iterations': results_no_cong['iterations'],
            'gap': results_no_cong['final_gap'],
            'total_cost': results_no_cong['total_cost'],
            'time_seconds': time_no_cong,
            'converged': results_no_cong['converged']
        },
        'congestion': {
            'iterations': results_cong['iterations'],
            'gap': results_cong['final_gap'],
            'total_cost': results_cong['total_cost'],
            'time_seconds': time_cong,
            'converged': results_cong['converged']
        },
        'comparison': {
            'cost_increase': cost_increase,
            'cost_increase_pct': cost_increase_pct,
            'time_difference': time_cong - time_no_cong,
            'congestion_levels': congestion_levels
        }
    }

    summary_path = Path('../components/models/data/processed/comparison_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   ✓ Resumen JSON guardado: {summary_path}")

    print("\n" + "="*80)
    print("✅ COMPARACIÓN COMPLETADA EXITOSAMENTE")
    print("="*80)

    return summary


if __name__ == '__main__':
    """
    Ejecutar comparación completa:
    
        python src/models/compare_assignments.py
    """
    try:
        summary = compare_assignments()
    except Exception as e:
        print(f"\n❌ Error durante la comparación: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
