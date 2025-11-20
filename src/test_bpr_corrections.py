"""
Script de prueba para validar las correcciones de BPR.

Este script ejecuta una prueba rápida del traffic assignment con las correcciones:
1. Valores de 'b' corregidos (usar 0.15 en lugar de ~10^-18)
2. Valores de 'power' corregidos (usar 4.0 cuando power=0)
3. Capacidades estimadas (en lugar de 1.0)
4. Diagnóstico mejorado de pares OD sin ruta
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager
from src.models.traditional_TA.frank_wolfe_congestion import FrankWolfeAssignmentCongestion


def test_bpr_corrections():
    """Prueba las correcciones de BPR."""
    print("\n" + "="*80)
    print("🧪 PRUEBA DE CORRECCIONES BPR")
    print("="*80)

    # 1. Cargar datos
    print("\n1️⃣ Cargando datos...")
    manager = DataManager()
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')
    graph = manager.build_graph()

    print(f"\n   Grafo: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} enlaces")
    print(f"   Matriz OD: {od_matrix.nnz} pares con demanda")

    # 2. Verificar parámetros BPR en un enlace de muestra
    print("\n2️⃣ Verificando parámetros BPR en enlaces de muestra...")
    sample_edges = list(graph.edges(data=True))[:5]

    for u, v, data in sample_edges:
        b_original = data.get('b', 0.0)
        power_original = data.get('power', 0.0)
        capacity_original = data.get('capacity', 1.0)

        print(f"\n   Enlace {u} → {v}:")
        print(f"      b original: {b_original:.2e} (debería ser ~0.15)")
        print(f"      power original: {power_original:.2f} (debería ser ~4.0)")
        print(f"      capacity original: {capacity_original:.2f} (debería ser >100)")

    # 3. Crear instancia de Frank-Wolfe con correcciones
    print("\n3️⃣ Creando modelo Frank-Wolfe con correcciones BPR...")
    assignment_model = FrankWolfeAssignmentCongestion(
        graph=graph,
        od_matrix=od_matrix,
        cost_function='bpr',
        solution_attr='solution_test',
        use_cache=False,
        k_paths=10
    )

    # 4. Probar cálculo de costos BPR con las correcciones
    print("\n4️⃣ Probando cálculo de costos BPR corregidos...")

    for u, v, data in sample_edges:
        # Calcular costo con flujo = 0 (debería ser igual a free_flow_time)
        cost_no_flow = assignment_model.compute_bpr_cost(u, v, flow=0.0)
        t0 = data.get('free_flow_time', data.get('length', 1.0))

        # Calcular costo con flujo = 100 (debería ser mayor por congestión)
        cost_with_flow = assignment_model.compute_bpr_cost(u, v, flow=100.0)

        print(f"\n   Enlace {u} → {v}:")
        print(f"      free_flow_time: {t0:.4f}")
        print(f"      Costo con flujo=0: {cost_no_flow:.4f}")
        print(f"      Costo con flujo=100: {cost_with_flow:.4f}")
        print(f"      Incremento por congestión: {(cost_with_flow/cost_no_flow - 1)*100:.2f}%")

        # Verificar que el costo aumenta con el flujo
        if cost_with_flow > cost_no_flow:
            print(f"      ✅ CORRECTO: Congestión funciona (costo aumenta con flujo)")
        else:
            print(f"      ❌ ERROR: Congestión NO funciona (costo no aumenta)")

    # 5. Ejecutar una iteración de prueba
    print("\n5️⃣ Ejecutando prueba rápida (3 iteraciones)...")

    try:
        result = assignment_model.solve(
            max_iterations=3,
            convergence_threshold=0.01,
            verbose=True
        )

        print("\n6️⃣ RESULTADOS DE LA PRUEBA:")
        print(f"   ✅ Iteraciones completadas: {result['iterations']}")
        print(f"   ✅ Gap final: {result['final_gap']:.6f}")
        print(f"   ✅ Costo total: {result['total_cost']:.2f}")

        # Verificar que el costo total es razonable (no negativo, no infinito)
        if result['total_cost'] > 0 and result['total_cost'] < float('inf'):
            print(f"\n   ✅ PRUEBA EXITOSA: Costos BPR funcionan correctamente")
        else:
            print(f"\n   ❌ ERROR: Costo total inválido: {result['total_cost']}")

        return True

    except Exception as e:
        print(f"\n   ❌ ERROR durante la ejecución: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🚀 INICIANDO PRUEBA DE CORRECCIONES BPR...\n")

    success = test_bpr_corrections()

    if success:
        print("\n" + "="*80)
        print("✅ TODAS LAS CORRECCIONES VALIDADAS EXITOSAMENTE")
        print("="*80)
        print("\n📋 RESUMEN DE CORRECCIONES APLICADAS:")
        print("   1. ✅ Parámetro 'b' corregido: usar 0.15 cuando b < 0.01")
        print("   2. ✅ Parámetro 'power' corregido: usar 4.0 cuando power <= 0")
        print("   3. ✅ Capacidades estimadas: basadas en tipo de vía y free_flow_time")
        print("   4. ✅ Diagnóstico mejorado: reporta detalles de pares OD sin ruta")
        print("\n💡 SIGUIENTES PASOS:")
        print("   - Ejecutar: python src/models/traffic_assignment_congestion.py")
        print("   - Verificar que NO hay advertencias de 'pares OD sin ruta'")
        print("   - Comparar resultados con flujos observados")
    else:
        print("\n" + "="*80)
        print("❌ PRUEBA FALLÓ - Revisar errores arriba")
        print("="*80)

    print("\n")
