"""
Script de prueba para verificar que DataManager funciona con SiouxFalls.

Ejecuta:
    python src/test_siouxfalls.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager


def test_siouxfalls():
    """Prueba carga de datos de SiouxFalls."""
    print("\n" + "="*80)
    print("🧪 TEST: DataManager con SiouxFalls")
    print("="*80)

    # Crear instancia para SiouxFalls
    manager = DataManager(network_name='SiouxFalls')

    # Cargar y combinar todos los datos
    unified_df, od_matrix = manager.load_and_merge(od_format='sparse')

    # Construir grafo con coordenadas
    graph = manager.build_graph(add_node_coords=True)

    # Verificar que las coordenadas se agregaron
    print("\n📍 VERIFICACIÓN DE COORDENADAS EN NODOS:")
    sample_nodes = list(graph.nodes(data=True))[:5]
    for node_id, attrs in sample_nodes:
        has_coords = 'x' in attrs and 'y' in attrs
        coords_str = f"({attrs.get('x', 'N/A'):.6f}, {attrs.get('y', 'N/A'):.6f})" if has_coords else "No disponibles"
        print(f"   Nodo {node_id}: {coords_str}")

    # Guardar grafo
    saved_path = manager.save_graph_pickle(
        output_path='data/processed/siouxfalls_network.gpickle',
        format='pickle'
    )

    # Resumen
    print("\n" + "="*80)
    print("📊 RESUMEN SIOUXFALLS:")
    print("="*80)
    print(f"   Red: {manager.network_df.shape[0]} enlaces")
    print(f"   Nodos con coordenadas: {len(manager.node_coords_df)}")
    print(f"   Grafo: {graph.number_of_nodes()} nodos, {graph.number_of_edges()} aristas")
    print(f"   Matriz OD: {od_matrix.shape}, {od_matrix.nnz} pares OD")

    # Verificar valores BPR
    print(f"\n🔍 VALORES BPR EN SIOUXFALLS:")
    print(f"   b (media): {unified_df['b'].mean():.4f}")
    print(f"   power (media): {unified_df['power'].mean():.4f}")
    print(f"   capacity (media): {unified_df['capacity'].mean():.2f}")

    print(f"\n✅ Test SiouxFalls completado exitosamente!")
    print(f"   Grafo guardado en: {saved_path}")

    return manager


if __name__ == '__main__':
    try:
        manager = test_siouxfalls()
    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
