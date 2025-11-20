"""
Script de prueba para verificar el cargador de matrices OD.
Muestra ejemplos de uso del ODMatrixGenerator.
"""
from pathlib import Path
import sys

# Añadir src al path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from od_matrix_generator import ODMatrixGenerator, load_od_matrix # type: ignore
from scipy import sparse
import numpy as np


def test_load_sparse():
    """Prueba carga de matriz sparse desde archivo guardado."""
    print("=== Test: Cargar matriz sparse guardada ===")

    matrix_path = Path('../data/processed/od_matrix_barcelona.npz')
    if matrix_path.exists():
        od_sparse = sparse.load_npz(matrix_path)
        print(f"✓ Matriz cargada: {od_sparse.shape}")
        print(f"  Elementos no-cero: {od_sparse.nnz}")
        print(f"  Tipo: {type(od_sparse)}")

        # Ejemplo: obtener flujo del par OD (1, 74)
        # Recordar: indexación base-0, entonces zona 1 = índice 0
        flow_1_74 = od_sparse[0, 73]  # zona 1 -> zona 74
        print(f"  Flujo zona 1 → zona 74: {flow_1_74}")

        return True
    else:
        print(f"✗ Archivo no encontrado: {matrix_path}")
        return False


def test_load_dataframe():
    """Prueba carga de DataFrame desde parquet."""
    print("\n=== Test: Cargar DataFrame OD ===")

    import pandas as pd
    df_path = Path('../data/processed/od_dataframe_barcelona.parquet')

    if df_path.exists():
        od_df = pd.read_parquet(df_path)
        print(f"✓ DataFrame cargado: {od_df.shape}")
        print(f"\nTop 5 pares OD por flujo:")
        print(od_df.nlargest(5, 'flow'))

        return True
    else:
        print(f"✗ Archivo no encontrado: {df_path}")
        return False


def test_convenience_function():
    """Prueba función de conveniencia load_od_matrix."""
    print("\n=== Test: Función de conveniencia ===")

    try:
        # Cargar como sparse
        od_sparse = load_od_matrix(format='sparse')
        print(f"✓ Sparse matrix: {od_sparse.shape}, {od_sparse.nnz} elementos")

        # Cargar como DataFrame
        od_df = load_od_matrix(format='dataframe')
        print(f"✓ DataFrame: {od_df.shape}")

        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_matrix_operations():
    """Ejemplos de operaciones útiles con la matriz OD."""
    print("\n=== Test: Operaciones con matriz OD ===")

    matrix_path = Path('../data/processed/od_matrix_barcelona.npz')
    if not matrix_path.exists():
        print("✗ Matriz no encontrada")
        return False

    od_sparse = sparse.load_npz(matrix_path)

    # 1. Total de viajes generados por cada zona (suma por fila)
    origins_total = np.array(od_sparse.sum(axis=1)).flatten()
    print(f"\n1. Viajes generados por zona (productions):")
    print(f"   Zona con más viajes generados: {origins_total.argmax() + 1} ({origins_total.max():.2f} viajes)")

    # 2. Total de viajes atraídos por cada zona (suma por columna)
    destinations_total = np.array(od_sparse.sum(axis=0)).flatten()
    print(f"\n2. Viajes atraídos por zona (attractions):")
    print(f"   Zona con más viajes atraídos: {destinations_total.argmax() + 1} ({destinations_total.max():.2f} viajes)")

    # 3. Verificar balance (suma de productions = suma de attractions)
    total_productions = origins_total.sum()
    total_attractions = destinations_total.sum()
    print(f"\n3. Balance de flujos:")
    print(f"   Total productions: {total_productions:.2f}")
    print(f"   Total attractions: {total_attractions:.2f}")
    print(f"   Diferencia: {abs(total_productions - total_attractions):.6f}")

    # 4. Zonas más conectadas (mayor número de destinos con flujo > 0)
    connectivity = (od_sparse > 0).sum(axis=1)
    connectivity = np.array(connectivity).flatten()
    print(f"\n4. Conectividad:")
    print(f"   Zona más conectada: {connectivity.argmax() + 1} ({connectivity.max()} destinos)")

    return True


if __name__ == '__main__':
    print("🧪 Tests de ODMatrixGenerator\n")

    results = []
    results.append(("Cargar sparse", test_load_sparse()))
    results.append(("Cargar DataFrame", test_load_dataframe()))
    results.append(("Función conveniencia", test_convenience_function()))
    results.append(("Operaciones matriz", test_matrix_operations()))

    print("\n" + "="*50)
    print("RESULTADOS:")
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    if all_passed:
        print("\n🎉 Todos los tests pasaron!")
    else:
        print("\n⚠️ Algunos tests fallaron")

