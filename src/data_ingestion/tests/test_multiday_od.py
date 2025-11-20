"""
Test y ejemplo de uso del MultidayODMatrixGenerator.

Este script demuestra cómo usar la clase MultidayODMatrixGenerator para procesar
archivos TNTP con matrices OD desagregadas por día y hora.

Formato esperado del archivo:
    <matrix> YYYY-MM-DD HH:MM Origin N
     dest1 : flow1 ; dest2 : flow2 ; ...

El generador:
1. Parsea todas las matrices horarias con timestamps
2. Agrega por día (suma horas 0-23 de cada día)
3. Calcula el promedio diario
"""

import sys
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(src_path))

from data_ingestion.processing_modules.od_matrix_generator import MultidayODMatrixGenerator


def create_test_multiday_file(filepath: Path):
    """
    Crea un archivo de ejemplo con matrices OD multiday para testing.
    """
    content = """<NUMBER OF ZONES> 5
<TOTAL OD FLOW> 1000.0
<END OF METADATA>

<matrix> 2022-09-26 00:00 Origin 1
 1 : 0.0; 2 : 10.0; 3 : 5.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-26 00:00 Origin 2
 1 : 8.0; 2 : 0.0; 3 : 12.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-26 01:00 Origin 1
 1 : 0.0; 2 : 15.0; 3 : 8.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-26 01:00 Origin 2
 1 : 6.0; 2 : 0.0; 3 : 10.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-27 00:00 Origin 1
 1 : 0.0; 2 : 12.0; 3 : 6.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-27 00:00 Origin 2
 1 : 7.0; 2 : 0.0; 3 : 11.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-27 01:00 Origin 1
 1 : 0.0; 2 : 14.0; 3 : 7.0; 4 : 0.0; 5 : 0.0;

<matrix> 2022-09-27 01:00 Origin 2
 1 : 9.0; 2 : 0.0; 3 : 13.0; 4 : 0.0; 5 : 0.0;
"""

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"✓ Archivo de test creado: {filepath}")


def test_multiday_generator():
    """
    Test completo del MultidayODMatrixGenerator.
    """
    print("🧪 Test de MultidayODMatrixGenerator\n")

    # Crear archivo de test
    test_file = Path(__file__).parent / "test_data" / "multiday_trips.tntp"
    create_test_multiday_file(test_file)

    try:
        # Inicializar generador
        print("\n1️⃣ Inicializando generador...")
        generator = MultidayODMatrixGenerator(test_file)
        print(f"   ✓ Generador creado para: {generator.path}")

        # Cargar y procesar datos
        print("\n2️⃣ Cargando y procesando matrices multiday...")
        od_df, metadata = generator.load()

        print(f"\n📋 METADATOS:")
        for key, value in metadata.items():
            if isinstance(value, list):
                print(f"   {key}: {', '.join(str(v) for v in value)}")
            elif isinstance(value, float):
                print(f"   {key}: {value:.2f}")
            else:
                print(f"   {key}: {value}")

        # Mostrar DataFrame resultante
        print(f"\n📊 MATRIZ OD PROMEDIO DIARIA:")
        print(f"   Forma: {od_df.shape}")
        print(f"   Columnas: {list(od_df.columns)}")
        print(f"\n   Datos completos:")
        print(od_df.to_string(index=False))

        # Estadísticas
        print("\n3️⃣ Calculando estadísticas...")
        stats = generator.get_statistics()
        print(f"\n📈 ESTADÍSTICAS:")
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        # Generar matriz sparse
        print("\n4️⃣ Generando matriz dispersa...")
        od_sparse = generator.to_sparse_matrix(format='csr')
        print(f"   ✓ Matriz sparse creada:")
        print(f"      - Forma: {od_sparse.shape}")
        print(f"      - Elementos no-cero: {od_sparse.nnz}")
        print(f"      - Densidad: {od_sparse.nnz / (od_sparse.shape[0] * od_sparse.shape[1]) * 100:.2f}%")

        # Guardar archivos
        print("\n5️⃣ Guardando archivos...")
        output_dir = Path(__file__).parent / "test_data"

        sparse_path = output_dir / "multiday_od_matrix.npz"
        generator.save_sparse(sparse_path, compressed=True)
        print(f"   ✓ Matriz sparse guardada: {sparse_path}")

        df_path = output_dir / "multiday_od_dataframe.parquet"
        generator.save_dataframe(df_path, format='parquet')
        print(f"   ✓ DataFrame guardado: {df_path}")

        # Verificar matrices diarias
        print("\n6️⃣ Matrices diarias individuales:")
        for date, daily_df in generator.daily_matrices.items():
            print(f"\n   📅 {date}:")
            print(f"      - Pares OD: {len(daily_df)}")
            print(f"      - Flujo total: {daily_df['flow'].sum():.2f}")
            print(f"      - Flujo promedio por par: {daily_df['flow'].mean():.2f}")

        print("\n✅ Test completado exitosamente!")

        # Verificación manual
        print("\n🔍 VERIFICACIÓN MANUAL:")
        print("   Día 2022-09-26:")
        print("      - Origin 1 -> Dest 2: hora 00 (10.0) + hora 01 (15.0) = 25.0")
        print("      - Origin 1 -> Dest 3: hora 00 (5.0) + hora 01 (8.0) = 13.0")
        print("   Día 2022-09-27:")
        print("      - Origin 1 -> Dest 2: hora 00 (12.0) + hora 01 (14.0) = 26.0")
        print("      - Origin 1 -> Dest 3: hora 00 (6.0) + hora 01 (7.0) = 13.0")
        print("   Promedio diario:")
        print("      - Origin 1 -> Dest 2: (25.0 + 26.0) / 2 = 25.5")
        print("      - Origin 1 -> Dest 3: (13.0 + 13.0) / 2 = 13.0")

        print("\n   Valores en DataFrame:")
        for _, row in od_df[od_df['origin'] == 1].iterrows():
            print(f"      - Origin {int(row['origin'])} -> Dest {int(row['destination'])}: {row['flow']:.1f}")

    except Exception as e:
        print(f"\n❌ Error en test: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    test_multiday_generator()
