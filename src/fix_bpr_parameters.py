"""
Script para corregir los parámetros BPR en los datos de red.

PROBLEMA IDENTIFICADO:
- Los valores de 'b' en Barcelona_net.tntp son extremadamente pequeños (10^-18 a 10^-9)
- Esto hace que la función BPR NO funcione correctamente
- Los valores deberían estar alrededor de 0.15

SOLUCIÓN:
- Escalar los valores de 'b' al rango correcto
- Usar 0.15 para enlaces sin valor válido de 'b'
- Validar capacidades realistas
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_ingestion.data_processing import DataManager


def analyze_bpr_values():
    """Analiza los valores actuales de BPR."""
    print("="*80)
    print("🔍 ANÁLISIS DE VALORES BPR ACTUALES")
    print("="*80)

    manager = DataManager()
    network_df = manager.load_network()

    print(f"\n📊 Estadísticas de 'b':")
    print(f"   Min: {network_df['b'].min():.2e}")
    print(f"   Max: {network_df['b'].max():.2e}")
    print(f"   Mean: {network_df['b'].mean():.2e}")
    print(f"   Median: {network_df['b'].median():.2e}")

    print(f"\n📊 Estadísticas de 'power':")
    print(f"   Min: {network_df['power'].min()}")
    print(f"   Max: {network_df['power'].max()}")
    print(f"   Mean: {network_df['power'].mean():.2f}")
    print(f"   Valores = 0: {(network_df['power'] == 0).sum()}")

    print(f"\n📊 Estadísticas de 'capacity':")
    print(f"   Min: {network_df['capacity'].min()}")
    print(f"   Max: {network_df['capacity'].max()}")
    print(f"   Mean: {network_df['capacity'].mean():.2f}")
    print(f"   Valores únicos: {network_df['capacity'].unique()}")

    # Análisis de free_flow_time
    print(f"\n📊 Estadísticas de 'free_flow_time':")
    print(f"   Min: {network_df['free_flow_time'].min():.2f}")
    print(f"   Max: {network_df['free_flow_time'].max():.2f}")
    print(f"   Mean: {network_df['free_flow_time'].mean():.2f}")

    return network_df


def propose_corrections(network_df):
    """Propone correcciones para los parámetros BPR."""
    print("\n" + "="*80)
    print("💡 CORRECCIONES PROPUESTAS")
    print("="*80)

    # 1. Análisis de 'b'
    print("\n1️⃣ CORRECCIÓN DE PARÁMETRO 'b':")
    print(f"   Problema: Valores de 'b' son {network_df['b'].mean():.2e} (demasiado pequeños)")
    print(f"   Valor estándar BPR: 0.15")

    # Calcular factor de escala necesario
    if network_df['b'].max() > 0:
        scale_factor = 0.15 / network_df['b'].median()
        print(f"   Factor de escala sugerido: {scale_factor:.2e}")

    print(f"\n   Estrategia recomendada:")
    print(f"   a) Usar 0.15 para todos los enlaces (estándar BPR)")
    print(f"   b) Escalar valores actuales al rango [0.10, 0.20]")
    print(f"   c) Usar valores específicos por tipo de vía si están disponibles")

    # 2. Análisis de 'power'
    print("\n2️⃣ CORRECCIÓN DE PARÁMETRO 'power':")
    enlaces_power_zero = (network_df['power'] == 0).sum()
    print(f"   Enlaces con power = 0: {enlaces_power_zero}")
    if enlaces_power_zero > 0:
        print(f"   Recomendación: Cambiar power = 0 → power = 4.0 (estándar BPR)")

    # 3. Análisis de 'capacity'
    print("\n3️⃣ CORRECCIÓN DE CAPACIDAD:")
    if (network_df['capacity'] == 1.0).all():
        print(f"   ⚠️ PROBLEMA: Todas las capacidades son 1.0 (incorrecto)")
        print(f"   Recomendación: Usar capacidades realistas basadas en:")
        print(f"      - Tipo de vía (link_type)")
        print(f"      - Número de carriles (si está disponible)")
        print(f"      - Capacidades típicas: autopista ~2000, arterial ~1000, local ~500")

    # 4. Análisis por tipo de vía
    print("\n4️⃣ ANÁLISIS POR TIPO DE VÍA:")
    if 'link_type' in network_df.columns:
        type_stats = network_df.groupby('link_type').agg({
            'b': ['count', 'mean', 'median'],
            'power': ['mean', 'median'],
            'capacity': ['mean', 'median'],
            'free_flow_time': ['mean', 'median']
        })
        print(type_stats)


def create_correction_recommendations():
    """Crea recomendaciones específicas de corrección."""
    print("\n" + "="*80)
    print("📋 RECOMENDACIONES FINALES")
    print("="*80)

    print("\n✅ ACCIÓN INMEDIATA REQUERIDA:")
    print("\n1. Modificar frank_wolfe_congestion.py:")
    print("   - Cambiar default_b de 0.15 a usar SIEMPRE 0.15")
    print("   - NO confiar en los valores de 'b' del archivo (están incorrectos)")
    print("   - Asegurar que power = 4.0 cuando power <= 0")

    print("\n2. Estimar capacidades realistas:")
    print("   - Usar free_flow_time o length como proxy")
    print("   - Capacidad típica = función(tipo_vía, longitud)")

    print("\n3. Validar resultados:")
    print("   - Ejecutar traffic_assignment_congestion.py")
    print("   - Verificar que V/C ratios sean realistas (0.5 - 1.5)")
    print("   - Comprobar que el costo con congestión > costo sin congestión")

    print("\n" + "="*80)


if __name__ == '__main__':
    print("\n🔧 ANÁLISIS Y CORRECCIÓN DE PARÁMETROS BPR\n")

    # Analizar valores actuales
    network_df = analyze_bpr_values()

    # Proponer correcciones
    propose_corrections(network_df)

    # Crear recomendaciones
    create_correction_recommendations()

    print("\n✅ Análisis completado. Revisar recomendaciones arriba.\n")
