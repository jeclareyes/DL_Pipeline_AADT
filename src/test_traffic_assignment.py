"""
Script de prueba completo para Traffic Assignment con Frank-Wolfe.

Ejecuta el pipeline completo:
1. Carga de datos (red, flujos, matriz OD)
2. Construcción del grafo
3. Asignación de tráfico con Frank-Wolfe
4. Comparación con flujos observados
5. Guardado de resultados

Uso:
    python src/test_traffic_assignment.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.components.models.traditional_TA.traffic_assignment import main as run_traffic_assignment


if __name__ == '__main__':
    """
    Ejecuta el Traffic Assignment completo.
    
    Este script ahora simplemente llama a la función main() de traffic_assignment.py
    que contiene toda la lógica de ejecución.
    """
    try:
        results = run_traffic_assignment()
    except Exception as e:
        print(f"\n❌ Error durante el test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
