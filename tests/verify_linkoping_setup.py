"""
Script de verificación pre-entrenamiento para Linköping.
Verifica que todos los archivos y configuraciones estén correctos.
"""
import sys
from pathlib import Path
import yaml

def check_file(path, description):
    """Verifica que un archivo existe."""
    p = Path(path)
    if p.exists():
        print(f"   ✓ {description}: {path}")
        return True
    else:
        print(f"   ✗ {description} NO ENCONTRADO: {path}")
        return False

def check_directory(path, description):
    """Verifica que un directorio existe."""
    p = Path(path)
    if p.exists() and p.is_dir():
        print(f"   ✓ {description}: {path}")
        return True
    else:
        print(f"   ✗ {description} NO ENCONTRADO: {path}")
        return False

def main():
    print("=" * 80)
    print("VERIFICACIÓN PRE-ENTRENAMIENTO - LINKÖPING")
    print("=" * 80)

    all_ok = True

    # 1. Verificar archivos de configuración
    print("\n1. ARCHIVOS DE CONFIGURACIÓN")
    all_ok &= check_file("../configs/linkoping.yaml", "Configuración")

    # 2. Verificar scripts
    print("\n2. SCRIPTS DE ENTRENAMIENTO")
    all_ok &= check_file("../src/components/models/Cyclic_Model/cyclic_model.py", "Modelo")
    all_ok &= check_file("../src/components/models/Cyclic_Model/cyclic_model_data_ingestion.py", "Data Loader")
    all_ok &= check_file("../src/components/models/Cyclic_Model/train_cyclic_model.py", "Script de entrenamiento")

    # 3. Verificar datos
    print("\n3. DATOS DE LINKÖPING")
    all_ok &= check_directory("../data/processed/Linköping", "Directorio de datos")

    # Intentar buscar archivos con diferentes encodings
    data_dir = Path("../data/processed")
    linkoping_found = False
    for item in data_dir.iterdir():
        if item.is_dir() and 'link' in item.name.lower():
            print(f"   ✓ Directorio encontrado: {item.name}")
            linkoping_dir = item
            linkoping_found = True

            # Verificar archivos dentro
            graph_files = list(linkoping_dir.glob('*_graph.pkl'))
            if graph_files:
                print(f"      ✓ Grafo: {graph_files[0].name}")
            else:
                print(f"      ✗ Grafo no encontrado")
                all_ok = False

            od_files = list(linkoping_dir.glob('*_od_matrix.npz'))
            if od_files:
                print(f"      ✓ Matriz OD: {od_files[0].name}")
            else:
                print(f"      ✗ Matriz OD no encontrada")
                all_ok = False

            link_files = list(linkoping_dir.glob('*_link_data.parquet'))
            if link_files:
                print(f"      ✓ Datos de enlaces: {link_files[0].name}")
            else:
                print(f"      ✗ Datos de enlaces no encontrados")
                all_ok = False

            routing_cache = linkoping_dir / 'routing_cache'
            if routing_cache.exists():
                print(f"      ✓ Routing cache: {routing_cache.name}")
                route_files = list(routing_cache.glob('*.pkl'))
                if route_files:
                    print(f"         ✓ Rutas: {route_files[0].name}")
                else:
                    print(f"         ✗ Archivo de rutas no encontrado")
                    all_ok = False
            else:
                print(f"      ✗ Routing cache no encontrado")
                all_ok = False

            break

    if not linkoping_found:
        print(f"   ✗ Directorio de Linköping no encontrado")
        all_ok = False

    # 4. Verificar configuración
    print("\n4. CONFIGURACIÓN")
    try:
        with open("../configs/linkoping.yaml", 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        print(f"   ✓ YAML cargado correctamente")
        print(f"      - Red: {config['data']['network_name']}")
        print(f"      - Año: {config['data']['volume_year']}")
        print(f"      - Épocas: {config['training']['epochs']}")
        print(f"      - Device: {config['training']['device']}")
        print(f"      - Flow rate: {config['sampling']['flow_rate']}")
        print(f"      - OD rate: {config['sampling']['od_rate']}")
    except Exception as e:
        print(f"   ✗ Error al cargar configuración: {e}")
        all_ok = False

    # 5. Verificar imports
    print("\n5. DEPENDENCIAS")
    try:
        import torch
        print(f"   ✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"      ✓ CUDA disponible: {torch.cuda.get_device_name(0)}")
        else:
            print(f"      ⚠ CUDA no disponible (usará CPU)")
    except ImportError:
        print(f"   ✗ PyTorch no instalado")
        all_ok = False

    try:
        import networkx as nx
        print(f"   ✓ NetworkX: {nx.__version__}")
    except ImportError:
        print(f"   ✗ NetworkX no instalado")
        all_ok = False

    try:
        import pandas as pd
        print(f"   ✓ Pandas: {pd.__version__}")
    except ImportError:
        print(f"   ✗ Pandas no instalado")
        all_ok = False

    try:
        import numpy as np
        print(f"   ✓ NumPy: {np.__version__}")
    except ImportError:
        print(f"   ✗ NumPy no instalado")
        all_ok = False

    try:
        import scipy
        print(f"   ✓ SciPy: {scipy.__version__}")
    except ImportError:
        print(f"   ✗ SciPy no instalado")
        all_ok = False

    # 6. Directorios de output
    print("\n6. DIRECTORIOS DE OUTPUT")
    output_dirs = [
        "outputs/models",
        "outputs/tables/Linköping",
        "outputs/logs"
    ]

    for dir_path in output_dirs:
        p = Path(dir_path)
        if not p.exists():
            try:
                p.mkdir(parents=True, exist_ok=True)
                print(f"   ✓ Creado: {dir_path}")
            except Exception as e:
                print(f"   ✗ Error al crear {dir_path}: {e}")
                all_ok = False
        else:
            print(f"   ✓ Existe: {dir_path}")

    # RESUMEN FINAL
    print("\n" + "=" * 80)
    if all_ok:
        print("✅ VERIFICACIÓN COMPLETADA - TODO LISTO PARA ENTRENAMIENTO")
        print("=" * 80)
        print("\nPara iniciar el entrenamiento, ejecutar:")
        print("   python src/models/Cyclic_Model/train_cyclic_model.py --config configs/linkoping.yaml")
        print("\nO usar el script rápido:")
        print("   train_linkoping.bat")
        return 0
    else:
        print("❌ VERIFICACIÓN FALLIDA - REVISAR ERRORES ARRIBA")
        print("=" * 80)
        return 1

if __name__ == '__main__':
    sys.exit(main())

