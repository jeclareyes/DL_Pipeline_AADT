"""
Script coordinador para entrenar modelos de asignación de tráfico.

Este script coordina el entrenamiento de diferentes modelos:
- cyclic_model: Modelo original
- cyclic_model_ultra: Modelo mejorado con arquitectura avanzada

Características:
- Carga configuración desde YAML
- Selección de modelo desde configuración
- Gestión automática de rutas de salida
- Soporte para múltiples modelos de manera extensible
"""

from pathlib import Path
import subprocess
import sys
import os
import yaml
from datetime import datetime
from typing import Dict, Any, Optional
import json


# =============================================================================
# REGISTRO DE MODELOS DISPONIBLES
# =============================================================================

AVAILABLE_MODELS = {
    'cyclic_model': {
        'script': 'src/models/Cyclic_Model/train_cyclic_model_deprecated.py',
        'module': 'models.Cyclic_Model.train_cyclic_model',
        'description': 'Modelo original de asignación cíclica',
        'class': 'CyclicODModel'
    },
    'cyclic_model_ultra': {
        'script': 'src/models/Cyclic_Model/train_cyclic_model_deprecated.py',
        'module': 'models.Cyclic_Model.train_cyclic_model',
        'description': 'Modelo mejorado con arquitectura avanzada',
        'class': 'CyclicODModelUltra'
    }
}


def get_next_run_number(output_dir: Path) -> int:
    """Obtiene el siguiente número consecutivo para el run."""
    if not output_dir.exists():
        return 1

    existing_runs = [d for d in output_dir.iterdir() if d.is_dir()]
    if not existing_runs:
        return 1

    # Extraer números de los directorios existentes
    numbers = []
    for run_dir in existing_runs:
        parts = run_dir.name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            numbers.append(int(parts[-1]))

    return max(numbers) + 1 if numbers else 1


def setup_output_directory(config: Dict[str, Any], model_type: str) -> Path:
    """
    Configura el directorio de salida con estructura: outputs/models/{case}/{model_name}_{consecutive}

    Args:
        config: Configuración cargada desde YAML
        model_type: Tipo de modelo a entrenar

    Returns:
        Path del directorio de salida
    """
    # Obtener nombre del caso
    case_name = config.get('routing', {}).get('case',
                          config.get('data', {}).get('network_name', 'default'))

    # Directorio base: outputs/models/{case}
    base_models_dir = Path(config['outputs']['base_dir']) / config['outputs']['models_dir'] / case_name
    base_models_dir.mkdir(parents=True, exist_ok=True)

    # Obtener siguiente número consecutivo
    run_number = get_next_run_number(base_models_dir)

    # Crear directorio para este run: {model_name}_{consecutive}
    run_dir = base_models_dir / f"{model_type}_{run_number:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Crear subdirectorios
    (run_dir / 'checkpoints').mkdir(exist_ok=True)
    (run_dir / 'logs').mkdir(exist_ok=True)
    (run_dir / 'metrics').mkdir(exist_ok=True)

    return run_dir


def save_run_metadata(run_dir: Path, config: Dict[str, Any], model_type: str):
    """Guarda metadata del run para trazabilidad."""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': model_type,
        'model_description': AVAILABLE_MODELS[model_type]['description'],
        'case': config.get('routing', {}).get('case',
                          config.get('data', {}).get('network_name', 'default')),
        'config_summary': {
            'network': config.get('data', {}).get('network_name'),
            'volume_year': config.get('data', {}).get('volume_year'),
            'epochs': config.get('training', {}).get('epochs'),
            'learning_rate': config.get('training', {}).get('learning_rate'),
            'od_rate': config.get('sampling', {}).get('od_rate'),
            'flow_rate': config.get('sampling', {}).get('flow_rate'),
            'k_paths': config.get('network', {}).get('k_paths'),
        },
        'full_config': config
    }

    metadata_path = run_dir / 'run_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"   ✓ Metadata guardada en: {metadata_path}")


def validate_config(config: Dict[str, Any]) -> None:
    """Valida que la configuración tenga todos los campos necesarios."""
    required_sections = ['data', 'model', 'training', 'outputs']

    for section in required_sections:
        if section not in config:
            raise ValueError(f"Sección requerida '{section}' no encontrada en configuración")

    # Validar tipo de modelo
    model_type = config['model'].get('type', 'cyclic_model')
    if model_type not in AVAILABLE_MODELS:
        available = ', '.join(AVAILABLE_MODELS.keys())
        raise ValueError(f"Modelo '{model_type}' no reconocido. Disponibles: {available}")


def load_and_validate_config(config_path: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Carga y valida la configuración desde YAML.

    Args:
        config_path: Ruta al archivo YAML (acepta rutas absolutas o relativas)
        overrides: Parámetros para sobrescribir

    Returns:
        Configuración validada
    """
    # Normalize to Path
    cfg_path = Path(config_path)

    # If the path is not absolute or doesn't exist, try some sensible fallbacks
    if not cfg_path.exists():
        project_root = Path(__file__).resolve().parents[2]

        # Try project root + provided path
        candidate = project_root / config_path
        if candidate.exists():
            cfg_path = candidate
        else:
            # Try project_root/configs/<basename>
            candidate2 = project_root / 'configs' / Path(config_path).name
            if candidate2.exists():
                cfg_path = candidate2
            else:
                # Try just configs/<config_path> relative to CWD
                cwd_candidate = Path.cwd() / config_path
                if cwd_candidate.exists():
                    cfg_path = cwd_candidate

    if not cfg_path.exists():
        # Helpful error listing attempted locations
        attempted = [
            str(Path(config_path).resolve()) if Path(config_path).is_absolute() else str(Path(config_path)),
            str(Path(__file__).resolve().parents[2] / config_path),
            str(Path(__file__).resolve().parents[2] / 'configs' / Path(config_path).name,
                ),
            str(Path.cwd() / config_path)
        ]
        # Deduplicate
        attempted = list(dict.fromkeys(attempted))
        raise FileNotFoundError(
            f"No se encontró el archivo de configuración: '{config_path}'.\n"
            f"Se intentaron las siguientes rutas:\n  - " + "\n  - ".join(attempted)
        )

    # Cargar YAML
    with open(cfg_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Aplicar overrides
    if overrides:
        for key, value in overrides.items():
            if '.' in key:
                # Soportar notación de punto: training.epochs
                parts = key.split('.')
                current = config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = value
            else:
                config[key] = value

    # Validar
    validate_config(config)

    return config


def update_config_with_output_dir(config: Dict[str, Any], run_dir: Path) -> Dict[str, Any]:
    """Actualiza la configuración para usar el directorio de salida específico del run."""
    # Actualizar paths de salida en la configuración
    config['outputs']['models_dir'] = str(run_dir)
    config['outputs']['logs_dir'] = str(run_dir / 'logs')
    config['outputs']['checkpoint_dir'] = str(run_dir / 'checkpoints')
    config['outputs']['metrics_dir'] = str(run_dir / 'metrics')

    return config


def run_training_with_config(config_path: str,
                             model_type: Optional[str] = None,
                             epochs: Optional[int] = None,
                             device: Optional[str] = None,
                             volume_year: Optional[int] = None) -> int:
    """
    Ejecuta entrenamiento con la configuración especificada.

    Args:
        config_path: Ruta al archivo de configuración YAML
        model_type: Tipo de modelo (sobrescribe config)
        epochs: Número de épocas (sobrescribe config)
        device: Device de cómputo (sobrescribe config)
        volume_year: Año de volumen (sobrescribe config)

    Returns:
        Código de salida del proceso
    """
    project_root = Path(__file__).resolve().parents[2]

    # Cargar y validar configuración
    overrides = {}
    if model_type:
        overrides['model.type'] = model_type
    if epochs:
        overrides['training.epochs'] = epochs
    if device:
        overrides['training.device'] = device
    if volume_year:
        overrides['data.volume_year'] = volume_year

    try:
        config = load_and_validate_config(config_path, overrides)
    except Exception as e:
        print(f"❌ Error cargando configuración: {e}")
        return 1

    # Obtener tipo de modelo
    selected_model = config['model'].get('type', 'cyclic_model')
    model_info = AVAILABLE_MODELS[selected_model]

    print("=" * 80)
    print("🚀 COORDINADOR DE ENTRENAMIENTO - MODELOS DE ASIGNACIÓN")
    print("=" * 80)
    print(f"\n📋 Configuración:")
    print(f"   Modelo: {selected_model}")
    print(f"   Descripción: {model_info['description']}")
    print(f"   Caso: {config.get('routing', {}).get('case', config.get('data', {}).get('network_name'))}")
    print(f"   Red: {config['data']['network_name']}")
    print(f"   Año: {config['data'].get('volume_year', 'N/A')}")
    print(f"   Épocas: {config['training']['epochs']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")
    print(f"   OD rate: {config['sampling']['od_rate']}")
    print(f"   Flow rate: {config['sampling']['flow_rate']}")

    # Configurar directorio de salida
    print(f"\n📁 Configurando directorios de salida...")
    run_dir = setup_output_directory(config, selected_model)
    print(f"   ✓ Directorio del run: {run_dir}")

    # Actualizar configuración con paths de salida
    config = update_config_with_output_dir(config, run_dir)

    # Guardar metadata
    save_run_metadata(run_dir, config, selected_model)

    # Guardar configuración actualizada
    updated_config_path = run_dir / 'config_used.yaml'
    with open(updated_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"   ✓ Configuración guardada en: {updated_config_path}")

    # Preparar script de entrenamiento
    script_path = project_root / model_info['script']
    if not script_path.exists():
        print(f"❌ Script no encontrado: {script_path}")
        return 1

    print(f"\n⚙️ Preparando entrenamiento...")
    print(f"   Script: {script_path.relative_to(project_root)}")

    # Construir argumentos
    args = [
        "--config", str(updated_config_path),
        "--model", selected_model
    ]

    # Ejecutar entrenamiento
    return _run_script_with_args(script_path, args, project_root)


def _run_script_with_args(script_path: Path, args: list, cwd: Path) -> int:
    """Ejecuta el script de entrenamiento con los argumentos especificados."""
    src_dir = cwd / "src"

    # Intentar ejecutar como módulo
    try:
        if src_dir in script_path.parents or (script_path.exists() and src_dir in script_path.resolve().parents):
            module_rel = script_path.resolve().relative_to(src_dir.resolve())
            module_parts = module_rel.with_suffix("").parts
            module_name = ".".join(module_parts)
            cmd = [sys.executable, "-m", module_name] + args

            print(f"\n{'='*80}")
            print(f"▶️ Ejecutando entrenamiento como módulo: {module_name}")
            print(f"{'='*80}\n")

            env = os.environ.copy()
            if src_dir.exists():
                env_pythonpath = env.get('PYTHONPATH', '')
                src_path_str = str(src_dir.resolve())
                if env_pythonpath:
                    env['PYTHONPATH'] = src_path_str + os.pathsep + env_pythonpath
                else:
                    env['PYTHONPATH'] = src_path_str

            result = subprocess.run(cmd, cwd=str(cwd), env=env)
            return result.returncode
    except Exception as e:
        print(f"⚠️ No se pudo ejecutar como módulo: {e}")

    # Fallback: ejecutar directamente
    cmd = [sys.executable, str(script_path)] + args

    env = os.environ.copy()
    if src_dir.exists():
        env_pythonpath = env.get('PYTHONPATH', '')
        src_path_str = str(src_dir.resolve())
        if env_pythonpath:
            env['PYTHONPATH'] = src_path_str + os.pathsep + env_pythonpath
        else:
            env['PYTHONPATH'] = src_path_str

    print(f"\n{'='*80}")
    print(f"▶️ Ejecutando entrenamiento (modo directo)")
    print(f"{'='*80}\n")

    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    return result.returncode


def list_available_models():
    """Muestra los modelos disponibles."""
    print("=" * 80)
    print("📚 MODELOS DISPONIBLES")
    print("=" * 80)

    for model_name, model_info in AVAILABLE_MODELS.items():
        print(f"\n🔹 {model_name}")
        print(f"   Descripción: {model_info['description']}")
        print(f"   Clase: {model_info['class']}")
        print(f"   Script: {model_info['script']}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Coordinador de entrenamiento para modelos de asignación de tráfico',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  # Entrenar con configuración de Linköping (modelo especificado en YAML)
  python src/train/run_training.py --config configs/Linköping.yaml

  # Entrenar con un modelo específico
  python src/train/run_training.py --config configs/Linköping.yaml --model cyclic_model_ultra

  # Test rápido con 10 épocas
  python src/train/run_training.py --config configs/Linköping.yaml --epochs 10

  # Listar modelos disponibles
  python src/train/run_training.py --list-models
        """
    )

    parser.add_argument('--config', type=str, default='configs/Linköping.yaml',
                       help='Ruta al archivo de configuración YAML')
    parser.add_argument('--model', type=str, default=None,
                       help='Tipo de modelo a entrenar (sobrescribe YAML)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas (sobrescribe YAML)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device de cómputo: cuda, cpu, auto (sobrescribe YAML)')
    parser.add_argument('--volume-year', type=int, default=None,
                       help='Año de volumen para datos (sobrescribe YAML)')
    parser.add_argument('--list-models', action='store_true',
                       help='Listar modelos disponibles y salir')

    args = parser.parse_args()

    if args.list_models:
        list_available_models()
        sys.exit(0)

    # Ejecutar entrenamiento
    exitcode = run_training_with_config(
        config_path=args.config,
        model_type=args.model,
        epochs=args.epochs,
        device=args.device,
        volume_year=args.volume_year
    )

    sys.exit(exitcode)

"""
python src/train/run_training.py --config configs/Linköping.yaml --model cyclic_model_ultra --epochs 100
"""
