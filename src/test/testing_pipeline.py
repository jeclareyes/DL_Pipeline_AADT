"""
Testing pipeline entrypoint.
Reads configs/testing/testing.yaml via hydra, finds checkpoint files under
outputs/runs/{model_to_test}, and evaluates one or multiple models using the
functions in _testing_functions.py using capability-based task dispatch.
"""
import logging
import os
from glob import glob

import hydra
from omegaconf import DictConfig

from src.contracts.runtime_contracts import ContractError, resolve_testing_dispatch_plan

# Importamos la función orquestadora desde tu archivo de funciones
from src.test._testing_functions import process_evaluation


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """
    Pipeline de Testing:
    Lee archivos 'eval_*.pt', calcula métricas y genera gráficos.
    """
    logging.basicConfig(level=logging.INFO)

    # 1. Configuración de Rutas
    # outputs/runs/NombreModelo
    model_dir = os.path.join(cfg.testing.root, cfg.testing.model_to_test)

    # Carpeta donde guardaremos los reportes (dentro de la carpeta del modelo)
    results_dir = os.path.join(model_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Iniciando Testing Pipeline en: {model_dir}")

    # Dispatch preflight: resolve callable tasks once before processing files.
    from src.test import evaluation_tasks

    available_task_names = sorted(
        name
        for name, obj in vars(evaluation_tasks).items()
        if callable(obj) and not name.startswith("_") and getattr(obj, "__module__", "") == evaluation_tasks.__name__
    )

    dispatch_plan = resolve_testing_dispatch_plan(
        cfg.testing,
        available_task_names=available_task_names,
    )
    resolved_task_names = list(dispatch_plan["tasks_callable"])

    if dispatch_plan["tasks_unknown"]:
        logging.warning(
            "Unknown tasks in capability_dispatch were ignored: %s",
            dispatch_plan["tasks_unknown"],
        )
    if dispatch_plan["capabilities_unused"]:
        logging.warning(
            "Unused capabilities defined in capability_dispatch for model '%s': %s",
            dispatch_plan["model_key"],
            dispatch_plan["capabilities_unused"],
        )

    logging.info(
        "Dispatch preflight | model=%s | capabilities=%s | tasks=%s",
        dispatch_plan["model_key"],
        dispatch_plan["capabilities_selected"],
        resolved_task_names,
    )

    target_files = []

    # 2. Lógica de Selección de Archivos
    if cfg.testing.eval_mode.single_model:
        # Modo Single: Buscamos un archivo específico
        instance_name = cfg.testing.eval_mode.instance_to_test

        # Asegurarnos de que buscamos el archivo 'eval_', no el checkpoint puro
        if not instance_name.startswith("eval_"):
            eval_name = f"eval_{instance_name}"
        else:
            eval_name = instance_name

        file_path = os.path.join(model_dir, eval_name)

        if os.path.exists(file_path):
            target_files.append(file_path)
        else:
            logging.error(f" No se encontró el archivo específico: {file_path}")
            # Intento de fallback: buscar si el usuario puso el nombre sin extensión
            if not file_path.endswith(".pt"):
                if os.path.exists(file_path + ".pt"):
                    target_files.append(file_path + ".pt")

    else:
        # Modo Batch: Todos los eval_*.pt en la carpeta
        pattern = os.path.join(model_dir, "eval_*.pt")
        target_files = glob(pattern)
        logging.info(f"Modo Batch: Se encontraron {len(target_files)} modelos para evaluar.")

    if not target_files:
        logging.warning("No hay archivos para procesar. Verifica paths y prefijos 'eval_'.")
        return

    # 3. Ejecución del Test
    for pt_file in target_files:
        try:
            process_evaluation(
                pt_file,
                results_dir,
                cfg.testing,
                resolved_task_names=resolved_task_names,
            )
        except ContractError as e:
            logging.error(f"Contract violation evaluating {os.path.basename(pt_file)}: {str(e)}")
        except Exception as e:
            logging.error(f"Error evaluando {os.path.basename(pt_file)}: {str(e)}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
