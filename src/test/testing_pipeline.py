"""
Testing pipeline entrypoint.
Reads configs/testing/testing.yaml via hydra, finds checkpoint files under
outputs/runs/{model_to_test}, and evaluates one or multiple models using the
functions in _testing_functions.py using capability-based task dispatch.
"""
import logging
import os
from glob import glob
import torch

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
    # outputs/models/[run_id]
    model_dir = cfg.testing.testing_route

    # Carpeta donde guardaremos los reportes (dentro de la carpeta del modelo)
    results_dir = os.path.join(model_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)

    logging.info(f"Buscando artefactos en: {model_dir}")

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
        instance = cfg.testing.eval_mode.instance_to_test
        if instance == "auto":
            pattern = os.path.join(model_dir, "*_best.pt")
            found = sorted(glob(pattern))
            if found:
                target_files.append(found[0])
                logging.info(f"[MATCH] Modo auto encontró: {os.path.basename(found[0])}")
            else:
                logging.error(f"[ERROR] No se encontraron archivos '*_best.pt' en {model_dir}")
        else:
            file_path = os.path.join(model_dir, instance if instance.endswith(".pt") else f"{instance}.pt")
            if os.path.exists(file_path):
                target_files.append(file_path)
            else:
                logging.error(f"[ERROR] No se encontró el archivo específico: {file_path}")
    else:
        pattern = os.path.join(model_dir, "*_best.pt")
        target_files = sorted(glob(pattern))
        logging.info(f"[BATCH] Se encontraron {len(target_files)} archivos para procesar.")

    if not target_files:
        logging.warning("No hay archivos en target_files. Abortando ejecución.")
        return

    # 3. Ejecución del Test con Logging de Proceso
    # --- MODIFIED SECTION START ---
    from src.data_ingestion.artifact_loaders.training_artifact_loader import TrainingArtifactLoader

    if not target_files:
        logging.warning("No files found in target_files. Aborting execution.")
        return

    # 1. Extract the data path from the first available checkpoint's configuration
    logging.info("Extracting data path from the first checkpoint to initialize ArtifactLoader...")
    first_bundle = torch.load(target_files[0], map_location='cpu', weights_only=False)
    
    # Handle config format (dict vs OmegaConf)
    original_cfg = first_bundle.get("config", {})
    
    if hasattr(original_cfg, "artifact") and getattr(original_cfg.artifact, "path", None):
        data_base_path = original_cfg.artifact.path
    elif isinstance(original_cfg, dict) and "artifact" in original_cfg and "path" in original_cfg["artifact"]:
        data_base_path = original_cfg["artifact"]["path"]
    elif hasattr(original_cfg, "data"):
        data_base_path = original_cfg.data.base_path
    elif isinstance(original_cfg, dict) and "data" in original_cfg:
        data_base_path = original_cfg["data"].get("base_path")
    else:
        logging.error("Could not locate 'data.base_path' or 'artifact.path' in the saved checkpoint config.")
        raise ValueError("Missing data_base_path in model checkpoint.")

    # 2. Instantiate Loader and load raw_data once for all testing tasks
    loader = TrainingArtifactLoader(data_base_path)
    raw_data = loader.load_artifact()
    logging.info("Successfully loaded raw_data into the testing environment.")

    # 3. Execution Loop
    logging.info(f"--- Starting processing of {len(target_files)} files ---")
    for pt_file in target_files:
        logging.info(f"[START] Evaluating file: {os.path.basename(pt_file)}")
        try:
            process_evaluation(
                file_path=pt_file,
                output_dir=results_dir,
                testing_cfg=cfg.testing,
                raw_data=raw_data, # Injecting raw_data here
                resolved_task_names=resolved_task_names,
            )
            logging.info(f"[SUCCESS] Evaluation finished for {os.path.basename(pt_file)}")
            logging.info("Access testing results in: " + results_dir)
        except Exception as e:
            logging.error(f"[CRITICAL] Evaluation failed for {os.path.basename(pt_file)}: {str(e)}")
            import traceback
            traceback.print_exc()
    # --- MODIFIED SECTION END ---

if __name__ == "__main__":
    main()
