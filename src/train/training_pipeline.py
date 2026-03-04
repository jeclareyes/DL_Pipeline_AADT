import logging
import os
import sys
import io
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold

# --- Custom Modules ---
from src.data_ingestion._artifact_loader import TrafficArtifactLoader
from src.components.models.common.adapters import RouteModelAdapter
from src.components.sampling.engine import SamplingEngine
from src.train.trainer import TrafficTrainer
from src.train._pipeline_utils import get_run_filenames, compute_od_cost_prior

# Fix for Windows console encoding
if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def run_pipeline(cfg: DictConfig):
    """
    Main orchestration function for the training pipeline.

    Responsibilities:
    1. Orchestrate Data Loading (Disk -> Memory).
    2. Adapt Data to Model Tensors (Memory -> PyTorch).
    3. Perform Sampling (Train/Test Split).
    4. Delegate Training Loop to the Trainer class (Standard or K-Fold).

    If K=1, the list has 1 task (Train Global vs Test Global).
    If K>1, the list has K tasks (Sub-Train vs Validation).
    """

    # ---------------------------------------------------------
    # 0. SETUP & INITIALIZATION
    # ---------------------------------------------------------
    # Hydra changes the working directory to the output folder automatically.
    output_dir = cfg.runs.dir

    # Determine computation device (CPU/GPU)
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logging.info(f"Pipeline started at: {output_dir} | Device: {device}")

    # Generate descriptive filenames for artifacts using helper utility
    # This keeps the logic consistent and clean.
    base_model_name, base_eval_name = get_run_filenames(cfg)
    logging.info(f"Experiment Base Name: {base_model_name}")

    # ---------------------------------------------------------
    # 1. DATA INGESTION & ADAPTATION
    # ---------------------------------------------------------
    logging.info("--- Phase 1: Data Ingestion & Feature Engineering ---")

    # A) Artifact Loading (I/O Layer)
    # The Loader is agnostic; it just retrieves Python objects (DFs, Graphs) from disk.
    loader = TrafficArtifactLoader(cfg.data.base_path)
    raw_data = loader.load_all()  # Returns dict with 'graph', 'od_matrix', 'link_data', 'routes_data'

    # --- NUEVO: Extraer Link Types para Visualización ---
    # Intentamos extraer la columna, con un fallback por si no existe en el dataset
    try:
        if 'link_type' in raw_data['link_data'].columns:
            link_types = raw_data['link_data']['link_type'].values.astype(str)
        else:
            logging.warning("Columna 'link_type' no encontrada. Usando 'Unknown'.")
            link_types = np.array(['Unknown'] * len(raw_data['link_data']))
    except Exception as e:
        logging.warning(f"No se pudieron extraer link_types: {e}")
        link_types = np.array(['Unknown'] * len(raw_data['link_data']))
    # ----------------------------------------------------

    # B) Model Adaptation (Tensor Layer)
    # The Adapter converts raw objects into the specific tensors required by Route-Based models.
    # It handles sparse matrix creation and physical attribute extraction (t0, capacity).
    adapter = RouteModelAdapter(device=device)
    network_params = adapter.transform(
        graph=raw_data['graph'],
        routes_data=raw_data['routes_data']
    )

    # --- NUEVO: Inyectar en network_params ---
    # Lo metemos aquí para que viaje cómodamente hasta el Trainer
    network_params['link_types_vis'] = link_types # TODO: Verificar si esto no es redudante, puesto que en RouteModelAdapter también se inyecta link_types para el modelo. Podríamos unificarlo.

    logging.info("Data successfully adapted to PyTorch Tensors.")

    # Extraer num_nodes del grafo o link_data
    graph_obj = raw_data['graph']  # El objeto NetworkX original  # O max(max(u), max(v)) + 1
    od_indices = network_params['od_pair_indices']  # Tensor [Num_OD, 2]

    # Calcular el vector de costos
    t0_od_costs = compute_od_cost_prior(
        link_df=raw_data['link_data'],
        od_indices_tensor=od_indices,
        graph_obj=graph_obj,  # Pasamos el grafo entero
        device=device
    )

    # C) Global Vector Preparation (Standardization)
    # We extract the observed flows and OD demands into flat vectors for the Loss function.

    # C.1: Prepare Flows
    link_df = raw_data['link_data']
    target_year = cfg.data.get('volume_year', 2022)
    if target_year is False:
        logging.info("No 'volume_year' specified. Going with default \"volume\" column.")
        vol_col = f'volume'
    else:  
        vol_col = f'Volume_{target_year}'

    # Fallback mechanism if specific year column is missing
    if vol_col not in link_df.columns and 'flow' in link_df.columns:
        vol_col = 'flow'
        logging.warning(f"Column '{vol_col}' not found. Falling back to 'flow'.")

    if vol_col not in link_df.columns:
        raise ValueError(f"Could not find volume data for year {target_year}")

    flows_raw = link_df[vol_col].values
    all_flows_np = np.nan_to_num(flows_raw, nan=0.0)
    # Mask: 1.0 if data exists, 0.0 if NaN (Unobserved)
    observed_flow_mask_np = (~np.isnan(flows_raw)).astype(np.float32)

    # C.2: Prepare OD Demand
    od_sparse = raw_data['od_matrix']
    od_dense = od_sparse.toarray().flatten()
    od_vector_np = np.nan_to_num(od_dense, nan=0.0)
    observed_od_mask_np = (~np.isnan(od_dense)).astype(np.float32)

    # Move global reference tensors to GPU once
    true_flows_t = torch.FloatTensor(all_flows_np).to(device)
    true_od_t = torch.FloatTensor(od_vector_np).to(device)

    # We always use the global observed OD mask for loss (assuming we trust known ODs)
    observed_od_mask_t = torch.FloatTensor(observed_od_mask_np).to(device)

    logging.info(f"Observed Links: {int(observed_flow_mask_np.sum())} / {len(all_flows_np)}")
    logging.info(f"Observed OD Pairs: {int(observed_od_mask_np.sum())}")

    # ---------------------------------------------------------
    # 2. GLOBAL SAMPLING
    # ---------------------------------------------------------
    logging.info("--- Phase 2: Sampling Strategy ---")
    # Decide which observed links are Global Train vs Global Test (Hold-out)
    sampling_engine = SamplingEngine(config=cfg)

    train_mask_global, train_od_mask_global = sampling_engine.run(
        override_graph=raw_data['graph'],
        override_observed_flow_mask=observed_flow_mask_np,
        override_observed_od_mask=observed_od_mask_np
    )

    # Calculate Global Test Mask (Hold-out set)
    # Logic: Test = Observed - Train
    test_mask_global = np.clip(observed_flow_mask_np - train_mask_global, 0.0, 1.0)

    logging.info(f"Global Train Set size: {int(train_mask_global.sum())} links")
    logging.info(f"Global Test Set size:  {int(test_mask_global.sum())} links")

    # ---------------------------------------------------------
    # 3. SPLIT GENERATION STRATEGY
    # ---------------------------------------------------------
    # Here we unify the logic. We create a list of 'tasks'.
    # Each task contains: (suffix, train_mask, val_mask)

    k_folds = cfg.training.get("k_folds", 1)
    training_tasks = []

    if k_folds > 1:
        logging.info(f"Strategy: K-Fold Cross Validation (K={k_folds})")
        # In K-Fold, we split the Global Train set into Sub-Train and Validation
        train_indices = np.where(train_mask_global > 0)[0]

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=cfg.data.get("random_seed", 42))

        for fold, (t_idx, v_idx) in enumerate(kf.split(train_indices)):
            # Create Fold Masks
            sub_train_mask = np.zeros_like(train_mask_global)
            val_mask = np.zeros_like(train_mask_global)

            sub_train_mask[train_indices[t_idx]] = 1.0
            val_mask[train_indices[v_idx]] = 1.0

            # Append Task
            training_tasks.append({
                "name": f"Fold {fold+1}",
                "suffix": f"_fold{fold}.pt",
                "train_mask": sub_train_mask,
                "val_mask": val_mask
            })
    else:
        logging.info("Strategy: Standard Training (Global Train, NO Validation, Global Test as Hold-out)")
        # En Standard mode (K=1), entrenamos sin validación
        training_tasks.append({
            "name": "Standard Run",
            "suffix": ".pt",
            "train_mask": train_mask_global,
            "val_mask": np.zeros_like(train_mask_global)  # 0.0 para indicar que no hay validación
        })

    # ---------------------------------------------------------
    # 4. UNIFIED EXECUTION LOOP
    # ---------------------------------------------------------
    metrics_history = []

    for task in training_tasks:
        logging.info(f"--- Starting: {task['name']} ---")

        # A. Resolve Filenames for this task
        current_model_name = base_model_name.replace(".pt", task['suffix'])
        current_eval_name = base_eval_name.replace(".pt", task['suffix'])

        # B. Prepare Tensors for this specific task
        task_train_tensors = {
            'flows': true_flows_t,
            'od': true_od_t,
            'mask': torch.FloatTensor(task['train_mask']).to(device),
            'od_mask': observed_od_mask_t # Usually we use all known ODs for training
        }

        task_val_tensors = {
            'mask': torch.FloatTensor(task['val_mask']).to(device)
        }

        # C. Instantiate a Fresh Model (Reset weights)
        # We use **network_params to automatically inject the physics (t0, capacity, masks)

        # --- CALCULAR EL PROMEDIO INICIAL ---
        # Solo sumamos los valores conocidos y dividimos por la cantidad de conocidos
        # Evitamos dividir por cero con epsilon
        sum_known = (od_vector_np * observed_od_mask_np).sum()
        count_known = observed_od_mask_np.sum()
        # Evitar división por cero explícita
        if count_known > 0:
            initial_demand_mean = float(sum_known / count_known)
        else:
            initial_demand_mean = 1.0  # Fallback seguro
            logging.warning("No hay ODs observados para calcular la media inicial.")

        logging.info(f"Demanda promedio en fracción conocida: {initial_demand_mean:.4f}")

        model_params = network_params.copy()
        model_params['initial_mean'] = initial_demand_mean

        # task_train_tensors['od'][task_train_tensors['od_mask'] == 0] = initial_demand_mean

        # --- NUEVO: INJECTAR PRE-SCALING SEGUN ESTRATEGIA CONFIGURADA ---
        try:
            from src.train._pipeline_utils import _inject_prescaling
            # Inject prescaling using numpy arrays prepared earlier
            link_scale_val, od_scale_val = _inject_prescaling(
                cfg=cfg,
                model_params=model_params,
                all_flows_np=all_flows_np,
                observed_flow_mask_np=observed_flow_mask_np,
                od_vector_np=od_vector_np,
                observed_od_mask_np=observed_od_mask_np,
                network_params=network_params
            )
            # model_params['link_scale'] = link_scale_val
            # model_params['od_scale'] = od_scale_val
            model_params['link_scale'] = 1.0
            model_params['od_scale'] = 1.0
        except Exception as e:
            logging.warning(f"Could not inject prescaling: {e}")

        model = hydra.utils.instantiate(
            cfg.model,
            **model_params,
            _recursive_=False
        ).to(device)

        # D. Execute Training via Trainer
        trainer = TrafficTrainer(cfg, device, output_dir, current_model_name, current_eval_name)

        # The .fit() method handles the loop, early stopping, and saving
        best_mse = trainer.fit(model,
                               network_params,
                               task_train_tensors,
                               task_val_tensors,
                               t0_od_costs=t0_od_costs)

        metrics_history.append(best_mse)

        # --- NUEVO: EVALUACIÓN FINAL DE HOLD-OUT (UNBIASED) ---
        logging.info("--- Phase 5: Final Hold-out Evaluation ---")
        with torch.no_grad():
            model.eval()  # Aseguramos modo evaluación

            # 1. Forward pass completo
            test_results = model(
                observed_flows=true_flows_t,
                flow_mask=torch.FloatTensor(task['train_mask']).to(device)
            )

            y_pred_all = test_results['reconstructed_flows'].detach().cpu().numpy().flatten()  #
            y_true_all = true_flows_t.detach().cpu().numpy().flatten()
            test_mask = test_mask_global.flatten().astype(bool)  # El hold-out puro

            # 2. Filtrar solo los datos del Hold-out (links que el modelo nunca vio)
            y_pred = y_pred_all[test_mask]
            y_true = y_true_all[test_mask]

            if len(y_true) > 0:
                # --- Cálculo de Métricas ---
                # MAE (Mean Absolute Error)
                mae = np.mean(np.abs(y_pred - y_true))

                # RMSE (Root Mean Squared Error)
                mse = np.mean((y_true - y_pred) ** 2)
                rmse = np.sqrt(mse)

                # R2 (Coefficient of Determination)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                r2 = 1 - (ss_res / (ss_tot + 1e-8))

                # MAPE (Mean Absolute Percentage Error)
                non_zero = y_true != 0
                mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100 if np.any(
                    non_zero) else 0.0

                # --- Reporte Final ---
                logging.info("\n" + "*" * 30)
                logging.info("  FINAL EVALUATION (HOLD-OUT)")
                logging.info("*" * 30)
                logging.info(f"  >> R2:   {r2:.4f}")
                logging.info(f"  >> MAE:  {mae:.2f}")
                logging.info(f"  >> RMSE: {rmse:.2f}")
                logging.info(f"  >> MAPE: {mape:.2f}%")
                logging.info("*" * 30)
            else:
                logging.warning("No se encontraron datos en el test_mask_global para evaluar.")

        # E. Cleanup to prevent memory leaks in loop
        del model, trainer
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 5. FINAL REPORTING
    # ---------------------------------------------------------
    final_score = np.mean(metrics_history)
    std_score = np.std(metrics_history)

    logging.info("\n" + "="*40)
    logging.info(f"PIPELINE COMPLETED.")
    logging.info(f"Final Average MSE: {final_score:.6f} (+/- {std_score:.6f})")
    logging.info("="*40)

    return final_score

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    # Para acceder a la configuración de Hydra en tiempo de ejecución, usamos HydraConfig
    from hydra.core.hydra_config import HydraConfig
    from hydra.types import RunMode
    try:
        hydra_cfg = HydraConfig.get()
        # Verificar si mode es MULTIRUN (ya sea Enum o String, por seguridad)
        is_multirun = hydra_cfg.mode == RunMode.MULTIRUN or str(hydra_cfg.mode) == "MULTIRUN"
        
        if cfg.training.get("tuning") and not is_multirun:
            logging.warning("MODO TUNING ACTIVADO - MODO MULTIRUN (-m) NO ACTIVADO.")
    except Exception:
        pass

    return run_pipeline(cfg)

if __name__ == "__main__":
    main()