import logging
import os
import sys
import io
import numpy as np
import torch
import hydra
from omegaconf import DictConfig

# --- Custom Modules ---
from src.data_ingestion._artifact_loader import TrafficArtifactLoader
from src.components.models.common.adapters import RouteModelAdapter
from src.components.sampling.engine import SamplingEngine
from src.train.trainer import TrafficTrainer
from src.train._pipeline_utils import (
    get_run_filenames,
    generate_training_tasks,
    _inject_prescaling,
    extract_link_types_for_visualization,
    compute_initial_demand_mean,
    apply_model_pre_instantiate_hook,
    evaluate_holdout_metrics,
    persist_pipeline_summary,
)

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

    strict_data = bool(cfg.training.get("strict_data", False))
    link_types = extract_link_types_for_visualization(
        raw_data['link_data'],
        strict_data=strict_data,
    )

    # B) Model Adaptation (Tensor Layer)
    # The Adapter converts raw objects into the specific tensors required by Route-Based models.
    # It handles sparse matrix creation and physical attribute extraction (t0, capacity).
    model_k_paths = int(cfg.model.get('k_paths', 10)) if hasattr(cfg, 'model') else 10
    adapter = RouteModelAdapter(device=device, k_paths=model_k_paths)
    network_params = adapter.transform(
        graph=raw_data['graph'],
        routes_data=raw_data['routes_data']
    )

    # --- NUEVO: Inyectar en network_params ---
    # Lo metemos aquí para que viaje cómodamente hasta el Trainer
    network_params['link_types_vis'] = link_types # TODO: Verificar si esto no es redudante, puesto que en RouteModelAdapter también se inyecta link_types para el modelo. Podríamos unificarlo.

    logging.info("Data successfully adapted to PyTorch Tensors.")

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

    # Optional OD-space projection: dense [N_nodes^2] -> sparse [N_OD with routes]
    # Enabled per-model to preserve backwards compatibility with legacy models.
    use_sparse_od_targets = bool(cfg.model.get('od_target_sparse', False)) if hasattr(cfg, 'model') else False
    if use_sparse_od_targets:
        num_nodes = int(raw_data['od_matrix'].shape[0])
        flat_idx = None
        valid_idx = None

        # Preferred path: map OD pairs using raw node labels from routes.
        # This keeps OD projection in the same space as the OD matrix when graph node indexing differs.
        od_pair_node_labels = network_params.get('od_pair_node_labels', None)
        if od_pair_node_labels is not None and len(od_pair_node_labels) > 0:
            labels_flat = [str(x) for pair in od_pair_node_labels for x in pair]

            def _sort_key(label: str):
                return (0, int(label)) if label.isdigit() else (1, label)

            unique_labels = sorted(set(labels_flat), key=_sort_key)

            if len(unique_labels) == num_nodes:
                label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
                pair_idx = []
                pair_valid = []
                for u, v in od_pair_node_labels:
                    su, sv = str(u), str(v)
                    if su in label_to_idx and sv in label_to_idx:
                        i = label_to_idx[su]
                        j = label_to_idx[sv]
                        pair_idx.append(i * num_nodes + j)
                        pair_valid.append(True)
                    else:
                        pair_idx.append(0)
                        pair_valid.append(False)

                flat_idx = np.asarray(pair_idx, dtype=np.int64)
                valid_idx = np.asarray(pair_valid, dtype=bool)

        # Fallback path: map using graph node index space.
        if flat_idx is None or valid_idx is None:
            od_pair_indices_np = network_params['od_pair_indices'].detach().cpu().numpy()  # [N_OD, 2]
            flat_idx = (od_pair_indices_np[:, 0] * num_nodes + od_pair_indices_np[:, 1]).astype(np.int64)
            valid_idx = (
                (od_pair_indices_np[:, 0] >= 0) & (od_pair_indices_np[:, 0] < num_nodes) &
                (od_pair_indices_np[:, 1] >= 0) & (od_pair_indices_np[:, 1] < num_nodes) &
                (flat_idx >= 0) & (flat_idx < od_vector_np.shape[0])
            )

        od_vector_for_training_np = np.zeros(len(flat_idx), dtype=np.float32)
        observed_od_mask_for_training_np = np.zeros(len(flat_idx), dtype=np.float32)
        od_vector_for_training_np[valid_idx] = od_vector_np[flat_idx[valid_idx]].astype(np.float32)
        observed_od_mask_for_training_np[valid_idx] = observed_od_mask_np[flat_idx[valid_idx]].astype(np.float32)

        invalid_count = int((~valid_idx).sum())
        if invalid_count > 0:
            logging.warning(
                f"Sparse OD projection dropped {invalid_count} invalid OD pairs (outside OD dense matrix range)."
            )
    else:
        od_vector_for_training_np = od_vector_np.astype(np.float32)
        observed_od_mask_for_training_np = observed_od_mask_np.astype(np.float32)

    true_od_t = torch.FloatTensor(od_vector_for_training_np).to(device)

    logging.info(f"Observed Links: {int(observed_flow_mask_np.sum())} / {len(all_flows_np)}")
    logging.info(f"Observed OD Pairs: {int(observed_od_mask_for_training_np.sum())}")

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

    # Align sampled OD supervision mask to the OD space used by the model.
    if train_od_mask_global is None:
        train_od_mask_for_training_np = observed_od_mask_for_training_np.astype(np.float32)
    else:
        sampled_od_mask_np = np.asarray(train_od_mask_global, dtype=np.float32).reshape(-1)
        if use_sparse_od_targets:
            if sampled_od_mask_np.shape[0] == od_vector_np.shape[0]:
                train_od_mask_for_training_np = np.zeros(len(flat_idx), dtype=np.float32)
                train_od_mask_for_training_np[valid_idx] = sampled_od_mask_np[flat_idx[valid_idx]]
            elif sampled_od_mask_np.shape[0] == od_vector_for_training_np.shape[0]:
                train_od_mask_for_training_np = sampled_od_mask_np.astype(np.float32)
            else:
                logging.warning(
                    "Sampled OD mask size does not match dense/sparse OD spaces. Falling back to observed OD mask."
                )
                train_od_mask_for_training_np = observed_od_mask_for_training_np.astype(np.float32)
        else:
            if sampled_od_mask_np.shape[0] == od_vector_for_training_np.shape[0]:
                train_od_mask_for_training_np = sampled_od_mask_np.astype(np.float32)
            else:
                logging.warning(
                    "Sampled OD mask size does not match OD vector. Falling back to observed OD mask."
                )
                train_od_mask_for_training_np = observed_od_mask_for_training_np.astype(np.float32)

    train_od_mask_for_training_np = np.clip(train_od_mask_for_training_np, 0.0, 1.0)
    train_od_mask_for_training_np = train_od_mask_for_training_np * observed_od_mask_for_training_np
    train_od_mask_t = torch.FloatTensor(train_od_mask_for_training_np).to(device)

    # Calculate Global Test Mask (Hold-out set)
    # Logic: Test = Observed - Train
    test_mask_global = np.clip(observed_flow_mask_np - train_mask_global, 0.0, 1.0)

    logging.info(f"Global Train Set size: {int(train_mask_global.sum())} links")
    logging.info(f"Global Test Set size:  {int(test_mask_global.sum())} links")
    logging.info(f"Supervised OD Pairs (Train): {int(train_od_mask_for_training_np.sum())}")

    # ---------------------------------------------------------
    # 3. SPLIT GENERATION STRATEGY
    # ---------------------------------------------------------
    # Here we unify the logic. We create a list of 'tasks'.
    # Each task contains: (suffix, train_mask, val_mask)

    k_folds = int(cfg.training.get("k_folds", 1))
    random_seed = int(cfg.data.get("random_seed", 42))
    if k_folds > 1:
        logging.info(f"Strategy: K-Fold Cross Validation (K={k_folds})")
    else:
        logging.info("Strategy: Standard Training (global train, no validation fold)")

    training_tasks = list(
        generate_training_tasks(
            train_mask_global=train_mask_global,
            k_folds=k_folds,
            random_seed=random_seed,
        )
    )

    # ---------------------------------------------------------
    # 4. UNIFIED EXECUTION LOOP
    # ---------------------------------------------------------
    metrics_history = []
    holdout_history = []

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
            'od_mask': train_od_mask_t,
        }

        task_val_tensors = {
            'mask': torch.FloatTensor(task['val_mask']).to(device)
        }

        # C. Instantiate a Fresh Model (Reset weights)
        # We use **network_params to automatically inject the physics (t0, capacity, masks)

        initial_demand_mean = compute_initial_demand_mean(
            od_vector_np=od_vector_for_training_np,
            observed_od_mask_np=observed_od_mask_for_training_np,
            fallback_value=1.0,
        )

        logging.info(f"Demanda promedio en fracción conocida: {initial_demand_mean:.4f}")

        model_params = network_params.copy()
        model_params['initial_mean'] = initial_demand_mean

        hook_updates = apply_model_pre_instantiate_hook(
            cfg=cfg,
            model_params=model_params,
            context={
                'task': task,
                'network_params': network_params,
                'raw_data': raw_data,
                'initial_demand_mean': initial_demand_mean,
            },
        )
        if hook_updates:
            logging.info(f"Applied model pre-instantiation hook updates: {hook_updates}")

        # task_train_tensors['od'][task_train_tensors['od_mask'] == 0] = initial_demand_mean

        # Inject prescaling using numpy arrays prepared earlier.
        # In strict mode this must fail-fast on errors.
        _inject_prescaling(
            cfg=cfg,
            model_params=model_params,
            all_flows_np=all_flows_np,
            observed_flow_mask_np=observed_flow_mask_np,
            od_vector_np=od_vector_for_training_np,
            observed_od_mask_np=observed_od_mask_for_training_np,
            network_params=network_params
        )

        model = hydra.utils.instantiate(
            cfg.model,
            **model_params,
            _recursive_=False
        ).to(device)

        # D. Execute Training via Trainer
        trainer = TrafficTrainer(cfg, device, output_dir, current_model_name, current_eval_name)

        # The .fit() method handles the loop, early stopping, and saving
        task_score = trainer.fit(
            model,
            network_params,
            task_train_tensors,
            task_val_tensors,
        )
        metrics_history.append(float(task_score))

        logging.info("--- Phase 5: Final Hold-out Evaluation ---")
        holdout_metrics = evaluate_holdout_metrics(
            model=model,
            true_flows_t=true_flows_t,
            train_mask_np=task['train_mask'],
            test_mask_np=test_mask_global,
            device=device,
        )
        holdout_metrics['task'] = task['name']
        holdout_history.append(holdout_metrics)

        if holdout_metrics.get('has_holdout', False):
            logging.info("\n" + "*" * 30)
            logging.info("  FINAL EVALUATION (HOLD-OUT)")
            logging.info("*" * 30)
            logging.info(f"  >> R2:   {holdout_metrics['r2']:.4f}")
            logging.info(f"  >> MAE:  {holdout_metrics['mae']:.2f}")
            logging.info(f"  >> RMSE: {holdout_metrics['rmse']:.2f}")
            logging.info(f"  >> MAPE: {holdout_metrics['mape']:.2f}%")
            logging.info("*" * 30)
        else:
            logging.warning("No hold-out links were available for evaluation.")

        # E. Cleanup to prevent memory leaks in loop
        del model, trainer
        if torch.cuda.is_available() and str(device).startswith("cuda"):
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

    summary = {
        'schema_version': 1,
        'base_model_name': base_model_name,
        'base_eval_name': base_eval_name,
        'device': str(device),
        'k_folds': int(k_folds),
        'num_tasks': int(len(training_tasks)),
        'task_scores': [float(x) for x in metrics_history],
        'score_mean': float(final_score),
        'score_std': float(std_score),
        'holdout_metrics': holdout_history,
    }
    summary_path = persist_pipeline_summary(output_dir, summary)
    logging.info(f"Pipeline metadata saved to: {summary_path}")

    return float(final_score)

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