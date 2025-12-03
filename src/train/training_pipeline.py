import logging
import os
import sys
import io
import numpy as np
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import KFold  # [NUEVO] Import necesario

# Importaciones de estructura
from src.components.models.Cyclic_Model.cyclic_model_data_ingestion import LinkopingDataLoader
from src.components.sampling.engine import SamplingEngine
from src.train._saving_handler import save_checkpoint

if os.name == 'nt':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Función auxiliar para instanciar Loss (reutilizada para evitar duplicidad)
def get_loss_function(cfg, device):
    if hasattr(cfg.model, 'loss') and '_target_' in cfg.model.loss:
        return hydra.utils.instantiate(cfg.model.loss).to(device)
    else:
        from src.components.models.Cyclic_Model.cyclic_model import PartialDataLoss
        return PartialDataLoss(
            w_flow=cfg.model.loss_weights.w_flow,
            w_od=cfg.model.loss_weights.w_od,
            w_reg=cfg.model.loss_weights.w_reg
        ).to(device)

def run_pipeline(cfg: DictConfig):
    # Hydra ya ha cambiado el directorio de trabajo a la carpeta de salida configurada en config.yaml
    output_dir = cfg.runs.dir
    logging.info(f"Directorio de salida del run: {output_dir}")

    # Configurar device
    device = cfg.training.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Usando device: {device}")

    logging.info(f"[HYDRA] Iniciando Pipeline. Modelo: {cfg.model._target_}")

    # ---------------------------------------------------------
    # 1. CONSTRUCCIÓN DEL NOMBRE DEL ARCHIVO (Largo y descriptivo)
    # ---------------------------------------------------------
    # Usamos getattr o get para evitar errores si alguna clave no existe
    # Asumimos que 'network.cost_function' se refiere a cfg.network.cost_function
    # Si tu config usa 'vdf' en lugar de 'network', ajusta abajo (ej. cfg.vdf.name)

    base_name = (
        f"Epochs_{cfg.training.epochs}_"
        f"VDF_{cfg.get('network', {}).get('cost_function', 'UnknownVDF')}_"
        f"Learning_Rate_{cfg.training.lr}_"
        f"Flow_Sampling_Rate_{cfg.sampling.flow_rate}_"
        f"Sampling_Strategy_{cfg.sampling.strategy}_"
        f"Sampling_Basis_{cfg.sampling.sampling_basis}"
    )

    model_filename = f"{base_name}.pt"
    eval_filename = (f"eval_{base_name}.pt")

    full_model_path = os.path.join(output_dir, model_filename)
    full_eval_path = os.path.join(output_dir, eval_filename)

    logging.info(f"El entrenamiento se guardará en: {model_filename}")
    logging.info(f"Las evaluaciones se guardarán en: {eval_filename}")

    # ---------------------------------------------------------
    # 1. DATA INGESTION (Load Universe & Observed)
    # ---------------------------------------------------------
    print("Loading Data...")
    # config entera. El loader sacará las rutas de ahí.
    loader = LinkopingDataLoader(cfg)
    loader.load_all()
    network_params = loader.prepare_network_parameters() # tensores estructurales (t0, capacity, masks)

    # A) Flujos: Universo total y máscara de observados (Known)
    all_flows, observed_flow_mask = loader.prepare_observed_flows()

    # B) OD: Universo total y máscara de observados (Known)
    od_vector, observed_od_mask = loader.prepare_od_demand_vector()

    # C) Calcular UNOBSERVED (Unknown) por complemento
    # Unobserved = 1 - Observed
    unobserved_flow_mask = 1.0 - observed_flow_mask
    unobserved_od_mask = 1.0 - observed_od_mask

    # ---------------------------------------------------------
    # 2. SAMPLING (Split Observed into Training & Testing)
    # ---------------------------------------------------------
    logging.info("Ejecutando Sampling Engine (Train/Test Split)...")
    sampling_engine = SamplingEngine(config=cfg)

    # El engine recibe LO OBSERVADO y devuelve LO DE ENTRENAMIENTO
    train_flow_mask_global, train_od_mask_global = sampling_engine.run(
        override_graph=loader.graph,
        override_observed_flow_mask=observed_flow_mask,
        override_observed_od_mask=observed_od_mask
    )

    # Calcular TESTING por sustracción:
    # Testing = Observed - Training
    # (Matemáticamente seguro porque train es subset de observed)
    test_flow_mask = observed_flow_mask - train_flow_mask_global
    test_flow_mask = np.clip(test_flow_mask, 0.0, 1.0)  # Validación de seguridad (evitar -1 por errores de redondeo)

    logging.info(f"Total Enlaces Train (Global): {int(train_flow_mask_global.sum())}")
    logging.info(f"Total Enlaces Test (Hold-out): {int(test_flow_mask.sum())}")

    true_flows_t = torch.FloatTensor(all_flows).to(device)
    true_od_t = torch.FloatTensor(od_vector).to(device)

    # test_od_mask = observed_od_mask - train_od_mask_global TODO: pendiente
    # test_od_mask = np.clip(test_od_mask, 0.0, 1.0)

    # ---------------------------------------------------------
    # 3. RESUMEN DE DATASETS (sanity check)
    # ---------------------------------------------------------
    """from src.train._pipeline_utils import log_dataset_summary

    # For LINKS
    log_dataset_summary("LINKS", len(all_flows), observed_flow_mask, train_flow_mask_global, test_flow_mask,
                        unobserved_flow_mask, logging)

    logging.info("-" * 40)

    # For OD PAIRS
    log_dataset_summary("OD PAIRS", len(od_vector), observed_od_mask, train_od_mask_global, test_od_mask, unobserved_od_mask,
                        logging)"""

    # ---------------------------------------------------------
    # 3. K-FOLD CROSS VALIDATION
    # ---------------------------------------------------------
    # Si no se define k_folds en config, usamos 1 (entrenamiento normal sin validación cruzada)
    k_folds = cfg.training.get("k_folds", 1)

    if k_folds > 1:
        logging.info(f"Iniciando K-Fold Cross Validation con K={k_folds}")

        # Obtenemos los ÍNDICES de los enlaces que pertenecen al Train Set Global
        train_indices_flat = np.where(train_flow_mask_global > 0)[0]

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=cfg.data.get("random_seed", 42))

        fold_metrics = []

        for fold, (train_idx_local, val_idx_local) in enumerate(kf.split(train_indices_flat)):
            logging.info(f"\n{'='*40}")
            logging.info(f"INICIANDO FOLD {fold + 1} / {k_folds}")
            logging.info(f"{'='*40}")

            # --- 1. PREPARACIÓN DE RUTAS Y MÁSCARAS ---

            # Definir nombres únicos para este fold (mantiene compatibilidad con testing_pipeline)
            fold_suffix = f"_fold{fold}"
            fold_model_path = os.path.join(output_dir, model_filename.replace('.pt', f'{fold_suffix}.pt'))
            fold_eval_path = os.path.join(output_dir, eval_filename.replace('.pt', f'{fold_suffix}.pt'))

            # Crear máscaras dinámicas para este fold
            indices_sub_train = train_indices_flat[train_idx_local]
            indices_val = train_indices_flat[val_idx_local]

            fold_train_mask = np.zeros_like(train_flow_mask_global)
            fold_val_mask = np.zeros_like(train_flow_mask_global)

            fold_train_mask[indices_sub_train] = 1.0
            fold_val_mask[indices_val] = 1.0

            # Enviar a GPU
            fold_train_mask_t = torch.FloatTensor(fold_train_mask).to(device)
            fold_val_mask_t = torch.FloatTensor(fold_val_mask).to(device)
            # Usamos la máscara OD global de entrenamiento para calcular loss
            fold_od_mask_t = torch.FloatTensor(train_od_mask_global).to(device)

            # --- 2. INICIALIZACIÓN DE ESTRUCTURAS DE GUARDADO ---
            # Estas estructuras son idénticas a las del modo sin k-folds
            # para que _testing_functions.py pueda leerlas sin cambios.
            fold_checkpoint = {
                'config': OmegaConf.to_container(cfg, resolve=True),
                'epochs_history': {}
            }

            fold_eval_bundle = {
                'config': OmegaConf.to_container(cfg, resolve=True),
                'static_data': {
                    'true_flows': torch.FloatTensor(all_flows).cpu(),
                    'true_od': torch.FloatTensor(od_vector).cpu(),
                    'capacity': network_params['capacity'].cpu(),
                    't0': network_params['t0'].cpu(),
                    'link_group': network_params['link_group'].cpu(),
                    'masks': {
                        'flow_observed': torch.BoolTensor(observed_flow_mask).cpu(),
                        # Guardamos las máscaras ESPECÍFICAS de este fold
                        # para saber qué se usó para entrenar y qué para validar
                        'flow_train': torch.BoolTensor(fold_train_mask).cpu(),
                        'flow_test': torch.BoolTensor(fold_val_mask).cpu(), # Aquí Test actúa como Validation del fold
                        'od_observed': torch.BoolTensor(observed_od_mask).cpu(),
                    }
                },
                'epochs_history': {}
            }

            # --- 3. INSTANCIACIÓN DEL MODELO (Desde Cero) ---
            model = hydra.utils.instantiate(
                cfg.model,
                num_links=network_params['num_links'],
                num_od_pairs=network_params['num_od_pairs'],
                t0=network_params['t0'].to(device),
                capacity=network_params['capacity'].to(device),
                route_masks=network_params['route_masks'].to(device),
                od_pair_indices=network_params['od_pair_indices'].to(device),
                num_link_groups=network_params['num_link_groups'],
                link_group=network_params['link_group'].to(device),
                _recursive_=False
            ).to(device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.training.lr,
                weight_decay=cfg.training.weight_decay
            )
            criterion = get_loss_function(cfg, device)

            # A. INSTANCIAR SCHEDULER
            # Usamos la config para decidir cuál usar. Por defecto ReduceLROnPlateau es robusto.
            scheduler = None
            if cfg.training.scheduler == "reduce_on_plateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=cfg.training.scheduler_params.factor,  # Ej: 0.5
                    patience=cfg.training.scheduler_params.patience,  # Ej: 10
                    min_lr=cfg.training.scheduler_params.min_lr  # Ej: 1e-6
                )

            # Variables para Early Stopping
            patience = cfg.training.early_stopping.patience
            epochs_no_improve = 0
            best_val_loss = float('inf')
            early_stop = False

            # B. AMP SCALER (Opcional, para velocidad)
            scaler = torch.cuda.amp.GradScaler(enabled=(device == 'cuda'))

            # --- 4. LOOP DE ENTRENAMIENTO ---
            best_val_loss = float('inf')

            model.train()
            for epoch in range(cfg.training.epochs):

                if early_stop:
                    logging.info(f"Early Stopping activado en época {epoch}")
                    # break

                model.train()
                optimizer.zero_grad()

                # C. LOSS WARMUP (Estrategia Dinámica)
                # Durante las primeras 20 épocas, anulamos el peso de OD para que aprenda física básica primero
                current_w_od = 0.0 if epoch < 20 else cfg.model.loss_weights.w_od

                # Actualizar el peso en la función de pérdida si es modificable
                if hasattr(criterion, 'w_od'):
                    criterion.w_od = torch.tensor(current_w_od, device=device)

                # D. CURRICULUM LEARNING (SUE Iterations)
                # Aumentamos iteraciones gradualmente: 2 -> 5 -> 10
                current_iters = int(min(10, 2 + epoch * 0.2))

                # Forward con AMP (Mixed Precision)
                with torch.cuda.amp.autocast(enabled=(device == 'cuda')):
                    outputs = model(
                        observed_flows=true_flows_t,
                        flow_mask=fold_train_mask_t,
                        true_od_demand=true_od_t,
                        warmup=(epoch < 5),
                        override_max_iters=current_iters  # Pasamos curriculum
                    )

                    loss_dict = criterion(
                        predicted_flows=outputs['reconstructed_flows'],
                        true_flows=true_flows_t,
                        flow_mask=fold_train_mask_t,
                        predicted_od=outputs['estimated_demand'],
                        true_od=true_od_t,
                        od_mask=fold_od_mask_t,
                        learned_alpha=outputs.get('learned_alpha'),
                        learned_beta=outputs.get('learned_beta')
                    )

                # Backward con Scaler (Maneja float16)
                scaler.scale(loss_dict['total_loss']).backward()

                # E. GRADIENT CLIPPING (Antes del step del optimizador)
                # Desescalamos primero para clipear los gradientes reales
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

                scaler.step(optimizer)
                scaler.update()

                current_loss = loss_dict['total_loss'].item()

                # --- LOGGING PERIÓDICO (Cada 10 epochs de ESTE fold) ---
                if (epoch + 1) % 10 == 0:
                    logging.info(f"[Fold {fold+1}][Epoch {epoch+1}] Train Loss: {current_loss:.4f}")

                # --- GUARDADO PERIÓDICO (Idéntico lógica original) ---
                if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.training.epochs:

                    # 1. Actualizar Checkpoint
                    epoch_state = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': current_loss,
                        'metrics': {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
                    }
                    fold_checkpoint['epochs_history'][epoch + 1] = epoch_state
                    torch.save(fold_checkpoint, fold_model_path)

                    # 2. Actualizar Eval Bundle
                    eval_snapshot = {
                        'pred_flows': outputs['reconstructed_flows'].detach().cpu(),
                        'pred_od': outputs['estimated_demand'].detach().cpu(),
                        'route_probs': outputs['route_probs'].detach().cpu() if outputs.get('route_probs') is not None else None,
                        'alpha': outputs.get('learned_alpha', torch.tensor(-1)).detach().cpu(),
                        'beta': outputs.get('learned_beta', torch.tensor(-1)).detach().cpu(),
                        'convergence': outputs.get('convergence_info', {})
                    }
                    fold_eval_bundle['epochs_history'][epoch + 1] = eval_snapshot
                    torch.save(fold_eval_bundle, fold_eval_path)

                # --- VALIDACIÓN INTERNA DEL FOLD (Para métrica final) ---
                if (epoch + 1) % cfg.training.val_frequency == 0:
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(
                            observed_flows=true_flows_t,
                            flow_mask=fold_val_mask_t,  # Input mask
                            warmup=False
                        )
                        # Calcular métrica de validación (MSE en links desconocidos)
                        pred_flows = val_outputs['reconstructed_flows']
                        val_loss = torch.sum((pred_flows - true_flows_t) ** 2 * fold_val_mask_t) / (
                                    torch.sum(fold_val_mask_t) + 1e-6)
                        val_loss_item = val_loss.item()

                        # F. SCHEDULER STEP
                        if scheduler is not None:
                            # ReduceLR necesita la métrica de validación para decidir
                            scheduler.step(val_loss_item)

                        # G. EARLY STOPPING CHECK
                        if val_loss_item < best_val_loss:
                            best_val_loss = val_loss_item
                            epochs_no_improve = 0
                            # Aquí guardarías el "best_model_fold_X.pt" si quisieras
                        else:
                            epochs_no_improve += 1
                            if cfg.training.early_stopping.enabled and epochs_no_improve >= patience:
                                early_stop = True

                    model.train()

            logging.info(f"Fin Fold {fold + 1}. Mejor Val MSE: {best_val_loss:.6f}")
            logging.info(f"Guardado: {os.path.basename(fold_eval_path)}")
            fold_metrics.append(best_val_loss)

            # Limpieza de memoria
            del model, optimizer, criterion, outputs, loss_dict
            torch.cuda.empty_cache()

        # F. Reporte Final de K-Fold
        avg_mse = np.mean(fold_metrics)
        std_mse = np.std(fold_metrics)
        logging.info("\n" + "="*60)
        logging.info(f"RESULTADOS K-FOLD (K={k_folds})")
        logging.info(f"MSE Promedio: {avg_mse:.6f} (+/- {std_mse:.6f})")
        logging.info("="*60)

        # Retornamos el promedio para que Optuna lo pueda minimizar
        return avg_mse

    else:
        # ---------------------------------------------------------
        # ENTRENAMIENTO ESTÁNDAR (Sin K-Fold, usando todo Train)
        # ---------------------------------------------------------
        logging.info("Entrenamiento estándar (Sin K-Fold) con estrategias inteligentes...")

        # 1. Instanciar Modelo
        model = hydra.utils.instantiate(
            cfg.model,
            num_links=network_params['num_links'],
            num_od_pairs=network_params['num_od_pairs'],
            t0=network_params['t0'].to(device),
            capacity=network_params['capacity'].to(device),
            route_masks=network_params['route_masks'].to(device),
            od_pair_indices=network_params['od_pair_indices'].to(device),
            num_link_groups=network_params['num_link_groups'],
            link_group=network_params['link_group'].to(device),
            _recursive_=False
        ).to(device)

        # 2. Optimizador y Loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay
        )
        criterion = get_loss_function(cfg, device)

        # --- A. SCHEDULER ---
        scheduler = None
        if cfg.training.scheduler == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=cfg.training.scheduler_params.factor,
                patience=cfg.training.scheduler_params.patience,
                min_lr=cfg.training.scheduler_params.min_lr
            )

        # --- B. VARIABLES DE CONTROL ---
        train_flow_mask_t = torch.FloatTensor(train_flow_mask_global).to(device)
        train_od_mask_t = torch.FloatTensor(train_od_mask_global).to(device)
        test_flow_mask_t = torch.FloatTensor(test_flow_mask).to(device)

        # Early Stopping
        patience = cfg.training.early_stopping.patience
        epochs_no_improve = 0
        best_val_loss = float('inf')
        early_stop = False

        # Estructuras de guardado
        master_checkpoint = {
            'config': OmegaConf.to_container(cfg, resolve=True),
            'epochs_history': {}
        }

        master_eval_bundle = {
            'config': OmegaConf.to_container(cfg, resolve=True),
            'static_data': {
                'true_flows': torch.FloatTensor(all_flows).cpu(),
                'true_od': torch.FloatTensor(od_vector).cpu(),
                'capacity': network_params['capacity'].cpu(),
                't0': network_params['t0'].cpu(),
                'link_group': network_params['link_group'].cpu(),
                'masks': {
                    'flow_observed': torch.BoolTensor(observed_flow_mask).cpu(),
                    'flow_train': torch.BoolTensor(train_flow_mask_global).cpu(),
                    'flow_test': torch.BoolTensor(test_flow_mask).cpu(),
                    'od_observed': torch.BoolTensor(observed_od_mask).cpu(),
                }
            },
            'epochs_history': {}
        }

        # 5. Loop de Entrenamiento
        for epoch in range(cfg.training.epochs):
            if early_stop:
                logging.info(f"Early Stopping activado en época {epoch}. Entrenamiento finalizado.")
                # break

            model.train()
            optimizer.zero_grad()

            # --- C. LOSS WARMUP ---
            # Primeras 20 épocas: Solo aprender física de flujos, ignorar OD
            current_w_od = 0.0 if epoch < 20 else cfg.model.loss_weights.w_od
            if hasattr(criterion, 'w_od'):
                criterion.w_od = torch.tensor(current_w_od, device=device)

            # --- D. CURRICULUM LEARNING ---
            # Aumentar complejidad del SUE gradualmente
            current_iters = int(min(10, 2 + epoch * 0.2))

            # Forward DIRECTO (Sin autocast)
            outputs = model(
                observed_flows=true_flows_t,
                flow_mask=train_flow_mask_t,
                true_od_demand=true_od_t,
                warmup=(epoch < 5),
                #override_max_iters=current_iters
            )

            loss_dict = criterion(
                predicted_flows=outputs['reconstructed_flows'],
                true_flows=true_flows_t,
                flow_mask=train_flow_mask_t,
                predicted_od=outputs['estimated_demand'],
                true_od=true_od_t,
                od_mask=train_od_mask_t,
                learned_alpha=outputs.get('learned_alpha'),
                learned_beta=outputs.get('learned_beta')
            )

            # Backward ESTÁNDAR (Sin scaler)
            loss_dict['total_loss'].backward()

            # --- E. GRADIENT CLIPPING ---
            # Ahora podemos clipear directamente
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip_norm)

            optimizer.step()

            current_loss = loss_dict['total_loss'].item()

            # --- VALIDACIÓN & SCHEDULER ---
            if (epoch + 1) % cfg.training.val_frequency == 0:
                model.eval()
                with torch.no_grad():
                    val_outputs = model(
                        observed_flows=true_flows_t,
                        flow_mask=test_flow_mask_t,
                        warmup=False
                    )

                    # Calcular MSE en datos NO vistos (Test)
                    pred_flows = val_outputs['reconstructed_flows']
                    val_loss = torch.sum((pred_flows - true_flows_t) ** 2 * test_flow_mask_t) / (
                                torch.sum(test_flow_mask_t) + 1e-6)
                    val_loss_item = val_loss.item()

                    # F. Actualizar Scheduler
                    if scheduler is not None:
                        scheduler.step(val_loss_item)

                    # G. Chequeo Early Stopping
                    if val_loss_item < best_val_loss:
                        best_val_loss = val_loss_item
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1
                        if cfg.training.early_stopping.enabled and epochs_no_improve >= patience:
                            early_stop = True

            # Logging periódico
            if (epoch + 1) % 10 == 0:
                logging.info(
                    f"Epoch {epoch + 1}: Loss {current_loss:.4f} | Val MSE {best_val_loss:.4f} | SUE Iters {current_iters}")

            # 6. Guardado de Checkpoints
            if (epoch + 1) == 1 or (epoch + 1) % 10 == 0 or (epoch + 1) == cfg.training.epochs or early_stop:
                epoch_state = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                    'metrics': {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in loss_dict.items()}
                }
                master_checkpoint['epochs_history'][epoch + 1] = epoch_state
                torch.save(master_checkpoint, full_model_path)

                eval_snapshot = {
                    'pred_flows': outputs['reconstructed_flows'].detach().cpu(),
                    'pred_od': outputs['estimated_demand'].detach().cpu(),
                    'route_probs': outputs['route_probs'].detach().cpu() if outputs.get(
                        'route_probs') is not None else None,
                    'alpha': outputs.get('learned_alpha', torch.tensor(-1)).detach().cpu(),
                    'beta': outputs.get('learned_beta', torch.tensor(-1)).detach().cpu(),
                    'convergence': outputs.get('convergence_info', {})
                }
                master_eval_bundle['epochs_history'][epoch + 1] = eval_snapshot
                torch.save(master_eval_bundle, full_eval_path)

                if early_stop:
                    logging.info(f"Guardando checkpoint final por Early Stopping en época {epoch + 1}")
                    # break

        return best_val_loss

@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    run_pipeline(cfg)


if __name__ == "__main__":
    main()