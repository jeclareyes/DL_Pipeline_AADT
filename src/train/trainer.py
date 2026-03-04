import numpy as np
import logging
import torch
import os
import hydra
from pathlib import Path
from omegaconf import OmegaConf


class TrafficTrainer:
    """
    Clase encargada de ejecutar el bucle de entrenamiento para UN modelo.
    Abstrae la lógica de épocas, optimizadores, guardado y validación.
    """

    def __init__(self, cfg, device, output_dir, model_filename, eval_filename):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.model_path = os.path.join(output_dir, model_filename)
        self.eval_path = os.path.join(output_dir, eval_filename)
        self.logger = logging.getLogger(__name__)

        # Parse warmup config (backward compatible)
        warmup_cfg = getattr(self.cfg.training, 'warmup', None) if hasattr(self.cfg, 'training') else None
        if warmup_cfg is None:
            # Backward compatibility: default to existing hard-coded behavior
            self.warmup_enabled = True
            self.warmup_epochs = 10
        else:
            self.warmup_enabled = bool(warmup_cfg.get('enabled', False))
            self.warmup_epochs = int(warmup_cfg.get('epochs', 0))

    def fit(self, model, network_params, train_tensors, val_tensors, t0_od_costs=None):
        """
        Ejecuta el entrenamiento completo (Epoch 1 -> N).

        Args:
            model: Instancia del modelo (UltraCyclic, etc.)
            network_params: Dict con parámetros estáticos de la red (t0, cap, etc.) para guardado.
            train_tensors: Dict con {flows, mask, od, od_mask} para entrenar.
            val_tensors: Dict con {flows, mask} para validar (puede ser Test set o Validation set).
        """
        # Detectar si hay validación activa
        has_val = val_tensors['mask'].sum() > 0

        # 1. Setup Optimizador y Loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )

        criterion = self._get_loss_function(model, t0_costs=t0_od_costs)
        scheduler = self._get_scheduler(optimizer)

        # 2. Estructuras de Guardado (State Containers)
        checkpoint_container = {
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'epochs_history': {}
        }

        # Preparamos datos estáticos para el eval bundle
        eval_container = {
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'static_data': self._pack_static_data(train_tensors, val_tensors, network_params),
            'epochs_history': {}
        }

        # Preparar static_info para el diagnosticador
        # Usamos network_params['link_types_vis'] que acabamos de inyectar
        static_info_vis = {
            'capacity': network_params['capacity'].cpu().numpy(),
            'link_types': network_params.get('link_types_vis', None)
        }

        # 3. Variables de Control
        best_monitored_loss = float('inf')
        early_stop = False
        epochs_no_improve = 0
        patience = self.cfg.training.early_stopping.patience

        # Herramienta de diagnóstico (opcional)
        #from src.components.models.CGAME_MLP_SUE import CGAMEDiagnosticTool
        #diagnostician = CGAMEDiagnosticTool(window_size=100)

        # AÑADIR ESTO (Carga dinámica desde el YAML):
        if hasattr(self.cfg.model, 'diagnostics'):
            diagnostician = hydra.utils.instantiate(self.cfg.model.diagnostics)
        else:
            # Fallback simple por si el YAML no tiene diagnostics
            from src.components.models.CGAME_DataDriven import TrainingDiagnostician
            diagnostician = TrainingDiagnostician(history_window=100)

        # --- BUCLE DE ÉPOCAS ---
        for epoch in range(self.cfg.training.epochs):
            if early_stop:
                break

            model.train()
            optimizer.zero_grad()

            current_iters = int(min(10, 2 + epoch * 0.2))  # Curriculum SUE

            # Determine warmup flag from config instead of hard-coded
            warmup_flag = (self.warmup_enabled and (epoch < self.warmup_epochs))

            # B. Forward & Backward
            outputs = model(
                observed_flows=train_tensors['flows'],
                flow_mask=train_tensors['mask'],

                true_od_demand=train_tensors['od'],
                od_mask=train_tensors['od_mask'],

                warmup=warmup_flag,
                # override_max_iters=current_iters TODO fix
            )

            if hasattr(criterion, 'w_entropy'):
                # --- NUEVA INVOCACIÓN DEL LOSS ---
            # Preparamos los diccionarios que espera TopoLoss
                outputs_dict = {
                    'reconstructed_flows': outputs['reconstructed_flows'],
                    'estimated_demand': outputs['estimated_demand'],
                    'route_flows': outputs.get('route_flows') # Esencial para la Entropía
                }
                
                targets_dict = {
                    'flows': train_tensors['flows'],
                    'od': train_tensors['od']
                }
                
                masks_dict = {
                    'flow_mask': train_tensors['mask'],
                    'od_mask': train_tensors['od_mask'],
                    'route_mask': network_params['route_validity_mask'] # <- Físicamente inyectada
                }

                loss_dict = criterion(outputs=outputs_dict, targets=targets_dict, masks=masks_dict)

            else:
                loss_dict = criterion(
                    predicted_flows=outputs['reconstructed_flows'],
                    true_flows=train_tensors['flows'],
                    flow_mask=train_tensors['mask'],
                    predicted_od=outputs['estimated_demand'],
                    true_od=train_tensors['od'],
                    od_mask=train_tensors['od_mask'],
                    learned_alpha=outputs.get('learned_alpha'),
                    learned_beta=outputs.get('learned_beta')
                )

            loss_dict['total_loss'].backward()

            # Antes del optimizer.step(), capturamos el estado del gradiente
            diagnostician.capture_gradient_history(model, epoch)

            # 2. Update diagnostics
            diagnostician.update(
                outputs=outputs,
                model=model,
                loss_dict=loss_dict,
                targets=train_tensors,  # {flows, mask, od, od_mask}
                static_info=static_info_vis
            )

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip_norm)
            optimizer.step()

            current_loss = loss_dict['total_loss'].item()

            # C. Validación
            val_loss, val_preds = self._validate(model, train_tensors['flows'], train_tensors['mask'], val_tensors['mask'])

            # Seleccionar métrica para scheduler/seguimiento
            monitored_loss = val_loss if has_val else current_loss
            monitor_name = "val_mse" if has_val else "train_loss"

            # Scheduler Step (siempre activo si existe scheduler)
            if scheduler:
                pre_lrs = [pg.get('lr', None) for pg in optimizer.param_groups]

                try:
                    scheduler.step(monitored_loss)
                except TypeError:
                    scheduler.step()

                post_lrs = [pg.get('lr', None) for pg in optimizer.param_groups]

                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        try:
                            return float(x.item())
                        except Exception:
                            return None

                pre_f = [_to_float(x) for x in pre_lrs]
                post_f = [_to_float(x) for x in post_lrs]

                for i, (p, q) in enumerate(zip(pre_f, post_f)):
                    if p is not None and q is not None and q < p:
                        self.logger.info(
                            f"Scheduler reduced LR for param_group {i}: {p:.6e} -> {q:.6e} "
                            f"(monitor={monitor_name}, value={monitored_loss:.6f})"
                        )

            # Seguimiento de mejor métrica
            prev_best_monitored_loss = best_monitored_loss
            improved = monitored_loss < prev_best_monitored_loss
            if improved:
                best_monitored_loss = monitored_loss

            # Early stopping solo aplica cuando hay validación
            if has_val:
                if improved:
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if self.cfg.training.early_stopping.enabled and epochs_no_improve >= patience:
                        early_stop = True
                        self.logger.info(f"Early stopping en época {epoch + 1}")

            # D. Logging & Saving
            log_freq = self.cfg.training.get('save_frequency', 10)

            if (epoch + 1) % log_freq == 0:
                # 1. Mensaje Base: Pérdida (Loss)
                log_msg = (f"Epoch {epoch + 1}: Train Loss {current_loss:.0f} | "
                           f"Flow Loss {loss_dict.get('l_flow', 0.0):.0f} | "
                           f"OD Loss {loss_dict.get('l_od', 0.0):.0f} | "
                           f"REG OD Loss {loss_dict.get('l_demand_reg', 0.0):.0f}")
                if has_val:
                    log_msg += f" | Val MSE {val_loss:.4f}"

                # 2. Métricas Detalladas (R2, MAE, MAPE)
                if self.cfg.training.get('metrics_on_training', False):
                    # Siempre calculamos métricas de Training
                    train_metrics = self._compute_metrics_snapshot(
                        outputs['reconstructed_flows'],
                        train_tensors['flows'],
                        train_tensors['mask']
                    )
                    # Include RMSE in the training log summary (backwards-compatible addition)
                    log_msg += (
                        f"\n    >> TRAIN: R2={train_metrics['R2']:.3f}, "
                        f"MAE={train_metrics['MAE']:.2f}, RMSE={train_metrics.get('RMSE', 0.0):.2f}, "
                        f"MAPE={train_metrics['MAPE']:.1f}%, "
                        f"LR={optimizer.param_groups[0]['lr']:.2e}"
                    )

                    # Solo calculamos métricas de VAL si existe el dataset de validación
                    if has_val and val_preds is not None:
                        val_metrics = self._compute_metrics_snapshot(
                            val_preds,
                            train_tensors['flows'],
                            val_tensors['mask']
                        )
                        log_msg += (
                            f"\n    >> VAL  : R2={val_metrics['R2']:.3f}, "
                            f"MAE={val_metrics['MAE']:.2f}, RMSE={val_metrics.get('RMSE', 0.0):.2f}, "
                            f"MAPE={val_metrics['MAPE']:.1f}%"
                        )
                    else:
                        # Opcional: Indicar que no hay validación para evitar confusiones
                        log_msg += "\n    >> VAL  : N/A (Hold-out mode)"

                self.logger.info(log_msg)

            self._save_checkpoint(
                epoch, model, optimizer, current_loss, loss_dict, outputs,
                checkpoint_container, eval_container, early_stop
            )

        diagnostician.finalize_and_plot("final_training_report.png")
        diagnostician.plot_evolution("final_evolution_report.png")
        diagnostician.plot_physics("final_physics_report.png")
        diagnostician.plot_gradient_health("gradient_health_report.png")

        return best_monitored_loss

    def _validate(self, model, all_flows, train_mask, val_mask):
        """
        Args:
            train_mask: Máscara de observación (input al encoder)
            val_mask: Máscara de evaluación (links hold-out)
        """
        # Si la máscara de validación está vacía (K=1), devolvemos un valor neutro
        if val_mask.sum() == 0:
            return 0.0, None

        model.eval()
        with torch.no_grad():
            val_out = model(
                observed_flows=all_flows,
                flow_mask=train_mask,  # ✅ Mismo input que training
                warmup=False
            )
            pred = val_out['reconstructed_flows']

            # Calcular error solo en hold-out
            mse = torch.sum((pred - all_flows) ** 2 * val_mask) / (torch.sum(val_mask) + 1e-6)

        model.train()
        return mse.item(), pred

    def _compute_metrics_snapshot(self, pred_tensor, target_tensor, mask_tensor):
        """Calcula R2, MAE, RMSE y MAPE rápido para logs.

        Returns a dict with keys: 'R2', 'MAE', 'RMSE', 'MAPE'.
        Backwards compatible: callers that don't use 'RMSE' will continue to work.
        """
        # Pasar a CPU y Numpy para cálculo
        y_pred = pred_tensor.detach().cpu().numpy().flatten()
        y_true = target_tensor.detach().cpu().numpy().flatten()
        mask = mask_tensor.detach().cpu().numpy().flatten().astype(bool)

        # Filtrar solo lo observado
        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(y_true) == 0:
            return {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}

        # MAE
        mae = np.mean(np.abs(y_pred - y_true))

        # R2
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        # MAPE (Evitando división por cero)
        non_zero = y_true != 0
        if np.any(non_zero):
            mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
        else:
            mape = 0.0

        # RMSE
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    def _get_scheduler(self, optimizer):
        if self.cfg.training.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.cfg.training.scheduler_params.factor,
                patience=self.cfg.training.scheduler_params.patience,
                min_lr=self.cfg.training.scheduler_params.min_lr
            )
        return None

    def _get_loss_function(self, model, t0_costs=None):
        # 1. Extraer escalas si el modelo las tiene (como buffers)
        # Si el modelo no las tiene, usamos 1.0 por defecto
        # l_scale = getattr(model, 'link_scale', torch.tensor(1.0)).item() TODO: Esto no funciona si link_scale es un escalar float, así que lo hacemos más robusto:
        l_scale = (v.detach().item() if torch.is_tensor(v := getattr(model, "link_scale", 1.0)) else float(v))
        o_scale = (v.detach().item() if torch.is_tensor(v := getattr(model, "od_scale", 1.0)) else float(v))

        # 2. Instanciación general
        # Pasamos los escaladores como 'kwargs'. Si la clase Loss en el YAML
        # no tiene estos argumentos en su __init__, Hydra los ignorará
        # SIEMPLE Y CUANDO uses _recursive_=False y la clase tenga **kwargs
        # o los argumentos definidos.
        return hydra.utils.instantiate(
            self.cfg.model.loss,
            link_scale=l_scale,
            od_scale=o_scale,
            t0_costs=t0_costs,
            _recursive_=False
        ).to(self.device)


    def _pack_static_data(self, train_t, val_t, net_p):
        """
        Empaqueta tensores estáticos necesarios para reconstruir el escenario en Testing.
        """

        data_bundle = {
            # Ground Truth (para comparar en testing)
            'true_flows': train_t['flows'].cpu(),
            'true_od': train_t['od'].cpu(),

            # Física de la red (necesaria para el simulador SUE en testing)
            'capacity': net_p['capacity'].cpu(),
            't0': net_p['t0'].cpu(),
            'od_pair_indices': net_p['od_pair_indices'].cpu(),

            'link_group': net_p['link_group'].cpu(),
            'num_link_groups': net_p.get('num_link_groups', -1),

            'trips_scaler': net_p.get('trips_scaler', 1.0),

            # Máscaras de entrenamiento vs evaluación
            'masks': {
                'flow_train': train_t['mask'].cpu().bool(),
                'flow_test': val_t['mask'].cpu().bool(),
                'od_mask': train_t['od_mask'].cpu().bool()
            }
        }


        # --- NUEVO: Soporte Híbrido Topológico ---
        if 'route_masks' in net_p: # Legado 3D
            data_bundle['route_masks'] = net_p['route_masks'].cpu()
        
        if 'delta_matrix' in net_p: # Topo-CGAME 2D
            data_bundle['delta_matrix'] = net_p['delta_matrix'].cpu()
            
        if 'route_validity_mask' in net_p: # Máscara de Atención
            data_bundle['route_validity_mask'] = net_p['route_validity_mask'].cpu()

        return data_bundle


    def _save_checkpoint(self, epoch, model, optimizer, loss, loss_dict, outputs,
                         ckpt_cont, eval_cont, early_stop):
        """
        Guarda checkpoints robustos para retomar entrenamiento y para evaluación posterior.

        Guarda:
        1. 'model_final.pt': El último estado (para resume).
        2. 'eval_model_final.pt': Historial completo y configuración (para análisis).
        3. 'best_model.pt': El mejor modelo según validación (opcional, pero recomendado).
        """

        # 1. Actualizar Contenedores de Estado (En Memoria)
        # Historial de entrenamiento (pesos y optimizador)
        ckpt_cont['epochs_history'][epoch + 1] = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            # Guardamos métricas escalares para trazabilidad rápida
            'metrics': {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
        }

        # Historial de Evaluación (Resultados físicos)
        # Importante: Usamos .detach().cpu() para no llenar la GPU de historial
        eval_cont['epochs_history'][epoch + 1] = {
            'pred_flows': outputs['reconstructed_flows'].detach().cpu(),
            'pred_od': outputs['estimated_demand'].detach().cpu(),
            'convergence': outputs.get('convergence_info', {}),
            'learned_alpha': outputs.get('learned_alpha'),
            'learned_beta': outputs.get('learned_beta')
        }

        # Agregar el Scaler al contenedor si está disponible en el modelo
        # Esto es vital para desnormalizar en testing
        if hasattr(model, 'validator') and hasattr(model.validator, 'max_trips_scaler'):
            eval_cont['max_trips_scaler'] = model.validator.max_trips_scaler

        # 2. Guardado en Disco
        # Estrategia: Guardar frecuentemente, al final, o si hay early stop.
        is_final = (epoch + 1) == self.cfg.training.epochs or early_stop
        is_frequent = (epoch + 1) % self.cfg.training.get('save_frequency', 10) == 0

        if is_final or is_frequent:
            # Crear directorios
            Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)

            # Guardar Checkpoint de Entrenamiento (Pesado, incluye optimizador)
            torch.save(ckpt_cont, self.model_path)

            # Guardar Container de Evaluación (Ligero en dependencias, incluye config y datos estáticos)
            torch.save(eval_cont, self.eval_path)

            self.logger.info(f"Checkpoint guardado en epoch {epoch + 1}: {Path(self.model_path).name}")

            # Opcional: Guardar una copia específica del "Mejor Modelo" si la lógica de validación lo dicta
            # (Esto requeriría pasar un flag 'is_best' desde fit)

