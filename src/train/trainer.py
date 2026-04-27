import numpy as np
import logging
import torch
import os
import hydra
import csv
from pathlib import Path
from omegaconf import OmegaConf

from src.contracts.runtime_contracts import (
    ArtifactSchemaError,
    ConfigurationContractError,
    ModelOutputContractError,
    validate_model_input_contract,
    validate_model_output_contract,
)
from src.train._post_training_diagnostics import run_post_training_diagnostics


class TrafficTrainer:
    """
    Clase encargada de ejecutar el bucle de entrenamiento para UN modelo.
    Abstrae la lógica de épocas, optimizadores, guardado y validación.
    """

    def __init__(self, cfg, device, output_dir, model_filename, eval_filename):
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        self.models_dir = os.path.join(output_dir, "models")
        self.training_diagnostics_dir = os.path.join(output_dir, "diagnostics", "training")
        self.physics_diagnostics_dir = os.path.join(output_dir, "diagnostics", "physics")
        self.model_path = os.path.join(self.models_dir, model_filename)
        self.eval_path = os.path.join(self.models_dir, eval_filename)
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

        # Runtime stability telemetry.
        self._clip_ratio_history = []
        self._skipped_updates = 0
        self._validation_skipped_logged = False

    def fit(self, model, network_params, train_tensors, val_tensors, **kwargs):
        """
        Ejecuta el entrenamiento completo (Epoch 1 -> N).

        Args:
            model: Instancia del modelo (UltraCyclic, etc.)
            network_params: Dict con parámetros estáticos de la red (t0, cap, etc.) para guardado.
            train_tensors: Dict con {flows, mask, od, od_mask} para entrenar.
            val_tensors: Dict con {flows, mask} para validar (puede ser Test set o Validation set).
        """
        legacy_t0_prior = kwargs.pop('t0_od_costs', None)
        if legacy_t0_prior is not None:
            self.logger.warning(
                "Legacy kwarg t0_od_costs is deprecated and ignored in TrafficTrainer.fit"
            )
        if kwargs:
            raise ConfigurationContractError(
                f"Unexpected fit kwargs: {sorted(kwargs.keys())}"
            )

        validate_model_input_contract(train_tensors, val_tensors)

        # Normalize tensor shapes to a consistent batched contract.
        train_flows_t = self._ensure_batch_dim(train_tensors['flows'])
        train_mask_t = self._ensure_batch_dim(train_tensors['mask'])
        train_od_t = self._ensure_batch_dim(train_tensors['od'])
        train_od_mask_t = self._ensure_batch_dim(train_tensors['od_mask'])
        val_mask_t = self._ensure_batch_dim(val_tensors['mask'])

        diagnostics_targets = {
            'flows': train_flows_t,
            'mask': train_mask_t,
            'flow_mask': train_mask_t,
            'od': train_od_t,
            'od_mask': train_od_mask_t,
        }

        # Detectar si hay validación activa
        has_val = val_mask_t.sum() > 0

        # Monitoring policy for scheduler and early stopping.
        scheduler_monitor_cfg = str(self.cfg.training.get('scheduler_monitor', 'auto')).lower()
        if scheduler_monitor_cfg not in {'auto', 'train', 'val'}:
            raise ConfigurationContractError(
                f"Unknown training.scheduler_monitor='{scheduler_monitor_cfg}'. "
                "Allowed values: auto|train|val"
            )

        if scheduler_monitor_cfg == 'val':
            scheduler_monitor_name = 'val_mse' if has_val else 'train_loss'
            if not has_val:
                self.logger.warning(
                    "scheduler_monitor='val' requested but validation mask is empty. Falling back to train_loss."
                )
        elif scheduler_monitor_cfg == 'train':
            scheduler_monitor_name = 'train_loss'
        else:
            scheduler_monitor_name = 'val_mse' if has_val else 'train_loss'

        es_cfg = self.cfg.training.get('early_stopping', {})
        early_enabled = bool(es_cfg.get('enabled', False))
        early_patience = int(es_cfg.get('patience', 20))
        early_min_delta = float(es_cfg.get('min_delta', 0.0))
        early_monitor_cfg = str(es_cfg.get('monitor', 'auto')).lower()
        if early_monitor_cfg not in {'auto', 'train', 'val'}:
            raise ConfigurationContractError(
                f"Unknown training.early_stopping.monitor='{early_monitor_cfg}'. "
                "Allowed values: auto|train|val"
            )

        if early_monitor_cfg == 'val':
            early_monitor_name = 'val_mse' if has_val else None
        elif early_monitor_cfg == 'train':
            early_monitor_name = 'train_loss'
        else:
            early_monitor_name = 'val_mse' if has_val else 'train_loss'

        early_active = early_enabled and (early_monitor_name is not None)
        if early_enabled and not early_active:
            self.logger.warning(
                "early_stopping.enabled=True but monitor requires validation and validation mask is empty. "
                "Early stopping disabled for this run."
            )

        self.logger.info(
            "Training controls | scheduler_monitor=%s | early_stopping=%s%s",
            scheduler_monitor_name,
            "on" if early_active else "off",
            f" (monitor={early_monitor_name}, patience={early_patience}, min_delta={early_min_delta:.2e})"
            if early_active else "",
        )

        # 1. Setup Optimizador y Loss
        base_lr = float(self.cfg.training.lr)
        weight_decay = float(self.cfg.training.weight_decay)

        optimizer_param_groups = None
        if hasattr(model, "get_optimizer_param_groups"):
            try:
                optimizer_param_groups = model.get_optimizer_param_groups(
                    base_lr=base_lr,
                    weight_decay=weight_decay,
                )
            except TypeError:
                optimizer_param_groups = model.get_optimizer_param_groups(base_lr)

        if optimizer_param_groups:
            optimizer = torch.optim.Adam(optimizer_param_groups)
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=base_lr,
                weight_decay=weight_decay
            )
        scheduler = self._get_scheduler(optimizer)

        # 2. Estructuras de Guardado (State Containers)
        checkpoint_container = {
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'epochs_history': {}
        }

        # Preparamos datos estáticos para el eval bundle
        eval_container = {
            'schema_version': 1,
            'config': OmegaConf.to_container(self.cfg, resolve=True),
            'static_data': self._pack_static_data(train_tensors, val_tensors, network_params),
            'epochs_history': {}
        }

        # Preparar static_info para el diagnosticador
        # Usamos network_params['link_types_vis'] que acabamos de inyectar
        static_info_vis = {
            'capacity': network_params['capacity'].cpu().numpy(),
            'link_types': network_params.get('link_types_vis', None),
            'od_pair_indices': network_params.get('od_pair_indices', None),
        }

        # 3. Variables de Control
        best_scheduler_monitor_loss = float('inf')
        best_early_monitor_loss = float('inf')
        early_stop = False
        epochs_no_improve = 0

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
        last_outputs = None
        last_epoch = 0
        for epoch in range(self.cfg.training.epochs):
            if early_stop:
                break

            model.train()
            optimizer.zero_grad()
            is_final_epoch = (epoch + 1) == self.cfg.training.epochs

            pre_clip_norm = float('nan')
            post_clip_norm = float('nan')
            clip_ratio = 1.0

            # Determine warmup flag from config instead of hard-coded
            warmup_flag = (self.warmup_enabled and (epoch < self.warmup_epochs))

            # B. Forward & Backward
            outputs = model(
                observed_flows=train_flows_t,
                flow_mask=train_mask_t,

                true_od_demand=train_od_t,
                od_mask=train_od_mask_t,

                warmup=warmup_flag,
                current_epoch=epoch,
            )

            last_outputs = outputs
            last_epoch = epoch + 1

            outputs = validate_model_output_contract(outputs)
            loss_dict = outputs['loss']

            total_loss = loss_dict['total_loss']
            if not bool(torch.isfinite(total_loss).all()):
                self._skipped_updates += 1
                self.logger.error(
                    "Skipping epoch %d update due to non-finite total_loss=%s",
                    epoch + 1,
                    str(total_loss.detach().cpu().item()),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            total_loss.backward()

            non_finite_entries = 0
            non_finite_params = []
            for param_name, param in model.named_parameters():
                if param.grad is None:
                    continue
                finite_mask = torch.isfinite(param.grad)
                if bool(torch.all(finite_mask)):
                    continue

                bad_count = int((~finite_mask).sum().item())
                non_finite_entries += bad_count
                non_finite_params.append(f"{param_name}({bad_count})")
                param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=0.0, neginf=0.0)

            if non_finite_entries > 0:
                self.logger.warning(
                    "Epoch %d had %d non-finite gradient entries across %d params; sanitized before clipping. %s",
                    epoch + 1,
                    non_finite_entries,
                    len(non_finite_params),
                    ", ".join(non_finite_params[:6]),
                )

            # Antes del optimizer.step(), capturamos el estado del gradiente
            diagnostician.capture_gradient_history(model, epoch)

            # 2. Update diagnostics
            diagnostician.update(
                outputs=outputs,
                model=model,
                loss_dict=loss_dict,
                targets=diagnostics_targets,
                static_info=static_info_vis,
                is_final=is_final_epoch,
                epoch=epoch + 1,
            )

            try:
                pre_clip_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.cfg.training.grad_clip_norm,
                        error_if_nonfinite=True,
                    )
                )
            except RuntimeError as exc:
                self._skipped_updates += 1
                self.logger.error(
                    "Skipping epoch %d update due to non-finite gradient norm during clipping: %s",
                    epoch + 1,
                    str(exc),
                )
                optimizer.zero_grad(set_to_none=True)
                continue

            post_clip_norm_sq = 0.0
            for param in model.parameters():
                if param.grad is None:
                    continue
                g_norm = float(torch.norm(param.grad.detach(), p=2).item())
                post_clip_norm_sq += g_norm * g_norm
            post_clip_norm = post_clip_norm_sq ** 0.5

            if pre_clip_norm > 0.0:
                clip_ratio = post_clip_norm / pre_clip_norm

            self._clip_ratio_history.append(float(clip_ratio))
            if len(self._clip_ratio_history) > 100:
                self._clip_ratio_history.pop(0)

            optimizer.step()

            current_loss = total_loss.item()

            # C. Validación
            val_loss, val_preds = self._validate(model, train_flows_t, train_mask_t, val_mask_t)

            scheduler_monitor_value = val_loss if scheduler_monitor_name == 'val_mse' else current_loss
            early_monitor_value = None
            if early_monitor_name is not None:
                early_monitor_value = val_loss if early_monitor_name == 'val_mse' else current_loss

            # Scheduler Step (siempre activo si existe scheduler)
            if scheduler:
                pre_lrs = [pg.get('lr', None) for pg in optimizer.param_groups]

                try:
                    scheduler.step(scheduler_monitor_value)
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
                            f"(monitor={scheduler_monitor_name}, value={scheduler_monitor_value:.6f})"
                        )

            # Seguimiento de mejor monitor para resumen final
            if scheduler_monitor_value < best_scheduler_monitor_loss:
                best_scheduler_monitor_loss = scheduler_monitor_value

            if early_active and early_monitor_value is not None:
                improved = early_monitor_value < (best_early_monitor_loss - early_min_delta)
                if improved:
                    best_early_monitor_loss = early_monitor_value
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= early_patience:
                        early_stop = True
                        self.logger.info(
                            f"Early stopping en época {epoch + 1} "
                            f"(monitor={early_monitor_name}, value={early_monitor_value:.6f}, "
                            f"best={best_early_monitor_loss:.6f}, min_delta={early_min_delta:.2e})."
                        )

            # D. Logging & Saving
            log_freq = self.cfg.training.get('save_frequency', 10)

            if (epoch + 1) % log_freq == 0:
                def _as_float(x):
                    if torch.is_tensor(x):
                        return float(x.detach().item())
                    try:
                        return float(x)
                    except Exception:
                        return 0.0

                flow_loss_val = _as_float(loss_dict.get('l_flow', 0.0))
                od_loss_val = _as_float(loss_dict.get('l_od', 0.0))
                reg_od_loss_val = _as_float(loss_dict.get('l_demand_reg', 0.0))

                # 1. Mensaje Base: Pérdida (Loss)
                log_msg = (
                    f"Epoch {epoch + 1}: Train Loss {current_loss:.6g} | "
                    f"Flow Loss {flow_loss_val:.6g} | "
                    f"OD Loss {od_loss_val:.6g} | "
                    f"REG OD Loss {reg_od_loss_val:.6g}"
                )

                conv_info = outputs.get('convergence_info', {}) if isinstance(outputs, dict) else {}
                if isinstance(conv_info, dict):
                    iters = conv_info.get('iterations', None)
                    gap = conv_info.get('final_gap', None)
                    mode = conv_info.get('mode', None)
                    if iters is not None or gap is not None:
                        iter_txt = f"{float(iters):.1f}" if iters is not None else "n/a"
                        gap_txt = f"{float(gap):.2e}" if gap is not None else "n/a"
                        mode_txt = f", mode={mode}" if mode is not None else ""
                        log_msg += f" | Eq iters={iter_txt}, gap={gap_txt}{mode_txt}"

                recent_clip = self._clip_ratio_history[-10:]
                clip_freq10 = 0.0
                if len(recent_clip) > 0:
                    clip_freq10 = float(sum(r < 0.999 for r in recent_clip) / len(recent_clip))
                log_msg += (
                    f" | GradNorm pre={pre_clip_norm:.2e}, post={post_clip_norm:.2e}, "
                    f"clip_ratio={clip_ratio:.3f}, clip_freq10={clip_freq10:.0%}"
                )
                if len(recent_clip) == 10 and clip_freq10 > 0.5:
                    self.logger.warning(
                        "Frequent gradient clipping in last 10 epochs (%.0f%%). "
                        "Consider increasing grad_clip_norm or lowering lr.",
                        100.0 * clip_freq10,
                    )

                if has_val:
                    log_msg += f" | Val MSE {val_loss:.4f}"

                # 2. Métricas Detalladas (R2, MAE, MAPE)
                if self.cfg.training.get('metrics_on_training', False):
                    # Siempre calculamos métricas de Training
                    train_metrics = self._compute_metrics_snapshot(
                        outputs['reconstructed_flows'],
                        train_flows_t,
                        train_mask_t,
                    )
                    lr_summary = [f"{float(pg['lr']):.2e}" for pg in optimizer.param_groups]
                    # Include RMSE in the training log summary (backwards-compatible addition)
                    log_msg += (
                        f"\n    >> TRAIN: R2={train_metrics['R2']:.3f}, "
                        f"MAE={train_metrics['MAE']:.2f}, RMSE={train_metrics.get('RMSE', 0.0):.2f}, "
                        f"MAPE={train_metrics['MAPE']:.1f}%, "
                        f"LR={lr_summary}"
                    )

                    # Solo calculamos métricas de VAL si existe el dataset de validación
                    if has_val and val_preds is not None:
                        val_metrics = self._compute_metrics_snapshot(
                            val_preds,
                            train_flows_t,
                            val_mask_t,
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

        final_evolution_path = os.path.join(self.training_diagnostics_dir, "final_evolution_report.png")
        final_physics_path = os.path.join(self.physics_diagnostics_dir, "final_physics_report.png")
        final_gradient_path = os.path.join(self.training_diagnostics_dir, "gradient_health_report.png")

        Path(self.training_diagnostics_dir).mkdir(parents=True, exist_ok=True)
        Path(self.physics_diagnostics_dir).mkdir(parents=True, exist_ok=True)

        diagnostician.plot_evolution(final_evolution_path)
        diagnostician.plot_physics(final_physics_path)
        diagnostician.plot_gradient_health(final_gradient_path)

        if hasattr(diagnostician, "save_summary"):
            summary_path = os.path.join(self.training_diagnostics_dir, "summary_metrics.json")
            diagnostician.save_summary(summary_path)

        # Model-agnostic diagnostics: always persist final comparison CSVs
        # and post-process them into metrics + assignment calibration artifacts.
        self._write_final_comparison_csvs(
            outputs=last_outputs,
            targets=diagnostics_targets,
            network_params=network_params,
            epoch=last_epoch,
        )

        if bool(self.cfg.training.get('enable_post_training_diagnostics', True)):
            run_post_training_diagnostics(self.training_diagnostics_dir, self.logger)
        else:
            self.logger.info(
                "Post-training diagnostics disabled by training.enable_post_training_diagnostics=false"
            )

        if early_active and best_early_monitor_loss < float('inf'):
            return best_early_monitor_loss
        return best_scheduler_monitor_loss

    def _validate(self, model, all_flows, train_mask, val_mask):
        """
        Args:
            train_mask: Máscara de observación (input al encoder)
            val_mask: Máscara de evaluación (links hold-out)
        """
        # If validation mask is empty, skip validation metrics in fail-fast-safe mode.
        if val_mask.sum() == 0:
            if not self._validation_skipped_logged:
                self.logger.info("Validation skipped: received an empty validation mask.")
                self._validation_skipped_logged = True
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

    @staticmethod
    def _ensure_batch_dim(tensor):
        """Normalize 1D tensors to [1, N] to avoid implicit broadcasting in losses."""
        if tensor is None:
            return None
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _get_max_epochs_history(self) -> int:
        """Read checkpoint history retention from config (0 disables pruning)."""
        retention_cfg = self.cfg.training.get('checkpoint_retention', {})
        if retention_cfg is None:
            return 0

        raw_value = retention_cfg.get('max_epochs_history', 0)
        try:
            max_epochs_history = int(raw_value)
        except Exception as exc:
            raise ConfigurationContractError(
                "training.checkpoint_retention.max_epochs_history must be an integer"
            ) from exc

        if max_epochs_history < 0:
            raise ConfigurationContractError(
                "training.checkpoint_retention.max_epochs_history must be >= 0"
            )
        return max_epochs_history

    def _prune_history_container(self, container: dict, max_epochs_history: int) -> int:
        """Prune oldest epochs in-place and return removed count."""
        if max_epochs_history <= 0:
            return 0

        history = container.get('epochs_history', None)
        if not isinstance(history, dict):
            return 0
        if len(history) <= max_epochs_history:
            return 0

        sorted_epochs = sorted(history.keys(), key=lambda x: int(x))
        to_remove = sorted_epochs[:-max_epochs_history]
        for epoch_key in to_remove:
            history.pop(epoch_key, None)
        return len(to_remove)

    def _write_csv_rows(self, path, fieldnames, rows):
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _write_final_comparison_csvs(self, outputs, targets, network_params, epoch):
        """Genera CSVs finales estandarizados de flujo y demanda de forma agnóstica al modelo."""
        if outputs is None or targets is None:
            return

        pred_flow = outputs.get('reconstructed_flows')
        true_flow = targets.get('flows')
        if pred_flow is None or true_flow is None:
            return

        train_dir = self.training_diagnostics_dir
        Path(train_dir).mkdir(parents=True, exist_ok=True)

        pred_flow_np = pred_flow.detach().cpu().numpy().reshape(-1)
        true_flow_np = true_flow.detach().cpu().numpy().reshape(-1)
        flow_mask_t = targets.get('mask', targets.get('flow_mask', None))
        n_flow = min(len(pred_flow_np), len(true_flow_np))

        if flow_mask_t is not None:
            flow_mask_np = flow_mask_t.detach().cpu().numpy().reshape(-1)
        else:
            flow_mask_np = np.ones(n_flow, dtype=np.float32)

        flow_rows = []
        m_flow = min(n_flow, len(flow_mask_np))
        for i in range(m_flow):
            flow_rows.append(
                {
                    'epoch': int(epoch),
                    'link_id': int(i),
                    'real_flow': float(true_flow_np[i]),
                    'estimated_flow': float(pred_flow_np[i]),
                    'is_observed_link': bool(flow_mask_np[i] > 0.5),
                }
            )

        pred_od = outputs.get('estimated_demand')
        true_od = targets.get('od')
        od_mask_t = targets.get('od_mask', None)
        demand_rows = []
        if pred_od is not None and true_od is not None:
            pred_od_np = pred_od.detach().cpu().numpy().reshape(-1)
            true_od_np = true_od.detach().cpu().numpy().reshape(-1)
            n_od = min(len(pred_od_np), len(true_od_np))

            if od_mask_t is not None:
                od_mask_np = od_mask_t.detach().cpu().numpy().reshape(-1)
            else:
                od_mask_np = np.ones(n_od, dtype=np.float32)

            od_pair_labels = None
            od_pair_indices = network_params.get('od_pair_indices', None)
            if od_pair_indices is not None:
                try:
                    if torch.is_tensor(od_pair_indices):
                        od_pairs_np = od_pair_indices.detach().cpu().numpy()
                    else:
                        od_pairs_np = np.asarray(od_pair_indices)
                    if od_pairs_np.ndim == 2 and od_pairs_np.shape[0] == 2 and od_pairs_np.shape[1] != 2:
                        od_pairs_np = od_pairs_np.T
                    if od_pairs_np.ndim == 2 and od_pairs_np.shape[1] >= 2:
                        od_pair_labels = [f"{int(o)}-{int(d)}" for o, d in od_pairs_np[:, :2]]
                except Exception:
                    od_pair_labels = None

            m_od = min(n_od, len(od_mask_np))
            for i in range(m_od):
                od_pair_id = od_pair_labels[i] if od_pair_labels is not None and i < len(od_pair_labels) else str(i)
                demand_rows.append(
                    {
                        'epoch': int(epoch),
                        'od_index': int(i),
                        'od_pair_id': str(od_pair_id),
                        'real_demand': float(true_od_np[i]),
                        'estimated_demand': float(pred_od_np[i]),
                        'is_known_demand': bool(od_mask_np[i] > 0.5),
                    }
                )

        flow_csv = os.path.join(train_dir, 'estimated_vs_real_flows.csv')
        self._write_csv_rows(
            flow_csv,
            fieldnames=['epoch', 'link_id', 'real_flow', 'estimated_flow', 'is_observed_link'],
            rows=flow_rows,
        )

        demand_csv = os.path.join(train_dir, 'estimated_vs_real_demand.csv')
        self._write_csv_rows(
            demand_csv,
            fieldnames=['epoch', 'od_index', 'od_pair_id', 'real_demand', 'estimated_demand', 'is_known_demand'],
            rows=demand_rows,
        )

    def _get_scheduler(self, optimizer):
        if self.cfg.training.scheduler == "reduce_on_plateau":
            scheduler_cfg = self.cfg.training.scheduler_params
            scheduler_kwargs = {
                'mode': 'min',
                'factor': float(scheduler_cfg.factor),
                'patience': int(scheduler_cfg.patience),
                'min_lr': float(scheduler_cfg.min_lr),
            }
            threshold = scheduler_cfg.get('threshold', None)
            if threshold is not None:
                scheduler_kwargs['threshold'] = float(threshold)

            threshold_mode = scheduler_cfg.get('threshold_mode', None)
            if threshold_mode is not None:
                scheduler_kwargs['threshold_mode'] = str(threshold_mode)

            cooldown = scheduler_cfg.get('cooldown', None)
            if cooldown is not None:
                scheduler_kwargs['cooldown'] = int(cooldown)

            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **scheduler_kwargs,
            )
        return None

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
        if hasattr(model, 'get_evaluation_artifacts'):
            artifacts = model.get_evaluation_artifacts(outputs)
            if not isinstance(artifacts, dict):
                raise ModelOutputContractError(
                    f"get_evaluation_artifacts must return dict, got {type(artifacts)}"
                )
        else:
            artifacts = {
                'pred_flows': outputs['reconstructed_flows'].detach().cpu(),
                'pred_od': outputs['estimated_demand'].detach().cpu(),
                'convergence': outputs.get('convergence_info', {}),
                'learned_alpha': outputs.get('learned_alpha'),
                'learned_beta': outputs.get('learned_beta')
            }

        if 'pred_flows' not in artifacts or 'pred_od' not in artifacts:
            raise ArtifactSchemaError(
                "Evaluation artifacts must contain required keys: pred_flows and pred_od"
            )

        eval_cont['epochs_history'][epoch + 1] = {
            'artifacts': artifacts,
        }

        # Enforce bounded history retention for both heavy/light containers.
        max_epochs_history = self._get_max_epochs_history()
        pruned_ckpt = self._prune_history_container(ckpt_cont, max_epochs_history)
        pruned_eval = self._prune_history_container(eval_cont, max_epochs_history)
        if (pruned_ckpt > 0 or pruned_eval > 0) and (epoch + 1) % self.cfg.training.get('save_frequency', 10) == 0:
            self.logger.info(
                "Checkpoint retention applied (max_epochs_history=%d): pruned ckpt=%d, eval=%d",
                max_epochs_history,
                pruned_ckpt,
                pruned_eval,
            )

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

