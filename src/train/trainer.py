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

    def fit(self, model, network_params, train_tensors, val_tensors):
        """
        Ejecuta el entrenamiento completo (Epoch 1 -> N).

        Args:
            model: Instancia del modelo (UltraCyclic, etc.)
            network_params: Dict con parámetros estáticos de la red (t0, cap, etc.) para guardado.
            train_tensors: Dict con {flows, mask, od, od_mask} para entrenar.
            val_tensors: Dict con {flows, mask} para validar (puede ser Test set o Validation set).
        """
        # 1. Setup Optimizador y Loss
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )

        criterion = self._get_loss_function()
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

        # 3. Variables de Control
        best_val_loss = float('inf')
        early_stop = False
        epochs_no_improve = 0
        patience = self.cfg.training.early_stopping.patience

        # --- BUCLE DE ÉPOCAS ---
        for epoch in range(self.cfg.training.epochs):
            if early_stop:
                break

            model.train()
            optimizer.zero_grad()

            # A. Lógica Dinámica (Warmup & Curriculum)
            current_w_od = 0.0 if epoch < 20 else self.cfg.model.loss_weights.w_od
            if hasattr(criterion, 'w_od'):
                criterion.w_od = torch.tensor(current_w_od, device=self.device)

            current_iters = int(min(10, 2 + epoch * 0.2))  # Curriculum SUE

            # B. Forward & Backward
            outputs = model(
                observed_flows=train_tensors['flows'],
                flow_mask=train_tensors['mask'],
                true_od_demand=train_tensors['od'],
                warmup=(epoch < 5),
                # override_max_iters=current_iters TODO fix
            )

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.training.grad_clip_norm)
            optimizer.step()

            current_loss = loss_dict['total_loss'].item()

            # C. Validación
            val_loss = self._validate(model, train_tensors['flows'], val_tensors['mask'])

            # Scheduler Step
            if scheduler:
                scheduler.step(val_loss)

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if self.cfg.training.early_stopping.enabled and epochs_no_improve >= patience:
                    early_stop = True
                    self.logger.info(f"Early stopping en época {epoch + 1}")

            # D. Logging & Saving
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch + 1}: Train Loss {current_loss:.4f} | Val MSE {val_loss:.4f}")

            self._save_checkpoint(
                epoch, model, optimizer, current_loss, loss_dict, outputs,
                checkpoint_container, eval_container, early_stop
            )

        return best_val_loss

    def _validate(self, model, all_flows, val_mask):
        model.eval()
        with torch.no_grad():
            val_out = model(observed_flows=all_flows, flow_mask=val_mask, warmup=False)
            pred = val_out['reconstructed_flows']
            # MSE en enlaces de validación
            mse = torch.sum((pred - all_flows) ** 2 * val_mask) / (torch.sum(val_mask) + 1e-6)
        model.train()
        return mse.item()

    def _get_loss_function(self):
        # Instancia Loss usando Hydra o Fallback
        if hasattr(self.cfg.model, 'loss') and '_target_' in self.cfg.model.loss:
            return hydra.utils.instantiate(self.cfg.model.loss).to(self.device)
        from src.components.models.CGAME_MLP_SUE import PartialDataLoss
        return PartialDataLoss(
            w_flow=self.cfg.model.loss_weights.w_flow,
            w_od=self.cfg.model.loss_weights.w_od,
            w_reg=self.cfg.model.loss_weights.w_reg
        ).to(self.device)

    def _get_scheduler(self, optimizer):
        if self.cfg.training.scheduler == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min',
                factor=self.cfg.training.scheduler_params.factor,
                patience=self.cfg.training.scheduler_params.patience,
                min_lr=self.cfg.training.scheduler_params.min_lr
            )
        return None

    def _pack_static_data(self, train_t, val_t, net_p):
        """
        Empaqueta tensores estáticos necesarios para reconstruir el escenario en Testing.
        """
        return {
            # Ground Truth (para comparar en testing)
            'true_flows': train_t['flows'].cpu(),
            'true_od': train_t['od'].cpu(),

            # Física de la red (necesaria para el simulador SUE en testing)
            'capacity': net_p['capacity'].cpu(),
            't0': net_p['t0'].cpu(),
            'route_masks': net_p['route_masks'].cpu(),  # <--- AGREGADO IMPORTANTE
            'od_pair_indices': net_p['od_pair_indices'].cpu(),  # <--- AGREGADO IMPORTANTE

            # Máscaras de entrenamiento vs validación
            'masks': {
                'flow_train': train_t['mask'].cpu().bool(),
                'flow_val': val_t['mask'].cpu().bool(),
                'od_mask': train_t['od_mask'].cpu().bool()
            }
        }


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