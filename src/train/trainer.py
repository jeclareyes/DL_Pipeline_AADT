# src/train/trainer.py

"""
General Trainer
===============

This module contains the generic training executor used by the training
pipeline.

Project context
---------------
After the data-processing refactor, the trainer should not know how to:

- read TNTP files;
- load training artifacts;
- build graphs;
- adapt routes into tensors;
- construct flow or OD targets;
- perform sampling;
- instantiate models.

Those responsibilities belong to:

- data_processing.py and TrainingArtifactBuilder;
- TrainingArtifactLoader;
- TrainingInputPreparer;
- training_pipeline.py.

The trainer receives already prepared objects:

    model
    network_params
    train_tensors
    val_tensors

and executes a training loop for one model and one task.

Design principles
-----------------
- Train one model for one task.
- Keep data ingestion outside the trainer.
- Keep model instantiation outside the trainer.
- Keep evaluation/export utilities outside the trainer.
- Save checkpoint and evaluation history.
- Return a structured training result.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
from omegaconf import DictConfig, OmegaConf
from torch.amp import autocast

from src.contracts.runtime_contracts import (
    ConfigurationContractError,
    ModelInputContractError,
    ModelOutputContractError,
    validate_model_input_contract,
    validate_model_output_contract,
)

from src.train.diagnostics.training_audits import (
    collect_forward_physics_audit,
    collect_gradient_audit,
    collect_loss_component_gradient_audit,
    should_collect_forward_physics_audit,
    should_collect_gradient_audit,
    should_collect_loss_gradient_audit,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingFitResult:
    """
    Result returned by GeneralTrainer.fit().

    Attributes
    ----------
    score : float
        Best monitored score. Lower is better.

    best_epoch : int
        Epoch with the best monitored score.

    last_epoch : int
        Last executed epoch.

    stopped_early : bool
        Whether early stopping was triggered.

    model_path : str
        Path to the saved checkpoint.

    eval_path : str
        Path to the saved evaluation history.

    history : Dict[str, Any]
        Lightweight epoch history.

    metadata : Dict[str, Any]
        Runtime metadata and diagnostics.
    """

    score: float
    best_epoch: int
    last_epoch: int
    stopped_early: bool
    model_path: str
    eval_path: str
    history: Dict[str, Any]
    metadata: Dict[str, Any]


class GeneralTrainer:
    """
    Execute the training loop for one model and one training task.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    device : str
        Torch device used for training.

    output_dir : str
        Directory where model checkpoints and eval files are saved.

    model_filename : str
        Name of the model checkpoint file.

    eval_filename : str
        Name of the evaluation history file.

    diagnostics_dir : Optional[str], default=None
        Optional directory for diagnostics. This trainer only creates the folder
        and records metadata. Heavy diagnostics should be handled elsewhere.

    is_multirun : bool, default=False
        Whether the current run is part of a Hydra multirun.
    """

    ALLOWED_MONITORS = {"auto", "train", "val"}

    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        output_dir: str,
        model_filename: str,
        eval_filename: str,
        diagnostics_dir: Optional[str] = None,
        is_multirun: bool = False,
    ) -> None:
        self.cfg = cfg
        self.device = str(device)
        self.output_dir = Path(output_dir)
        self.models_dir = Path(output_dir)
        self.model_filename = str(model_filename)
        self.eval_filename = str(eval_filename)
        self.is_multirun = bool(is_multirun)

        self.model_path = self.models_dir / self.model_filename
        self.eval_path = self.models_dir / self.eval_filename

        if diagnostics_dir is None:
            diagnostics_dir = self.output_dir / "diagnostics"

        self.diagnostics_dir = Path(diagnostics_dir)
        self.training_diagnostics_dir = self.diagnostics_dir / "training"

        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.training_diagnostics_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)

        warmup_cfg = self.cfg.training.get("warmup", {})

        # Model warmup:
        # During these epochs, the trainer passes warmup=True to the model.
        # The model decides how to interpret this flag internally.
        model_warmup_cfg = warmup_cfg.get("model", warmup_cfg)
        self.warmup_enabled = bool(model_warmup_cfg.get("enabled", False))
        self.warmup_epochs = int(model_warmup_cfg.get("epochs", 0))

        # LR warmup:
        # This is handled entirely by the trainer/scheduler and is model-agnostic.
        lr_warmup_cfg = warmup_cfg.get("lr", {})
        self.lr_warmup_enabled = bool(lr_warmup_cfg.get("enabled", False))
        self.lr_warmup_epochs = int(lr_warmup_cfg.get("epochs", 0))
        self.lr_warmup_start_factor = float(lr_warmup_cfg.get("start_factor", 0.05))

        # Runtime stability telemetry.
        self._skipped_updates = 0
        self._validation_skipped_logged = False
        self._skip_update_warning_count = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        model: torch.nn.Module,
        network_params: Dict[str, Any],
        train_tensors: Dict[str, torch.Tensor],
        val_tensors: Dict[str, torch.Tensor],
        **kwargs: Any,
    ) -> TrainingFitResult:
        """
        Train one model for one task.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train. It must return a dictionary with at least:
            - loss;
            - reconstructed_flows.

        network_params : Dict[str, Any]
            Model-ready network parameters. Used only for checkpoint metadata.

        train_tensors : Dict[str, torch.Tensor]
            Training tensors prepared by TrainingInputPreparer.

            Required keys:
            - flows
            - mask
            - od
            - od_mask

        val_tensors : Dict[str, torch.Tensor]
            Validation tensors.

            Required keys:
            - mask

        **kwargs : Any
            Unexpected kwargs raise an error to avoid hidden legacy behavior.

        Returns
        -------
        TrainingFitResult
            Structured training result.
        """

        if kwargs:
            raise ConfigurationContractError(
                f"Unexpected GeneralTrainer.fit kwargs: {sorted(kwargs.keys())}"
            )

        validate_model_input_contract(train_tensors, val_tensors)

        model = model.to(self.device)

        train_flows_t = self._ensure_batch_dim(train_tensors["flows"])
        train_mask_t = self._ensure_batch_dim(train_tensors["mask"])
        train_od_t = self._ensure_batch_dim(train_tensors["od"])
        train_od_mask_t = self._ensure_batch_dim(train_tensors["od_mask"])
        val_mask_t = self._ensure_batch_dim(val_tensors["mask"])

        has_val = bool(val_mask_t.sum().item() > 0)

        training_controls = self._resolve_training_controls(
            has_val=has_val,
        )

        optimizer = self._build_optimizer(model)
        scheduler = self._build_scheduler(optimizer)

        checkpoint_container = self._build_checkpoint_container(
            network_params=network_params,
        )
        
        eval_container = self._build_eval_container(
            train_tensors=train_tensors,
            val_tensors=val_tensors,
            network_params=network_params,
        )

        history: Dict[str, Any] = {}

        best_scheduler_score = float("inf")
        best_early_score = float("inf")
        best_epoch = 0
        last_epoch = 0
        epochs_no_improve = 0
        stopped_early = False

        self.logger.info(
            "Training started | epochs=%d | lr=%.6g | weight_decay=%.6g | device=%s",
            int(self.cfg.training.epochs),
            float(self.cfg.training.lr),
            float(self.cfg.training.weight_decay),
            self.device,
        )

        self._audit_optimizer_registration(
            optimizer=optimizer,
        )

        for epoch_idx in range(1, int(self.cfg.training.epochs) + 1):
            last_epoch = epoch_idx

            train_result = self._train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_flows_t=train_flows_t,
                train_mask_t=train_mask_t,
                train_od_t=train_od_t,
                train_od_mask_t=train_od_mask_t,
                epoch_idx=epoch_idx,
            )

            if train_result["skipped_update"]:
                self._skipped_updates += 1
                history[epoch_idx] = train_result
                continue

            val_result = self._validate(
                model=model,
                all_flows=train_flows_t,
                train_mask=train_mask_t,
                val_mask=val_mask_t,
            )

            train_loss = float(train_result["train_loss"])
            val_mse = val_result["val_mse"]

            monitor_score = self._select_monitor_score(
                train_loss=train_loss,
                val_mse=val_mse,
                monitor_name=training_controls["scheduler_monitor_name"],
            )

            early_score = self._select_monitor_score(
                train_loss=train_loss,
                val_mse=val_mse,
                monitor_name=training_controls["early_monitor_name"],
            )

            self._step_scheduler(
                scheduler=scheduler,
                monitor_score=monitor_score,
            )

            epoch_record = {
                **train_result,
                **val_result,
                "monitor_score": float(monitor_score),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }

            history[epoch_idx] = epoch_record
            checkpoint_container["epochs_history"][epoch_idx] = epoch_record
            eval_container["epochs_history"][epoch_idx] = epoch_record

            if monitor_score < best_scheduler_score:
                best_scheduler_score = float(monitor_score)

            if early_score < best_early_score - training_controls["early_min_delta"]:
                best_early_score = float(early_score)
                best_epoch = epoch_idx
                epochs_no_improve = 0

                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch_idx=epoch_idx,
                    score=early_score,
                    checkpoint_container=checkpoint_container,
                    eval_container=eval_container,
                    is_best=True,
                )
            else:
                epochs_no_improve += 1

            if self._should_save_epoch(epoch_idx):
                self._save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch_idx=epoch_idx,
                    score=monitor_score,
                    checkpoint_container=checkpoint_container,
                    eval_container=eval_container,
                    is_best=False,
                )

            self.logger.info(
                "Epoch %d/%d | train_loss=%.6f | val_mse=%s | monitor=%.6f | lr=%.3e",
                epoch_idx,
                int(self.cfg.training.epochs),
                train_loss,
                f"{val_mse:.6f}" if val_mse is not None else "None",
                float(monitor_score),
                float(optimizer.param_groups[0]["lr"]),
            )

            if self._should_stop_early(
                training_controls=training_controls,
                epochs_no_improve=epochs_no_improve,
            ):
                stopped_early = True
                self.logger.info(
                    "Early stopping triggered at epoch %d after %d epochs without improvement.",
                    epoch_idx,
                    epochs_no_improve,
                )
                break

        final_score = (
            best_early_score
            if math.isfinite(best_early_score)
            else best_scheduler_score
        )

        self._save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch_idx=last_epoch,
            score=final_score,
            checkpoint_container=checkpoint_container,
            eval_container=eval_container,
            is_best=False,
            force_filename=self.model_filename,
        )

        self._save_eval_container(eval_container)

        result = TrainingFitResult(
            score=float(final_score),
            best_epoch=int(best_epoch),
            last_epoch=int(last_epoch),
            stopped_early=bool(stopped_early),
            model_path=str(self.model_path),
            eval_path=str(self.eval_path),
            history=history,
            metadata={
                "device": self.device,
                "has_validation": has_val,
                "skipped_updates": int(self._skipped_updates),
                "scheduler_monitor": training_controls["scheduler_monitor_name"],
                "early_monitor": training_controls["early_monitor_name"],
                "warmup_enabled": self.warmup_enabled,
                "warmup_epochs": self.warmup_epochs,
                "lr_warmup_enabled": bool(self.lr_warmup_enabled),
                "lr_warmup_epochs": int(self.lr_warmup_epochs),
                "gradient_control": OmegaConf.to_container(
                    self.cfg.training.get("gradient_control", {}),
                    resolve=True,
                ),
            },
        )

        self.logger.info(
            "Training completed | score=%.6f | best_epoch=%d | last_epoch=%d | stopped_early=%s",
            result.score,
            result.best_epoch,
            result.last_epoch,
            result.stopped_early,
        )

        return result

    # ------------------------------------------------------------------
    # Epoch execution
    # ------------------------------------------------------------------

    def _train_one_epoch(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        train_flows_t: torch.Tensor,
        train_mask_t: torch.Tensor,
        train_od_t: torch.Tensor,
        train_od_mask_t: torch.Tensor,
        epoch_idx: int,
    ) -> Dict[str, Any]:
        """
        Execute one training epoch.

        Parameters
        ----------
        model : torch.nn.Module
            Model to train.

        optimizer : torch.optim.Optimizer
            Optimizer.

        train_flows_t : torch.Tensor
            Flow target tensor.

        train_mask_t : torch.Tensor
            Flow training mask.

        train_od_t : torch.Tensor
            OD target tensor.

        train_od_mask_t : torch.Tensor
            OD supervision mask.

        epoch_idx : int
            One-based epoch index.

        Returns
        -------
        Dict[str, Any]
            Epoch training metrics.
        """

        model.train()
        optimizer.zero_grad(set_to_none=True)

        warmup_flag = self.warmup_enabled and epoch_idx <= self.warmup_epochs

        amp_enabled = self._amp_enabled()
        amp_device_type = "cuda" if self.device.startswith("cuda") else "cpu"

        with autocast(
            device_type=amp_device_type,
            enabled=amp_enabled,
        ):
            outputs = model(
                observed_flows=train_flows_t,
                flow_mask=train_mask_t,
                true_od_demand=train_od_t,
                od_mask=train_od_mask_t,
                warmup=warmup_flag,
                current_epoch=epoch_idx,
            )

        outputs = validate_model_output_contract(outputs)

        # -- Block for starting diagnostics collection. -----------------
        forward_physics_audit = {}

        if should_collect_forward_physics_audit(
            cfg=self.cfg,
            epoch_idx=epoch_idx,
        ):
            forward_physics_audit = collect_forward_physics_audit(
                outputs=outputs,
            )

        # -- End of diagnostics collection block -----------------------

        loss_dict = outputs.get("loss")
        if not isinstance(loss_dict, dict):
            raise ModelOutputContractError(
                "Model output must contain loss as a dictionary."
            )

        if "total_loss" not in loss_dict:
            raise ModelOutputContractError(
                "Model loss dictionary must contain 'total_loss'."
            )

        total_loss = loss_dict["total_loss"]

        if not torch.is_tensor(total_loss):
            raise ModelOutputContractError(
                "loss['total_loss'] must be a torch.Tensor."
            )

        if not bool(torch.isfinite(total_loss).all().item()):
            self.logger.error(
                "Skipping epoch %d because total_loss is non-finite: %s",
                epoch_idx,
                str(total_loss.detach().cpu()),
            )

            optimizer.zero_grad(set_to_none=True)

            return {
                "epoch": int(epoch_idx),
                "train_loss": float("inf"),
                "skipped_update": True,
                "warmup": bool(warmup_flag),
                "loss": self._detach_loss_dict(loss_dict),
            }


        # -- Collection of loss gradient for audit purposes. -----------------
        # Optional per-loss gradient diagnostics.
        # This must run before total_loss.backward() because it uses autograd.grad().
        loss_gradient_audit = {} 

        if should_collect_loss_gradient_audit(
            cfg=self.cfg,
            epoch_idx=epoch_idx,
        ):
            loss_gradient_audit = collect_loss_component_gradient_audit(
                model=model,
                loss_dict=loss_dict,
            )

        total_loss.backward()

        gradient_audit = {}

        if should_collect_gradient_audit(
            cfg=self.cfg,
            epoch_idx=epoch_idx,
        ):
            gradient_audit = collect_gradient_audit(
                model=model,
            )

        # -- End of loss gradient collection block --------------------------------

        # ------------------------------------------------------------------
        # Gradient health check
        # ------------------------------------------------------------------
        # Non-finite gradients are destructive and should still trigger skip logic.
        # Large but finite gradients should first be handled by clipping.
        non_finite_gradients = self._sanitize_non_finite_gradients(model)

        pre_clip_grad_norm = self._compute_gradient_norm(model)

        skip_due_to_non_finite, skip_reason = self._should_skip_optimizer_update(
            grad_norm=pre_clip_grad_norm,
            non_finite_gradients=non_finite_gradients,
        )

        if skip_due_to_non_finite and non_finite_gradients > 0:
            self._log_skipped_update(
                epoch_idx=epoch_idx,
                reason=skip_reason,
                grad_norm=pre_clip_grad_norm,
                non_finite_gradients=non_finite_gradients,
            )

            optimizer.zero_grad(set_to_none=True)

            return {
                "epoch": int(epoch_idx),
                "train_loss": float(total_loss.detach().cpu().item()),
                "skipped_update": True,
                "skip_reason": str(skip_reason),
                "warmup": bool(warmup_flag),
                "grad_norm": float(pre_clip_grad_norm),
                "pre_clip_grad_norm": float(pre_clip_grad_norm),
                "post_clip_grad_norm": float("nan"),
                "clip_applied": False,
                "non_finite_gradients": int(non_finite_gradients),
                "loss": self._detach_loss_dict(loss_dict),
                "gradient_audit": gradient_audit,
                "loss_gradient_audit": loss_gradient_audit,
                "forward_physics_audit": forward_physics_audit,
            }

        # ------------------------------------------------------------------
        # Clip finite gradients before deciding whether they are destructive.
        # ------------------------------------------------------------------
        clip_result = self._clip_gradients(model)

        post_clip_grad_norm = float(clip_result["post_clip_grad_norm"])

        skip_update, skip_reason = self._should_skip_optimizer_update(
            grad_norm=post_clip_grad_norm,
            non_finite_gradients=0,
        )

        if skip_update:
            self._log_skipped_update(
                epoch_idx=epoch_idx,
                reason=f"post_clip_{skip_reason}",
                grad_norm=post_clip_grad_norm,
                non_finite_gradients=0,
            )

            optimizer.zero_grad(set_to_none=True)

            return {
                "epoch": int(epoch_idx),
                "train_loss": float(total_loss.detach().cpu().item()),
                "skipped_update": True,
                "skip_reason": f"post_clip_{skip_reason}",
                "warmup": bool(warmup_flag),
                "grad_norm": float(pre_clip_grad_norm),
                "pre_clip_grad_norm": float(pre_clip_grad_norm),
                "post_clip_grad_norm": float(post_clip_grad_norm),
                "clip_applied": bool(clip_result["clip_applied"]),
                "non_finite_gradients": int(non_finite_gradients),
                "loss": self._detach_loss_dict(loss_dict),
                "gradient_audit": gradient_audit,
                "loss_gradient_audit": loss_gradient_audit,
                "forward_physics_audit": forward_physics_audit,
            }

        optimizer.step()

        ###########################################################################

        """
        return {
            "epoch": int(epoch_idx),
            "train_loss": float(total_loss.detach().cpu().item()),
            "skipped_update": False,
            "skip_reason": None,
            "warmup": bool(warmup_flag),
            "grad_norm": float(clip_result["pre_clip_grad_norm"]),
            "pre_clip_grad_norm": float(clip_result["pre_clip_grad_norm"]),
            "post_clip_grad_norm": float(clip_result["post_clip_grad_norm"]),
            "clip_applied": bool(clip_result["clip_applied"]),
            "non_finite_gradients": int(non_finite_gradients),
            "loss": self._detach_loss_dict(loss_dict),
            "gradient_audit": gradient_audit,
            "loss_gradient_audit": loss_gradient_audit,
        }"""

        ###########################################################################

        return {
            "epoch": int(epoch_idx),
            "train_loss": float(total_loss.detach().cpu().item()),
            "skipped_update": False,
            "skip_reason": None,
            "warmup": bool(warmup_flag),
            "grad_norm": float(pre_clip_grad_norm),
            "pre_clip_grad_norm": float(pre_clip_grad_norm),
            "post_clip_grad_norm": float(post_clip_grad_norm),
            "clip_applied": bool(clip_result["clip_applied"]),
            "non_finite_gradients": int(non_finite_gradients),
            "loss": self._detach_loss_dict(loss_dict),
            "gradient_audit": gradient_audit,
            "loss_gradient_audit": loss_gradient_audit,
            "forward_physics_audit": forward_physics_audit,
        }

        ##############################################################################


    def _validate(
        self,
        model: torch.nn.Module,
        all_flows: torch.Tensor,
        train_mask: torch.Tensor,
        val_mask: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Validate the model on held-out flow links.

        Parameters
        ----------
        model : torch.nn.Module
            Trained model.

        all_flows : torch.Tensor
            Full flow target tensor.

        train_mask : torch.Tensor
            Mask used as model input.

        val_mask : torch.Tensor
            Validation mask.

        Returns
        -------
        Dict[str, Any]
            Validation metrics.
        """

        if val_mask.sum().item() <= 0:
            if not self._validation_skipped_logged:
                self.logger.info(
                    "Validation skipped because validation mask is empty."
                )
                self._validation_skipped_logged = True

            return {
                "val_mse": None,
                "val_mae": None,
                "val_count": 0,
            }

        model.eval()

        with torch.no_grad():
            outputs = model(
                observed_flows=all_flows,
                flow_mask=train_mask,
                warmup=False,
            )

        outputs = validate_model_output_contract(outputs)

        if "reconstructed_flows" not in outputs:
            raise ModelOutputContractError(
                "Model output must contain 'reconstructed_flows' during validation."
            )

        prediction = outputs["reconstructed_flows"]

        prediction = self._ensure_batch_dim(prediction)
        all_flows = self._ensure_batch_dim(all_flows)
        val_mask = self._ensure_batch_dim(val_mask)

        mask_bool = val_mask > 0

        y_pred = prediction[mask_bool]
        y_true = all_flows[mask_bool]

        if y_true.numel() == 0:
            return {
                "val_mse": None,
                "val_mae": None,
                "val_count": 0,
            }

        mse = torch.mean((y_pred - y_true) ** 2)
        mae = torch.mean(torch.abs(y_pred - y_true))

        return {
            "val_mse": float(mse.detach().cpu().item()),
            "val_mae": float(mae.detach().cpu().item()),
            "val_count": int(y_true.numel()),
        }

    # ------------------------------------------------------------------
    # Optimizer and scheduler
    # ------------------------------------------------------------------

    def _build_optimizer(
        self,
        model: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        """
        Build the optimizer from YAML configuration.

        The trainer remains model-agnostic:
        - If the model exposes get_optimizer_param_groups(), those groups are used.
        - Otherwise, all trainable model parameters are optimized together.

        Supported optimizers:
        - adam
        - adamw
        - sgd
        - nadam
        - radam
        """

        optimizer_cfg = self.cfg.training.get("optimizer", "adam")

        if isinstance(optimizer_cfg, str):
            optimizer_name = optimizer_cfg.lower()
            optimizer_options = {}
        else:
            optimizer_name = str(optimizer_cfg.get("name", "adam")).lower()
            optimizer_options = dict(optimizer_cfg)

        base_lr = float(optimizer_options.get("lr", self.cfg.training.get("lr", 1e-3)))
        weight_decay = float(
            optimizer_options.get(
                "weight_decay",
                self.cfg.training.get("weight_decay", 0.0),
            )
        )

        param_groups = None

        allow_model_param_groups = bool(
            optimizer_options.get("allow_model_param_groups", True)
        )

        if allow_model_param_groups and hasattr(model, "get_optimizer_param_groups"):
            try:
                param_groups = model.get_optimizer_param_groups(
                    base_lr=base_lr,
                    weight_decay=weight_decay,
                )
            except TypeError:
                param_groups = model.get_optimizer_param_groups(base_lr)

        params = param_groups if param_groups else model.parameters()

        if optimizer_name in {"adam", "adamw", "nadam", "radam"}:
            betas = tuple(optimizer_options.get("betas", (0.9, 0.999)))
            eps = float(optimizer_options.get("eps", 1e-8))

            if optimizer_name == "adam":
                return torch.optim.Adam(
                    params,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                    amsgrad=bool(optimizer_options.get("amsgrad", False)),
                )

            if optimizer_name == "adamw":
                return torch.optim.AdamW(
                    params,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                    amsgrad=bool(optimizer_options.get("amsgrad", False)),
                )

            if optimizer_name == "nadam":
                return torch.optim.NAdam(
                    params,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                    momentum_decay=float(optimizer_options.get("momentum_decay", 0.004)),
                )

            if optimizer_name == "radam":
                return torch.optim.RAdam(
                    params,
                    lr=base_lr,
                    weight_decay=weight_decay,
                    betas=betas,
                    eps=eps,
                )

        if optimizer_name == "sgd":
            return torch.optim.SGD(
                params,
                lr=base_lr,
                weight_decay=weight_decay,
                momentum=float(optimizer_options.get("momentum", self.cfg.training.get("momentum", 0.0))),
                nesterov=bool(optimizer_options.get("nesterov", False)),
            )

        raise ConfigurationContractError(
            f"Unsupported optimizer '{optimizer_name}'. "
            "Supported values: adam|adamw|sgd|nadam|radam."
        )
    
    
    def _build_scheduler(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
        """
        Build the learning-rate scheduler.

        Supported scheduler types:
        - reduce_on_plateau
        - step
        - cosine
        - cosine_with_warmup
        - one_cycle
        - none
        """

        scheduler_cfg = self.cfg.training.get("scheduler", {})
        enabled = bool(scheduler_cfg.get("enabled", True))

        if not enabled:
            return None

        scheduler_type = str(
            scheduler_cfg.get("type", "reduce_on_plateau")
        ).lower()

        if scheduler_type == "none":
            return None

        if scheduler_type == "reduce_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=float(scheduler_cfg.get("factor", 0.5)),
                patience=int(scheduler_cfg.get("patience", 10)),
                min_lr=float(scheduler_cfg.get("min_lr", 1e-7)),
                threshold=float(scheduler_cfg.get("threshold", 1e-4)),
                threshold_mode=str(scheduler_cfg.get("threshold_mode", "rel")),
                cooldown=int(scheduler_cfg.get("cooldown", 0)),
            )

        if scheduler_type == "step":
            return torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=int(scheduler_cfg.get("step_size", 50)),
                gamma=float(scheduler_cfg.get("gamma", 0.5)),
            )

        if scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=int(scheduler_cfg.get("t_max", self.cfg.training.epochs)),
                eta_min=float(scheduler_cfg.get("min_lr", 1e-7)),
            )

        if scheduler_type == "cosine_with_warmup":
            warmup_epochs = int(
                scheduler_cfg.get(
                    "warmup_epochs",
                    self.lr_warmup_epochs if self.lr_warmup_enabled else 0,
                )
            )

            total_epochs = int(self.cfg.training.epochs)
            min_lr = float(scheduler_cfg.get("min_lr", 1e-7))

            if warmup_epochs <= 0:
                return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_epochs, 1),
                    eta_min=min_lr,
                )

            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=float(
                    scheduler_cfg.get("warmup_start_factor", self.lr_warmup_start_factor)
                ),
                end_factor=1.0,
                total_iters=max(warmup_epochs, 1),
            )

            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(total_epochs - warmup_epochs, 1),
                eta_min=min_lr,
            )

            return torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )

        if scheduler_type == "one_cycle":
            max_lr = scheduler_cfg.get("max_lr", None)

            if max_lr is None:
                max_lr = max(
                    float(group.get("lr", self.cfg.training.get("lr", 1e-3)))
                    for group in optimizer.param_groups
                )

            return torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(max_lr),
                epochs=int(self.cfg.training.epochs),
                steps_per_epoch=1,
                pct_start=float(scheduler_cfg.get("pct_start", 0.15)),
                div_factor=float(scheduler_cfg.get("div_factor", 25.0)),
                final_div_factor=float(scheduler_cfg.get("final_div_factor", 1000.0)),
                anneal_strategy=str(scheduler_cfg.get("anneal_strategy", "cos")),
            )

        raise ConfigurationContractError(
            f"Unsupported scheduler type '{scheduler_type}'. "
            "Supported values: reduce_on_plateau|step|cosine|cosine_with_warmup|one_cycle|none."
        )


    @staticmethod
    def _step_scheduler(
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
        monitor_score: float,
    ) -> None:
        """
        Step the scheduler.

        ReduceLROnPlateau requires a monitored score.
        Other schedulers advance once per optimizer update/epoch.
        """

        if scheduler is None:
            return

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(float(monitor_score))
            return

        scheduler.step()

    # ------------------------------------------------------------------
    # Monitoring and early stopping
    # ------------------------------------------------------------------

    def _resolve_training_controls(
        self,
        has_val: bool,
    ) -> Dict[str, Any]:
        """
        Resolve scheduler and early-stopping monitor configuration.

        Parameters
        ----------
        has_val : bool
            Whether validation data is available.

        Returns
        -------
        Dict[str, Any]
            Training control configuration.
        """

        scheduler_monitor_cfg = str(
            self.cfg.training.get("scheduler_monitor", "auto")
        ).lower()

        if scheduler_monitor_cfg not in self.ALLOWED_MONITORS:
            raise ConfigurationContractError(
                f"Unknown training.scheduler_monitor='{scheduler_monitor_cfg}'. "
                "Allowed values: auto|train|val"
            )

        scheduler_monitor_name = self._resolve_monitor_name(
            requested=scheduler_monitor_cfg,
            has_val=has_val,
            fallback_to_train=True,
        )

        early_cfg = self.cfg.training.get("early_stopping", {})
        early_enabled = bool(early_cfg.get("enabled", False))
        early_monitor_cfg = str(early_cfg.get("monitor", "auto")).lower()

        if early_monitor_cfg not in self.ALLOWED_MONITORS:
            raise ConfigurationContractError(
                f"Unknown training.early_stopping.monitor='{early_monitor_cfg}'. "
                "Allowed values: auto|train|val"
            )

        early_monitor_name = self._resolve_monitor_name(
            requested=early_monitor_cfg,
            has_val=has_val,
            fallback_to_train=True,
        )

        early_patience = int(early_cfg.get("patience", 20))
        early_min_delta = float(early_cfg.get("min_delta", 0.0))
        early_active = early_enabled and early_monitor_name is not None

        self.logger.info(
            "Training controls | scheduler_monitor=%s | early_stopping=%s",
            scheduler_monitor_name,
            "on" if early_active else "off",
        )

        return {
            "scheduler_monitor_name": scheduler_monitor_name,
            "early_monitor_name": early_monitor_name,
            "early_active": bool(early_active),
            "early_patience": int(early_patience),
            "early_min_delta": float(early_min_delta),
        }

    @staticmethod
    def _resolve_monitor_name(
        requested: str,
        has_val: bool,
        fallback_to_train: bool,
    ) -> Optional[str]:
        """
        Resolve monitor name from user configuration.

        Parameters
        ----------
        requested : str
            Requested monitor mode: auto, train or val.

        has_val : bool
            Whether validation data is available.

        fallback_to_train : bool
            Whether to fall back to train_loss when validation is unavailable.

        Returns
        -------
        Optional[str]
            Monitor metric name.
        """

        if requested == "train":
            return "train_loss"

        if requested == "val":
            if has_val:
                return "val_mse"
            return "train_loss" if fallback_to_train else None

        if requested == "auto":
            return "val_mse" if has_val else "train_loss"

        return None

    @staticmethod
    def _select_monitor_score(
        train_loss: float,
        val_mse: Optional[float],
        monitor_name: Optional[str],
    ) -> float:
        """
        Select score used for scheduler or early stopping.

        Parameters
        ----------
        train_loss : float
            Training loss.

        val_mse : Optional[float]
            Validation MSE.

        monitor_name : Optional[str]
            Selected monitor name.

        Returns
        -------
        float
            Score. Lower is better.
        """

        if monitor_name == "val_mse" and val_mse is not None:
            return float(val_mse)

        return float(train_loss)

    @staticmethod
    def _should_stop_early(
        training_controls: Dict[str, Any],
        epochs_no_improve: int,
    ) -> bool:
        """
        Decide whether early stopping should stop training.

        Parameters
        ----------
        training_controls : Dict[str, Any]
            Training control configuration.

        epochs_no_improve : int
            Number of epochs without improvement.

        Returns
        -------
        bool
            True if early stopping should stop.
        """

        if not training_controls["early_active"]:
            return False

        return epochs_no_improve >= int(training_controls["early_patience"])

    # ------------------------------------------------------------------
    # Saving
    # ------------------------------------------------------------------

    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch_idx: int,
        score: float,
        checkpoint_container: Dict[str, Any],
        eval_container: Dict[str, Any],
        is_best: bool,
        force_filename: Optional[str] = None,
    ) -> None:
        """
        Save model checkpoint and eval container.

        Parameters
        ----------
        model : torch.nn.Module
            Model.

        optimizer : torch.optim.Optimizer
            Optimizer.

        epoch_idx : int
            Current epoch.

        score : float
            Monitored score.

        checkpoint_container : Dict[str, Any]
            Checkpoint metadata.

        eval_container : Dict[str, Any]
            Evaluation metadata.

        is_best : bool
            Whether this is the best checkpoint.

        force_filename : Optional[str], default=None
            Optional checkpoint filename.
        """

        filename = force_filename or (
            self.model_filename
            if not is_best
            else self._best_model_filename(self.model_filename)
        )

        path = self.models_dir / filename

        checkpoint = {
            "schema_version": 2,
            "epoch": int(epoch_idx),
            "score": float(score),
            "is_best": bool(is_best),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "metadata": checkpoint_container,
        }

        torch.save(checkpoint, path)
        self._save_eval_container(eval_container)

    def _save_eval_container(
        self,
        eval_container: Dict[str, Any],
    ) -> None:
        """
        Save evaluation history container.

        Parameters
        ----------
        eval_container : Dict[str, Any]
            Evaluation history container.
        """

        torch.save(eval_container, self.eval_path)

    def _should_save_epoch(
        self,
        epoch_idx: int,
    ) -> bool:
        """
        Decide whether to save an intermediate checkpoint.

        Parameters
        ----------
        epoch_idx : int
            Current epoch.

        Returns
        -------
        bool
            True if checkpoint should be saved.
        """

        checkpoint_cfg = self.cfg.training.get("checkpointing", {})
        save_every = int(checkpoint_cfg.get("save_every", 0))

        if save_every <= 0:
            return False

        return epoch_idx % save_every == 0

    @staticmethod
    def _best_model_filename(model_filename: str) -> str:
        """
        Build best-model checkpoint filename.

        Parameters
        ----------
        model_filename : str
            Base model filename.

        Returns
        -------
        str
            Best-model filename.
        """

        path = Path(model_filename)

        if path.suffix:
            return f"{path.stem}_best{path.suffix}"

        return f"{model_filename}_best.pt"

    # ------------------------------------------------------------------
    # Containers and metadata
    # ------------------------------------------------------------------

    def _build_checkpoint_container(
        self,
        network_params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build base checkpoint metadata container.

        If network_params is provided, store alignment metadata so that a saved
        checkpoint can be interpreted without relying only on an external artifact.

        Returns
        -------
        Dict[str, Any]
            Checkpoint metadata.
        """

        container = {
        "config": OmegaConf.to_container(self.cfg, resolve=True),
        "epochs_history": {},
    }

        if network_params is not None:
            container["static_data"] = {
                "num_links": int(network_params.get("num_links", -1)),
                "num_od_pairs": int(network_params.get("num_od_pairs", -1)),
                "k_paths": int(network_params.get("k_paths", -1)),
                "alignment": self._build_alignment_payload(
                    network_params=network_params,
                ),
            }

        return container

    def _build_eval_container(
        self,
        train_tensors: Dict[str, torch.Tensor],
        val_tensors: Dict[str, torch.Tensor],
        network_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build evaluation history container.

        Parameters
        ----------
        train_tensors : Dict[str, torch.Tensor]
            Training tensors.

        val_tensors : Dict[str, torch.Tensor]
            Validation tensors.

        network_params : Dict[str, Any]
            Network parameters.

        Returns
        -------
        Dict[str, Any]
            Evaluation container.
        """

        return {
            "schema_version": 2,
            "config": OmegaConf.to_container(self.cfg, resolve=True),
            "static_data": self._pack_static_data(
                train_tensors=train_tensors,
                val_tensors=val_tensors,
                network_params=network_params,
            ),
            "epochs_history": {},
        }

    def _pack_static_data(
        self,
        train_tensors: Dict[str, torch.Tensor],
        val_tensors: Dict[str, torch.Tensor],
        network_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Pack lightweight static data for evaluation files.

        This payload is intentionally small but must preserve enough alignment
        metadata to interpret reconstructed_flows[i] and od[i] outside the live
        training process.

        Parameters
        ----------
        train_tensors : Dict[str, torch.Tensor]
            Training tensors.

        val_tensors : Dict[str, torch.Tensor]
            Validation tensors.

        network_params : Dict[str, Any]
            Network parameters.

        Returns
        -------
        Dict[str, Any]
            Static metadata.
        """

        alignment_payload = self._build_alignment_payload(
            network_params=network_params,
        )

        return {
            "num_links": int(network_params.get("num_links", -1)),
            "num_od_pairs": int(network_params.get("num_od_pairs", -1)),
            "k_paths": int(network_params.get("k_paths", -1)),
            "train_flows_shape": tuple(train_tensors["flows"].shape),
            "train_mask_shape": tuple(train_tensors["mask"].shape),
            "od_shape": tuple(train_tensors["od"].shape),
            "od_mask_shape": tuple(train_tensors["od_mask"].shape),
            "val_mask_shape": tuple(val_tensors["mask"].shape),
            "network_keys": sorted(network_params.keys()),
            "alignment": alignment_payload,
        }


    def _build_alignment_payload(
        self,
        network_params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build a JSON/checkpoint-safe alignment payload.

        The model operates positionally. This payload records what each position
        means:
            - reconstructed_flows[i] corresponds to link_pair_indices[i]
            - od[i] corresponds to od_pairs[i]

        Large tensors are converted to CPU lists for portability.
        """

        payload: Dict[str, Any] = {}

        if "link_pair_indices" in network_params:
            payload["link_pair_indices"] = self._to_cpu_list(
                network_params["link_pair_indices"]
            )

        if "od_pairs" in network_params:
            payload["od_pairs"] = self._normalize_od_pairs_for_storage(
                network_params["od_pairs"]
            )

        if "od_pair_indices" in network_params:
            payload["od_pair_indices"] = self._to_cpu_list(
                network_params["od_pair_indices"]
            )

        if "node_id_to_idx" in network_params:
            payload["node_id_to_idx"] = self._normalize_scalar_mapping_for_storage(
                network_params["node_id_to_idx"]
            )

        if "idx_to_node_id" in network_params:
            payload["idx_to_node_id"] = self._normalize_scalar_mapping_for_storage(
                network_params["idx_to_node_id"]
            )

        if "edge_to_idx" in network_params:
            payload["edge_to_idx"] = self._normalize_pair_mapping_for_storage(
                network_params["edge_to_idx"]
            )

        if "idx_to_edge" in network_params:
            payload["idx_to_edge"] = self._normalize_idx_to_pair_mapping_for_storage(
                network_params["idx_to_edge"]
            )

        return payload   

    # ------------------------------------------------------------------
    # Tensor helpers
    # ------------------------------------------------------------------

    def _ensure_batch_dim(
        self,
        tensor: torch.Tensor,
    ) -> torch.Tensor:
        """
        Ensure a tensor has batch dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Tensor with batch dimension.
        """

        if not torch.is_tensor(tensor):
            raise ModelInputContractError(
                f"Expected torch.Tensor, got {type(tensor)}."
            )

        tensor = tensor.to(self.device)

        if tensor.ndim == 1:
            return tensor.unsqueeze(0)

        return tensor


    def _clip_gradients(
        self,
        model: torch.nn.Module,
    ) -> Dict[str, Any]:
        """
        Clip gradients according to YAML configuration.

        Supported configuration styles
        ------------------------------
        Legacy:
            training.grad_clip_norm: 5.0

        New:
            training.gradient_control.clip.enabled: true
            training.gradient_control.clip.max_norm: 5.0
            training.gradient_control.clip.norm_type: 2.0

        Returns
        -------
        Dict[str, Any]
            Gradient clipping telemetry.
        """

        gradient_cfg = self.cfg.training.get("gradient_control", {})
        clip_cfg = gradient_cfg.get("clip", {})

        legacy_clip_norm = self.cfg.training.get("grad_clip_norm", 0.0)

        clip_enabled = bool(
            clip_cfg.get(
                "enabled",
                legacy_clip_norm is not None and float(legacy_clip_norm) > 0.0,
            )
        )

        max_norm = float(
            clip_cfg.get(
                "max_norm",
                legacy_clip_norm if legacy_clip_norm is not None else 0.0,
            )
        )

        norm_type = float(clip_cfg.get("norm_type", 2.0))

        pre_clip_norm = self._compute_gradient_norm(
            model=model,
            norm_type=norm_type,
        )

        if not clip_enabled or max_norm <= 0.0:
            return {
                "pre_clip_grad_norm": float(pre_clip_norm),
                "post_clip_grad_norm": float(pre_clip_norm),
                "clip_applied": False,
            }

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=False,
        )

        post_clip_norm = self._compute_gradient_norm(
            model=model,
            norm_type=norm_type,
        )

        return {
            "pre_clip_grad_norm": float(pre_clip_norm),
            "post_clip_grad_norm": float(post_clip_norm),
            "clip_applied": bool(pre_clip_norm > max_norm),
        }


    def _compute_gradient_norm(
            self,
            model: torch.nn.Module,
            norm_type: float = 2.0,
        ) -> float:
        """
        Compute the global gradient norm without modifying gradients.
        """

        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.grad is not None
        ]

        if not parameters:
            return 0.0

        device = parameters[0].grad.device

        if norm_type == float("inf"):
            total_norm = max(
                parameter.grad.detach().abs().max().to(device)
                for parameter in parameters
            )
            return float(total_norm.detach().cpu().item())

        norms = torch.stack(
            [
                torch.norm(parameter.grad.detach(), p=norm_type).to(device)
                for parameter in parameters
            ]
        )

        total_norm = torch.norm(norms, p=norm_type)

        return float(total_norm.detach().cpu().item())


    def _should_skip_optimizer_update(
        self,
        grad_norm: float,
        non_finite_gradients: int,
    ) -> Tuple[bool, str]:
        """
        Decide whether to skip the optimizer update for the current epoch.

        This is a safety mechanism for unstable training. It is intentionally
        model-agnostic: it only observes gradient health.
        """

        gradient_cfg = self.cfg.training.get("gradient_control", {})
        skip_cfg = gradient_cfg.get("skip_update", {})

        enabled = bool(skip_cfg.get("enabled", False))

        if not enabled:
            return False, ""

        if bool(skip_cfg.get("on_non_finite", True)) and non_finite_gradients > 0:
            return True, "non_finite_gradients"

        max_grad_norm = skip_cfg.get("max_grad_norm", None)

        if max_grad_norm is not None:
            max_grad_norm = float(max_grad_norm)

            if grad_norm > max_grad_norm:
                return True, "grad_norm_above_threshold"

        return False, ""


    def _log_skipped_update(
        self,
        epoch_idx: int,
        reason: str,
        grad_norm: float,
        non_finite_gradients: int,
    ) -> None:
        """
        Log skipped optimizer updates with rate limiting to avoid noisy logs.
        """

        gradient_cfg = self.cfg.training.get("gradient_control", {})
        skip_cfg = gradient_cfg.get("skip_update", {})

        log_enabled = bool(skip_cfg.get("log", True))
        max_warnings = int(skip_cfg.get("max_logged_warnings", 20))

        if not log_enabled:
            return

        if self._skip_update_warning_count >= max_warnings:
            return

        self._skip_update_warning_count += 1

        self.logger.warning(
            "Skipping optimizer update at epoch %d | reason=%s | grad_norm=%.6e | non_finite_gradients=%d",
            int(epoch_idx),
            str(reason),
            float(grad_norm),
            int(non_finite_gradients),
        )

    @staticmethod
    def _sanitize_non_finite_gradients(
        model: torch.nn.Module,
    ) -> int:
        """
        Replace non-finite gradient entries with zero.

        Parameters
        ----------
        model : torch.nn.Module
            Model.

        Returns
        -------
        int
            Number of sanitized gradient entries.
        """

        total_bad_entries = 0

        for parameter in model.parameters():
            if parameter.grad is None:
                continue

            finite_mask = torch.isfinite(parameter.grad)

            if bool(torch.all(finite_mask).item()):
                continue

            bad_entries = int((~finite_mask).sum().item())
            total_bad_entries += bad_entries

            parameter.grad.data = torch.nan_to_num(
                parameter.grad.data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        return total_bad_entries

    @staticmethod
    def _detach_loss_dict(
        loss_dict: Dict[str, Any],
    ) -> Dict[str, float]:
        """
        Convert loss dictionary to plain floats where possible.

        Parameters
        ----------
        loss_dict : Dict[str, Any]
            Model loss dictionary.

        Returns
        -------
        Dict[str, float]
            Detached scalar loss dictionary.
        """

        detached = {}

        for key, value in loss_dict.items():
            if torch.is_tensor(value):
                if value.numel() == 1:
                    detached[key] = float(value.detach().cpu().item())
                else:
                    detached[key] = float(value.detach().cpu().mean().item())
            else:
                try:
                    detached[key] = float(value)
                except (TypeError, ValueError):
                    continue

        return detached

    def _amp_enabled(self) -> bool:
        """
        Decide whether automatic mixed precision is enabled.

        Returns
        -------
        bool
            True if AMP should be used.
        """

        amp_cfg = self.cfg.training.get("amp", {})
        enabled = bool(amp_cfg.get("enabled", False))

        return enabled and self.device.startswith("cuda")

    def _audit_optimizer_registration(
        self,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """
        Audit optimizer parameter registration.

        Parameters
        ----------
        optimizer : torch.optim.Optimizer
            Optimizer.
        """

        if not bool(self.cfg.training.get("audit_optimizer", False)):
            return

        self.logger.info("--- Optimizer registration audit ---")

        for group_idx, group in enumerate(optimizer.param_groups):
            for param_idx, param in enumerate(group["params"]):
                if not param.is_leaf:
                    self.logger.error(
                        "Optimizer group %d param %d is not a leaf tensor.",
                        group_idx,
                        param_idx,
                    )

                if not param.requires_grad:
                    self.logger.error(
                        "Optimizer group %d param %d does not require gradients.",
                        group_idx,
                        param_idx,
                    )

    @staticmethod
    def _to_cpu_list(value: Any) -> Any:
        """
        Convert tensor/array-like values to CPU Python lists.
        """

        if torch.is_tensor(value):
            return value.detach().cpu().tolist()

        try:
            return value.tolist()
        except AttributeError:
            return value


    @staticmethod
    def _normalize_od_pairs_for_storage(value: Any) -> list[list[int]]:
        """
        Normalize OD pairs to a JSON/checkpoint-safe list of [origin, destination].
        """

        if torch.is_tensor(value):
            raw_pairs = value.detach().cpu().tolist()
        else:
            raw_pairs = value

        return [
            [int(origin), int(destination)]
            for origin, destination in raw_pairs
        ]


    @staticmethod
    def _normalize_scalar_mapping_for_storage(value: Any) -> Dict[str, int]:
        """
        Normalize scalar-key mappings to string-key dictionaries for storage.

        This is safe for checkpoint metadata because it is only for inspection and
        reconstruction, not for direct model instantiation.
        """

        return {
            str(int(key)): int(item)
            for key, item in dict(value).items()
        }


    @staticmethod
    def _normalize_pair_mapping_for_storage(value: Any) -> Dict[str, int]:
        """
        Normalize pair-key mappings such as {(u, v): idx} for storage.
        """

        normalized = {}

        for key, item in dict(value).items():
            if isinstance(key, str):
                storage_key = key
            else:
                u, v = key
                storage_key = f"{int(u)}->{int(v)}"

            normalized[storage_key] = int(item)

        return normalized


    @staticmethod
    def _normalize_idx_to_pair_mapping_for_storage(value: Any) -> Dict[str, list[int]]:
        """
        Normalize mappings such as {idx: (u, v)} for storage.
        """

        normalized = {}

        for key, item in dict(value).items():
            u, v = item
            normalized[str(int(key))] = [int(u), int(v)]

        return normalized
