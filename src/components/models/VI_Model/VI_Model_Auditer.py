from __future__ import annotations

import logging
from typing import Dict, Optional
import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class VIDiagnostician:
    """
    Training diagnostics companion for `VariationalInferenceModel`.

    Responsibilities:
        - track flow fit metrics (R2, MAE) and loss components across epochs
        - monitor solver convergence signals (`iterations`, `final_gap`, mode)
        - monitor gradient health for key modules and OD logits
        - audit forward/backward consistency when IMD is enabled
        - export summary JSON, diagnostic CSV tables, and plotting artifacts
    """

    def __init__(
        self,
        history_window: int = 100,
        gap_alert_threshold: float = 1e-6,
        od_grad_low_threshold: float = 1e-8,
        r2_stall_window: int = 10,
        r2_stall_delta: float = 1e-4,
        enabled: bool = True,
        enable_assignment_audit: bool = True,
        enable_route_flow_history: bool = True,
        enable_flow_conservation_audit: bool = True,
        enable_mass_conservation_audit: bool = True,
        enable_link_type_audit: bool = True,
        enable_flow_demand_history: bool = True,
        enable_forward_backward_audit: bool = True,
        enable_imd_gap_history: bool = True,
        enable_gradient_history: bool = True,
    ):
        # Global toggle used by every diagnostic method.
        self.enabled = bool(enabled)
        # Individual toggles for optional diagnostics.
        self.enable_assignment_audit = bool(enable_assignment_audit)
        self.enable_route_flow_history = bool(enable_route_flow_history)
        self.enable_flow_conservation_audit = bool(enable_flow_conservation_audit)
        self.enable_mass_conservation_audit = bool(enable_mass_conservation_audit)
        self.enable_link_type_audit = bool(enable_link_type_audit)
        self.enable_flow_demand_history = bool(enable_flow_demand_history)
        self.enable_forward_backward_audit = bool(enable_forward_backward_audit)
        self.enable_imd_gap_history = bool(enable_imd_gap_history)
        self.enable_gradient_history = bool(enable_gradient_history)

        self.window = int(history_window)
        self.gap_alert_threshold = float(gap_alert_threshold)
        self.od_grad_low_threshold = float(od_grad_low_threshold)
        self.r2_stall_window = int(r2_stall_window)
        self.r2_stall_delta = float(r2_stall_delta)
        self.tb_logger = None
        self.val_freq = 10

        self.full_history = {
            "r2_flow": [],
            "mae_flow": [],
            "total_loss": [],
            "l_flow": [],
            "l_od": [],
            "l_demand_reg": [],
            "iterations": [],
            "final_gap": [],
            "converged": [],
            "mode": [],
            "max_grad": [],
            "od_logits_grad": [],
            "grad_norms": {},
            "alerts": [],
            "learned_theta": [],
        }
        self.window_history = {
            "r2_flow": [],
            "mae_flow": [],
            "max_grad": [],
        }

        self.gradient_detailed_rows = []
        self.link_type_epoch_rows = []
        self.flow_comparison_rows = []
        self.demand_comparison_rows = []
        self.forward_backward_audit_rows = []
        self.imd_relative_gap_rows = []
        self.od_matrix_rows = []

        # Initialize route flow tracking if the model provides route-level outputs
        self.route_flow_rows = []
        self.flow_conservation_rows = []
        # Threshold for link-level flow conservation checks (absolute error).
        self.flow_conservation_tolerance = 1e-2

        # Initialize mass balance tracking for OD demand
        self.mass_conservation_rows = []
        # Threshold for node-level mass conservation checks (absolute error).
        self.mass_conservation_tolerance = 1e-2

    def attach_tensorboard(self, tb_logger, val_freq: int):
        """
        Attach a TensorBoard logger for lightweight/periodic diagnostics.

        Diagnostics meaning:
            Adds scalar/histogram summaries for solver, loss, and physics
            parameters to the TensorBoard run.

        Storage:
            TensorBoard event files handled by the caller's logger.
        """
        self.tb_logger = tb_logger
        self.val_freq = val_freq

    def _to_float(self, value, default: float = 0.0) -> float:
        """
        Normalize arbitrary inputs to a scalar float for history storage.

        Diagnostics meaning:
            Keeps history arrays compact and safe even when tensors are empty.

        Storage:
            Stored only in memory (history buffers).
        """
        if value is None:
            return float(default)
        if torch.is_tensor(value):
            if value.numel() == 0:
                return float(default)
            return float(value.detach().mean().cpu().item())
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    def _append_scalar(self, key: str, value, default: float = 0.0):
        """
        Append a scalar value to the full history buffer.

        Diagnostics meaning:
            Records scalar metrics for long-term trend analysis.

        Storage:
            Stored only in memory; exported in save_summary().
        """
        self.full_history[key].append(self._to_float(value, default=default))

    def _push_window(self, key: str, value: float):
        """
        Push a scalar into the rolling window history buffer.

        Diagnostics meaning:
            Keeps a recent window for alert heuristics (stall detection).

        Storage:
            Stored only in memory; used for alert logic.
        """
        self.window_history[key].append(value)
        if len(self.window_history[key]) > self.window:
            self.window_history[key].pop(0)

    def update(self, outputs: Dict, targets: Dict, model=None, **kwargs):
        """
        Ingest one training/eval step and append all tracked diagnostics.

        Diagnostics meaning:
            Computes flow fit (R2/MAE), loss components, solver convergence,
            and optional physics/IMD audits for the current epoch.

        Storage:
            In-memory buffers that are exported by save_summary() into
            diagnostics CSV/JSON files.
        """
        if not self.enabled:
            return
        with torch.no_grad():
            epoch = int(kwargs.get("epoch", len(self.full_history["r2_flow"]) + 1))
            is_final = bool(kwargs.get("is_final", False)) # <--- Extract final-epoch flag
            is_heavy_log_epoch = epoch % self.val_freq == 0

            pred_flow = outputs.get("reconstructed_flows")
            true_flow = targets.get("flows")
            mask_flow = targets.get("flow_mask")
            if mask_flow is None:
                mask_flow = targets.get("mask")

            if pred_flow is None or true_flow is None:
                return
            if mask_flow is None:
                mask_flow = torch.ones_like(true_flow)

            if pred_flow.dim() == 2 and pred_flow.shape[0] == 1:
                pred_flow = pred_flow.squeeze(0)
            if true_flow.dim() == 2 and true_flow.shape[0] == 1:
                true_flow = true_flow.squeeze(0)
            if mask_flow.dim() == 2 and mask_flow.shape[0] == 1:
                mask_flow = mask_flow.squeeze(0)

            mask_bool = mask_flow > 0
            if mask_bool.sum() <= 0:
                return

            y_pred = pred_flow[mask_bool].detach().cpu().numpy()
            y_true = true_flow[mask_bool].detach().cpu().numpy()
            mae = float(np.mean(np.abs(y_true - y_pred)))

            if len(y_true) > 1 and np.var(y_true) > 0:
                ss_res = float(np.sum((y_true - y_pred) ** 2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
            else:
                r2 = 0.0

            # --- TensorBoard Scalar Logging (Lightweight, every epoch) ---
            if self.tb_logger is not None:
                self.tb_logger.log_scalar("Metrics_Flow/R2", r2, epoch)
                self.tb_logger.log_scalar("Metrics_Flow/MAE", mae, epoch)
                
                loss_dict = kwargs.get("loss_dict") or outputs.get("loss") or {}
                if isinstance(loss_dict, dict):
                    self.tb_logger.log_scalar("Loss/Total", loss_dict.get("total_loss", 0), epoch)
                    self.tb_logger.log_scalar("Loss/Flow", loss_dict.get("l_flow", 0), epoch)
                    self.tb_logger.log_scalar("Loss/OD_Prior", loss_dict.get("l_od", 0), epoch)
                    self.tb_logger.log_scalar("Loss/Demand_Reg", loss_dict.get("l_demand_reg", 0), epoch)

                conv_info = outputs.get("convergence_info", {}) or {}
                self.tb_logger.log_scalar("Equilibrium/Iterations", conv_info.get("iterations", 0), epoch)
                self.tb_logger.log_scalar("Equilibrium/Final_Gap", conv_info.get("final_gap", 0), epoch)
                
                # Check for IMD consistency tracking
                if conv_info.get("implicit_grad", False) and model is not None:
                    try:
                        bwd_info = model.equilibrium_solver._imd_last_backward_info
                        self.tb_logger.log_scalar("IMD/Fixed_Point_Residual", bwd_info.get("fixed_point_residual", 0), epoch)
                    except AttributeError:
                        pass
                
                # --- TensorBoard Heavy Logging (Gated) ---
                if is_heavy_log_epoch:
                    if outputs.get("learned_alpha") is not None:
                        self.tb_logger.log_histogram("Physics/Alpha", outputs["learned_alpha"], epoch)
                    if outputs.get("learned_beta") is not None:
                        self.tb_logger.log_histogram("Physics/Beta", outputs["learned_beta"], epoch)
                    if outputs.get("learned_capacity_multiplier") is not None:
                        self.tb_logger.log_histogram("Physics/Capacity_Multiplier", outputs["learned_capacity_multiplier"], epoch)

            # -----------------------------------------------------

            self.full_history["r2_flow"].append(r2)
            self.full_history["mae_flow"].append(mae)
            self._push_window("r2_flow", r2)
            self._push_window("mae_flow", mae)

            loss_dict = kwargs.get("loss_dict") or outputs.get("loss") or {}
            if isinstance(loss_dict, dict):
                self._append_scalar("total_loss", loss_dict.get("total_loss"), default=0.0)
                self._append_scalar("l_flow", loss_dict.get("l_flow"), default=0.0)
                self._append_scalar("l_od", loss_dict.get("l_od"), default=0.0)
                self._append_scalar("l_demand_reg", loss_dict.get("l_demand_reg"), default=0.0)

            conv_info = outputs.get("convergence_info", {}) or {}
            self._append_scalar("iterations", conv_info.get("iterations"), default=0.0)
            self._append_scalar("final_gap", conv_info.get("final_gap"), default=0.0)
            self._append_scalar("converged", conv_info.get("converged"), default=0.0)
            self._append_scalar("learned_theta", outputs.get("learned_theta"), default=0.0)
            self.full_history["mode"].append(str(conv_info.get("mode", "physics_stage1")))
            if self.enable_imd_gap_history:
                self._collect_imd_relative_gap_rows(epoch=epoch, conv_info=conv_info)

            static_info = kwargs.get("static_info", {}) or {}
            if self.enable_link_type_audit:
                self._collect_link_type_rows(
                    epoch=epoch,
                    outputs=outputs,
                    static_info=static_info,
                )
            if self.enable_flow_demand_history:
                self._collect_flow_demand_rows(
                    epoch=epoch,
                    outputs=outputs,
                    targets=targets,
                    static_info=static_info,
                )
            if self.enable_forward_backward_audit:
                self._collect_forward_backward_audit_row(
                    epoch=epoch,
                    outputs=outputs,
                    model=model,
                )

            self._record_alerts(epoch=len(self.full_history["r2_flow"]) - 1)

        # --- ROUTE FLOW HISTORY (optional) ---
        # By default we only capture the final epoch to keep files small.
        if (
            is_final
            and model is not None
            and (self.enable_route_flow_history or self.enable_flow_conservation_audit)
        ):
                route_flows = outputs.get("route_flows")
                pred_link_flows = outputs.get("reconstructed_flows")
                solver = getattr(model, "equilibrium_solver", None)
                
                if route_flows is not None and pred_link_flows is not None and solver is not None:
                    # 1. Route flow table (per OD, per path).
                    if self.enable_route_flow_history:
                        mask = solver.route_validity_mask 
                        num_od, k_paths = mask.shape

                        rf_np = route_flows.detach().cpu().numpy().reshape(-1)
                        mask_np = mask.detach().cpu().numpy().reshape(-1)

                        for i in range(len(rf_np)):
                            if mask_np[i]:
                                self.route_flow_rows.append({
                                    "epoch": epoch,
                                    "od_index": i // k_paths,
                                    "route_index": i % k_paths,
                                    "route_flow": float(rf_np[i])
                                })

                    # 2. Flow conservation audit: f = Delta * h.
                    if self.enable_flow_conservation_audit:
                        delta_matrix = solver.delta_matrix
                    
                        # Ensure dimensions [Routes] and [Links].
                        rf_t = route_flows.detach().squeeze()
                        pf_t = pred_link_flows.detach().squeeze()
                    
                        if rf_t.dim() == 1 and pf_t.dim() == 1:
                            # Sparse matrix multiplication: [Links, Routes] x [Routes, 1].
                            rf_col = rf_t.unsqueeze(1) 
                            f_recon_t = torch.sparse.mm(delta_matrix, rf_col).squeeze(1) 

                            pf_np = pf_t.cpu().numpy()
                            f_recon_np = f_recon_t.cpu().numpy()

                            abs_err = np.abs(pf_np - f_recon_np)
                            # Protect against division by zero for relative errors.
                            rel_err = abs_err / np.maximum(pf_np, 1e-9)

                            tolerance = self.flow_conservation_tolerance

                            for link_idx in range(len(pf_np)):
                                self.flow_conservation_rows.append({
                                    "epoch": epoch,
                                    "link_id": link_idx,
                                    "estimated_flow": float(pf_np[link_idx]),
                                    "reconstructed_flow": float(f_recon_np[link_idx]),
                                    "abs_error": float(abs_err[link_idx]),
                                    "rel_error": float(rel_err[link_idx]),
                                    "status": "OK" if abs_err[link_idx] <= tolerance else "FAIL"
                                })

        # --- NODE MASS CONSERVATION AUDIT (optional): A * f = E ---
        if (
            self.enable_mass_conservation_audit
            and is_final
            and model is not None
        ):
            link_nodes = static_info.get("link_pair_indices") 
            
            # Use raw string labels if available to avoid tensor index mapping mismatches.
            od_nodes = static_info.get("od_pair_node_labels")
            if od_nodes is None:
                od_nodes = static_info.get("od_pair_indices")
            
            if link_nodes is not None and od_nodes is not None:
                net_demand = {}
                pred_od_np = outputs.get("estimated_demand").detach().cpu().numpy().reshape(-1)
                
                # 1. Vector E: net demand per node (string labels).
                for idx, pair in enumerate(od_nodes):
                    # Force string casting to unify the ID space
                    o, d = str(pair[0]), str(pair[1]) 
                    dem = float(pred_od_np[idx])
                    net_demand[o] = net_demand.get(o, 0.0) + dem
                    net_demand[d] = net_demand.get(d, 0.0) - dem

                net_flows = {}
                pred_flow_np = outputs.get("reconstructed_flows").detach().cpu().numpy().reshape(-1)
                
                # 2. Vector A*f: net flow per node (string labels).
                for idx, pair in enumerate(link_nodes):
                    u, v = str(pair[0]), str(pair[1])
                    f = float(pred_flow_np[idx])
                    net_flows[u] = net_flows.get(u, 0.0) + f
                    net_flows[v] = net_flows.get(v, 0.0) - f

                # 3. Vector comparison.
                all_nodes = set(net_demand.keys()).union(set(net_flows.keys()))
                
                for node in all_nodes:
                    d_net = net_demand.get(node, 0.0)
                    f_net = net_flows.get(node, 0.0)
                    abs_err = abs(d_net - f_net)
                    
                    self.mass_conservation_rows.append({
                        "epoch": epoch,
                        "node_id": node, 
                        "net_demand_generated": d_net,
                        "net_flow_routed": f_net,
                        "abs_error": abs_err,
                        "status": "OK" if abs_err <= self.mass_conservation_tolerance else "FAIL"
                    })


    def _collect_forward_backward_audit_row(self, epoch: int, outputs: Dict, model: Optional[nn.Module]):
        """
        Record IMD forward/backward consistency and residual indicators.

        Diagnostics meaning:
            Checks whether the implicit gradient backward linearization matches
            the forward iteration state for IMD epochs.

        Storage:
            Stored in forward_backward_audit_rows and exported to
            forward_backward_consistency_audit.csv in save_summary().
        """
        conv_info = outputs.get("convergence_info", {}) if isinstance(outputs, dict) else {}
        if not isinstance(conv_info, dict):
            conv_info = {}

        mode = str(conv_info.get("mode", "unknown"))
        implicit_grad = bool(conv_info.get("implicit_grad", False))

        fwd_iterations = self._to_float(conv_info.get("iterations"), default=0.0)
        fwd_final_gap = self._to_float(conv_info.get("final_gap"), default=0.0)
        fwd_linearization_iter = self._to_float(conv_info.get("linearization_iter"), default=0.0)

        bwd_available = False
        bwd_iter_idx = 0.0
        bwd_fixed_point_residual = 0.0
        bwd_z_norm = 0.0
        bwd_grad_output_norm = 0.0
        bwd_iters = 0.0
        bwd_damping = 0.0

        if model is not None and hasattr(model, "equilibrium_solver"):
            solver = getattr(model, "equilibrium_solver")
            if hasattr(solver, "_imd_last_info") and isinstance(solver._imd_last_info, dict):
                if fwd_linearization_iter <= 0.0:
                    fwd_linearization_iter = self._to_float(
                        solver._imd_last_info.get("linearization_iter"),
                        default=fwd_iterations,
                    )
            if hasattr(solver, "_imd_last_backward_info") and isinstance(solver._imd_last_backward_info, dict):
                bwd_info = solver._imd_last_backward_info
                bwd_available = bool(self._to_float(bwd_info.get("available"), default=0.0) > 0.5)
                bwd_iter_idx = self._to_float(bwd_info.get("iter_idx"), default=0.0)
                bwd_fixed_point_residual = self._to_float(bwd_info.get("fixed_point_residual"), default=0.0)
                bwd_z_norm = self._to_float(bwd_info.get("z_norm"), default=0.0)
                bwd_grad_output_norm = self._to_float(bwd_info.get("grad_output_norm"), default=0.0)
                bwd_iters = self._to_float(bwd_info.get("backward_iters"), default=0.0)
                bwd_damping = self._to_float(bwd_info.get("damping"), default=0.0)

        iter_match = False
        if implicit_grad and bwd_available:
            iter_match = int(round(fwd_linearization_iter)) == int(round(bwd_iter_idx))
            if not iter_match:
                self.full_history["alerts"].append((int(epoch), "imd_iter_mismatch", float(abs(fwd_linearization_iter - bwd_iter_idx))))

        status = "non_imd"
        if implicit_grad and bwd_available:
            status = "ok" if iter_match else "iter_mismatch"
        elif implicit_grad and (not bwd_available):
            status = "missing_backward_info"

        self.forward_backward_audit_rows.append(
            {
                "epoch": int(epoch),
                "mode": mode,
                "implicit_grad": int(implicit_grad),
                "forward_iterations": float(fwd_iterations),
                "forward_final_gap": float(fwd_final_gap),
                "forward_linearization_iter": float(fwd_linearization_iter),
                "backward_info_available": int(bwd_available),
                "backward_iter_idx": float(bwd_iter_idx),
                "iter_match": int(iter_match) if implicit_grad else 1,
                "backward_fixed_point_residual": float(bwd_fixed_point_residual),
                "backward_z_norm": float(bwd_z_norm),
                "backward_grad_output_norm": float(bwd_grad_output_norm),
                "backward_iters": float(bwd_iters),
                "backward_damping": float(bwd_damping),
                "status": status,
            }
        )

    def _collect_imd_relative_gap_rows(self, epoch: int, conv_info: Dict):
        """
        Store per-iteration relative-gap traces for IMD epochs.

        Diagnostics meaning:
            Captures solver convergence trajectories inside IMD for
            post-mortem analysis.

        Storage:
            Stored in imd_relative_gap_rows and exported to
            imd_relative_gap_history_by_epoch.csv in save_summary().
        """
        if not isinstance(conv_info, dict):
            return
        raw_history = conv_info.get("relative_gap_history", None)
        if raw_history is None:
            return
        if isinstance(raw_history, np.ndarray):
            history = raw_history.tolist()
        elif isinstance(raw_history, (list, tuple)):
            history = list(raw_history)
        else:
            return

        mode = str(conv_info.get("mode", "unknown"))
        implicit_grad = int(bool(conv_info.get("implicit_grad", False)))
        for iter_idx, gap_val in enumerate(history, start=1):
            try:
                gap_float = float(gap_val)
            except (TypeError, ValueError):
                continue
            self.imd_relative_gap_rows.append(
                {
                    "epoch": int(epoch),
                    "imd_iteration": int(iter_idx),
                    "relative_gap": float(gap_float),
                    "mode": mode,
                    "implicit_grad": int(implicit_grad),
                }
            )

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        """
        Capture module-wise and parameter-wise gradient norms for health checks.

        Diagnostics meaning:
            Monitors gradient magnitude and numerical stability to detect
            vanishing/exploding gradients.

        Storage:
            Stored in gradient_detailed_rows and exported to
            gradient_history_detailed.csv in save_summary().
        """
        if not self.enabled or not self.enable_gradient_history:
            return
        key_modules = {
            "supply_net": getattr(model, "supply_net", None),
            "equilibrium_solver": getattr(model, "equilibrium_solver", None),
        }

        max_grad = 0.0
        is_heavy_log_epoch = (epoch + 1) % self.val_freq == 0  
              
        for name, module in key_modules.items():
            if module is None:
                continue
            total_norm_sq = 0.0
            named_params = list(module.named_parameters())
            if len(named_params) == 0:
                self.gradient_detailed_rows.append(
                    {
                        "epoch": int(epoch) + 1,
                        "module": str(name),
                        "param_name": "__no_trainable_params__",
                        "norm2": 0.0,
                        "max_abs": 0.0,
                        "mean_abs": 0.0,
                        "finite_ratio": 0.0,
                        "non_finite_count": 0,
                    }
                )
            for param_name, p in named_params:
                row = self._build_gradient_row(
                    epoch=int(epoch) + 1,
                    module=name,
                    param_name=f"{name}.{param_name}",
                    grad=(p.grad.detach() if p.grad is not None else None),
                )
                self.gradient_detailed_rows.append(row)
                 
                if p.grad is not None:
                    g = p.grad.data.norm(2).item()
                    total_norm_sq += g * g
            total_norm = total_norm_sq ** 0.5
            self.full_history["grad_norms"].setdefault(name, []).append(total_norm)
            max_grad = max(max_grad, total_norm)

        od_logits_grad = 0.0
        if hasattr(model, "od_logits") and getattr(model, "od_logits") is not None:
            grad = model.od_logits.grad
            if grad is not None:
                od_logits_grad = float(grad.detach().norm(2).item())
            self.gradient_detailed_rows.append(
                self._build_gradient_row(
                    epoch=int(epoch) + 1,
                    module="od_logits",
                    param_name="od_logits",
                    grad=(grad.detach() if grad is not None else None),
                )
            )

            if self.tb_logger is not None and is_heavy_log_epoch and grad is not None:
                self.tb_logger.log_histogram("Gradients/od_logits", grad, epoch + 1)

        self.full_history["max_grad"].append(float(max_grad))
        self.full_history["od_logits_grad"].append(float(od_logits_grad))
        self._push_window("max_grad", float(max_grad))

        if epoch >= 2 and od_logits_grad < self.od_grad_low_threshold:
            self.full_history["alerts"].append((int(epoch), "od_logits_low_grad", float(od_logits_grad)))

    def _record_alerts(self, epoch: int):
        """
        Emit heuristic alerts for poor convergence or stalled validation dynamics.

        Diagnostics meaning:
            Adds warning events when solver gap is high or R2 stalls.

        Storage:
            Stored in full_history["alerts"] and summarized in save_summary().
        """
        final_gap = self.full_history["final_gap"][-1] if self.full_history["final_gap"] else 0.0
        if final_gap > self.gap_alert_threshold:
            self.full_history["alerts"].append((int(epoch), "high_final_gap", float(final_gap)))

        if len(self.full_history["r2_flow"]) >= self.r2_stall_window:
            tail = self.full_history["r2_flow"][-self.r2_stall_window:]
            drift = float(max(tail) - min(tail))
            if drift < self.r2_stall_delta:
                self.full_history["alerts"].append((int(epoch), "r2_stall", drift))

    def _build_gradient_row(self, epoch: int, module: str, param_name: str, grad: Optional[torch.Tensor]) -> Dict[str, float]:
        """
        Build one gradient diagnostics row for CSV export.

        Diagnostics meaning:
            Provides per-parameter gradient norms and finite ratios.

        Storage:
            Stored in gradient_detailed_rows and exported in save_summary().
        """
        if grad is None:
            return {
                "epoch": int(epoch),
                "module": str(module),
                "param_name": str(param_name),
                "norm2": 0.0,
                "max_abs": 0.0,
                "mean_abs": 0.0,
                "finite_ratio": 0.0,
                "non_finite_count": 0,
            }

        g = grad.detach().reshape(-1)
        finite_mask = torch.isfinite(g)
        finite_count = int(finite_mask.sum().item())
        total_count = int(g.numel())
        non_finite_count = int(total_count - finite_count)
        finite_ratio = float(finite_count / max(total_count, 1))

        if finite_count > 0:
            g_finite = g[finite_mask]
            norm2 = float(torch.norm(g_finite, p=2).item())
            max_abs = float(torch.max(torch.abs(g_finite)).item())
            mean_abs = float(torch.mean(torch.abs(g_finite)).item())
        else:
            norm2 = 0.0
            max_abs = 0.0
            mean_abs = 0.0

        return {
            "epoch": int(epoch),
            "module": str(module),
            "param_name": str(param_name),
            "norm2": norm2,
            "max_abs": max_abs,
            "mean_abs": mean_abs,
            "finite_ratio": finite_ratio,
            "non_finite_count": non_finite_count,
        }

    def _collect_link_type_rows(self, epoch: int, outputs: Dict, static_info: Dict):
        """
        Aggregate alpha/beta/volume-capacity stats grouped by link type.

        Diagnostics meaning:
            Summarizes how learned physics parameters vary by road category.

        Storage:
            Stored in link_type_epoch_rows and exported to
            physics/link_type_alpha_beta_vc_history.csv in save_summary().
        """
        alpha = outputs.get("learned_alpha")
        beta = outputs.get("learned_beta")
        pred_flow = outputs.get("reconstructed_flows")
        capacity = static_info.get("capacity", None)
        link_types = static_info.get("link_types", None)

        cap_mult = outputs.get("learned_capacity_multiplier", None)

        if alpha is None or beta is None or pred_flow is None or capacity is None or link_types is None:
            return

        alpha_np = alpha.detach().cpu().numpy().reshape(-1)
        beta_np = beta.detach().cpu().numpy().reshape(-1)

        pred_flow_np = pred_flow.detach().cpu().numpy()
        if pred_flow_np.ndim > 1:
            pred_flow_np = pred_flow_np.reshape(pred_flow_np.shape[0], -1)[0]
        else:
            pred_flow_np = pred_flow_np.reshape(-1)

        cap_np = np.asarray(capacity).reshape(-1)



        types_np = np.asarray(link_types).reshape(-1).astype(str)

        n = min(len(alpha_np), len(beta_np), len(pred_flow_np), len(cap_np), len(types_np))
        if n <= 0:
            return

        cap_np = np.clip(cap_np[:n], a_min=1e-9, a_max=None)
        # Ajustar la capacidad base con el multiplicador aprendido
        if cap_mult is not None:
            cap_mult_np = cap_mult.detach().cpu().numpy().reshape(-1)[:n]
            adj_cap_np = cap_np * cap_mult_np
        else:
            adj_cap_np = cap_np

        alpha_np = alpha_np[:n]
        beta_np = beta_np[:n]
        pred_flow_np = pred_flow_np[:n]
        cap_np = np.clip(cap_np[:n], a_min=1e-9, a_max=None)
        types_np = types_np[:n]

        vc_np = pred_flow_np / np.clip(adj_cap_np, a_min=1e-9, a_max=None)

        for link_type in sorted(set(types_np.tolist())):
            mask = types_np == link_type
            if not np.any(mask):
                continue
            self.link_type_epoch_rows.append(
                {
                    "epoch": int(epoch),
                    "link_type": str(link_type),
                    "count_links": int(mask.sum()),
                    "alpha_mean": float(np.mean(alpha_np[mask])),
                    "beta_mean": float(np.mean(beta_np[mask])),
                    "estimated_volume_mean": float(np.mean(pred_flow_np[mask])),
                    "capacity_mean": float(np.mean(cap_np[mask])),
                    "vc_mean": float(np.mean(vc_np[mask])),
                    "vc_p50": float(np.percentile(vc_np[mask], 50)),
                    "vc_p95": float(np.percentile(vc_np[mask], 95)),
                }
            )

    def _collect_flow_demand_rows(self, epoch: int, outputs: Dict, targets: Dict, static_info: Optional[Dict] = None):
        """
        Store row-wise comparison tables for estimated vs target flow and demand.

        Diagnostics meaning:
            Creates per-link and per-OD comparison tables to audit errors
            at the finest granularity.

        Storage:
            Stored in flow_comparison_rows / demand_comparison_rows and
            exported to true_vs_estimated_*_history.csv in save_summary().
        """
        static_info = static_info or {}
        pred_flow = outputs.get("reconstructed_flows")
        true_flow = targets.get("flows")
        flow_mask = targets.get("mask", targets.get("flow_mask", None))

        if pred_flow is not None and true_flow is not None:
            pred_flow_np = pred_flow.detach().cpu().numpy().reshape(-1)
            true_flow_np = true_flow.detach().cpu().numpy().reshape(-1)
            n = min(len(pred_flow_np), len(true_flow_np))
            if n > 0:
                if flow_mask is not None:
                    flow_mask_np = flow_mask.detach().cpu().numpy().reshape(-1)
                else:
                    flow_mask_np = np.ones(n, dtype=np.float32)
                flow_mask_np = np.asarray(flow_mask_np).reshape(-1)
                m = min(n, len(flow_mask_np))
                for i in range(m):
                    self.flow_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "link_index": int(i),
                            "true_flow": float(true_flow_np[i]),
                            "estimated_flow": float(pred_flow_np[i]),
                            "flow_mask": float(flow_mask_np[i]),
                        }
                    )

        pred_od = outputs.get("estimated_demand")
        true_od = targets.get("od")
        od_mask = targets.get("od_mask")
        if pred_od is not None and true_od is not None:
            pred_od_np = pred_od.detach().cpu().numpy().reshape(-1)
            true_od_np = true_od.detach().cpu().numpy().reshape(-1)
            n = min(len(pred_od_np), len(true_od_np))
            if n > 0:
                if od_mask is not None:
                    od_mask_np = od_mask.detach().cpu().numpy().reshape(-1)
                else:
                    od_mask_np = np.ones(n, dtype=np.float32)
                od_mask_np = np.asarray(od_mask_np).reshape(-1)
                m = min(n, len(od_mask_np))
                od_pair_indices = static_info.get("od_pair_indices", None)
                if torch.is_tensor(od_pair_indices):
                    od_pair_indices = od_pair_indices.detach().cpu().numpy()
                if od_pair_indices is not None:
                    od_pair_indices = np.asarray(od_pair_indices)
                known_rank = 0
                unknown_rank = 0
                for i in range(m):
                    mask_val = float(od_mask_np[i])
                    is_known = int(mask_val > 0.5)
                    if is_known:
                        known_rank += 1
                        known_od_index = int(known_rank)
                        unknown_od_index = -1
                        od_status = "known"
                    else:
                        unknown_rank += 1
                        known_od_index = -1
                        unknown_od_index = int(unknown_rank)
                        od_status = "unknown"

                    origin_index = -1
                    destination_index = -1
                    if od_pair_indices is not None and od_pair_indices.ndim >= 2 and i < od_pair_indices.shape[0] and od_pair_indices.shape[1] >= 2:
                        origin_index = int(od_pair_indices[i, 0])
                        destination_index = int(od_pair_indices[i, 1])

                    self.demand_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "true_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                            "od_mask": mask_val,
                        }
                    )
                    self.od_matrix_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "origin_index": int(origin_index),
                            "destination_index": int(destination_index),
                            "od_status": od_status,
                            "known_od_index": int(known_od_index),
                            "unknown_od_index": int(unknown_od_index),
                            "od_mask": mask_val,
                            "true_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                        }
                    )

    def _write_csv(self, path: str, fieldnames, rows):
        """
        Write diagnostics rows to a semicolon-delimited CSV file.

        Diagnostics meaning:
            Centralized CSV exporter used by save_summary() and plots.

        Storage:
            Writes to the provided path, creating parent folders as needed.
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        def format_value(v):
            if isinstance(v, float):
                return str(v).replace(".", ",")
            return v

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                delimiter=";"
            )
            writer.writeheader()

            for row in rows:
                formatted_row = {k: format_value(v) for k, v in row.items()}
                writer.writerow(formatted_row)

    def plot_evolution(self, filename: str):
        """
        Plot R2 and MAE trends across epochs.

        Diagnostics meaning:
            Visual summary of fit quality across training.

        Storage:
            Writes an image file to the provided filename.
        """
        if not self.enabled:
            return
        if len(self.full_history["r2_flow"]) == 0:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.plot(self.full_history["r2_flow"], label="R2 Flow", color="tab:blue")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("R2", color="tab:blue")
        ax1.grid(alpha=0.2)

        ax2 = ax1.twinx()
        ax2.plot(self.full_history["mae_flow"], label="MAE Flow", color="tab:orange")
        ax2.set_ylabel("MAE", color="tab:orange")

        plt.title("VI Training Evolution")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_physics(self, filename: str):
        """
        Plot loss decomposition and solver convergence trajectories.

        Diagnostics meaning:
            Shows how loss terms and equilibrium solver metrics evolve.

        Storage:
            Writes an image file to the provided filename.
        """
        if not self.enabled:
            return
        if len(self.full_history["iterations"]) == 0 and len(self.full_history["l_flow"]) == 0:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
        ax1.plot(self.full_history["l_flow"], label="l_flow", color="tab:blue", alpha=0.85)
        ax1.plot(self.full_history["l_od"], label="l_od", color="tab:green", alpha=0.85)
        ax1.plot(self.full_history["l_demand_reg"], label="l_demand_reg", color="tab:orange", alpha=0.85)
        ax1.plot(self.full_history["total_loss"], label="total_loss", color="tab:red", alpha=0.65)
        ax1.set_ylabel("Loss")
        ax1.grid(alpha=0.2)
        ax1.legend(loc="upper right")

        ax2.plot(self.full_history["iterations"], label="iterations", color="tab:purple", alpha=0.85)
        ax2.plot(self.full_history["final_gap"], label="final_gap", color="tab:brown", alpha=0.85)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Solver")
        ax2.grid(alpha=0.2)
        ax2.legend(loc="upper right")

        plt.suptitle("VI Physics Diagnostics")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_gradient_health(self, filename: str):
        """
        Plot gradient norms over epochs (log scale when values are positive).

        Diagnostics meaning:
            Visual check for vanishing/exploding gradients across modules.

        Storage:
            Writes an image file to the provided filename.
        """
        if not self.enabled or not self.enable_gradient_history:
            return
        grads = self.full_history.get("grad_norms", {})
        if (not grads) and len(self.full_history.get("od_logits_grad", [])) == 0:
            return

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        plt.figure(figsize=(12, 6))
        for name, values in grads.items():
            if values:
                plt.plot(values, label=name)

        if len(self.full_history.get("od_logits_grad", [])) > 0:
            plt.plot(self.full_history["od_logits_grad"], label="od_logits_grad", linewidth=1.6)

        if len(self.full_history.get("max_grad", [])) > 0:
            plt.plot(self.full_history["max_grad"], label="max_grad", linewidth=1.2, alpha=0.8)

        all_vals = []
        for values in grads.values():
            all_vals.extend([float(v) for v in values])
        all_vals.extend([float(v) for v in self.full_history.get("od_logits_grad", [])])
        all_vals.extend([float(v) for v in self.full_history.get("max_grad", [])])
        if any(v > 0.0 for v in all_vals):
            plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Gradient Norm")
        plt.title("VI Gradient Health")
        plt.grid(alpha=0.2)
        plt.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_summary(self, filename: str):
        """
        Export summary JSON and detailed diagnostic CSV artifacts.

        Diagnostics meaning:
            Collects all diagnostic tables and writes them to the standard
            training diagnostics folder hierarchy.

        Storage:
            Writes the JSON summary to the provided filename and CSVs to
            the same folder or its diagnostics root.
        """
        if not self.enabled:
            return
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        alerts = self.full_history.get("alerts", [])
        alert_counts = {}
        for _, kind, _ in alerts:
            alert_counts[kind] = alert_counts.get(kind, 0) + 1

        summary = {
            "steps": len(self.full_history.get("r2_flow", [])),
            "metrics": {
                "r2_flow_mean": float(np.mean(self.full_history.get("r2_flow", [0.0]))),
                "mae_flow_mean": float(np.mean(self.full_history.get("mae_flow", [0.0]))),
                "total_loss_mean": float(np.mean(self.full_history.get("total_loss", [0.0]))),
                "l_flow_mean": float(np.mean(self.full_history.get("l_flow", [0.0]))),
                "l_od_mean": float(np.mean(self.full_history.get("l_od", [0.0]))),
                "iterations_mean": float(np.mean(self.full_history.get("iterations", [0.0]))),
                "final_gap_mean": float(np.mean(self.full_history.get("final_gap", [0.0]))),
                "od_logits_grad_mean": float(np.mean(self.full_history.get("od_logits_grad", [0.0]))),
            },
            "alerts": {
                "total": len(alerts),
                "counts": alert_counts,
                "examples": [
                    {"epoch": int(epoch), "type": str(kind), "value": float(value)}
                    for epoch, kind, value in alerts[-10:]
                ],
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Export detailed gradient history in CGAME_PhysicsMirrorDescent style.
        if self.enable_gradient_history:
            grad_csv = os.path.join(out_dir, "gradient_history_detailed.csv")
            self._write_csv(
                grad_csv,
                fieldnames=[
                    "epoch",
                    "module",
                    "param_name",
                    "norm2",
                    "max_abs",
                    "mean_abs",
                    "finite_ratio",
                    "non_finite_count",
                ],
                rows=self.gradient_detailed_rows,
            )

        # Export a single forward/backward consistency audit file across all epochs.
        if self.enable_forward_backward_audit:
            fb_audit_csv = os.path.join(out_dir, "forward_backward_consistency_audit.csv")
            self._write_csv(
                fb_audit_csv,
                fieldnames=[
                    "epoch",
                    "mode",
                    "implicit_grad",
                    "forward_iterations",
                    "forward_final_gap",
                    "forward_linearization_iter",
                    "backward_info_available",
                    "backward_iter_idx",
                    "iter_match",
                    "backward_fixed_point_residual",
                    "backward_z_norm",
                    "backward_grad_output_norm",
                    "backward_iters",
                    "backward_damping",
                    "status",
                ],
                rows=self.forward_backward_audit_rows,
            )

        # Export diagnostics requested at run-level diagnostics root.
        diagnostics_root = os.path.dirname(out_dir)
        if self.enable_imd_gap_history:
            imd_gap_csv = os.path.join(diagnostics_root, "imd_relative_gap_history_by_epoch.csv")
            self._write_csv(
                imd_gap_csv,
                fieldnames=["epoch", "imd_iteration", "relative_gap", "mode", "implicit_grad"],
                rows=self.imd_relative_gap_rows,
            )

        epoch_loss_rows = []
        n_epochs = max(
            len(self.full_history.get("l_flow", [])),
            len(self.full_history.get("l_od", [])),
            len(self.full_history.get("total_loss", [])),
            len(self.full_history.get("l_demand_reg", [])),
        )
        for idx in range(n_epochs):
            l_flow = float(self.full_history.get("l_flow", [0.0])[idx]) if idx < len(self.full_history.get("l_flow", [])) else 0.0
            l_od = float(self.full_history.get("l_od", [0.0])[idx]) if idx < len(self.full_history.get("l_od", [])) else 0.0
            total_loss = float(self.full_history.get("total_loss", [0.0])[idx]) if idx < len(self.full_history.get("total_loss", [])) else 0.0
            l_demand_reg = float(self.full_history.get("l_demand_reg", [0.0])[idx]) if idx < len(self.full_history.get("l_demand_reg", [])) else 0.0
            epoch_loss_rows.append(
                {
                    "epoch": int(idx + 1),
                    "l_flow": l_flow,
                    "l_od": l_od,
                    "total_loss": total_loss,
                    "l_demand_reg": l_demand_reg,
                }
            )

        loss_epoch_csv = os.path.join(diagnostics_root, "epoch_loss_evolution.csv")
        self._write_csv(
            loss_epoch_csv,
            fieldnames=["epoch", "l_flow", "l_od", "total_loss", "l_demand_reg"],
            rows=epoch_loss_rows,
        )

        od_matrix_csv = os.path.join(diagnostics_root, "od_matrix_history_by_epoch.csv")
        self._write_csv(
            od_matrix_csv,
            fieldnames=[
                "epoch",
                "od_index",
                "origin_index",
                "destination_index",
                "od_status",
                "known_od_index",
                "unknown_od_index",
                "od_mask",
                "true_demand",
                "estimated_demand",
            ],
            rows=self.od_matrix_rows,
        )

        # Export link-type alpha/beta/vc summaries in a single historical CSV.
        if self.enable_link_type_audit:
            physics_dir = os.path.join(diagnostics_root, "physics")
            link_type_all_csv = os.path.join(physics_dir, "link_type_alpha_beta_vc_history.csv")
            self._write_csv(
                link_type_all_csv,
                fieldnames=[
                    "epoch",
                    "link_type",
                    "count_links",
                    "alpha_mean",
                    "beta_mean",
                    "estimated_volume_mean",
                    "capacity_mean",
                    "vc_mean",
                    "vc_p50",
                    "vc_p95",
                ],
                rows=self.link_type_epoch_rows,
            )

        # Export comparison tables for flows and demands.
        if self.enable_flow_demand_history:
            flow_csv = os.path.join(out_dir, "true_vs_estimated_flow_history.csv")
            self._write_csv(
                flow_csv,
                fieldnames=["epoch", "link_index", "true_flow", "estimated_flow", "flow_mask"],
                rows=self.flow_comparison_rows,
            )

            demand_csv = os.path.join(out_dir, "true_vs_estimated_demand_history.csv")
            self._write_csv(
                demand_csv,
                fieldnames=["epoch", "od_index", "true_demand", "estimated_demand", "od_mask"],
                rows=self.demand_comparison_rows,
            )

        # Route flow history.
        if self.route_flow_rows and self.enable_route_flow_history:
            route_csv = os.path.join(out_dir, "route_flow_history.csv")
            self._write_csv(
                route_csv,
                fieldnames=["epoch", "od_index", "route_index", "route_flow"],
                rows=self.route_flow_rows
            )

        # Link-level flow conservation audit.
        if (
            self.enable_flow_conservation_audit
            and hasattr(self, "flow_conservation_rows")
            and self.flow_conservation_rows
        ):
            cons_csv = os.path.join(out_dir, "flow_conservation_audit.csv")
            self._write_csv(
                cons_csv,
                fieldnames=["epoch", "link_id", "estimated_flow", "reconstructed_flow", "abs_error", "rel_error", "status"],
                rows=self.flow_conservation_rows
            )

        # Node-level mass conservation audit.
        if (
            self.enable_mass_conservation_audit
            and hasattr(self, "mass_conservation_rows")
            and self.mass_conservation_rows
        ):
            mass_csv = os.path.join(out_dir, "node_mass_conservation_audit.csv")
            self._write_csv(
                mass_csv,
                fieldnames=["epoch", "node_id", "net_demand_generated", "net_flow_routed", "abs_error", "status"],
                rows=self.mass_conservation_rows
            )

    def _audit_ensure_2d(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ensure tensors have shape [B, N] for audit computations.

        Diagnostics meaning:
            Normalizes batch dimension to keep audit logic vectorized.

        Storage:
            Used only in memory inside run_assignment_audit().
        """
        if x.dim() == 1:
            return x.unsqueeze(0)
        return x

    def _audit_scalar(self, x: torch.Tensor) -> float:
        """
        Convert a scalar tensor to a Python float for audit summaries.

        Diagnostics meaning:
            Ensures audit metrics are JSON-serializable.

        Storage:
            Used only in memory inside run_assignment_audit().
        """
        return float(x.detach().cpu().item())

    def _audit_stats(self, x: torch.Tensor, prefix: str) -> Dict[str, float]:
        """
        Return basic detached statistics for a tensor.

        Diagnostics meaning:
            Summarizes scale, min/max, and count for audit traces.

        Storage:
            Returned to the caller of run_assignment_audit().
        """
        x = x.detach().float().reshape(-1)

        if x.numel() == 0:
            return {
                f"{prefix}_count": 0.0,
                f"{prefix}_sum": 0.0,
                f"{prefix}_mean": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
            }

        return {
            f"{prefix}_count": float(x.numel()),
            f"{prefix}_sum": self._audit_scalar(x.sum()),
            f"{prefix}_mean": self._audit_scalar(x.mean()),
            f"{prefix}_min": self._audit_scalar(x.min()),
            f"{prefix}_max": self._audit_scalar(x.max()),
        }

    def run_assignment_audit(
        self,
        model: torch.nn.Module,
        observed_flows: torch.Tensor,
        flow_mask: torch.Tensor,
        true_od_demand: Optional[torch.Tensor],
        od_mask: Optional[torch.Tensor],
    ) -> Dict[str, float]:
        """
        Run a non-destructive VI assignment audit.

        Diagnostics meaning:
            Checks for mass loss through OD alignment, demand completion,
            OD-to-route feasibility, route-to-link projection, fixed-point
            residual, and link-flow scale consistency.

        Storage:
            Returns a dictionary for JSON export by the caller.
        """
        if not self.enabled or not self.enable_assignment_audit:
            return {}

        if not hasattr(model, "equilibrium_solver"):
            raise RuntimeError(
                "run_assignment_audit requires a model with an equilibrium_solver."
            )

        was_training = model.training
        model.eval()

        solver = model.equilibrium_solver

        observed_flows = self._audit_ensure_2d(observed_flows.detach())
        flow_mask = self._audit_ensure_2d(flow_mask.detach()).float()

        if true_od_demand is not None:
            true_od_demand = self._audit_ensure_2d(true_od_demand.detach())

        if od_mask is not None:
            od_mask = self._audit_ensure_2d(od_mask.detach()).float()

        with torch.no_grad():
            outputs = model(
                observed_flows=observed_flows,
                flow_mask=flow_mask,
                true_od_demand=true_od_demand,
                od_mask=od_mask,
                warmup=False,
                is_pure_inference=True,
            )

        estimated_od = self._audit_ensure_2d(outputs["estimated_demand"].detach())
        route_flows = self._audit_ensure_2d(outputs["route_flows"].detach())
        pred_link_flows = self._audit_ensure_2d(outputs["reconstructed_flows"].detach())

        valid = solver.route_validity_mask.to(
            device=route_flows.device,
            dtype=route_flows.dtype,
        )
        valid_3d = valid.unsqueeze(0).expand(
            route_flows.shape[0],
            -1,
            -1,
        )

        num_od, k_paths = valid.shape
        route_flows_3d = route_flows.view(-1, num_od, k_paths)

        # ------------------------------------------------------------------
        # 1. OD target alignment audit
        # ------------------------------------------------------------------
        audit: Dict[str, float] = {}

        audit.update(self._audit_stats(estimated_od, "estimated_od"))

        if true_od_demand is not None:
            aligned_true_od, aligned_od_mask = model._align_od_targets(
                true_od_demand,
                od_mask,
            )
            aligned_true_od = self._audit_ensure_2d(aligned_true_od.detach())
            aligned_od_mask = self._audit_ensure_2d(aligned_od_mask.detach()).float()

            known_mask = aligned_od_mask > 0.5

            audit.update(self._audit_stats(aligned_true_od, "aligned_true_od"))
            audit["aligned_known_od_count"] = float(known_mask.sum().item())
            audit["aligned_known_od_total"] = self._audit_scalar(
                aligned_true_od[known_mask].sum()
                if known_mask.any()
                else torch.tensor(0.0, device=estimated_od.device)
            )

            if known_mask.any():
                known_abs_error = torch.abs(
                    estimated_od[known_mask] - aligned_true_od[known_mask]
                )
                audit["known_od_abs_error_mean"] = self._audit_scalar(
                    known_abs_error.mean()
                )
                audit["known_od_abs_error_max"] = self._audit_scalar(
                    known_abs_error.max()
                )
            else:
                audit["known_od_abs_error_mean"] = 0.0
                audit["known_od_abs_error_max"] = 0.0

        # ------------------------------------------------------------------
        # 2. OD-to-route feasibility
        # ------------------------------------------------------------------
        valid_route_flows = route_flows_3d * valid_3d
        assigned_by_od = valid_route_flows.sum(dim=2)

        od_abs_error = torch.abs(assigned_by_od - estimated_od)
        od_rel_error = od_abs_error / torch.clamp(torch.abs(estimated_od), min=1.0)

        invalid_route_flow = torch.where(
            valid_3d.bool(),
            torch.zeros_like(route_flows_3d),
            torch.abs(route_flows_3d),
        )

        audit["route_feasibility_abs_error_mean"] = self._audit_scalar(
            od_abs_error.mean()
        )
        audit["route_feasibility_abs_error_max"] = self._audit_scalar(
            od_abs_error.max()
        )
        audit["route_feasibility_rel_error_mean"] = self._audit_scalar(
            od_rel_error.mean()
        )
        audit["route_feasibility_rel_error_max"] = self._audit_scalar(
            od_rel_error.max()
        )
        audit["invalid_route_flow_max"] = self._audit_scalar(
            invalid_route_flow.max()
        )

        # ------------------------------------------------------------------
        # 3. Route-to-link projection: f = Delta h
        # ------------------------------------------------------------------
        projected_link_flows = torch.sparse.mm(
            solver.delta_matrix,
            route_flows.t(),
        ).t()

        projection_abs_error = torch.abs(projected_link_flows - pred_link_flows)
        projection_rel_error = projection_abs_error / torch.clamp(
            torch.abs(pred_link_flows),
            min=1.0,
        )

        audit["delta_projection_abs_error_mean"] = self._audit_scalar(
            projection_abs_error.mean()
        )
        audit["delta_projection_abs_error_max"] = self._audit_scalar(
            projection_abs_error.max()
        )
        audit["delta_projection_rel_error_mean"] = self._audit_scalar(
            projection_rel_error.mean()
        )
        audit["delta_projection_rel_error_max"] = self._audit_scalar(
            projection_rel_error.max()
        )

        # ------------------------------------------------------------------
        # 4. Link-flow mass scale
        # ------------------------------------------------------------------
        observed_link_mask = flow_mask > 0.5

        audit.update(self._audit_stats(pred_link_flows, "pred_link_flow"))
        audit.update(self._audit_stats(observed_flows, "true_link_flow"))

        if observed_link_mask.any():
            pred_obs_total = pred_link_flows[observed_link_mask].sum()
            true_obs_total = observed_flows[observed_link_mask].sum()

            audit["pred_observed_link_total"] = self._audit_scalar(pred_obs_total)
            audit["true_observed_link_total"] = self._audit_scalar(true_obs_total)
            audit["observed_link_total_ratio_pred_over_true"] = self._audit_scalar(
                pred_obs_total / torch.clamp(true_obs_total, min=1.0)
            )
        else:
            audit["pred_observed_link_total"] = 0.0
            audit["true_observed_link_total"] = 0.0
            audit["observed_link_total_ratio_pred_over_true"] = 0.0

        # Weighted average number of links used per unit of OD demand.
        path_link_counts = torch.sparse.sum(
            solver.delta_matrix,
            dim=0,
        ).to_dense().to(route_flows.device)

        total_od = estimated_od.sum()
        total_link_mass_from_routes = (route_flows * path_link_counts.unsqueeze(0)).sum()

        audit["weighted_average_path_length_links"] = self._audit_scalar(
            total_link_mass_from_routes / torch.clamp(total_od, min=1.0)
        )

        if audit["weighted_average_path_length_links"] > 0.0:
            audit["implied_required_od_total_from_observed_links"] = (
                audit["true_observed_link_total"]
                / audit["weighted_average_path_length_links"]
            )
            audit["implied_od_scale_factor_needed"] = (
                audit["implied_required_od_total_from_observed_links"]
                / max(audit["estimated_od_sum"], 1.0)
            )
        else:
            audit["implied_required_od_total_from_observed_links"] = 0.0
            audit["implied_od_scale_factor_needed"] = 0.0

        # ------------------------------------------------------------------
        # 5. One-step fixed-point residual
        # ------------------------------------------------------------------
        bpr_params = {
            "alpha": outputs["learned_alpha"],
            "beta": outputs["learned_beta"],
            "capacity_multiplier": outputs["learned_capacity_multiplier"],
        }

        if "learned_theta" in outputs:
            bpr_params["theta"] = outputs["learned_theta"]

        one_step_route_flows = solver._fixed_point_step(
            route_flows=route_flows,
            od_demands=estimated_od,
            bpr_params=bpr_params,
            iter_idx=int(outputs.get("convergence_info", {}).get("iterations", 1)),
        )

        fixed_point_abs = torch.norm(one_step_route_flows - route_flows, p=2)
        fixed_point_den = torch.norm(route_flows, p=2).clamp(min=1.0)

        audit["one_step_fixed_point_relative_residual"] = self._audit_scalar(
            fixed_point_abs / fixed_point_den
        )

        # ------------------------------------------------------------------
        # 6. Existing convergence telemetry
        # ------------------------------------------------------------------
        conv_info = outputs.get("convergence_info", {}) or {}

        for key in [
            "iterations",
            "converged",
            "final_gap",
            "wardrop_gap",
            "relative_flow_change",
            "feasibility_abs_error",
            "feasibility_rel_error",
            "invalid_route_flow",
            "implicit_grad",
            "linearization_iter",
        ]:
            value = conv_info.get(key)
            if value is not None:
                audit[f"solver_{key}"] = float(value)

        if "learned_theta" in outputs:
            audit["learned_theta"] = self._audit_scalar(outputs["learned_theta"])

        if was_training:
            model.train()

        return audit


def run_vi_assignment_audit(
    model: torch.nn.Module,
    observed_flows: torch.Tensor,
    flow_mask: torch.Tensor,
    true_od_demand: Optional[torch.Tensor],
    od_mask: Optional[torch.Tensor],
) -> Dict[str, float]:
    """
    Backwards-compatible wrapper for the assignment audit.

    This keeps existing import paths intact and delegates to VIDiagnostician.
    """
    diagnostician = VIDiagnostician()
    return diagnostician.run_assignment_audit(
        model=model,
        observed_flows=observed_flows,
        flow_mask=flow_mask,
        true_od_demand=true_od_demand,
        od_mask=od_mask,
    )