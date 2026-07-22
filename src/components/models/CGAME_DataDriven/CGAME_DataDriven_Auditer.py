from __future__ import annotations

import csv
import json
import os
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class CGAMEDataDrivenDiagnostician:
    """
    Training diagnostics companion for CGAME_DataDriven.

    Responsibilities:
        - track flow fit metrics (R2, MAE)
        - export per-link and per-OD comparison tables
        - monitor gradient health for key modules
        - export summaries and optional plots
    """

    def __init__(
        self,
        history_window: int = 100,
        enabled: bool = True,
        enable_flow_demand_history: bool = True,
        enable_gradient_history: bool = True,
        enable_plots: bool = True,
    ):
        # Global toggle used by every diagnostic method.
        self.enabled = bool(enabled)
        # Individual toggles for optional diagnostics.
        self.enable_flow_demand_history = bool(enable_flow_demand_history)
        self.enable_gradient_history = bool(enable_gradient_history)
        self.enable_plots = bool(enable_plots)

        self.window = int(history_window)
        self.full_history = {
            "r2_flow": [],
            "mae_flow": [],
            "grad_norms": {},
        }
        self.window_history = {
            "r2_flow": [],
            "mae_flow": [],
        }

        self.flow_comparison_rows = []
        self.demand_comparison_rows = []

    def update(self, outputs: Dict, targets: Dict, model=None, **kwargs):
        """
        Ingest one training/eval step and append tracked diagnostics.

        Diagnostics meaning:
            Computes flow fit metrics and collects row-wise comparisons for
            flow and OD demand.

        Storage:
            Stored in memory buffers and exported by save_summary().
        """
        if not self.enabled:
            return

        with torch.no_grad():
            pred_flow = outputs.get("reconstructed_flows")
            true_flow = targets.get("flows")
            mask_flow = targets.get("mask", targets.get("flow_mask"))

            if pred_flow is None or true_flow is None:
                return
            if mask_flow is None:
                mask_flow = torch.ones_like(true_flow)

            mask_bool = mask_flow > 0

            if pred_flow.dim() == 2 and pred_flow.shape[0] == 1:
                pred_flow = pred_flow.squeeze(0)
            if true_flow.dim() == 2 and true_flow.shape[0] == 1:
                true_flow = true_flow.squeeze(0)
            if mask_bool.dim() == 2 and mask_bool.shape[0] == 1:
                mask_bool = mask_bool.squeeze(0)

            if mask_bool.sum() > 0:
                y_pred = pred_flow[mask_bool].detach().cpu().numpy()
                y_true = true_flow[mask_bool].detach().cpu().numpy()

                mae = float(np.mean(np.abs(y_true - y_pred)))

                if len(y_true) > 1 and np.var(y_true) > 0:
                    ss_res = float(np.sum((y_true - y_pred) ** 2))
                    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))
                else:
                    r2 = 0.0

                self.full_history["r2_flow"].append(r2)
                self.full_history["mae_flow"].append(mae)
                self._push_window("r2_flow", r2)
                self._push_window("mae_flow", mae)

            epoch = int(kwargs.get("epoch", len(self.full_history["r2_flow"])))
            static_info = kwargs.get("static_info", {})
            if self.enable_flow_demand_history:
                self._collect_flow_demand_rows(
                    epoch=epoch,
                    outputs=outputs,
                    targets=targets,
                    static_info=static_info,
                )

    def _collect_flow_demand_rows(
        self,
        epoch: int,
        outputs: Dict,
        targets: Dict,
        static_info: Dict,
    ):
        """
        Store row-wise comparison tables for estimated vs target flow and demand.

        Diagnostics meaning:
            Creates per-link and per-OD comparison tables to audit errors
            at the finest granularity.

        Storage:
            Stored in flow_comparison_rows / demand_comparison_rows and
            exported to CSVs in save_summary().
        """
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
                            "link_id": int(i),
                            "real_flow": float(true_flow_np[i]),
                            "estimated_flow": float(pred_flow_np[i]),
                            "is_observed_link": bool(flow_mask_np[i] > 0.5),
                        }
                    )

        pred_od = outputs.get("estimated_demand")
        true_od = targets.get("od")
        od_mask = targets.get("od_mask")
        od_pair_indices = static_info.get("od_pair_indices", None)

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

                od_labels = None
                if od_pair_indices is not None:
                    try:
                        if torch.is_tensor(od_pair_indices):
                            od_pairs_np = od_pair_indices.detach().cpu().numpy()
                        else:
                            od_pairs_np = np.asarray(od_pair_indices)
                        if od_pairs_np.ndim == 2 and od_pairs_np.shape[0] == 2 and od_pairs_np.shape[1] != 2:
                            od_pairs_np = od_pairs_np.T
                        if od_pairs_np.ndim == 2 and od_pairs_np.shape[1] >= 2:
                            od_labels = [f"{int(o)}-{int(d)}" for o, d in od_pairs_np[:m, :2]]
                    except Exception:
                        od_labels = None

                for i in range(m):
                    od_id = od_labels[i] if od_labels is not None and i < len(od_labels) else int(i)
                    self.demand_comparison_rows.append(
                        {
                            "epoch": int(epoch),
                            "od_index": int(i),
                            "od_pair_id": od_id,
                            "real_demand": float(true_od_np[i]),
                            "estimated_demand": float(pred_od_np[i]),
                            "is_known_demand": bool(od_mask_np[i] > 0.5),
                        }
                    )

    def _write_csv(self, path: str, fieldnames, rows):
        """
        Write diagnostics rows to CSV.

        Diagnostics meaning:
            Centralized CSV exporter used by save_summary().

        Storage:
            Writes to the provided path, creating parent folders as needed.
        """
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def save_summary(self, filename: str):
        """
        Export summary JSON and diagnostic CSV artifacts.

        Diagnostics meaning:
            Creates summary metrics and comparison tables for auditing.

        Storage:
            Writes JSON summary to filename and CSVs to the same folder.
        """
        if not self.enabled:
            return

        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        summary = {
            "steps": len(self.full_history.get("r2_flow", [])),
            "metrics": {
                "r2_flow_mean": float(np.mean(self.full_history.get("r2_flow", [0.0]))),
                "mae_flow_mean": float(np.mean(self.full_history.get("mae_flow", [0.0]))),
            },
            "rows": {
                "flow_rows": len(self.flow_comparison_rows),
                "demand_rows": len(self.demand_comparison_rows),
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        if self.enable_flow_demand_history:
            latest_flow_rows = self.flow_comparison_rows
            if latest_flow_rows:
                latest_epoch = max(int(r.get("epoch", 0)) for r in latest_flow_rows)
                latest_flow_rows = [
                    r for r in latest_flow_rows if int(r.get("epoch", 0)) == latest_epoch
                ]

            flow_csv = os.path.join(out_dir, "estimated_vs_real_flows.csv")
            self._write_csv(
                flow_csv,
                fieldnames=["epoch", "link_id", "real_flow", "estimated_flow", "is_observed_link"],
                rows=latest_flow_rows,
            )

            latest_demand_rows = self.demand_comparison_rows
            if latest_demand_rows:
                latest_epoch = max(int(r.get("epoch", 0)) for r in latest_demand_rows)
                latest_demand_rows = [
                    r for r in latest_demand_rows if int(r.get("epoch", 0)) == latest_epoch
                ]

            demand_csv = os.path.join(out_dir, "estimated_vs_real_demand.csv")
            self._write_csv(
                demand_csv,
                fieldnames=[
                    "epoch",
                    "od_index",
                    "od_pair_id",
                    "real_demand",
                    "estimated_demand",
                    "is_known_demand",
                ],
                rows=latest_demand_rows,
            )

    def check_gradients(self, model: nn.Module) -> str:
        """
        Provide a compact gradient health report.

        Diagnostics meaning:
            Flags extremely large or small gradient magnitudes.

        Storage:
            Returns a string intended for logger output only.
        """
        max_grad = 0.0
        params_checked = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.data.norm(2).item())
                params_checked += 1
        if params_checked == 0:
            return "No Grads"

        status = []
        if max_grad > 1000:
            status.append("High Grads")
        if max_grad < 1e-6:
            status.append("Low Grads")
        if not status:
            return f"Grads OK ({max_grad:.2e})"
        return " | ".join(status)

    def capture_gradient_history(self, model: nn.Module, epoch: int):
        """
        Capture gradient norms by module for health checks.

        Diagnostics meaning:
            Tracks gradient magnitude for each model block over epochs.

        Storage:
            Stored in full_history["grad_norms"] and plotted/exported.
        """
        if not self.enabled or not self.enable_gradient_history:
            return

        if "grad_norms" not in self.full_history:
            self.full_history["grad_norms"] = {}

        key_modules = {
            "Fwd Enc": getattr(model, "f_encoder", None),
            "Fwd Dec": getattr(model, "f_decoder", None),
            "Matcher": getattr(model, "matcher", None),
            "Bwd Enc": getattr(model, "b_encoder", None),
            "Bwd Dec": getattr(model, "b_decoder", None),
        }

        for name, module in key_modules.items():
            if module is None:
                continue
            total_norm = 0.0
            for param in module.parameters():
                if param.grad is not None:
                    total_norm += param.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            if name not in self.full_history["grad_norms"]:
                self.full_history["grad_norms"][name] = []
            self.full_history["grad_norms"][name].append(total_norm)

    def get_report(self) -> str:
        """
        Return a short rolling window report.

        Diagnostics meaning:
            Provides average R2 and MAE over the recent window.

        Storage:
            Returns a string intended for logger output only.
        """
        if not self.window_history["r2_flow"]:
            return "Init..."
        avg_r2 = np.mean(self.window_history["r2_flow"])
        avg_mae = np.mean(self.window_history["mae_flow"])
        return f"R2: {avg_r2:.3f} | MAE: {avg_mae:.1f}"

    def _push_window(self, key: str, value: float):
        self.window_history[key].append(value)
        if len(self.window_history[key]) > self.window:
            self.window_history[key].pop(0)

    # --- Plotting methods ---

    def finalize_and_plot(self, filename_prefix: str = "final_report"):
        """
        Generate all plots with a shared filename prefix.

        Diagnostics meaning:
            Produces a set of diagnostic plots for offline inspection.

        Storage:
            Writes image files next to filename_prefix.
        """
        if not self.enabled or not self.enable_plots:
            return
        plt.switch_backend("Agg")
        self.plot_evolution(filename_prefix.replace(".png", "_evolution.png"))
        self.plot_physics(filename_prefix.replace(".png", "_physics.png"))
        self.plot_gradient_health(filename_prefix.replace(".png", "_gradients.png"))

    def plot_evolution(self, filename: str):
        """
        Plot R2 and MAE trends across epochs.

        Diagnostics meaning:
            Visual summary of flow fit quality over time.

        Storage:
            Writes an image file to filename.
        """
        if not self.enabled or not self.enable_plots:
            return
        r2 = self.full_history["r2_flow"]
        mae = self.full_history["mae_flow"]
        if not r2:
            return

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("R2 Score", color="tab:blue")
        ax1.plot(r2, color="tab:blue", label="R2 Flow", alpha=0.7)
        ax1.set_ylim(-1, 1)

        ax2 = ax1.twinx()
        ax2.set_ylabel("MAE Flow", color="tab:orange")
        ax2.plot(mae, color="tab:orange", label="MAE Flow", alpha=0.7)

        plt.title("Training Evolution")
        plt.savefig(filename)
        plt.close()

    def plot_physics(self, filename: str):
        """
        Plot reconstruction error trends across epochs.

        Diagnostics meaning:
            Shows overall reconstruction error as a proxy for consistency.

        Storage:
            Writes an image file to filename.
        """
        if not self.enabled or not self.enable_plots:
            return
        data = self.full_history["mae_flow"]
        if not data:
            return
        plt.figure(figsize=(10, 6))
        plt.plot(data, label="MAE Flow", color="gray", alpha=0.5)
        plt.title("Reconstruction Error")
        plt.savefig(filename)
        plt.close()

    def plot_gradient_health(self, filename: str):
        """
        Plot gradient norms over epochs.

        Diagnostics meaning:
            Visual check for vanishing or exploding gradients.

        Storage:
            Writes an image file to filename.
        """
        if not self.enabled or not self.enable_plots or not self.enable_gradient_history:
            return
        grads = self.full_history.get("grad_norms", {})
        if not grads:
            return
        plt.figure(figsize=(12, 6))
        for name, values in grads.items():
            if values:
                plt.plot(values, label=name)
        plt.yscale("log")
        plt.legend()
        plt.title("Gradient Health")
        plt.savefig(filename)
        plt.close()
