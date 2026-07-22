import csv
import logging
import os
from typing import Dict

import numpy as np
import torch


class CGAMEDataDrivenDelegator:
    """
    Orchestrates model-specific logistics for CGAME_DataDriven.

    Responsibilities:
        - format epoch logs
        - compute flow metrics
        - export standard CSV artifacts
    """

    def __init__(self, model_instance):
        self.model = model_instance
        self.logger = logging.getLogger(__name__)
        self.logger.info(
            "CGAMEDataDrivenDelegator initialized for model: %s",
            type(model_instance).__name__,
        )

    def format_epoch_log(
        self,
        epoch: int,
        current_loss: float,
        loss_dict: Dict,
        outputs: Dict,
        grad_stats: Dict,
        has_val: bool,
        val_loss: float,
    ) -> str:
        """
        Format a compact log message for one epoch.

        Diagnostics meaning:
            Summarizes loss components and gradient health at a glance.

        Storage:
            Returned string is intended for logger output only.
        """
        def _as_float(x):
            if torch.is_tensor(x):
                return float(x.detach().item())
            try:
                return float(x)
            except Exception:
                return 0.0

        flow_loss_val = _as_float(loss_dict.get("l_flow", 0.0))
        od_loss_val = _as_float(loss_dict.get("l_od", 0.0))

        log_msg = (
            f"Epoch {epoch + 1}: Train Loss {current_loss:.6g} | "
            f"Flow Loss {flow_loss_val:.6g} | "
            f"OD Loss {od_loss_val:.6g}"
        )

        log_msg += (
            f" | GradNorm pre={grad_stats['pre_clip_norm']:.2e}, "
            f"post={grad_stats['post_clip_norm']:.2e}, "
            f"clip_ratio={grad_stats['clip_ratio']:.3f}, "
            f"clip_freq10={grad_stats['clip_freq10']:.0%}"
        )

        if has_val:
            log_msg += f" | Val MSE {val_loss:.4f}"

        return log_msg

    def compute_metrics(
        self,
        pred_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Compute basic regression metrics for flow prediction.

        Diagnostics meaning:
            Provides R2, MAE, RMSE, and MAPE for masked link flows.

        Storage:
            Returns a dictionary for immediate logging or export.
        """
        y_pred = pred_tensor.detach().cpu().numpy().flatten()
        y_true = target_tensor.detach().cpu().numpy().flatten()
        mask = mask_tensor.detach().cpu().numpy().flatten().astype(bool)

        y_pred = y_pred[mask]
        y_true = y_true[mask]

        if len(y_true) == 0:
            return {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0}

        mae = np.mean(np.abs(y_pred - y_true))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))

        non_zero = y_true != 0
        mape = (
            np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100
            if np.any(non_zero)
            else 0.0
        )

        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

    def generate_final_csvs(
        self,
        output_dir: str,
        outputs: Dict,
        targets: Dict,
        network_params: Dict,
        epoch: int,
    ) -> None:
        """
        Export standardized CSVs for flows and OD demand.

        Diagnostics meaning:
            Provides comparable artifacts across models for evaluation.

        Storage:
            Writes CSV files to output_dir.
        """
        if outputs is None or targets is None:
            return

        os.makedirs(output_dir, exist_ok=True)

        # 1. Link flow export
        pred_flow = outputs.get("reconstructed_flows")
        true_flow = targets.get("flows")
        flow_mask_t = targets.get("mask", targets.get("flow_mask"))

        if pred_flow is not None and true_flow is not None:
            pred_flow_np = pred_flow.detach().cpu().numpy().reshape(-1)
            true_flow_np = true_flow.detach().cpu().numpy().reshape(-1)
            flow_mask_np = (
                flow_mask_t.detach().cpu().numpy().reshape(-1)
                if flow_mask_t is not None
                else np.ones_like(true_flow_np)
            )

            flow_rows = []
            for i in range(len(pred_flow_np)):
                flow_rows.append(
                    {
                        "epoch": int(epoch),
                        "link_id": int(i),
                        "real_flow": float(true_flow_np[i]),
                        "estimated_flow": float(pred_flow_np[i]),
                        "is_observed_link": bool(flow_mask_np[i] > 0.5),
                    }
                )

            self._write_csv(
                os.path.join(output_dir, "estimated_vs_real_flows.csv"),
                ["epoch", "link_id", "real_flow", "estimated_flow", "is_observed_link"],
                flow_rows,
            )

        # 2. OD demand export
        pred_od = outputs.get("estimated_demand")
        true_od = targets.get("od")
        od_mask_t = targets.get("od_mask")

        if pred_od is not None and true_od is not None:
            pred_od_np = pred_od.detach().cpu().numpy().reshape(-1)
            true_od_np = true_od.detach().cpu().numpy().reshape(-1)
            od_mask_np = (
                od_mask_t.detach().cpu().numpy().reshape(-1)
                if od_mask_t is not None
                else np.ones_like(true_od_np)
            )

            od_pair_labels = self._resolve_od_labels(network_params, len(pred_od_np))

            demand_rows = []
            for i in range(len(pred_od_np)):
                demand_rows.append(
                    {
                        "epoch": int(epoch),
                        "od_index": int(i),
                        "od_pair_id": od_pair_labels[i],
                        "real_demand": float(true_od_np[i]),
                        "estimated_demand": float(pred_od_np[i]),
                        "is_known_demand": bool(od_mask_np[i] > 0.5),
                    }
                )

            self._write_csv(
                os.path.join(output_dir, "estimated_vs_real_demand.csv"),
                [
                    "epoch",
                    "od_index",
                    "od_pair_id",
                    "real_demand",
                    "estimated_demand",
                    "is_known_demand",
                ],
                demand_rows,
            )

    def _write_csv(self, path: str, fieldnames, rows) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _resolve_od_labels(self, network_params: Dict, n_od: int):
        od_pair_indices = network_params.get("od_pair_indices")
        if od_pair_indices is not None:
            try:
                pairs = od_pair_indices.detach().cpu().numpy()
                return [f"{int(o)}-{int(d)}" for o, d in pairs]
            except Exception:
                pass
        return [str(i) for i in range(n_od)]
