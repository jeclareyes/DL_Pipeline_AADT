from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class LinkTypeTACalibrationService:
    """Calibrates alpha/beta by link group using differentiable assignment."""

    def __init__(self, device: str | torch.device = "cpu", logger=None):
        self.device = torch.device(device)
        self.logger = logger

    def _log_warning(self, msg: str, *args) -> None:
        if self.logger is not None:
            self.logger.warning(msg, *args)

    def _log_info(self, msg: str, *args) -> None:
        if self.logger is not None:
            self.logger.info(msg, *args)

    def _compute_metrics_array(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
        y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
        n = int(min(len(y_true), len(y_pred)))
        if n <= 0:
            return {"R2": 0.0, "MAE": 0.0, "RMSE": 0.0, "MAPE": 0.0, "count": 0}

        y_true = y_true[:n]
        y_pred = y_pred[:n]

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1.0 - (ss_res / (ss_tot + 1e-8)))

        non_zero = np.abs(y_true) > 1e-12
        if np.any(non_zero):
            mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0)
        else:
            mape = 0.0

        return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape, "count": n}

    def _run_soft_route_assignment(
        self,
        q: torch.Tensor,
        delta: torch.Tensor,
        validity_mask: torch.Tensor,
        t0: torch.Tensor,
        capacity: torch.Tensor,
        link_group: torch.Tensor,
        alpha_group: torch.Tensor,
        beta_group: torch.Tensor,
        num_iters: int = 15,
        theta: float | torch.Tensor = 1.0,
        eta: float = 1.0,
        damping: float = 0.5,
        update_rule: str = "mirror",
    ) -> torch.Tensor:
        """Differentiable assignment with logit or mirror updates over route shares."""
        num_od, k_paths = validity_mask.shape
        valid = validity_mask.bool()
        valid_f = valid.float()

        denom0 = torch.clamp(valid_f.sum(dim=1, keepdim=True), min=1.0)
        sigma = valid_f / denom0

        cap_safe = torch.clamp(capacity, min=1e-6)
        alpha_link = alpha_group[link_group]
        beta_link = beta_group[link_group]

        for _ in range(max(int(num_iters), 1)):
            route_flows = sigma * q.unsqueeze(1)
            link_flows = torch.sparse.mm(delta, route_flows.reshape(-1, 1)).squeeze(1)

            vc = torch.clamp(link_flows / cap_safe, min=0.0, max=8.0)
            link_costs = t0 * (1.0 + alpha_link * torch.pow(vc + 1e-8, beta_link))

            route_costs = torch.sparse.mm(delta.transpose(0, 1), link_costs.unsqueeze(1)).squeeze(1)
            route_costs = route_costs.reshape(num_od, k_paths)
            route_costs = torch.where(valid, route_costs, torch.full_like(route_costs, 1e9))

            c_min = route_costs.min(dim=1, keepdim=True).values
            shifted = torch.where(valid, route_costs - c_min, torch.zeros_like(route_costs))

            if str(update_rule).lower() == "logit":
                if torch.is_tensor(theta):
                    theta_t = theta.to(device=shifted.device, dtype=shifted.dtype)
                else:
                    theta_t = torch.tensor(float(theta), device=shifted.device, dtype=shifted.dtype)

                logits = torch.clamp(-theta_t * shifted, min=-40.0, max=40.0)
                exp_logits = torch.exp(logits) * valid_f
                denom = exp_logits.sum(dim=1, keepdim=True)
                sigma_next = torch.where(denom > 0, exp_logits / (denom + 1e-9), sigma)
            else:
                exp_term = torch.exp(torch.clamp(-float(eta) * shifted, min=-30.0, max=30.0))
                sigma_next = sigma * exp_term
                sigma_next = sigma_next * valid_f
                denom = sigma_next.sum(dim=1, keepdim=True)
                sigma_next = torch.where(denom > 0, sigma_next / (denom + 1e-9), sigma)

            sigma = float(damping) * sigma + (1.0 - float(damping)) * sigma_next
            sigma = sigma * valid_f
            sigma = sigma / (sigma.sum(dim=1, keepdim=True) + 1e-9)

        route_flows = sigma * q.unsqueeze(1)
        link_flows = torch.sparse.mm(delta, route_flows.reshape(-1, 1)).squeeze(1)
        return link_flows

    @staticmethod
    def _raw_from_bounded_value(value: float, min_v: float, max_v: float) -> float:
        p = (float(value) - float(min_v)) / (float(max_v) - float(min_v) + 1e-12)
        p = float(np.clip(p, 1e-4, 1.0 - 1e-4))
        return float(np.log(p / (1.0 - p)))

    def calibrate(
        self,
        network_params: dict[str, Any],
        estimated_demand: np.ndarray,
        target_flow: np.ndarray,
        target_flow_mask: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        """Runs link-type calibration and returns per-group rows."""
        required_keys = ["delta_matrix", "route_validity_mask", "t0", "capacity", "link_group"]
        for key in required_keys:
            if key not in network_params:
                self._log_warning("Calibration skipped: network_params missing key '%s'", key)
                return []

        if estimated_demand is None or target_flow is None:
            self._log_warning("Calibration skipped: missing estimated demand/flow vectors.")
            return []

        device = self.device
        delta = network_params["delta_matrix"].to(device).coalesce()
        validity = network_params["route_validity_mask"].to(device).bool()
        t0 = network_params["t0"].to(device).float().reshape(-1)
        capacity = network_params["capacity"].to(device).float().reshape(-1)
        link_group = network_params["link_group"].to(device).long().reshape(-1)

        num_od = int(validity.shape[0])
        num_links = int(t0.numel())

        q_np = np.asarray(estimated_demand, dtype=np.float32).reshape(-1)
        tgt_np = np.asarray(target_flow, dtype=np.float32).reshape(-1)
        if target_flow_mask is None:
            mask_np = np.ones(num_links, dtype=bool)
        else:
            mask_np = np.asarray(target_flow_mask, dtype=bool).reshape(-1)

        q = torch.zeros(num_od, dtype=torch.float32, device=device)
        m_od = min(num_od, len(q_np))
        if m_od > 0:
            q[:m_od] = torch.from_numpy(np.clip(q_np[:m_od], 0.0, None)).to(device)

        target = torch.zeros(num_links, dtype=torch.float32, device=device)
        mask = torch.zeros(num_links, dtype=torch.bool, device=device)
        m_links = min(num_links, len(tgt_np), len(mask_np))
        if m_links > 0:
            target[:m_links] = torch.from_numpy(tgt_np[:m_links]).to(device)
            mask[:m_links] = torch.from_numpy(mask_np[:m_links]).to(device)

        if int(mask.sum().item()) <= 0:
            self._log_warning("Calibration skipped: no valid target flow entries.")
            return []

        num_link_groups = int(network_params.get("num_link_groups", int(link_group.max().item()) + 1))
        alpha_min, alpha_max = 0.01, 1.50
        beta_min, beta_max = 1.10, 10.00

        candidate_cfgs = [
            {
                "name": "logit_learned_temp_scale",
                "update_rule": "logit",
                "assign_iters": 60,
                "eta": 1.0,
                "theta": 1.0,
                "damping": 0.35,
                "demand_scale": 1.0,
                "lr": 0.02,
                "steps": 340,
                "huber_delta": 140.0,
                "loss_mode": "legacy",
                "train_theta": True,
                "train_demand_scale": True,
            },
            {
                "name": "legacy_baseline",
                "update_rule": "logit",
                "assign_iters": 20,
                "eta": 1.0,
                "theta": 1.0,
                "damping": 0.50,
                "demand_scale": 1.0,
                "lr": 0.05,
                "steps": 140,
                "huber_delta": 150.0,
                "loss_mode": "legacy",
                "train_theta": False,
                "train_demand_scale": False,
            },
            {
                "name": "mirror_1",
                "update_rule": "mirror",
                "assign_iters": 50,
                "eta": 1.0,
                "theta": 1.0,
                "damping": 0.15,
                "demand_scale": 0.8,
                "lr": 0.03,
                "steps": 260,
                "huber_delta": 120.0,
                "loss_mode": "hybrid",
                "train_theta": False,
                "train_demand_scale": False,
            },
            {
                "name": "mirror_2",
                "update_rule": "mirror",
                "assign_iters": 80,
                "eta": 1.6,
                "theta": 1.0,
                "damping": 0.10,
                "demand_scale": 1.0,
                "lr": 0.02,
                "steps": 320,
                "huber_delta": 100.0,
                "loss_mode": "hybrid",
                "train_theta": False,
                "train_demand_scale": False,
            },
            {
                "name": "logit_1",
                "update_rule": "logit",
                "assign_iters": 60,
                "eta": 1.0,
                "theta": 1.5,
                "damping": 0.25,
                "demand_scale": 0.6,
                "lr": 0.025,
                "steps": 280,
                "huber_delta": 120.0,
                "loss_mode": "hybrid",
                "train_theta": False,
                "train_demand_scale": False,
            },
            {
                "name": "logit_2",
                "update_rule": "logit",
                "assign_iters": 100,
                "eta": 1.0,
                "theta": 2.3,
                "damping": 0.10,
                "demand_scale": 1.2,
                "lr": 0.015,
                "steps": 360,
                "huber_delta": 90.0,
                "loss_mode": "hybrid",
                "train_theta": False,
                "train_demand_scale": False,
            },
        ]

        best_solution: dict[str, Any] | None = None

        init_alpha_raw = self._raw_from_bounded_value(0.15, alpha_min, alpha_max)
        init_beta_raw = self._raw_from_bounded_value(4.0, beta_min, beta_max)

        target_abs_mean = torch.mean(torch.abs(target[mask])) + 1e-6

        for cfg in candidate_cfgs:
            alpha_raw = torch.nn.Parameter(torch.full((num_link_groups,), init_alpha_raw, dtype=torch.float32, device=device))
            beta_raw = torch.nn.Parameter(torch.full((num_link_groups,), init_beta_raw, dtype=torch.float32, device=device))
            opt_params = [alpha_raw, beta_raw]

            theta_raw = None
            demand_scale_raw = None
            if bool(cfg.get("train_theta", False)):
                theta_init = float(cfg.get("theta", 1.0))
                theta_raw_init = self._raw_from_bounded_value(theta_init, 0.2, 6.0)
                theta_raw = torch.nn.Parameter(torch.tensor(theta_raw_init, dtype=torch.float32, device=device))
                opt_params.append(theta_raw)

            if bool(cfg.get("train_demand_scale", False)):
                ds_init = float(cfg.get("demand_scale", 1.0))
                ds_raw_init = self._raw_from_bounded_value(ds_init, 0.2, 2.5)
                demand_scale_raw = torch.nn.Parameter(torch.tensor(ds_raw_init, dtype=torch.float32, device=device))
                opt_params.append(demand_scale_raw)

            optimizer = torch.optim.AdamW(opt_params, lr=float(cfg["lr"]), weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.6,
                patience=20,
                min_lr=5e-4,
            )

            local_best_loss = float("inf")
            local_best_alpha = None
            local_best_beta = None
            local_best_theta = float(cfg.get("theta", 1.0))
            local_best_demand_scale = float(cfg.get("demand_scale", 1.0))
            stall_count = 0

            for _ in range(int(cfg["steps"])):
                optimizer.zero_grad()

                alpha = alpha_min + (alpha_max - alpha_min) * torch.sigmoid(alpha_raw)
                beta = beta_min + (beta_max - beta_min) * torch.sigmoid(beta_raw)

                if theta_raw is not None:
                    theta_current = 0.2 + (6.0 - 0.2) * torch.sigmoid(theta_raw)
                else:
                    theta_current = torch.tensor(float(cfg.get("theta", 1.0)), dtype=torch.float32, device=device)

                if demand_scale_raw is not None:
                    demand_scale_current = 0.2 + (2.5 - 0.2) * torch.sigmoid(demand_scale_raw)
                else:
                    demand_scale_current = torch.tensor(float(cfg.get("demand_scale", 1.0)), dtype=torch.float32, device=device)

                q_scaled = q * demand_scale_current

                pred_flow = self._run_soft_route_assignment(
                    q=q_scaled,
                    delta=delta,
                    validity_mask=validity,
                    t0=t0,
                    capacity=capacity,
                    link_group=link_group,
                    alpha_group=alpha,
                    beta_group=beta,
                    num_iters=int(cfg["assign_iters"]),
                    theta=theta_current,
                    eta=float(cfg["eta"]),
                    damping=float(cfg["damping"]),
                    update_rule=str(cfg["update_rule"]),
                )

                err = pred_flow[mask] - target[mask]
                loss_mode = str(cfg.get("loss_mode", "hybrid")).lower()
                if loss_mode == "legacy":
                    loss_mse = torch.mean(err ** 2)
                    reg = 1e-4 * torch.mean((alpha - 0.15) ** 2) + 1e-4 * torch.mean((beta - 4.0) ** 2)
                    loss = loss_mse + reg
                else:
                    loss_huber = F.huber_loss(pred_flow[mask], target[mask], reduction="mean", delta=float(cfg["huber_delta"]))
                    rel_err = err / (torch.abs(target[mask]) + 25.0)
                    loss_rel = torch.mean(rel_err ** 2)
                    loss_mse_norm = torch.mean((err / target_abs_mean) ** 2)
                    reg = 5e-4 * torch.mean((alpha - 0.15) ** 2) + 5e-4 * torch.mean((beta - 4.0) ** 2)
                    loss = loss_huber + 0.35 * loss_rel + 0.15 * loss_mse_norm + reg

                if not bool(torch.isfinite(loss)):
                    break

                loss.backward()
                torch.nn.utils.clip_grad_norm_([alpha_raw, beta_raw], max_norm=3.0)
                optimizer.step()
                scheduler.step(float(loss.detach().item()))

                loss_val = float(loss.detach().item())
                if loss_val + 1e-9 < local_best_loss:
                    local_best_loss = loss_val
                    local_best_alpha = alpha.detach().clone()
                    local_best_beta = beta.detach().clone()
                    local_best_theta = float(theta_current.detach().item())
                    local_best_demand_scale = float(demand_scale_current.detach().item())
                    stall_count = 0
                else:
                    stall_count += 1

                if stall_count >= 60:
                    break

            if local_best_alpha is None or local_best_beta is None:
                continue

            with torch.no_grad():
                pred_final = self._run_soft_route_assignment(
                    q=q * float(local_best_demand_scale),
                    delta=delta,
                    validity_mask=validity,
                    t0=t0,
                    capacity=capacity,
                    link_group=link_group,
                    alpha_group=local_best_alpha,
                    beta_group=local_best_beta,
                    num_iters=max(int(cfg["assign_iters"]), 120),
                    theta=float(local_best_theta),
                    eta=float(cfg["eta"]),
                    damping=float(cfg["damping"]),
                    update_rule=str(cfg["update_rule"]),
                )

            pred_np_local = pred_final.detach().cpu().numpy().reshape(-1)
            target_np_local = target.detach().cpu().numpy().reshape(-1)
            mask_np_local = mask.detach().cpu().numpy().reshape(-1).astype(bool)
            metrics_local = self._compute_metrics_array(target_np_local[mask_np_local], pred_np_local[mask_np_local])

            candidate_solution = {
                "cfg": cfg,
                "alpha": local_best_alpha,
                "beta": local_best_beta,
                "theta": float(local_best_theta),
                "demand_scale": float(local_best_demand_scale),
                "pred_np": pred_np_local,
                "metrics": metrics_local,
                "objective": float(local_best_loss),
            }

            if best_solution is None:
                best_solution = candidate_solution
            else:
                old = best_solution["metrics"]
                new = candidate_solution["metrics"]
                if (new["R2"] > old["R2"] + 1e-8) or (
                    abs(new["R2"] - old["R2"]) <= 1e-8 and new["MAE"] < old["MAE"]
                ):
                    best_solution = candidate_solution

        if best_solution is None:
            self._log_warning("Calibration did not converge to finite parameters.")
            return []

        best_alpha = best_solution["alpha"]
        best_beta = best_solution["beta"]
        pred_np = best_solution["pred_np"]
        target_np = target.detach().cpu().numpy().reshape(-1)
        mask_np = mask.detach().cpu().numpy().reshape(-1).astype(bool)
        metrics = best_solution["metrics"]

        link_group_np = link_group.detach().cpu().numpy().reshape(-1)
        alpha_np = best_alpha.detach().cpu().numpy().reshape(-1)
        beta_np = best_beta.detach().cpu().numpy().reshape(-1)

        link_types_vis = network_params.get("link_types_vis", None)
        if link_types_vis is not None:
            link_types_arr = np.asarray(link_types_vis).reshape(-1).astype(str)
        else:
            link_types_arr = None

        best_cfg = best_solution["cfg"]

        rows: list[dict[str, Any]] = []
        for group in range(num_link_groups):
            group_mask = (link_group_np == group) & mask_np
            n_links = int(np.sum(group_mask))
            if link_types_arr is not None and len(link_types_arr) >= len(link_group_np) and n_links > 0:
                labels, counts = np.unique(link_types_arr[group_mask], return_counts=True)
                link_type_name = str(labels[int(np.argmax(counts))])
            else:
                link_type_name = f"group_{group}"

            group_metrics = self._compute_metrics_array(target_np[group_mask], pred_np[group_mask]) if n_links > 0 else {
                "R2": 0.0,
                "MAE": 0.0,
                "MAPE": 0.0,
                "RMSE": 0.0,
            }

            rows.append(
                {
                    "link_group": int(group),
                    "link_type": link_type_name,
                    "num_links": n_links,
                    "alpha": float(alpha_np[group]),
                    "beta": float(beta_np[group]),
                    "r2": float(group_metrics["R2"]),
                    "mae": float(group_metrics["MAE"]),
                    "mape": float(group_metrics["MAPE"]),
                    "rmse": float(group_metrics["RMSE"]),
                    "global_r2": float(metrics["R2"]),
                    "global_mae": float(metrics["MAE"]),
                    "global_mape": float(metrics["MAPE"]),
                    "global_rmse": float(metrics["RMSE"]),
                    "objective_value": float(best_solution["objective"]),
                    "update_rule": str(best_cfg["update_rule"]),
                    "assign_iters": int(best_cfg["assign_iters"]),
                    "theta": float(best_solution.get("theta", best_cfg["theta"])),
                    "eta": float(best_cfg["eta"]),
                    "damping": float(best_cfg["damping"]),
                    "demand_scale": float(best_solution.get("demand_scale", best_cfg.get("demand_scale", 1.0))),
                    "optimizer_lr": float(best_cfg["lr"]),
                    "optimizer_steps": int(best_cfg["steps"]),
                }
            )

        self._log_info(
            "Best calibration config=%s | global R2=%.4f | MAE=%.2f | RMSE=%.2f",
            str(best_cfg.get("name", best_cfg)),
            float(metrics["R2"]),
            float(metrics["MAE"]),
            float(metrics["RMSE"]),
        )

        return rows
