from __future__ import annotations

# src/train/oracle/vi_oracle_debug.py

"""
A. La masa global predicha y la masa target no coinciden.
B. La masa global sí coincide, pero los links están desalineados por posición.
C. La matriz Delta o la proyección route → link está mal.
D. Las rutas/OD están bien, pero el solver VI no replica el SUE-MSA generador.
E. El target de flows no corresponde al mismo escenario/artifact.
"""

"""
Temporary VI oracle diagnostics
===============================

This module contains temporary diagnostics for validating whether VI_Model,
under oracle supply parameters, can reproduce the synthetic link-flow targets.

It should be removed once the oracle/debugging phase is complete.

Responsibilities
----------------
- Run a model forward pass with known OD.
- Compare oracle-predicted flows against target flows.
- Check global mass consistency.
- Check route-to-link projection consistency.
- Detect possible link-order misalignment.
- Run an independent SUE-MSA assignment in artifact space.

This module does not train the model and does not modify model parameters.
"""


import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.train._pipeline_utils import to_json_serializable


# =============================================================================
# Generic helpers
# =============================================================================


def _ensure_2d(x: torch.Tensor) -> torch.Tensor:
    """Ensure tensor has shape [B, N]."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    return x


def _tensor_to_np(x: torch.Tensor) -> np.ndarray:
    """Detach a tensor and convert it to a flattened numpy array."""
    return x.detach().cpu().float().reshape(-1).numpy()


def _float(value: torch.Tensor | float | int) -> float:
    """Convert scalar tensor or numeric value to Python float."""
    if torch.is_tensor(value):
        return float(value.detach().cpu().item())
    return float(value)


def _stats_np(values: np.ndarray, prefix: str) -> Dict[str, float]:
    """Return basic statistics for a numpy vector."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)

    if values.size == 0:
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_sum": 0.0,
            f"{prefix}_mean": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }

    return {
        f"{prefix}_count": float(values.size),
        f"{prefix}_sum": float(np.nansum(values)),
        f"{prefix}_mean": float(np.nanmean(values)),
        f"{prefix}_min": float(np.nanmin(values)),
        f"{prefix}_max": float(np.nanmax(values)),
    }


def _masked_metrics_np(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    prefix: str,
) -> Dict[str, float]:
    """Compute basic masked regression diagnostics."""
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.asarray(mask, dtype=np.float64).reshape(-1) > 0.5

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: {y_true.shape[0]} vs {y_pred.shape[0]}"
        )

    if y_true.shape[0] != mask.shape[0]:
        raise ValueError(
            f"y_true and mask length mismatch: {y_true.shape[0]} vs {mask.shape[0]}"
        )

    if not mask.any():
        return {
            f"{prefix}_count": 0.0,
            f"{prefix}_mae": 0.0,
            f"{prefix}_rmse": 0.0,
            f"{prefix}_mean_true": 0.0,
            f"{prefix}_mean_pred": 0.0,
            f"{prefix}_sum_true": 0.0,
            f"{prefix}_sum_pred": 0.0,
            f"{prefix}_sum_ratio_pred_over_true": 0.0,
        }

    yt = y_true[mask]
    yp = y_pred[mask]
    err = yp - yt

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))

    denom = float(np.sum(yt))
    ratio = float(np.sum(yp) / denom) if abs(denom) > 1e-12 else 0.0

    return {
        f"{prefix}_count": float(mask.sum()),
        f"{prefix}_mae": mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_mse": mse,
        f"{prefix}_mean_true": float(np.mean(yt)),
        f"{prefix}_mean_pred": float(np.mean(yp)),
        f"{prefix}_sum_true": float(np.sum(yt)),
        f"{prefix}_sum_pred": float(np.sum(yp)),
        f"{prefix}_sum_ratio_pred_over_true": ratio,
        f"{prefix}_max_abs_error": float(np.max(np.abs(err))),
    }


# =============================================================================
# Main oracle forward
# =============================================================================


def run_vi_oracle_forward(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    flow_mask_t: torch.Tensor,
    true_od_t: torch.Tensor,
    od_mask_t: torch.Tensor,
) -> Dict[str, Any]:
    """
    Run one oracle forward pass.

    The model should already be configured with:
    - supply_mode='oracle';
    - hard-anchored OD;
    - oracle alpha/beta/capacity/theta.

    Returns the raw model outputs.
    """

    was_training = model.training
    model.eval()

    true_flows_t = _ensure_2d(true_flows_t)
    flow_mask_t = _ensure_2d(flow_mask_t)
    true_od_t = _ensure_2d(true_od_t)
    od_mask_t = _ensure_2d(od_mask_t)

    with torch.no_grad():
        outputs = model(
            observed_flows=true_flows_t,
            flow_mask=flow_mask_t,
            true_od_demand=true_od_t,
            od_mask=od_mask_t,
            warmup=False,
            is_pure_inference=True,
        )

    if was_training:
        model.train()

    return outputs


# =============================================================================
# Test A/B/C: global mass, link alignment, Delta projection
# =============================================================================


def build_link_oracle_diagnostics(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    true_flows_t: torch.Tensor,
    flow_train_mask_np: np.ndarray,
    flow_observed_mask_np: np.ndarray,
    link_metadata: Optional[pd.DataFrame],
    output_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    """
    Compare oracle model predictions against target link flows.

    Produces:
    - global JSON summary;
    - link-by-link CSV sorted by absolute error;
    - sorted-flow diagnostic CSV to detect ordering problems.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_flow = _tensor_to_np(true_flows_t)
    pred_flow = _tensor_to_np(outputs["reconstructed_flows"])

    train_mask = np.asarray(flow_train_mask_np, dtype=np.float32).reshape(-1)
    observed_mask = np.asarray(flow_observed_mask_np, dtype=np.float32).reshape(-1)

    if true_flow.shape[0] != pred_flow.shape[0]:
        raise ValueError(
            f"Flow length mismatch: true={true_flow.shape[0]}, pred={pred_flow.shape[0]}"
        )

    if train_mask.shape[0] != true_flow.shape[0]:
        raise ValueError(
            f"train_mask length mismatch: mask={train_mask.shape[0]}, flows={true_flow.shape[0]}"
        )

    if observed_mask.shape[0] != true_flow.shape[0]:
        raise ValueError(
            f"observed_mask length mismatch: mask={observed_mask.shape[0]}, flows={true_flow.shape[0]}"
        )

    # ------------------------------------------------------------------
    # Global and masked metrics
    # ------------------------------------------------------------------
    summary: Dict[str, Any] = {}

    summary.update(_stats_np(true_flow, "true_flow"))
    summary.update(_stats_np(pred_flow, "oracle_pred_flow"))

    summary.update(
        _masked_metrics_np(
            y_true=true_flow,
            y_pred=pred_flow,
            mask=np.ones_like(true_flow),
            prefix="all_links",
        )
    )

    summary.update(
        _masked_metrics_np(
            y_true=true_flow,
            y_pred=pred_flow,
            mask=train_mask,
            prefix="train_links",
        )
    )

    summary.update(
        _masked_metrics_np(
            y_true=true_flow,
            y_pred=pred_flow,
            mask=observed_mask,
            prefix="observed_links",
        )
    )

    # ------------------------------------------------------------------
    # Link-by-link table
    # ------------------------------------------------------------------
    link_df = pd.DataFrame(
        {
            "link_position": np.arange(true_flow.shape[0], dtype=int),
            "true_flow": true_flow,
            "oracle_pred_flow": pred_flow,
        }
    )

    # ------------------------------------------------------------------
    # TEMPORARY: link-order fingerprint
    # ------------------------------------------------------------------
    # This checks whether the target vector appears to be sorted by link_id,
    # while the model prediction follows another internal order.
    # ------------------------------------------------------------------

    if link_metadata is not None and len(link_metadata) == len(link_df):
        metadata = link_metadata.reset_index(drop=True).copy()

        if "link_id" in metadata.columns:
            link_ids = metadata["link_id"].to_numpy()

            summary["metadata_link_id_is_monotonic_increasing"] = bool(
                np.all(np.diff(link_ids.astype(float)) >= 0)
            )

            summary["metadata_first_20_link_ids"] = [
                int(x) for x in link_ids[:20]
            ]

            # Compare target sorted by current position vs target sorted by link_id.
            order_by_link_id = np.argsort(link_ids.astype(float))

            true_by_link_id = true_flow[order_by_link_id]
            pred_by_link_id = pred_flow[order_by_link_id]

            summary["mae_after_sorting_both_by_metadata_link_id"] = float(
                np.mean(np.abs(pred_by_link_id - true_by_link_id))
            )

            # Fingerprint: top target links and top predicted links by position.
            top_true_idx = np.argsort(-true_flow)[:20]
            top_pred_idx = np.argsort(-pred_flow)[:20]

            summary["top_true_flow_positions"] = [
                {
                    "position": int(idx),
                    "link_id": int(link_ids[idx]),
                    "true_flow": float(true_flow[idx]),
                    "pred_flow_at_same_position": float(pred_flow[idx]),
                }
                for idx in top_true_idx
            ]

            summary["top_pred_flow_positions"] = [
                {
                    "position": int(idx),
                    "link_id": int(link_ids[idx]),
                    "pred_flow": float(pred_flow[idx]),
                    "true_flow_at_same_position": float(true_flow[idx]),
                }
                for idx in top_pred_idx
            ]


    # ------------------------------------------------------------------
    # TEMPORARY: nearest-flow permutation check
    # ------------------------------------------------------------------
    # If each predicted flow can be matched to a target flow with almost zero
    # difference, then the vectors are permutations of each other.
    # ------------------------------------------------------------------

    target_sorted_indices = np.argsort(true_flow)
    pred_sorted_indices = np.argsort(pred_flow)

    permutation_df = pd.DataFrame(
        {
            "rank": np.arange(true_flow.shape[0], dtype=int),
            "target_position": target_sorted_indices,
            "pred_position": pred_sorted_indices,
            "target_flow_sorted": true_flow[target_sorted_indices],
            "pred_flow_sorted": pred_flow[pred_sorted_indices],
            "sorted_abs_diff": np.abs(
                true_flow[target_sorted_indices] - pred_flow[pred_sorted_indices]
            ),
        }
    )

    if link_metadata is not None and len(link_metadata) == len(link_df):
        metadata = link_metadata.reset_index(drop=True)

        if "link_id" in metadata.columns:
            link_ids = metadata["link_id"].to_numpy()

            permutation_df["target_link_id_at_target_position"] = [
                int(link_ids[idx]) for idx in target_sorted_indices
            ]

            permutation_df["metadata_link_id_at_pred_position"] = [
                int(link_ids[idx]) for idx in pred_sorted_indices
            ]

    permutation_path = output_dir / f"{prefix}_oracle_flow_permutation_check.csv"
    permutation_df.to_csv(permutation_path, index=False)

    summary["flow_permutation_check_csv"] = str(permutation_path)
    summary["flow_permutation_sorted_abs_diff_mean"] = float(
        permutation_df["sorted_abs_diff"].mean()
    )
    summary["flow_permutation_sorted_abs_diff_max"] = float(
        permutation_df["sorted_abs_diff"].max()
    )

    link_df["error"] = link_df["oracle_pred_flow"] - link_df["true_flow"]
    link_df["abs_error"] = link_df["error"].abs()
    link_df["rel_error"] = link_df["abs_error"] / np.maximum(
        link_df["true_flow"].abs(),
        1.0,
    )
    link_df["is_train"] = train_mask > 0.5
    link_df["is_observed"] = observed_mask > 0.5

    if link_metadata is not None and len(link_metadata) == len(link_df):
        metadata = link_metadata.reset_index(drop=True).copy()
        link_df = pd.concat([metadata, link_df], axis=1)

    link_df_sorted = link_df.sort_values(
        by="abs_error",
        ascending=False,
    )

    link_csv_path = output_dir / f"{prefix}_oracle_link_errors.csv"
    link_df_sorted.to_csv(link_csv_path, index=False)

    summary["link_error_csv"] = str(link_csv_path)

    # ------------------------------------------------------------------
    # Sorted-flow fingerprint
    # ------------------------------------------------------------------
    # If sorted true and sorted prediction are similar but positional errors are
    # huge, the likely problem is link-order mismatch.
    sorted_true = np.sort(true_flow)
    sorted_pred = np.sort(pred_flow)

    sorted_err = sorted_pred - sorted_true

    sorted_df = pd.DataFrame(
        {
            "rank": np.arange(true_flow.shape[0], dtype=int),
            "sorted_true_flow": sorted_true,
            "sorted_oracle_pred_flow": sorted_pred,
            "sorted_error": sorted_err,
            "sorted_abs_error": np.abs(sorted_err),
        }
    )

    sorted_csv_path = output_dir / f"{prefix}_oracle_sorted_flow_fingerprint.csv"
    sorted_df.to_csv(sorted_csv_path, index=False)

    summary["sorted_flow_fingerprint_csv"] = str(sorted_csv_path)
    summary["sorted_flow_mae"] = float(np.mean(np.abs(sorted_err)))
    summary["sorted_flow_rmse"] = float(np.sqrt(np.mean(sorted_err ** 2)))

    positional_mae = float(np.mean(np.abs(pred_flow - true_flow)))
    summary["positional_mae_over_sorted_mae"] = (
        positional_mae / max(summary["sorted_flow_mae"], 1e-12)
    )

    # ------------------------------------------------------------------
    # Delta projection check
    # ------------------------------------------------------------------
    solver = getattr(model, "equilibrium_solver", None)

    if solver is not None and "route_flows" in outputs:
        route_flows = _ensure_2d(outputs["route_flows"].detach())
        pred_flows_t = _ensure_2d(outputs["reconstructed_flows"].detach())

        projected = torch.sparse.mm(
            solver.delta_matrix,
            route_flows.t(),
        ).t()

        projection_error = torch.abs(projected - pred_flows_t)

        summary["delta_projection_abs_error_mean"] = _float(
            projection_error.mean()
        )
        summary["delta_projection_abs_error_max"] = _float(
            projection_error.max()
        )

    # ------------------------------------------------------------------
    # Save summary
    # ------------------------------------------------------------------
    summary_path = output_dir / f"{prefix}_oracle_link_diagnostics.json"
    summary_path.write_text(
        json.dumps(
            to_json_serializable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# =============================================================================
# Test C: OD-to-route feasibility
# =============================================================================


def build_route_mass_diagnostics(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    output_dir: Path,
    prefix: str,
) -> Dict[str, Any]:
    """
    Check OD-to-route mass conservation in the model route space.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    solver = getattr(model, "equilibrium_solver", None)

    if solver is None:
        raise RuntimeError("Model does not expose equilibrium_solver.")

    if "route_flows" not in outputs:
        raise RuntimeError("Model outputs do not contain route_flows.")

    if "estimated_demand" not in outputs:
        raise RuntimeError("Model outputs do not contain estimated_demand.")

    route_flows = _ensure_2d(outputs["route_flows"].detach())
    od = _ensure_2d(outputs["estimated_demand"].detach())

    valid = solver.route_validity_mask.to(
        device=route_flows.device,
        dtype=route_flows.dtype,
    )

    num_od, k_paths = valid.shape
    route_3d = route_flows.view(-1, num_od, k_paths)

    valid_3d = valid.unsqueeze(0).expand_as(route_3d)
    invalid_flow = torch.where(
        valid_3d.bool(),
        torch.zeros_like(route_3d),
        torch.abs(route_3d),
    )

    assigned_by_od = (route_3d * valid_3d).sum(dim=2)
    od_abs_error = torch.abs(assigned_by_od - od)
    od_rel_error = od_abs_error / torch.clamp(torch.abs(od), min=1.0)

    active_routes = ((route_3d > 1e-6) & valid_3d.bool()).sum(dim=2)

    summary = {
        "num_od": float(num_od),
        "k_paths": float(k_paths),
        "od_total": _float(od.sum()),
        "assigned_route_total": _float(assigned_by_od.sum()),
        "od_route_abs_error_mean": _float(od_abs_error.mean()),
        "od_route_abs_error_max": _float(od_abs_error.max()),
        "od_route_rel_error_mean": _float(od_rel_error.mean()),
        "od_route_rel_error_max": _float(od_rel_error.max()),
        "invalid_route_flow_max": _float(invalid_flow.max()),
        "active_routes_mean": _float(active_routes.float().mean()),
        "active_routes_min": _float(active_routes.float().min()),
        "active_routes_max": _float(active_routes.float().max()),
    }

    summary_path = output_dir / f"{prefix}_oracle_route_mass_diagnostics.json"
    summary_path.write_text(
        json.dumps(
            to_json_serializable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# =============================================================================
# Test D: independent SUE-MSA in artifact space
# =============================================================================


def _compute_link_costs_bpr(
    link_flows: torch.Tensor,
    t0: torch.Tensor,
    capacity: torch.Tensor,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    capacity_multiplier: torch.Tensor,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Compute BPR link costs:
        t = t0 * (1 + alpha * (x / capacity_eff) ** beta)
    """

    cap_eff = torch.clamp(
        capacity * capacity_multiplier,
        min=eps,
    )

    vc = torch.clamp(link_flows / cap_eff, min=0.0)

    return t0 * (1.0 + alpha * torch.pow(vc, beta))


def run_independent_sue_msa(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    max_iterations: int = 5000,
    theta: Optional[float] = None,
    rel_gap_tol: float = 1.0e-7,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Run an independent route-based SUE-MSA assignment using the model artifact space.

    This is not used for training. It checks whether a classical SUE-MSA fixed
    point, using the same Delta/OD/BPR parameters as the model, reproduces the
    target flows better than the VI solver.

    Returns
    -------
    link_flows : torch.Tensor
        Final MSA link flows with shape [L].

    info : Dict[str, Any]
        Convergence metadata.
    """

    solver = getattr(model, "equilibrium_solver", None)

    if solver is None:
        raise RuntimeError("Model does not expose equilibrium_solver.")

    device = solver.t0.device
    dtype = solver.t0.dtype

    valid = solver.route_validity_mask.to(device=device)
    num_od, k_paths = valid.shape

    od = _ensure_2d(outputs["estimated_demand"].detach()).to(
        device=device,
        dtype=dtype,
    )[0]

    alpha = outputs["learned_alpha"].detach().to(device=device, dtype=dtype)
    beta = outputs["learned_beta"].detach().to(device=device, dtype=dtype)
    capacity_multiplier = outputs["learned_capacity_multiplier"].detach().to(
        device=device,
        dtype=dtype,
    )

    if theta is None:
        if "learned_theta" in outputs:
            theta_value = float(outputs["learned_theta"].detach().cpu().reshape(-1)[0].item())
        else:
            theta_value = 1.0
    else:
        theta_value = float(theta)

    theta_t = torch.tensor(theta_value, device=device, dtype=dtype)

    # Initial route flows: uniform over valid routes for each OD.
    valid_float = valid.float()
    num_valid = valid_float.sum(dim=1).clamp(min=1.0)

    route_2d = (od[:, None] * valid_float) / num_valid[:, None]
    route_flat = route_2d.reshape(-1)

    last_rel_gap = float("inf")

    for iteration in range(1, int(max_iterations) + 1):
        link_flows = torch.sparse.mm(
            solver.delta_matrix,
            route_flat[:, None],
        ).squeeze(1)

        link_costs = _compute_link_costs_bpr(
            link_flows=link_flows,
            t0=solver.t0,
            capacity=solver.capacity,
            alpha=alpha,
            beta=beta,
            capacity_multiplier=capacity_multiplier,
        )

        route_costs = torch.sparse.mm(
            solver.delta_matrix.transpose(0, 1),
            link_costs[:, None],
        ).squeeze(1)

        route_costs_2d = route_costs.view(num_od, k_paths)

        # Invalid routes must receive zero probability.
        masked_costs = route_costs_2d.masked_fill(
            ~valid,
            float("inf"),
        )

        logits = -theta_t * masked_costs

        # Stabilize softmax by replacing invalid logits with a very negative value.
        logits = logits.masked_fill(~valid, -1.0e30)

        probs = torch.softmax(logits, dim=1)
        probs = probs * valid_float
        probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-12)

        aux_route_2d = od[:, None] * probs
        aux_route_flat = aux_route_2d.reshape(-1)

        diff = aux_route_flat - route_flat
        rel_gap = torch.norm(diff, p=1) / torch.clamp(
            torch.norm(route_flat, p=1),
            min=1.0,
        )

        last_rel_gap = float(rel_gap.detach().cpu().item())

        step = 1.0 / float(iteration)
        route_flat = route_flat + step * diff

        if last_rel_gap <= rel_gap_tol:
            break

    final_link_flows = torch.sparse.mm(
        solver.delta_matrix,
        route_flat[:, None],
    ).squeeze(1)

    info = {
        "msa_iterations": float(iteration),
        "msa_final_rel_gap_l1": float(last_rel_gap),
        "msa_theta": float(theta_value),
    }

    return final_link_flows, info


def build_msa_vs_vi_diagnostics(
    model: torch.nn.Module,
    outputs: Dict[str, Any],
    true_flows_t: torch.Tensor,
    observed_mask_np: np.ndarray,
    output_dir: Path,
    prefix: str,
    msa_max_iterations: int = 5000,
) -> Dict[str, Any]:
    """
    Compare:
    - target link flows;
    - VI solver oracle prediction;
    - independent SUE-MSA prediction in artifact space.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    true_flow = _tensor_to_np(true_flows_t)
    vi_flow = _tensor_to_np(outputs["reconstructed_flows"])

    msa_flow_t, msa_info = run_independent_sue_msa(
        model=model,
        outputs=outputs,
        max_iterations=msa_max_iterations,
    )

    msa_flow = _tensor_to_np(msa_flow_t)

    observed_mask = np.asarray(observed_mask_np, dtype=np.float32).reshape(-1)

    summary: Dict[str, Any] = dict(msa_info)

    summary.update(
        _masked_metrics_np(
            y_true=true_flow,
            y_pred=vi_flow,
            mask=observed_mask,
            prefix="vi_vs_target_observed",
        )
    )

    summary.update(
        _masked_metrics_np(
            y_true=true_flow,
            y_pred=msa_flow,
            mask=observed_mask,
            prefix="msa_vs_target_observed",
        )
    )

    summary.update(
        _masked_metrics_np(
            y_true=vi_flow,
            y_pred=msa_flow,
            mask=observed_mask,
            prefix="msa_vs_vi_observed",
        )
    )

    comparison_df = pd.DataFrame(
        {
            "link_position": np.arange(true_flow.shape[0], dtype=int),
            "true_flow": true_flow,
            "vi_oracle_flow": vi_flow,
            "msa_oracle_flow": msa_flow,
            "vi_error": vi_flow - true_flow,
            "msa_error": msa_flow - true_flow,
            "msa_minus_vi": msa_flow - vi_flow,
            "observed": observed_mask > 0.5,
        }
    )

    comparison_df["vi_abs_error"] = comparison_df["vi_error"].abs()
    comparison_df["msa_abs_error"] = comparison_df["msa_error"].abs()
    comparison_df["msa_vi_abs_diff"] = comparison_df["msa_minus_vi"].abs()

    comparison_path = output_dir / f"{prefix}_oracle_vi_vs_msa_vs_target.csv"
    comparison_df.sort_values(
        by="vi_abs_error",
        ascending=False,
    ).to_csv(comparison_path, index=False)

    summary["vi_msa_target_csv"] = str(comparison_path)

    summary_path = output_dir / f"{prefix}_oracle_msa_vs_vi_summary.json"
    summary_path.write_text(
        json.dumps(
            to_json_serializable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# =============================================================================
# One-shot orchestrator
# =============================================================================


def run_full_vi_oracle_diagnostics(
    model: torch.nn.Module,
    true_flows_t: torch.Tensor,
    flow_train_mask_np: np.ndarray,
    flow_observed_mask_np: np.ndarray,
    true_od_t: torch.Tensor,
    od_mask_t: torch.Tensor,
    link_metadata: Optional[pd.DataFrame],
    output_dir: Path,
    prefix: str = "standard_run",
) -> Dict[str, Any]:
    """
    Run all temporary VI oracle diagnostics and write JSON/CSV artifacts.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device

    true_flows_t = _ensure_2d(true_flows_t.to(device))
    flow_mask_t = _ensure_2d(
        torch.as_tensor(
            flow_train_mask_np,
            dtype=torch.float32,
            device=device,
        )
    )
    true_od_t = _ensure_2d(true_od_t.to(device))
    od_mask_t = _ensure_2d(od_mask_t.to(device))

    outputs = run_vi_oracle_forward(
        model=model,
        true_flows_t=true_flows_t,
        flow_mask_t=flow_mask_t,
        true_od_t=true_od_t,
        od_mask_t=od_mask_t,
    )

    link_summary = build_link_oracle_diagnostics(
        model=model,
        outputs=outputs,
        true_flows_t=true_flows_t,
        flow_train_mask_np=flow_train_mask_np,
        flow_observed_mask_np=flow_observed_mask_np,
        link_metadata=link_metadata,
        output_dir=output_dir,
        prefix=prefix,
    )

    route_summary = build_route_mass_diagnostics(
        model=model,
        outputs=outputs,
        output_dir=output_dir,
        prefix=prefix,
    )

    msa_summary = build_msa_vs_vi_diagnostics(
        model=model,
        outputs=outputs,
        true_flows_t=true_flows_t,
        observed_mask_np=flow_observed_mask_np,
        output_dir=output_dir,
        prefix=prefix,
    )

    combined = {
        "link_summary": link_summary,
        "route_summary": route_summary,
        "msa_summary": msa_summary,
    }

    combined_path = output_dir / f"{prefix}_oracle_diagnostics_combined.json"
    combined_path.write_text(
        json.dumps(
            to_json_serializable(combined),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return combined