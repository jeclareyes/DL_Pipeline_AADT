import csv
import os
from typing import Iterable

import numpy as np


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _pick_metric_key(candidates: Iterable[str], rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for key in candidates:
        if key in keys:
            return key
    return None


def _select_latest_epoch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows
    if "epoch" not in rows[0]:
        return rows

    epochs: list[int] = []
    for row in rows:
        try:
            epochs.append(int(float(row.get("epoch", 0))))
        except Exception:
            continue

    if not epochs:
        return rows

    latest_epoch = max(epochs)
    selected: list[dict[str, str]] = []
    for row in rows:
        try:
            if int(float(row.get("epoch", latest_epoch))) == latest_epoch:
                selected.append(row)
        except Exception:
            continue
    return selected


def _compute_metrics_array(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
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


def _write_csv_rows(path: str, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_post_training_diagnostics(training_diagnostics_dir: str, logger) -> bool:
    """Builds consolidated flow/demand metrics CSV from final comparison files.

    Returns True when metrics CSV was generated, False otherwise.
    """
    flow_csv_candidates = [
        os.path.join(training_diagnostics_dir, "estimated_vs_real_flows.csv"),
        os.path.join(training_diagnostics_dir, "true_vs_estimated_flow_history.csv"),
    ]
    demand_csv_candidates = [
        os.path.join(training_diagnostics_dir, "estimated_vs_real_demand.csv"),
        os.path.join(training_diagnostics_dir, "true_vs_estimated_demand_history.csv"),
    ]

    flow_csv = next((p for p in flow_csv_candidates if os.path.exists(p)), None)
    demand_csv = next((p for p in demand_csv_candidates if os.path.exists(p)), None)

    if flow_csv is None or demand_csv is None:
        logger.warning(
            "Post-training diagnostics skipped: missing comparison CSVs in %s",
            training_diagnostics_dir,
        )
        return False

    try:
        flow_rows = _select_latest_epoch_rows(_read_csv_rows(flow_csv))
        demand_rows = _select_latest_epoch_rows(_read_csv_rows(demand_csv))
    except Exception as exc:
        logger.warning("Could not parse diagnostics CSVs: %s", str(exc))
        return False

    flow_true_key = _pick_metric_key(["real_flow", "true_flow"], flow_rows)
    flow_est_key = _pick_metric_key(["estimated_flow", "pred_flow"], flow_rows)
    demand_true_key = _pick_metric_key(["real_demand", "true_demand"], demand_rows)
    demand_est_key = _pick_metric_key(["estimated_demand", "pred_demand"], demand_rows)

    if flow_true_key is None or flow_est_key is None or demand_true_key is None or demand_est_key is None:
        logger.warning("Post-training diagnostics skipped: expected metric columns were not found.")
        return False

    flow_true = np.array([float(row[flow_true_key]) for row in flow_rows], dtype=np.float64)
    flow_est = np.array([float(row[flow_est_key]) for row in flow_rows], dtype=np.float64)
    demand_true = np.array([float(row[demand_true_key]) for row in demand_rows], dtype=np.float64)
    demand_est = np.array([float(row[demand_est_key]) for row in demand_rows], dtype=np.float64)

    flow_metrics = _compute_metrics_array(flow_true, flow_est)
    demand_metrics = _compute_metrics_array(demand_true, demand_est)

    metrics_csv = os.path.join(training_diagnostics_dir, "estimated_vs_real_metrics.csv")
    metrics_rows: list[dict[str, float | int | str]] = [
        {
            "scope": "flows",
            "count": int(flow_metrics["count"]),
            "r2": float(flow_metrics["R2"]),
            "mae": float(flow_metrics["MAE"]),
            "mape": float(flow_metrics["MAPE"]),
            "rmse": float(flow_metrics["RMSE"]),
        },
        {
            "scope": "demand",
            "count": int(demand_metrics["count"]),
            "r2": float(demand_metrics["R2"]),
            "mae": float(demand_metrics["MAE"]),
            "mape": float(demand_metrics["MAPE"]),
            "rmse": float(demand_metrics["RMSE"]),
        },
    ]
    _write_csv_rows(
        metrics_csv,
        fieldnames=["scope", "count", "r2", "mae", "mape", "rmse"],
        rows=metrics_rows,
    )

    logger.info("Post-training metrics saved to %s", metrics_csv)
    return True
