# src/train/diagnostics/_post_training_diagnostics.py

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def _pick_metric_key(candidates: Iterable[str], rows: list[dict[str, str]]) -> str | None:
    if not rows:
        return None
    keys = set(rows[0].keys())
    for key in candidates:
        if key in keys:
            return key
    return None


def _compute_metrics_array(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """
    Compute regression metrics for two already aligned arrays.

    This function intentionally fails when array lengths differ. Truncating to
    the shortest length would hide alignment errors between true and predicted
    values.
    """

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            "Cannot compute diagnostics metrics because y_true and y_pred "
            "have different lengths. This may indicate an alignment error. "
            f"len(y_true)={y_true.shape[0]}, len(y_pred)={y_pred.shape[0]}."
        )

    n = int(y_true.shape[0])

    if n <= 0:
        return {
            "R2": 0.0,
            "MAE": 0.0,
            "RMSE": 0.0,
            "MAPE": 0.0,
            "count": 0,
        }

    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError(
            "Cannot compute diagnostics metrics because y_true or y_pred "
            "contains NaN or infinite values."
        )

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))

    r2 = float(1.0 - (ss_res / (ss_tot + 1.0e-8)))

    non_zero = np.abs(y_true) > 1.0e-12

    if np.any(non_zero):
        mape = float(
            np.mean(
                np.abs(
                    (y_true[non_zero] - y_pred[non_zero])
                    / y_true[non_zero]
                )
            )
            * 100.0
        )
    else:
        mape = 0.0

    return {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "count": n,
    }

def _write_csv_rows(path: str | Path, fieldnames: list[str], rows: list[dict[str, float | int | str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def run_post_training_diagnostics(training_diagnostics_dir: str, logger) -> bool:
    """Builds consolidated flow/demand metrics CSV from final comparison files.

    Returns True when metrics CSV was generated, False otherwise.
    """
    training_diagnostics_dir = Path(training_diagnostics_dir)
    flow_csv_candidates = [
        training_diagnostics_dir / "estimated_vs_real_flows.csv",
        training_diagnostics_dir / "true_vs_estimated_flow_history.csv",
    ]
    demand_csv_candidates = [
        training_diagnostics_dir / "estimated_vs_real_demand.csv",
        training_diagnostics_dir / "true_vs_estimated_demand_history.csv",
    ]

    flow_csv = next((p for p in flow_csv_candidates if p.exists()), None)
    demand_csv = next((p for p in demand_csv_candidates if p.exists()), None)

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

    # -------------------------------------------------------------------------
    # EL NÚCLEO DEL ARREGLO:
    # Usamos explícitamente _safe_float() en lugar de float() para que la coma
    # se reemplace automáticamente por un punto numérico antes de crear el array.
    # -------------------------------------------------------------------------
    flow_true = np.array([_safe_float(row[flow_true_key]) for row in flow_rows], dtype=np.float64)
    flow_est = np.array([_safe_float(row[flow_est_key]) for row in flow_rows], dtype=np.float64)
    demand_true = np.array([_safe_float(row[demand_true_key]) for row in demand_rows], dtype=np.float64)
    demand_est = np.array([_safe_float(row[demand_est_key]) for row in demand_rows], dtype=np.float64)
    
    try:
        flow_metrics = _compute_metrics_array(
            y_true=flow_true,
            y_pred=flow_est,
        )

        demand_metrics = _compute_metrics_array(
            y_true=demand_true,
            y_pred=demand_est,
        )

    except ValueError as exc:
        logger.warning(
            "Post-training diagnostics skipped because metric arrays are invalid: %s",
            str(exc),
        )
        return False

    metrics_csv = training_diagnostics_dir / "estimated_vs_real_metrics.csv"
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

def _safe_float(val: str) -> float:
    """Safely converts strings to floats, handling European comma-decimals."""
    if val is None:
        return 0.0

    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return 0.0


def _read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    with Path(path).open("r", encoding="utf-8") as fh:
        first_line = fh.readline()
        fh.seek(0)

        detected_delimiter = ";" if ";" in first_line else ","

        reader = csv.DictReader(
            fh,
            delimiter=detected_delimiter,
        )

        for row in reader:
            rows.append(row)

    return rows


def _select_latest_epoch_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    if not rows:
        return rows

    if "epoch" not in rows[0]:
        return rows

    epochs: list[int] = []

    for row in rows:
        epochs.append(
            int(
                _safe_float(
                    row.get("epoch", 0)
                )
            )
        )

    if not epochs:
        return rows

    latest_epoch = max(epochs)

    selected: list[dict[str, str]] = []

    for row in rows:
        if int(_safe_float(row.get("epoch", latest_epoch))) == latest_epoch:
            selected.append(row)

    return selected
