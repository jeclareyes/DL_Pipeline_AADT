import argparse
import csv
import logging
import os
from pathlib import Path

import numpy as np
import torch

from src.contracts.runtime_contracts import EvalBundleContractError, validate_eval_bundle_contract
from src.train._ta_calibration import LinkTypeTACalibrationService


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path.cwd() / path


def _read_csv_rows(path: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(row)
    return rows


def _pick_metric_key(candidates: list[str], rows: list[dict[str, str]]) -> str | None:
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


def _write_csv_rows(path: str, fieldnames: list[str], rows: list[dict]) -> None:
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _build_vectors(flow_rows: list[dict[str, str]], demand_rows: list[dict[str, str]], static_data: dict):
    flow_est_key = _pick_metric_key(["estimated_flow", "pred_flow"], flow_rows)
    flow_idx_key = _pick_metric_key(["link_id", "link_index"], flow_rows)
    flow_obs_key = _pick_metric_key(["is_observed_link", "flow_mask"], flow_rows)
    demand_est_key = _pick_metric_key(["estimated_demand", "pred_demand"], demand_rows)
    demand_idx_key = _pick_metric_key(["od_index"], demand_rows)

    if flow_est_key is None or demand_est_key is None:
        raise ValueError("Could not find required estimated flow/demand columns in diagnostics CSVs")

    n_links = int(static_data["t0"].numel())
    n_od = int(static_data["route_validity_mask"].shape[0])

    flow_by_link = np.zeros(n_links, dtype=np.float32)
    flow_mask = np.zeros(n_links, dtype=bool)

    for i, row in enumerate(flow_rows):
        raw_idx = row.get(flow_idx_key, None) if flow_idx_key is not None else None
        try:
            idx = int(float(raw_idx)) if raw_idx not in (None, "") else i
        except Exception:
            idx = i
        if not (0 <= idx < n_links):
            continue

        flow_by_link[idx] = float(row[flow_est_key])
        if flow_obs_key is not None:
            obs_raw = str(row.get(flow_obs_key, "")).strip().lower()
            flow_mask[idx] = obs_raw in {"1", "true", "t", "yes"}
        else:
            flow_mask[idx] = True

    if int(flow_mask.sum()) <= 0:
        flow_mask[:] = True

    demand_by_od = np.zeros(n_od, dtype=np.float32)
    for i, row in enumerate(demand_rows):
        raw_idx = row.get(demand_idx_key, None) if demand_idx_key is not None else None
        try:
            idx = int(float(raw_idx)) if raw_idx not in (None, "") else i
        except Exception:
            idx = i
        if not (0 <= idx < n_od):
            continue
        demand_by_od[idx] = float(row[demand_est_key])

    return demand_by_od, flow_by_link, flow_mask


def _extract_network_params(static_data: dict) -> dict:
    required = ["delta_matrix", "route_validity_mask", "t0", "capacity", "link_group"]
    for key in required:
        if key not in static_data:
            raise EvalBundleContractError(f"static_data missing required key for calibration: {key}")

    network_params = {
        "delta_matrix": static_data["delta_matrix"],
        "route_validity_mask": static_data["route_validity_mask"],
        "t0": static_data["t0"],
        "capacity": static_data["capacity"],
        "link_group": static_data["link_group"],
        "num_link_groups": static_data.get("num_link_groups", None),
        "link_types_vis": static_data.get("link_types_vis", None),
    }
    return network_params


def main() -> int:
    parser = argparse.ArgumentParser(description="Separate TA link-type calibration pipeline")
    parser.add_argument("--eval-path", required=True, help="Path to eval bundle (*.pt)")
    parser.add_argument("--flows-csv", required=True, help="Path to estimated_vs_real_flows.csv")
    parser.add_argument("--demand-csv", required=True, help="Path to estimated_vs_real_demand.csv")
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Output CSV path (default: <flows_csv_dir>/calibrated_link_type_assignment.csv)",
    )
    parser.add_argument("--device", default="cpu", help="Calibration device: cpu|cuda")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("calibration_pipeline")

    eval_path = _resolve_path(args.eval_path)
    flows_csv = _resolve_path(args.flows_csv)
    demand_csv = _resolve_path(args.demand_csv)

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval bundle not found: {eval_path}")
    if not flows_csv.exists():
        raise FileNotFoundError(f"Flows CSV not found: {flows_csv}")
    if not demand_csv.exists():
        raise FileNotFoundError(f"Demand CSV not found: {demand_csv}")

    bundle_raw = torch.load(eval_path, map_location="cpu")
    bundle = validate_eval_bundle_contract(bundle_raw)
    static_data = bundle["static_data"]

    flow_rows = _select_latest_epoch_rows(_read_csv_rows(str(flows_csv)))
    demand_rows = _select_latest_epoch_rows(_read_csv_rows(str(demand_csv)))

    network_params = _extract_network_params(static_data)
    demand_by_od, flow_by_link, flow_mask = _build_vectors(flow_rows, demand_rows, static_data)

    service = LinkTypeTACalibrationService(device=args.device, logger=logger)
    rows = service.calibrate(
        network_params=network_params,
        estimated_demand=demand_by_od,
        target_flow=flow_by_link,
        target_flow_mask=flow_mask,
    )

    if not rows:
        logger.warning("Calibration finished without rows.")
        return 1

    output_csv = _resolve_path(args.output_csv) if args.output_csv else flows_csv.parent / "calibrated_link_type_assignment.csv"
    _write_csv_rows(
        str(output_csv),
        fieldnames=[
            "link_group",
            "link_type",
            "num_links",
            "alpha",
            "beta",
            "r2",
            "mae",
            "mape",
            "rmse",
            "global_r2",
            "global_mae",
            "global_mape",
            "global_rmse",
            "objective_value",
            "update_rule",
            "assign_iters",
            "theta",
            "eta",
            "damping",
            "demand_scale",
            "optimizer_lr",
            "optimizer_steps",
        ],
        rows=rows,
    )

    logger.info("Calibration CSV written to: %s", output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
