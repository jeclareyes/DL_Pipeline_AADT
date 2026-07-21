import argparse
import csv
import logging
import os
from pathlib import Path
import glob

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


from omegaconf import OmegaConf

def main() -> int:
    parser = argparse.ArgumentParser(description="TA link-type calibration pipeline")
    parser.add_argument(
        "--config", 
        default="configs/calibration/calibration.yaml", 
        help="Path to YAML config file (e.g. configs/calibration/calibration.yaml)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("calibration_pipeline")

    cfg_path = _resolve_path(args.config)
    if not cfg_path.exists():
        logger.error(f"Configuration file not found: {cfg_path}")
        return 1

    cfg = OmegaConf.load(str(cfg_path))
    logger.info(f"Loaded calibration config: {cfg_path}")
    
    # Map config paths
    # 1. Resolvemos el path del eval_bundle que tiene el asterisco
    raw_eval_path = _resolve_path(cfg.inputs.eval_bundle_path)
    
    # Lógica para resolver el asterisco (*)
    matches = glob.glob(str(raw_eval_path))
    if not matches:
        raise FileNotFoundError(f"No se encontró ningún archivo que coincida con: {raw_eval_path}")
    
    # Tomamos el primero que coincida (el que termina en eval.pt)
    eval_path = Path(matches[0]) 
    logger.info(f"Archivo eval resuelto: {eval_path.name}")

    ##############################################################
    
    flows_csv = _resolve_path(cfg.inputs.flows_csv_path)
    demand_csv = _resolve_path(cfg.inputs.demand_csv_path)

    calibrate_params = cfg.get("calibrate_parameters", True)

    if not eval_path.exists():
        raise FileNotFoundError(f"Eval bundle not found: {eval_path}")
    if not flows_csv.exists():
        raise FileNotFoundError(f"Flows CSV not found: {flows_csv}")
    if not demand_csv.exists():
        raise FileNotFoundError(f"Demand CSV not found: {demand_csv}")

    bundle_raw = torch.load(eval_path, map_location="cpu", weights_only=False)
    bundle = validate_eval_bundle_contract(bundle_raw)
    static_data = bundle["static_data"]

    flow_rows = _select_latest_epoch_rows(_read_csv_rows(str(flows_csv)))
    demand_rows = _select_latest_epoch_rows(_read_csv_rows(str(demand_csv)))

    network_params = _extract_network_params(static_data)
    demand_by_od, flow_by_link, flow_mask = _build_vectors(flow_rows, demand_rows, static_data)

    output_csv_cfg = cfg.outputs.get("output_csv_path")
    output_csv = _resolve_path(output_csv_cfg) if output_csv_cfg else flows_csv.parent / "calibrated_link_type_assignment.csv"

    if calibrate_params:
        logger.info("Initializing LinkTypeTACalibrationService to calibrate parameters...")
        service = LinkTypeTACalibrationService(device=cfg.hardware.get("device", "cpu"), logger=logger)
        
        # Obtenemos los min/max de los yaml (si existieran) o pasamos default
        alpha_min = cfg.get("tuner", {}).get("alpha_min", 0.1)
        alpha_max = cfg.get("tuner", {}).get("alpha_max", 0.5)
        beta_min = cfg.get("tuner", {}).get("beta_min", 1.0)
        beta_max = cfg.get("tuner", {}).get("beta_max", 6.0)
        
        rows = service.calibrate(
            network_params=network_params,
            estimated_demand=demand_by_od,
            target_flow=flow_by_link,
            target_flow_mask=flow_mask,
            # Aquí podrías inyectar min/max si _ta_calibration.py lo permitiera por kwargs.
        )
    else:
        logger.info("Checking for physical parameters in eval bundle artifacts...")
        
        history = bundle_raw.get("epochs_history", {})
        if not history:
            logger.error("No epochs_history found inside eval_bundle.")
            return 1
            
        latest_epoch = max(history.keys())
        artifacts = history[latest_epoch].get("artifacts", {})
        
        # 1. Verificamos si este modelo produjo parámetros BPR
        if "learned_alpha" not in artifacts or "learned_beta" not in artifacts:
            logger.warning(
                "Model evaluation artifacts do not contain 'learned_alpha' or 'learned_beta'. "
                "This model (e.g., Data-Driven) does not support physical parameter extraction. Skipping."
            )
            return 0
            
        logger.info("Physical parameters found. Extracting and running consistency check...")
        
        alpha_t = artifacts["learned_alpha"]
        beta_t = artifacts["learned_beta"]
        
        # Asegurarnos de que estén en el device correcto
        device = torch.device(cfg.hardware.get("device", "cpu"))
        alpha_t = alpha_t.to(device, dtype=torch.float32)
        beta_t = beta_t.to(device, dtype=torch.float32)
        
        service = LinkTypeTACalibrationService(device=device, logger=logger)
        
        # Preparamos los tensores para el simulador
        q_t = torch.as_tensor(demand_by_od, dtype=torch.float32, device=device)
        delta_t = network_params["delta_matrix"].to(device).coalesce()
        validity_t = network_params["route_validity_mask"].to(device).bool()
        t0_t = network_params["t0"].to(device).float().reshape(-1)
        cap_t = network_params["capacity"].to(device).float().reshape(-1)
        link_group_t = network_params["link_group"].to(device).long().reshape(-1)
        
        # 2. EJECUTAMOS LA ASIGNACIÓN FÍSICA
        logger.info("Running stochastic assignment with extracted parameters...")
        with torch.no_grad():
            assigned_flows_t = service._run_soft_route_assignment(
                q=q_t,
                delta=delta_t,
                validity_mask=validity_t,
                t0=t0_t,
                capacity=cap_t,
                link_group=link_group_t,
                alpha_group=alpha_t,
                beta_group=beta_t,
                num_iters=100,
                update_rule="mirror",
                damping=0.5
            )
            
        assigned_flows_np = assigned_flows_t.cpu().numpy()
        
        # --- PREPARACIÓN DE LAS 3 TABLAS ---
        
        # TABLA 1: Métricas Globales
        consistency_metrics = service._compute_metrics_array(flow_by_link, assigned_flows_np)
        
        # TABLA 2: Parámetros Extraídos
        alpha_vals = alpha_t.cpu().detach().numpy()
        beta_vals = beta_t.cpu().detach().numpy()
        param_rows = [{"link_group": i, "alpha": float(alpha_vals[i]), "beta": float(beta_vals[i])} for i in range(len(alpha_vals))]
        
        # TABLA 3: Métricas a Nivel Link (Fila por fila)
        link_rows = []
        for i in range(len(flow_by_link)):
            nn_val = float(flow_by_link[i])          # Lo que estimó la red neuronal
            phys_val = float(assigned_flows_np[i])   # Lo que dictan las leyes de la física
            
            abs_err = abs(nn_val - phys_val)
            rel_err = abs_err / (abs(phys_val) + 1e-6)  # 1e-6 previene división por cero
            
            link_rows.append({
                "link_index": i,
                "link_group": int(link_group_t[i].item()),
                "nn_estimated_flow": nn_val,
                "physics_assigned_flow": phys_val,
                "absolute_error": abs_err,
                "relative_error": rel_err,
            })
            
        # --- ESCRITURA MULTI-TABLA EN EL MISMO CSV ---
        output_csv_path = str(output_csv)
        out_dir = os.path.dirname(output_csv_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
            # Escribir Tabla 1
            f.write("--- TABLA 1: METRICAS GLOBALES DE CONSISTENCIA FISICA ---\n")
            global_writer = csv.DictWriter(f, fieldnames=["R2", "MAE", "RMSE", "MAPE"])
            global_writer.writeheader()
            global_writer.writerow({
                "R2": consistency_metrics["R2"],
                "MAE": consistency_metrics["MAE"],
                "RMSE": consistency_metrics["RMSE"],
                "MAPE": consistency_metrics["MAPE"]
            })
            
            f.write("\n\n") # Espaciado visual en Excel
            
            # Escribir Tabla 2
            f.write("--- TABLA 2: PARAMETROS FISICOS APRENDIDOS ---\n")
            param_writer = csv.DictWriter(f, fieldnames=["link_group", "alpha", "beta"])
            param_writer.writeheader()
            param_writer.writerows(param_rows)
            
            f.write("\n\n")
            
            # Escribir Tabla 3
            f.write("--- TABLA 3: ANALISIS DETALLADO NIVEL LINK ---\n")
            link_writer = csv.DictWriter(f, fieldnames=link_rows[0].keys())
            link_writer.writeheader()
            link_writer.writerows(link_rows)
            
        logger.info(f"Reporte detallado de consistencia guardado en: {output_csv_path}")
        
        # IMPORTANTE: Retornamos 0 aquí para terminar el programa de forma segura, 
        # evitando que el viejo código '_write_csv_rows' (que borraste) intente ejecutarse.
        return 0
    
    if not rows:
        logger.warning("Calibration finished without rows.")
        return 1

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
