"""
Helper evaluation functions for testing pipeline.
Provides utilities to instantiate models from saved checkpoints and evaluate them
using the same loss used in training (PartialDataLoss).

This module is intentionally independent from hydra's main so it can be imported
without side effects.
"""
import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import torch
from omegaconf import OmegaConf
import hydra

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Imported here to avoid circular imports at package import time in some cases
from src.components.models.Cyclic_Model.cyclic_model import PartialDataLoss


def _get_latest_epoch_state(master_checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    """Return the epoch-state dict for the latest epoch saved in master_checkpoint.
    master_checkpoint is expected to have an 'epochs_history' mapping from str(epoch)->state.
    """
    epochs = master_checkpoint.get("epochs_history", {})
    if not epochs:
        raise ValueError("Checkpoint contains no 'epochs_history' entries")
    # keys may be strings; convert to ints then back to str for lookup
    try:
        max_epoch = max(int(k) for k in epochs.keys())
        return epochs[str(max_epoch)]
    except Exception:
        # fallback: take the last by insertion order
        last_key = list(epochs.keys())[-1]
        return epochs[last_key]


def instantiate_model_from_checkpoint(master_checkpoint: Dict[str, Any],
                                      loader,
                                      device: torch.device) -> torch.nn.Module:
    """Instantiate the model defined in the checkpoint's saved config and return it.

    Args:
        master_checkpoint: loaded object saved by training (contains 'config').
        loader: the LinkopingDataLoader (or compatible) already used to prepare network params.
        device: torch.device to move the model to.

    Returns:
        model (torch.nn.Module)
    """
    if "config" not in master_checkpoint:
        raise ValueError("master_checkpoint has no 'config' key; cannot reconstruct model")

    ckpt_cfg = OmegaConf.create(master_checkpoint["config"]) if not isinstance(master_checkpoint["config"], OmegaConf) else master_checkpoint["config"]

    # Prepare network parameters from the loader (same as training)
    network_params = loader.prepare_network_parameters()

    # Instantiate model using hydra so any targets in config are resolved
    model = hydra.utils.instantiate(
        ckpt_cfg.model,
        num_links=network_params["num_links"],
        num_od_pairs=network_params["num_od_pairs"],
        t0=network_params["t0"].to(device),
        capacity=network_params["capacity"].to(device),
        route_masks=network_params["route_masks"].to(device),
        od_pair_indices=network_params["od_pair_indices"].to(device),
        num_link_groups=network_params.get("num_link_groups", None),
        link_group=(network_params.get("link_group") and network_params.get("link_group").to(device)),
        _recursive_=False,
    )

    model.to(device)
    return model


def evaluate_model_from_checkpoint(master_checkpoint: Dict[str, Any],
                                   model: torch.nn.Module,
                                   tensors: Dict[str, torch.Tensor],
                                   device: torch.device) -> Dict[str, Any]:
    """Load state dict from checkpoint (latest epoch) into model and evaluate.

    Args:
        master_checkpoint: checkpoint dict loaded from torch.load().
        model: instantiated model (matching checkpoint architecture).
        tensors: dict with keys: true_flows_t, true_od_t, flow_mask_t, od_mask_t
        device: torch.device

    Returns:
        dict with numeric metrics (losses) and meta info
    """
    epoch_state = _get_latest_epoch_state(master_checkpoint)
    state_dict = epoch_state.get("model_state_dict") or epoch_state.get("state_dict")
    if state_dict is None:
        raise ValueError("No 'model_state_dict' found in epoch state")

    # Load state dict
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    true_flows_t = tensors["true_flows_t"]
    true_od_t = tensors["true_od_t"]
    flow_mask_t = tensors["flow_mask_t"]
    od_mask_t = tensors["od_mask_t"]

    # Loss function - build from the checkpoint's stored config if available
    # Try to get loss_weights from saved config, else fall back to defaults on model
    loss_cfg = None
    if "config" in master_checkpoint:
        cfg = master_checkpoint["config"]
        # cfg may be plain dict
        loss_cfg = (cfg.get("model", {}) or {}).get("loss_weights")

    if loss_cfg is not None:
        w_flow = loss_cfg.get("w_flow", 1.0)
        w_od = loss_cfg.get("w_od", 1.0)
        w_reg = loss_cfg.get("w_reg", 0.0)
    else:
        # sensible defaults
        w_flow, w_od, w_reg = 1.0, 1.0, 0.0

    criterion = PartialDataLoss(w_flow=w_flow, w_od=w_od, w_reg=w_reg).to(device)

    with torch.no_grad():
        outputs = model(observed_flows=true_flows_t, flow_mask=flow_mask_t, warmup=False)

        loss_dict = criterion(
            predicted_flows=outputs["reconstructed_flows"],
            true_flows=true_flows_t,
            flow_mask=flow_mask_t,
            predicted_od=outputs["estimated_demand"],
            true_od=true_od_t,
            od_mask=od_mask_t,
            learned_alpha=outputs.get("learned_alpha"),
            learned_beta=outputs.get("learned_beta"),
        )

    # Convert any tensors to python numbers
    metrics = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in loss_dict.items()}

    meta = {
        # timezone-aware UTC timestamp
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_latest_epoch": None,
    }
    # try to pull epoch number
    try:
        epochs = list(master_checkpoint.get("epochs_history", {}).keys())
        if epochs:
            # store as string to avoid static-type warnings and keep JSON-safe
            meta["checkpoint_latest_epoch"] = str(max(int(k) for k in epochs))
    except Exception:
        meta["checkpoint_latest_epoch"] = None

    return {"metrics": metrics, "meta": meta}


def evaluate_checkpoint_file(checkpoint_path: str,
                             loader,
                             tensors: Dict[str, torch.Tensor],
                             device: torch.device,
                             save_dir: Optional[str] = None) -> Dict[str, Any]:
    """Load a .pt checkpoint file, instantiate the saved model, evaluate and optionally
    save results to save_dir as JSON.

    Returns the results dict.
    """
    logging.info(f"Evaluating checkpoint: {checkpoint_path}")

    raw = torch.load(checkpoint_path, map_location=device)

    # instantiate
    model = instantiate_model_from_checkpoint(raw, loader, device)

    results = evaluate_model_from_checkpoint(raw, model, tensors, device)
    # put checkpoint location into meta to keep top-level structure consistent
    if "meta" not in results:
        results["meta"] = {}
    results["meta"]["checkpoint_path"] = checkpoint_path

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.basename(checkpoint_path)
        name = os.path.splitext(base)[0]
        out_file = os.path.join(save_dir, f"{name}_eval.json")
        with open(out_file, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
        logging.info(f"Saved evaluation results to: {out_file}")

    return results

####

def load_eval_bundle(file_path: str) -> Dict[str, Any]:
    """Carga el archivo .pt de evaluación (prefijo eval_)."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    logging.info(f"📂 Cargando datos de evaluación: {file_path}")
    return torch.load(file_path, map_location='cpu')


def get_latest_epoch_data(bundle: Dict[str, Any]) -> tuple:
    """Extrae los datos de la última época registrada en el historial."""
    history = bundle.get('epochs_history', {})
    if not history:
        raise ValueError("El archivo de evaluación no tiene historial de épocas.")

    # Ordenar por número de época (las claves pueden ser int o str)
    sorted_epochs = sorted(history.keys(), key=lambda x: int(x))
    latest_epoch = sorted_epochs[-1]

    logging.info(f"⏳ Usando datos de la época {latest_epoch} para el análisis.")
    return int(latest_epoch), history[latest_epoch]


def calculate_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Calcula R2, MAE, RMSE, MAPE ignorando NaNs y ceros en el target para MAPE."""
    # Aplanar arrays
    y_pred = pred.flatten()
    y_true = target.flatten()

    # RMSE
    mse = np.mean((y_pred - y_true) ** 2)
    rmse = np.sqrt(mse)

    # MAE
    mae = np.mean(np.abs(y_pred - y_true))

    # R2 Score
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    # MAPE (Evitar división por cero)
    mask = y_true != 0
    if np.any(mask):
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan

    return {
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def plot_link_histograms(df: pd.DataFrame, model_name: str, output_dir: str):
    """
    Genera histogramas de Volumen y V/C Ratio agrupados por Link Type.
    Guarda dos imágenes separadas.
    """
    sns.set_style("whitegrid")

    # 1. Histograma de Volumen
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="Pred_Volume", hue="Link_Type", element="step", bins=30, common_norm=False)
    plt.title(f"Distribución de Flujos Predichos por Tipo de Link\n({model_name})")
    plt.xlabel("Volumen (vech/h)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_hist_volume.png"))
    plt.close()

    # 2. Histograma de Volume/Capacity (V/C)
    plt.figure(figsize=(12, 6))
    # Filtramos outliers extremos de V/C para que el gráfico sea legible (ej. > 2.0)
    df_filtered = df[df["VC_Ratio"] <= 2.0]

    sns.histplot(data=df_filtered, x="VC_Ratio", hue="Link_Type", element="step", bins=30, common_norm=False)
    plt.axvline(1.0, color='red', linestyle='--', label='Capacidad (1.0)')
    plt.title(f"Distribución de V/C Ratio por Tipo de Link\n({model_name})")
    plt.xlabel("Volume / Capacity Ratio")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_hist_vc_ratio.png"))
    plt.close()


def process_evaluation(file_path: str, output_dir: str):
    """Función principal que orquesta la evaluación de un solo archivo."""

    model_name = os.path.basename(file_path).replace("eval_", "").replace(".pt", "")
    bundle = load_eval_bundle(file_path)

    # 1. Obtener Datos Estáticos y Dinámicos
    static = bundle['static_data']
    epoch_num, dynamic = get_latest_epoch_data(bundle)

    # 2. Preparar Datos para Links
    # Convertir a numpy
    pred_flows = dynamic['pred_flows'].numpy()
    true_flows = static['true_flows'].numpy()
    capacity = static['capacity'].numpy()
    link_groups = static['link_group'].numpy()

    # Máscaras
    mask_test = static['mask_flow_test'].numpy().astype(bool)
    mask_train = static['mask_flow_train'].numpy().astype(bool)

    # --- MÉTRICAS DE LINKS (Solo en Test Set para validación rigurosa) ---
    if np.any(mask_test):
        link_metrics = calculate_metrics(pred_flows[mask_test], true_flows[mask_test])
        subset_name = "TEST_SET"
    else:
        logging.warning("⚠️ No se encontró máscara de test. Usando TODOS los links.")
        link_metrics = calculate_metrics(pred_flows, true_flows)
        subset_name = "FULL_SET"

    logging.info(f"📊 Métricas Links ({subset_name}): {link_metrics}")

    # --- MÉTRICAS DE OD (Si existe Ground Truth) ---
    od_metrics = {}
    if 'true_od' in static and static['true_od'] is not None:
        true_od = static['true_od'].numpy()
        pred_od = dynamic['pred_od'].numpy()

        # Usar máscara de OD conocidos si existe (para evaluar lo desconocido)
        # O evaluar todo. Generalmente se evalúa todo el OD estimado vs real.
        od_metrics = calculate_metrics(pred_od, true_od)
        logging.info(f"📊 Métricas OD Matrix: {od_metrics}")

    # 3. Guardar Métricas en CSV
    metrics_df = pd.DataFrame([link_metrics])
    metrics_df['type'] = 'Links_Test'
    metrics_df['model'] = model_name

    if od_metrics:
        od_df = pd.DataFrame([od_metrics])
        od_df['type'] = 'OD_Pairs'
        od_df['model'] = model_name
        metrics_df = pd.concat([metrics_df, od_df], ignore_index=True)

    csv_path = os.path.join(output_dir, f"{model_name}_metrics.csv")
    metrics_df.to_csv(csv_path, index=False)

    # 4. Generar DataFrame para Gráficos (Usamos TODOS los links para ver congestión global)
    # Mapeo simple de grupos si son enteros (puedes personalizar esto si tienes un dict de nombres)
    group_map = {0: 'Highway', 1: 'Arterial', 2: 'Collector', 3: 'Local'}  # Ejemplo genérico
    link_types_mapped = [group_map.get(g, f'Type_{g}') for g in link_groups]

    # Evitar división por cero en capacidad
    safe_capacity = capacity.copy()
    safe_capacity[safe_capacity == 0] = 1.0  # Evitar error, aunque capacidad 0 es raro

    df_vis = pd.DataFrame({
        'Pred_Volume': pred_flows,
        'Capacity': capacity,
        'VC_Ratio': pred_flows / safe_capacity,
        'Link_Type': link_types_mapped,
        'Is_Test': mask_test
    })

    plot_link_histograms(df_vis, model_name, output_dir)
    logging.info(f"✅ Evaluación completada para: {model_name}")