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

from omegaconf import OmegaConf, DictConfig
import hydra

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.contracts.runtime_contracts import (
    EvalBundleContractError,
    TaskDispatchContractError,
    resolve_testing_dispatch_plan,
    validate_artifacts_contract,
    validate_eval_bundle_contract,
)

# Imported here to avoid circular imports at package import time in some cases


# Utility: coerce tensors/arrays to 1-D numpy arrays before building DataFrames
def _to_1d(arr, name: Optional[str] = None):
    """Coerce input to a 1-D numpy array.

    Args:
        arr: torch.Tensor, numpy array, list, or None.
        name: optional name used in error messages.

    Returns:
        1-D numpy.ndarray or None if arr is None.
    """
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr = arr.cpu().numpy()
    arr = np.asarray(arr)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.ravel()


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

    Now creates the Loss function dynamically based on the saved config.
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

    # --- CORRECCIÓN: Instanciación Dinámica de la Loss ---
    criterion = None

    # 1. Recuperar la config del checkpoint
    if "config" in master_checkpoint:
        # Aseguramos que sea OmegaConf para poder navegar fácil
        ckpt_cfg = master_checkpoint["config"]
        if not isinstance(ckpt_cfg, (DictConfig, list)):
            ckpt_cfg = OmegaConf.create(ckpt_cfg)

        # 2. Buscar la definición de la loss en la config guardada
        if hasattr(ckpt_cfg, "model") and hasattr(ckpt_cfg.model, "loss"):
            try:
                # Instancia EXACTAMENTE la misma loss que se usó al entrenar
                criterion = hydra.utils.instantiate(ckpt_cfg.model.loss).to(device)
                logging.info(f"Loss function instantiated dynamically: {ckpt_cfg.model.loss._target_}")
            except Exception as e:
                logging.warning(f"Could not instantiate loss from config: {e}. Metrics based on Loss will be skipped.")

    # Si falló la carga o es un checkpoint muy antiguo sin config de loss
    if criterion is None:
        logging.warning("No loss config found. Using a generic MSE evaluator as fallback.")
        # Fallback genérico simple (solo MSE) para no romper el código
        criterion = torch.nn.MSELoss()
        # Nota: Esto no generará el diccionario detallado (l_flow, l_od),
        # así que el bloque siguiente necesita un try-catch o lógica condicional.

    with torch.no_grad():
        outputs = model(observed_flows=true_flows_t, flow_mask=flow_mask_t, warmup=False)

        # Calculamos la loss solo si tenemos un criterio válido y complejo
        # Asumimos que si instanciamos via Hydra, cumple la interfaz estándar (acepta kwargs)
        try:
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
        except TypeError:
            # Fallback por si criterion es un simple nn.MSELoss()
            loss_val = criterion(outputs["reconstructed_flows"], true_flows_t)
            loss_dict = {"total_loss": loss_val}

    # Convert any tensors to python numbers
    metrics = {k: (v.item() if isinstance(v, torch.Tensor) else float(v)) for k, v in loss_dict.items()}

    meta = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_latest_epoch": None,
    }

    try:
        epochs = list(master_checkpoint.get("epochs_history", {}).keys())
        if epochs:
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

    logging.info(f"Cargando datos de evaluación: {file_path}")
    return torch.load(file_path, map_location='cpu')


def get_latest_epoch_data(bundle: Dict[str, Any]) -> tuple:
    """Extrae los datos de la última época registrada en el historial."""
    history = bundle.get('epochs_history', {})
    if not history:
        raise ValueError("El archivo de evaluación no tiene historial de épocas.")

    # Ordenar por número de época (las claves pueden ser int o str)
    sorted_epochs = sorted(history.keys(), key=lambda x: int(x))
    latest_epoch = sorted_epochs[-1]

    logging.info(f"Usando datos de la época {latest_epoch} para el análisis.")
    return int(latest_epoch), history[latest_epoch]


def plot_scatter_comparison(
        pred: np.ndarray,
        target: np.ndarray,
        mask: Optional[np.ndarray],
        mask_label: str,
        title: str,
        xlabel: str,
        ylabel: str,
        output_path: str,
        log_scale: bool = True
):
    """
    Genera un scatter plot comparando valores reales vs estimados.

    Args:
        mask: Array booleano. Si se provee, separa los datos en dos grupos
              (ej. Train vs Test, o Conocido vs Desconocido).
        mask_label: Etiqueta para los datos donde mask == True.
    """
    plt.figure(figsize=(10, 8))
    sns.set_style("whitegrid")

    # Coerce inputs to 1-D numpy arrays
    pred_arr = _to_1d(pred, name="pred")
    true_arr = _to_1d(target, name="target")

    if pred_arr.shape != true_arr.shape:
        raise ValueError(f"pred and target shapes differ: {pred_arr.shape} vs {true_arr.shape}")

    if mask is not None:
        # Si hay máscara, creamos una columna de categoría
        # mask == True -> mask_label (ej. "Test Set" o "Known OD")
        # mask == False -> "Others" (ej. "Train Set" o "Unknown OD")
        mask_arr = _to_1d(mask, name="mask")
        if mask_arr.shape != true_arr.shape:
            raise ValueError(f"Mask shape {mask_arr.shape} != data shape {true_arr.shape}")
        labels = np.where(mask_arr, mask_label, 'Others')
        data = {'True': true_arr, 'Predicted': pred_arr, 'Type': labels}
        hue = 'Type'
        palette = {mask_label: '#FF0054', 'Others': '#0077B6'}  # Rojo para destacado, Azul para resto
    else:
        data = {'True': true_arr, 'Predicted': pred_arr}
        hue = None
        palette = None

    df_plot = pd.DataFrame(data)

    # Filtrar ceros si vamos a usar escala logarítmica para evitar errores
    if log_scale:
        df_plot = df_plot[(df_plot['True'] > 0) & (df_plot['Predicted'] > 0)]

    # Scatter Plot
    sns.scatterplot(
        data=df_plot,
        x='True',
        y='Predicted',
        hue=hue,
        palette=palette,
        alpha=0.6,
        edgecolor=None
    )

    # Línea de identidad (Perfect Fit)
    min_val = min(df_plot['True'].min(), df_plot['Predicted'].min())
    max_val = max(df_plot['True'].max(), df_plot['Predicted'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.5, label='Perfect Fit')

    if log_scale:
        plt.xscale('log')
        plt.yscale('log')
        plt.title(f"{title} (Log Scale)")
    else:
        plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_link_histograms(df: pd.DataFrame, model_name: str, output_dir: str):
    """Genera histogramas de Volumen y V/C Ratio (Igual que antes)."""
    sns.set_style("whitegrid")

    # 1. Volumen
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df, x="Pred_Volume", hue="Link_Type", element="step", bins=50, common_norm=False)
    plt.title(f"Distribución de Flujos Predichos por Tipo de Link\n({model_name})")
    plt.xlabel("Volumen (veh/h)")
    plt.savefig(os.path.join(output_dir, f"{model_name}_hist_volume.png"))
    plt.close()

    # 2. V/C Ratio
    plt.figure(figsize=(12, 6))
    df_filtered = df[df["VC_Ratio"] <= 2.0]
    sns.histplot(data=df_filtered, x="VC_Ratio", hue="Link_Type", element="step", bins=50, common_norm=False)
    plt.axvline(1.0, color='red', linestyle='--', label='Capacidad (1.0)')
    plt.title(f"Distribución de V/C Ratio por Tipo de Link\n({model_name})")
    plt.xlabel("Volume / Capacity Ratio")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"{model_name}_hist_vc_ratio.png"))
    plt.close()

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


# =============================================================================
# FUNCIONES DE EXPORTACIÓN (Nuevas)
# =============================================================================

def export_flows_csv(
        pred_flows: np.ndarray,
        true_flows: np.ndarray,
        mask_observed: np.ndarray,
        output_path: str,
        extras: Dict[str, np.ndarray] = None
):
    """
    Exporta un CSV con comparación de flujos y variables físicas opcionales.

    Args:
        pred_flows: Array de flujos estimados.
        true_flows: Array de flujos reales (ground truth).
        mask_observed: Array booleano (True donde hay lectura real).
        output_path: Ruta donde guardar el CSV.
        extras: Diccionario opcional con {nombre_columna: array_datos}
                ej: {'Capacity': cap_array, 'Link_Type': type_array}
    """
    # 1. Columnas Base (aseguramos 1-D)
    pred_vec = _to_1d(pred_flows, name='pred_flows')
    true_vec = _to_1d(true_flows, name='true_flows')
    mask_vec = _to_1d(mask_observed, name='mask_observed')

    df = pd.DataFrame({
        'estimated_flow': pred_vec,
        'true_flow': true_vec,
        'is_observed': mask_vec.astype(bool)
    })

    # 2. Añadir columnas opcionales (Capacity, t0, Link_Type, etc.)
    if extras:
        for col_name, data in extras.items():
            flat = _to_1d(data, name=col_name)
            if flat is None:
                continue
            # Aseguramos que tenga la misma longitud
            if flat.shape[0] == df.shape[0]:
                df[col_name] = flat
            else:
                logging.warning(
                    f"La variable extra '{col_name}' no tiene la misma longitud que los flujos. Se omite.")

    # 3. Guardar
    df.to_csv(output_path, index_label='link_index')
    logging.info(f"Flujos exportados a: {output_path}")


def _classify_pair_type(o_node: int, d_node: int, num_tazs: int) -> str:
    """Clasifica el par según si sus nodos son TAZ (Centroides) o Auxiliares."""
    # Asumimos que los primeros 'num_tazs' nodos (0 a num_tazs-1) son los centroides
    o_is_taz = o_node < num_tazs
    d_is_taz = d_node < num_tazs

    if o_is_taz and d_is_taz:
        return 'taz to taz'
    elif o_is_taz and not d_is_taz:
        return 'taz to aux'
    elif not o_is_taz and d_is_taz:
        return 'aux to taz'
    else:
        return 'aux to aux'


def export_od_analysis_csv(
        pred_demand: np.ndarray,
        true_demand: np.ndarray,
        od_indices: np.ndarray,
        output_path: str,
        mask_known: Optional[np.ndarray] = None,
        num_centroids: int = 0
):
    """
    Exporta CSV detallado de pares OD con clasificación de tipo de nodo.

    Args:
        pred_demand: Array de demanda estimada.
        true_demand: Array de demanda real.
        od_indices: Array/Tensor [N_pairs, 2] con los IDs de nodo (Origen, Destino).
        output_path: Ruta de salida.
        mask_known: Array booleano indicando qué pares eran conocidos (input).
        num_centroids: Número de centroides (TAZs) para clasificar tipos de par.
    """
    # 1. Crear DataFrame Base (asegurando 1-D para pred/true y formato para índices)
    pred_vec = _to_1d(pred_demand, name='pred_demand')
    true_vec = _to_1d(true_demand, name='true_demand') if true_demand is not None else np.zeros_like(pred_vec)

    # Asegurar od_indices es numpy array con forma (N,2)
    if isinstance(od_indices, torch.Tensor):
        indices_np = od_indices.cpu().numpy()
    else:
        indices_np = np.asarray(od_indices)

    if indices_np.ndim != 2 or indices_np.shape[1] < 2:
        raise ValueError(f"od_indices must be shape (N,2), got {indices_np.shape}")

    if len(pred_vec) != indices_np.shape[0]:
        raise ValueError(f"Desajuste: {len(pred_vec)} demandas vs {indices_np.shape[0]} índices OD.")

    df = pd.DataFrame({
        'origin_node': indices_np[:, 0],
        'destination_node': indices_np[:, 1],
        'estimated_od': pred_vec,
        'known_od_value': true_vec
    })

    # 2. Columna Booleana de "Conocido"
    if mask_known is not None:
        mask_vec = _to_1d(mask_known, name='mask_known')
        if mask_vec.shape[0] != df.shape[0]:
            logging.warning("La máscara 'mask_known' no coincide en longitud con los pares OD; se ignorará.")
        else:
            df['is_known_pair'] = mask_vec.astype(bool)
    else:
        df['is_known_pair'] = False  # Por defecto todo desconocido si no se pasa máscara

    # 3. Clasificación de Pares (TAZ vs Aux)
    # Vectorizamos la operación para eficiencia
    if num_centroids > 0:
        # Convertimos a numpy para iterar rápido si no lo es
        if isinstance(od_indices, torch.Tensor):
            indices_np = od_indices.cpu().numpy()
        else:
            indices_np = od_indices

        df['pair_type'] = [
            _classify_pair_type(o, d, num_centroids)
            for o, d in indices_np
        ]
    else:
        df['pair_type'] = 'unknown (no num_centroids provided)'

    # 4. Reordenar columnas para que origin/dest/type queden primero
    cols = ['origin_node', 'destination_node', 'pair_type', 'estimated_od', 'known_od_value', 'is_known_pair']
    df = df[cols]

    df.to_csv(output_path, index=False)
    logging.info(f"Análisis OD exportado a: {output_path}")


def process_evaluation(
        file_path: str,
        output_dir: str,
        testing_cfg,
        resolved_task_names: Optional[list[str]] = None,
) -> None:
    """Config-driven evaluation dispatcher.

    The pipeline loads artifacts and runs task functions resolved from
    capability-based dispatch in configs/testing/testing.yaml.

    Args:
        resolved_task_names: Optional preflight-resolved callable task list.
            If omitted, tasks are resolved in-place with resolver safeguards.
    """
    from src.test import evaluation_tasks

    model_name = os.path.basename(file_path).replace("eval_", "").replace(".pt", "")
    bundle = load_eval_bundle(file_path)
    bundle = validate_eval_bundle_contract(bundle)

    static = dict(bundle["static_data"])
    masks = dict(static.get("masks", {}))

    # Canonical OD mask convention: masks.od_mask (legacy alias: static.mask_od_known).
    if "od_mask" not in masks and "mask_od_known" in static:
        masks["od_mask"] = static["mask_od_known"]
    if "mask_od_known" not in static and "od_mask" in masks:
        static["mask_od_known"] = masks["od_mask"]

    _, dynamic = get_latest_epoch_data(bundle)

    # Strict canonical artifacts contract (legacy fallback disabled by design).
    artifacts = dynamic.get("artifacts")
    if artifacts is None:
        raise EvalBundleContractError(
            "Evaluation bundle epoch payload missing required key 'artifacts'. "
            "Legacy key fallback is disabled in strict mode."
        )

    artifacts = validate_artifacts_contract(artifacts)

    if resolved_task_names is not None:
        task_names = [str(t).strip() for t in resolved_task_names if str(t).strip()]
    else:
        available_task_names = sorted(
            name
            for name, obj in vars(evaluation_tasks).items()
            if callable(obj) and not name.startswith("_") and getattr(obj, "__module__", "") == evaluation_tasks.__name__
        )
        dispatch_plan = resolve_testing_dispatch_plan(
            testing_cfg,
            available_task_names=available_task_names,
        )
        task_names = list(dispatch_plan["tasks_callable"])

        if dispatch_plan["tasks_unknown"]:
            logging.warning(
                "Unknown tasks in capability_dispatch were ignored: %s",
                dispatch_plan["tasks_unknown"],
            )
        if dispatch_plan["capabilities_unused"]:
            logging.warning(
                "Unused capabilities defined in capability_dispatch for model '%s': %s",
                dispatch_plan["model_key"],
                dispatch_plan["capabilities_unused"],
            )

    task_names = list(dict.fromkeys(task_names))
    if len(task_names) == 0:
        raise TaskDispatchContractError(
            f"No callable evaluation tasks resolved for model '{testing_cfg.model_to_test}'"
        )

    valid_task_count = 0
    for task_name in task_names:
        task_fn = getattr(evaluation_tasks, task_name, None)
        if not callable(task_fn):
            logging.warning(
                "Configured task '%s' is not defined in src.test.evaluation_tasks. Skipping.",
                task_name,
            )
            continue

        valid_task_count += 1
        task_fn(
            artifacts=artifacts,
            static=static,
            masks=masks,
            output_dir=output_dir,
            model_name=model_name,
        )

    if valid_task_count == 0:
        raise TaskDispatchContractError(
            f"No callable evaluation tasks resolved for model '{testing_cfg.model_to_test}'"
        )

    logging.info(f"Evaluation completed for: {model_name}")