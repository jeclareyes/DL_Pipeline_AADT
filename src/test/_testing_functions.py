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

    # Crear DataFrame para facilitar el plot con Seaborn
    data = {'True': target, 'Predicted': pred}

    if mask is not None:
        # Si hay máscara, creamos una columna de categoría
        # mask == True -> mask_label (ej. "Test Set" o "Known OD")
        # mask == False -> "Others" (ej. "Train Set" o "Unknown OD")
        labels = np.where(mask, mask_label, 'Others')
        data['Type'] = labels
        hue = 'Type'
        palette = {mask_label: '#FF0054', 'Others': '#0077B6'}  # Rojo para destacado, Azul para resto
    else:
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
    # 1. Columnas Base
    df = pd.DataFrame({
        'estimated_flow': pred_flows.flatten(),
        'true_flow': true_flows.flatten(),
        'is_observed': mask_observed.flatten().astype(bool)
    })

    # 2. Añadir columnas opcionales (Capacity, t0, Link_Type, etc.)
    if extras:
        for col_name, data in extras.items():
            # Aseguramos que tenga la misma longitud
            if len(data.flatten()) == len(df):
                df[col_name] = data.flatten()
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
    # Validar dimensiones
    if len(pred_demand) != len(od_indices):
        raise ValueError(f"Desajuste: {len(pred_demand)} demandas vs {len(od_indices)} índices OD.")

    # 1. Crear DataFrame Base
    df = pd.DataFrame({
        'origin_node': od_indices[:, 0],
        'destination_node': od_indices[:, 1],
        'estimated_od': pred_demand,
        'known_od_value': true_demand if true_demand is not None else np.zeros_like(pred_demand)
    })

    # 2. Columna Booleana de "Conocido"
    if mask_known is not None:
        df['is_known_pair'] = mask_known.astype(bool)
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


def process_evaluation(file_path: str, output_dir: str):
    """Función principal orquestadora (Actualizada con Scatter Plots)."""

    model_name = os.path.basename(file_path).replace("eval_", "").replace(".pt", "")
    bundle = load_eval_bundle(file_path)

    static = bundle['static_data']
    mask = static['masks']
    epoch_num, dynamic = get_latest_epoch_data(bundle)

    # --- DATOS FLOWS ---
    pred_flows = dynamic['pred_flows'].numpy()
    true_flows = static['true_flows'].numpy()
    capacity = static['capacity'].numpy()
    link_groups = static['link_group'].numpy()
    mask_test = mask['flow_test'].numpy().astype(bool)



    # --- MÉTRICAS FLOWS ---
    # Calcular métricas globales y específicas de Test
    if np.any(mask_test):
        link_metrics = calculate_metrics(pred_flows[mask_test], true_flows[mask_test])
        subset_name = "TEST_SET"
        scatter_mask = mask_test
        scatter_label = "Test Links"
    else:
        link_metrics = calculate_metrics(pred_flows, true_flows)
        subset_name = "FULL_SET"
        scatter_mask = None
        scatter_label = "All Links"

    logging.info(f"Métricas Links ({subset_name}): {link_metrics}")

    # GRÁFICO 1: Scatter Flows (Real vs Pred)
    plot_scatter_comparison(
        pred=pred_flows,
        target=true_flows,
        mask=scatter_mask,
        mask_label=scatter_label,
        title=f"Traffic Flows: True vs Estimated ({model_name})",
        xlabel="True Flow (veh/h)",
        ylabel="Estimated Flow (veh/h)",
        output_path=os.path.join(output_dir, f"{model_name}_scatter_flows.png"),
        log_scale=True  # Recomendado para flujos de tráfico
    )

    # --- DATOS OD ---
    od_metrics = {}
    if 'true_od' in static and static['true_od'] is not None:
        true_od = static['true_od'].numpy()
        pred_od = dynamic['pred_od'].numpy()

        # Máscara de OD Conocidos
        if 'mask_od_known' in static:
            mask_od = static['mask_od_known'].numpy().astype(bool)
        else:
            mask_od = None

        # Métricas sobre TODO el conjunto (generalmente queremos ver si recuperó la matriz entera)
        od_metrics = calculate_metrics(pred_od, true_od)
        logging.info(f"Métricas OD Matrix (Global): {od_metrics}")

        # GRÁFICO 2: Scatter OD (Real vs Pred)
        # Filtramos para resaltar los conocidos ("que solo se conocían obviamente")
        plot_scatter_comparison(
            pred=pred_od,
            target=true_od,
            mask=mask_od,
            mask_label="Known OD (Input)",  # Etiqueta para los datos que SÍ se conocían
            title=f"OD Demand: True vs Estimated ({model_name})",
            xlabel="True Demand",
            ylabel="Estimated Demand",
            output_path=os.path.join(output_dir, f"{model_name}_scatter_od.png"),
            log_scale=True
        )

    # 3. Guardar CSV (Igual que antes)
    metrics_df = pd.DataFrame([link_metrics])
    metrics_df['type'] = 'Links_Test'
    metrics_df['model'] = model_name

    if od_metrics:
        od_df = pd.DataFrame([od_metrics])
        od_df['type'] = 'OD_Pairs_Global'
        od_df['model'] = model_name
        metrics_df = pd.concat([metrics_df, od_df], ignore_index=True)

    metrics_df.to_csv(os.path.join(output_dir, f"{model_name}_metrics.csv"), index=False)

    # 4. Histogramas (Igual que antes)
    group_map = {0: 'Multi-Lane', 1: 'Motorway', 2: 'Two_Lane', 3: 'Rural', 4: 'Connectors'}
    link_types_mapped = [group_map.get(g, f'Type_{g}') for g in link_groups]
    safe_capacity = capacity.copy()
    safe_capacity[safe_capacity == 0] = 1.0

    df_vis = pd.DataFrame({
        'Pred_Volume': pred_flows,
        'Capacity': capacity,
        'VC_Ratio': pred_flows / safe_capacity,
        'Link_Type': link_types_mapped,
        'Is_Test': mask_test
    })

    plot_link_histograms(df_vis, model_name, output_dir)
    logging.info(f"Evaluación (Gráficos y Métricas) completada para: {model_name}")

    # =====================================================
    # NUEVO: EXPORTACIÓN DE FLUJOS (CSV)
    # =====================================================
    # Preparamos las variables opcionales del usuario
    flow_extras = {
        'capacity': static['capacity'].numpy(),
        't0': static['t0'].numpy(),
        'link_group': static['link_group'].numpy()
    }

    # Usamos la máscara de test como "is_observed" para distinguir
    # OJO: Si tienes una máscara global de "sensores reales", úsala aquí.
    # Usualmente mask_flow_train | mask_flow_test = todos los sensores.
    mask_observed_total = (
            static['masks']['flow_train'].numpy().astype(bool) |
            static['masks']['flow_test'].numpy().astype(bool)
    )

    export_flows_csv(
        pred_flows=pred_flows,
        true_flows=true_flows,
        mask_observed=mask_observed_total,
        output_path=os.path.join(output_dir, f"{model_name}_flows_detailed.csv"),
        extras=flow_extras
    )

    # =====================================================
    # EXPORTACIÓN DE OD PAIRS (CSV) - ACTUALIZADO
    # =====================================================
    if 'true_od' in static and static['true_od'] is not None:

        # VERIFICACIÓN: ¿Tenemos los índices guardados?
        if 'od_pair_indices' in static:
            od_indices = static['od_pair_indices'].numpy()
            num_centroids = static.get('num_centroids', 0)  # Leemos del archivo, no del loader

            true_od = static['true_od'].numpy()
            pred_od = dynamic['pred_od'].numpy()

            mask_od_known = None
            if 'mask_od_known' in static:
                mask_od_known = static['mask_od_known'].numpy()

            export_od_analysis_csv(
                pred_demand=pred_od,
                true_demand=true_od,
                od_indices=od_indices,
                output_path=os.path.join(output_dir, f"{model_name}_od_analysis.csv"),
                mask_known=mask_od_known,
                num_centroids=num_centroids
            )
        else:
            # Fallback por si intentas evaluar un modelo viejo que no tenía estos datos guardados
            logging.warning(
                f"El archivo {model_name} no contiene 'od_indices'. No se puede exportar el análisis OD detallado. (Re-entrena el modelo con el nuevo pipeline).")

    logging.info(f"Evaluación completa para: {model_name}")