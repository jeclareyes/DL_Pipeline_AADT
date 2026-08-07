# src/train/_pipeline_utils.py
import logging
import json
from pathlib import Path
from typing import Any, Dict, Iterator

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from sklearn.model_selection import KFold

from src.contracts.runtime_contracts import (
    ConfigurationContractError,
    ModelInputContractError,
)

def to_json_serializable(value: Any) -> Any:
    """
    Convert common Python, NumPy, PyTorch and OmegaConf objects into JSON-safe
    Python objects.

    This helper is used by the training pipeline, run context and evaluator
    before writing summaries or metrics to disk.

    Parameters
    ----------
    value : Any
        Object to convert.

    Returns
    -------
    Any
        JSON-serializable object.
    """

    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, DictConfig) or isinstance(value, ListConfig):
        return to_json_serializable(
            OmegaConf.to_container(
                value,
                resolve=True,
            )
        )

    if isinstance(value, dict):
        return {
            str(key): to_json_serializable(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            to_json_serializable(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        detached = value.detach().cpu()

        if detached.numel() == 1:
            return detached.item()

        return detached.tolist()

    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)



def extract_link_types_for_visualization(link_df, strict_data: bool = False) -> np.ndarray:
    """Extract link types from link_df, with configurable strictness."""
    if link_df is None:
        raise ModelInputContractError("link_df must not be None")

    if 'link_type' not in link_df.columns:
        if strict_data:
            raise ModelInputContractError("Column 'link_type' is required when training.strict_data=true")
        logging.warning("Column 'link_type' not found. Falling back to 'Unknown'.")
        return np.array(['Unknown'] * len(link_df), dtype=str)

    try:
        return link_df['link_type'].astype(str).values
    except Exception as exc:
        if strict_data:
            raise ModelInputContractError(f"Failed to extract 'link_type': {exc}") from exc
        logging.warning("Could not extract link_type values. Falling back to 'Unknown'.")
        return np.array(['Unknown'] * len(link_df), dtype=str)


def maybe_compute_t0_od_cost_prior(cfg, raw_data: dict, network_params: dict, device):
    """Compute t0 OD prior only when explicitly enabled in config."""
    compute_prior = bool(cfg.training.get('compute_t0_od_cost_prior', False))
    if not compute_prior:
        return None

    graph_obj = raw_data['graph']
    od_indices = network_params['od_pair_indices']
    return compute_od_cost_prior(
        link_df=raw_data['link_data'],
        od_indices_tensor=od_indices,
        graph_obj=graph_obj,
        device=device,
        node_id_to_idx=network_params.get["node_id_to_idx"]
    )


def compute_initial_demand_mean(
    od_vector_np: np.ndarray,
    observed_od_mask_np: np.ndarray,
    fallback_value: float = 1.0,
) -> float:
    """Compute initial mean demand over observed OD entries."""
    sum_known = float((od_vector_np * observed_od_mask_np).sum())
    count_known = float(observed_od_mask_np.sum())
    if count_known > 0:
        return float(sum_known / count_known)

    logging.warning("No observed ODs available to compute initial mean demand. Using fallback value.")
    return float(fallback_value)


def vi_od_initialization_hook(cfg, model_params: dict, context: dict | None = None) -> dict:
    """Model hook for VI OD initialization policy without model-name branching in pipeline."""
    # These values are constructor kwargs for the active VI model. Previously
    # the hook returned them as metadata, but ODDemandCompletionNet ignored
    # them, so unknown cells started at softplus(0) * od_scale.
    _ = context
    od_init_cfg = cfg.training.get('od_initialization', {})
    od_init_mode = str(od_init_cfg.get('mode', 'known_mean')).lower()

    if od_init_mode == 'low_unknown':
        low_unknown_val = float(od_init_cfg.get('low_unknown_value', 0.01))
        return {
            'init_unknown_od_low': True,
            'unknown_od_init_value': low_unknown_val,
        }
    if od_init_mode == 'known_mean':
        return {
            'init_unknown_od_low': False,
            'unknown_od_init_value': float(model_params.get('od_known_mean', 1.0)),
        }

    raise ConfigurationContractError(
        f"Unknown training.od_initialization.mode='{od_init_mode}'. "
        "Allowed values: known_mean|low_unknown"
    )


def apply_model_pre_instantiate_hook(cfg, model_params: dict, context: dict | None = None) -> dict:
    """Apply an optional pre-instantiation hook declared in model config."""
    hook_path = None
    if hasattr(cfg, 'model'):
        hook_path = cfg.model.get('pipeline_pre_instantiate_hook', None)

    if not hook_path:
        return {}

    try:
        hook_fn = hydra.utils.get_method(str(hook_path))
    except Exception as exc:
        raise ConfigurationContractError(
            f"Could not resolve model hook '{hook_path}'"
        ) from exc

    if not callable(hook_fn):
        raise ConfigurationContractError(f"Configured model hook '{hook_path}' is not callable")

    updates = hook_fn(cfg=cfg, model_params=model_params, context=context or {})
    if updates is None:
        updates = {}
    if not isinstance(updates, dict):
        raise ConfigurationContractError(
            f"Model hook '{hook_path}' must return dict or None, got {type(updates)}"
        )

    model_params.update(updates)
    return updates


def evaluate_holdout_metrics(model, true_flows_t, train_mask_np, test_mask_np, device):
    """Evaluate hold-out metrics in a dedicated reusable function."""
    with torch.no_grad():
        model.eval()
        test_results = model(
            observed_flows=true_flows_t,
            flow_mask=torch.FloatTensor(train_mask_np).to(device),
        )

        y_pred_all = test_results['reconstructed_flows'].detach().cpu().numpy().flatten()
        y_true_all = true_flows_t.detach().cpu().numpy().flatten()
        test_mask = np.asarray(test_mask_np).flatten().astype(bool)

        y_pred = y_pred_all[test_mask]
        y_true = y_true_all[test_mask]

        if len(y_true) == 0:
            return {'has_holdout': False, 'count': 0}

        mae = float(np.mean(np.abs(y_pred - y_true)))
        mse = float(np.mean((y_true - y_pred) ** 2))
        rmse = float(np.sqrt(mse))

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        r2 = float(1 - (ss_res / (ss_tot + 1e-8)))

        non_zero = y_true != 0
        mape = float(np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100) if np.any(non_zero) else 0.0

        return {
            'has_holdout': True,
            'count': int(len(y_true)),
            'r2': r2,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
        }


def persist_pipeline_summary(
        output_dir: str | Path,
        summary: dict,
        filename: str = "pipeline_summary.json",
    ) -> str:
    """
    Persist the pipeline summary as a JSON file.

    Parameters
    ----------
    output_dir : str | os.PathLike
        Output directory.

    summary : dict
        Pipeline summary payload.

    filename : str, default="pipeline_summary.json"
        Output filename.

    Returns
    -------
    str
        Path to the saved JSON file.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_path = output_dir / filename
    out_path.write_text(
        json.dumps(
            to_json_serializable(summary),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return str(out_path)

def summarize_training_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a compact JSON-safe summary of a training task.

    The full task dictionary may contain large NumPy masks. This function keeps
    only lightweight metadata and mask counts, which are useful in pipeline
    summaries.

    Parameters
    ----------
    task : Dict[str, Any]
        Training task generated by generate_training_tasks().

    Returns
    -------
    Dict[str, Any]
        Compact task summary.
    """

    if task is None:
        return {}

    train_mask = task.get("train_mask", task.get("flow_train_input_mask"))
    val_mask = task.get("val_mask")

    summary = {
        "name": str(task.get("name", "Unnamed Task")),
        "suffix": str(task.get("suffix", ".pt")),
    }

    if train_mask is not None:
        train_arr = _to_numpy(train_mask).reshape(-1)
        summary["num_train_links"] = int(np.asarray(train_arr).sum())
        summary["train_mask_size"] = int(train_arr.size)

    if val_mask is not None:
        val_arr = _to_numpy(val_mask).reshape(-1)
        summary["num_val_links"] = int(np.asarray(val_arr).sum())
        summary["val_mask_size"] = int(val_arr.size)

    return summary


def generate_training_tasks(
    train_mask_global: np.ndarray,
    k_folds: int,
    random_seed: int,
) -> Iterator[Dict[str, Any]]:
    """Yield training tasks for standard training or k-fold cross-validation.

    Contract:
    - If k_folds <= 1, yields one task with an empty validation mask.
    - If k_folds > 1, yields one task per fold.
    """
    if train_mask_global is None:
        raise ModelInputContractError("train_mask_global must not be None")
    if not isinstance(k_folds, int):
        raise ConfigurationContractError(f"k_folds must be int, got {type(k_folds)}")
    if k_folds < 1:
        raise ConfigurationContractError(f"k_folds must be >= 1, got {k_folds}")

    train_mask_global = np.asarray(train_mask_global, dtype=np.float32)
    if train_mask_global.ndim != 1:
        raise ModelInputContractError("train_mask_global must be a 1D mask")

    train_indices = np.where(train_mask_global > 0)[0]
    if train_indices.size <= 0:
        raise ModelInputContractError("Global training mask is empty")

    if k_folds <= 1:
        yield {
            "name": "Standard Run",
            "suffix": ".pt",
            "train_mask": train_mask_global.copy(),
            "val_mask": np.zeros_like(train_mask_global, dtype=np.float32),
        }
        return

    if k_folds > train_indices.size:
        raise ConfigurationContractError(
            f"k_folds ({k_folds}) cannot exceed available training samples ({train_indices.size})"
        )

    splitter = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)
    for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(train_indices), start=1):
        fold_train_mask = np.zeros_like(train_mask_global, dtype=np.float32)
        fold_val_mask = np.zeros_like(train_mask_global, dtype=np.float32)

        fold_train_mask[train_indices[train_idx]] = 1.0
        fold_val_mask[train_indices[val_idx]] = 1.0

        if fold_train_mask.shape != fold_val_mask.shape:
            raise ModelInputContractError("Fold masks shape mismatch")
        if fold_train_mask.sum() <= 0:
            raise ModelInputContractError(f"Fold {fold_idx} has empty training mask")

        yield {
            "name": f"Fold {fold_idx}",
            "suffix": f"_fold{fold_idx - 1}.pt",
            "train_mask": fold_train_mask,
            "val_mask": fold_val_mask,
        }

### Funciones de estadísticas para máscaras de sampleo ###

def _to_numpy(arr: Any) -> np.ndarray:
    """
    Convert input (torch.Tensor, list, np.ndarray, etc.) to a NumPy array on CPU.
    """
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def stats(mask: Any) -> str:
    """
    Return formatted stats string for a mask:
    \"count/total (pct%)\" matching the original formatting.
    Accepts torch.Tensor, np.ndarray or Python sequences.
    """
    arr = _to_numpy(mask)
    # Treat boolean or numeric masks: sum counts truthy elements
    count = int(arr.sum())
    total = int(arr.size)
    pct = (count / total) * 100 if total > 0 else 0.0
    return f"{count:5d}/{total:<5d} ({pct:4.1f}%)"


def stats_relative(mask: Any, base_mask: Any) -> str:
    """
    Return formatted relative stats string: \"num/den (pct%)\" where pct is relative to base_mask.
    """
    arr = _to_numpy(mask)
    base = _to_numpy(base_mask)
    num = int(arr.sum())
    den = int(base.sum())
    pct = (num / den) * 100 if den > 0 else 0.0
    return f"{num:4d}/{den:<4d} ({pct:4.1f}%)"

def log_dataset_summary(
    label: str,
    total_count: int,
    observed_mask,
    train_mask,
    test_mask,
    unobserved_mask=None,
    logger=logging,
    stats_fn=stats,
    stats_rel_fn=stats_relative,
):
    """
    Compact helper to log dataset/mask summaries.
    """
    if unobserved_mask is None:
        unobserved_mask = np.clip(1.0 - observed_mask, 0.0, 1.0)

    logger.info("=" * 60)
    logger.info(f" {label} SUMMARY ")
    logger.info("=" * 60)
    logger.info(f" {label} TOTALES: {total_count}")
    logger.info(f"  |-- [1] OBSERVED (Known):        {stats_fn(observed_mask)}")
    logger.info(
        f"  |    |-- Training:               "
        f"{stats_fn(train_mask)}   "
        f"[{stats_rel_fn(train_mask, observed_mask)} de OBS]"
    )
    logger.info(
        f"  |    |-- Testing:                "
        f"{stats_fn(test_mask)}   "
        f"[{stats_rel_fn(test_mask, observed_mask)} de OBS]"
    )
    logger.info(f"  |-- [0] UNOBSERVED (Unknown):    {stats_fn(unobserved_mask)}")
    logger.info("=" * 60)


### Función de chequeo visual intermedio (durante entrenamiento) de scatter plot

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


def check_intermediate_convergence(model, val_loader, scaler, epoch, writer=None):
    """
    Realiza un chequeo visual y estadístico del modelo.
    Args:
        model: Tu instancia de UltraCyclicODModel.
        val_loader: DataLoader de validación.
        scaler: El valor de max_trips_scaler usado para desnormalizar.
        epoch: Número de época actual.
        writer: (Opcional) Instancia de SummaryWriter (TensorBoard) o WandB.
    """
    model.eval()
    all_pred_flows = []
    all_true_flows = []

    # 1. Recolectar datos (sin gradientes)
    with torch.no_grad():
        for batch in val_loader:
            obs_counts = batch['observed_counts']  # [Batch, Links]
            # Asegúrate de pasar datos al device correcto
            obs_counts = obs_counts.to(model.forward_encoder.network[0].weight.device)

            # Forward pass
            out = model(obs_counts, warmup=False)

            # Obtener flujos (Normalizados [0,1])
            pred = out['reconstructed_flows'].cpu().numpy()
            true = obs_counts.cpu().numpy()

            # Desnormalizar a Vehículos/Hora reales
            pred_real = pred * scaler
            true_real = true * scaler

            # Aplanar y guardar solo donde hay mediciones (mask > 0 si aplica)
            # Aquí asumimos que obs_counts tiene 0 donde no hay sensor, o usas una máscara externa
            mask = true_real > 0
            all_pred_flows.append(pred_real[mask])
            all_true_flows.append(true_real[mask])

    # Concatenar todo
    y_pred = np.concatenate(all_pred_flows)
    y_true = np.concatenate(all_true_flows)

    # 2. Calcular Métricas
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # Calcular pendiente (Slope) m en y = mx + c
    # Si m < 1 subestima, m > 1 sobreestima
    slope = np.polyfit(y_true, y_pred, 1)[0]

    print(f"\n--- Chequeo Epoca {epoch} ---")
    print(f"R2 Score: {r2:.4f} (Objetivo: -> 1.0)")
    print(f"Slope:    {slope:.4f} (Objetivo: -> 1.0)")
    print(f"RMSE:     {rmse:.2f} veh/h")

    # 3. Generar Gráfico Scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, alpha=0.3, s=1, c='blue', label='Predicciones')

    # Línea de identidad (Ideal)
    lims = [0, max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'r--', alpha=0.75, label='Ideal (y=x)')

    ax.set_xlabel('Flujos Reales (Aforos)')
    ax.set_ylabel('Flujos Estimados (Modelo)')
    ax.set_title(f'Converge Check - Epoch {epoch}\nR2={r2:.3f}')
    ax.legend()
    ax.grid(True)

    # 4. Guardar en Tensorboard/WandB o Disco
    if writer:
        writer.add_figure('Convergence/Scatter_Flows', fig, epoch)
        writer.add_scalar('Convergence/R2', r2, epoch)
        writer.add_scalar('Convergence/Slope', slope, epoch)
    else:
        plt.show()  # O plt.savefig(f'check_epoch_{epoch}.png')
        plt.close()

    model.train()  # Volver a modo entrenamiento


### Lógica de nomenclatura y asignación de directorios a los archivos de modelos entrenados .pt según sea K-Fold y entrenamiento estándar

def get_run_filenames(cfg, run_hash: str) -> tuple:
    """
    Genera nombres de archivo descriptivos basados en la configuración del experimento.
    Recupera la lógica original de construcción de strings largos.

    Retorna:
        (model_filename, eval_filename)
    """
    # Construcción del nombre base
    model_name = cfg.model.get('model_name', 'Model')
    base_name = f"{model_name}_{run_hash}"

    model_filename = f"{base_name}.pt"
    eval_filename = f"{base_name}_eval.pt"

    return model_filename, eval_filename


def inject_prescaling(
        cfg,
        model_params: dict,
        all_flows_np: Any,
        observed_flow_mask_np: Any,
        od_vector_np: Any,
        observed_od_mask_np: Any,
        network_params: dict = None,
    ) -> tuple[float, float]:
    """
    Compute and inject link and OD scale values into model_params.

    The function supports three scaling strategies configured through:

        training.scaling_strategy

    Supported strategies
    --------------------
    standard
        Use the mean of observed flow and OD values.

    capacity
        Use the mean link capacity as the link and OD scale, when capacity is
        available.

    max
        Use the maximum observed flow and OD value.

    The computed values are injected in-place into:

        model_params["link_scale"]
        model_params["od_scale"]

    Parameters
    ----------
    cfg : Any
        Hydra configuration or dictionary-like configuration.

    model_params : dict
        Model parameter dictionary that will be updated in-place.

    all_flows_np : Any
        Link-flow target vector.

    observed_flow_mask_np : Any
        Binary observed-flow mask.

    od_vector_np : Any
        OD target vector.

    observed_od_mask_np : Any
        Binary observed-OD mask.

    network_params : dict, optional
        Optional network parameters. If present, capacity may be used by the
        capacity scaling strategy.

    Returns
    -------
    tuple[float, float]
        Computed link scale and OD scale.
    """

    def _cfg_get_any(
        cfg: Any,
        dotted_key: str,
        default: Any = None,
    ) -> Any:
        """
        Safely read a nested value from DictConfig, dict or object attributes.

        Parameters
        ----------
        cfg : Any
            Configuration object.

        dotted_key : str
            Dotted key path, for example "training.scaling_strategy".

        default : Any, default=None
            Value returned when the path does not exist.

        Returns
        -------
        Any
            Retrieved value or default.
        """

        current = cfg

        for part in dotted_key.split("."):
            if current is None:
                return default

            if isinstance(current, DictConfig):
                if part not in current:
                    return default
                current = current[part]
                continue

            if isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
                continue

            if hasattr(current, part):
                current = getattr(current, part)
                continue

            return default

        return current


    def _dict_get_any(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
        """
        Safely read one key from a dict-like object, DictConfig or plain object.
        """

        if obj is None:
            return default

        if isinstance(obj, DictConfig):
            if key not in obj:
                return default
            return obj[key]

        if isinstance(obj, dict):
            return obj.get(key, default)

        if hasattr(obj, key):
            return getattr(obj, key)

        return default


    def _as_numpy_or_none(value: Any) -> np.ndarray | None:
        """
        Convert tensors or array-like objects to NumPy arrays.

        Parameters
        ----------
        value : Any
            Input object.

        Returns
        -------
        Optional[np.ndarray]
            NumPy array or None.
        """

        if value is None:
            return None

        if torch.is_tensor(value):
            return value.detach().cpu().numpy()

        return np.asarray(value)


    def _masked_values(
        values: np.ndarray | None,
        mask: np.ndarray | None,
    ) -> np.ndarray:
        """
        Extract finite values selected by a binary mask.

        Parameters
        ----------
        values : Optional[np.ndarray]
            Value vector.

        mask : Optional[np.ndarray]
            Binary mask.

        Returns
        -------
        np.ndarray
            Masked finite values. Empty when inputs are missing or invalid.
        """

        if values is None or mask is None:
            return np.array([], dtype=np.float64)

        values = np.asarray(values, dtype=np.float64).reshape(-1)
        mask = np.asarray(mask, dtype=np.float32).reshape(-1)

        if values.size == 0 or mask.size == 0:
            return np.array([], dtype=np.float64)

        if values.shape[0] != mask.shape[0]:
            logging.warning(
                "Cannot compute masked values because values and mask have different lengths: "
                "%d vs %d.",
                values.shape[0],
                mask.shape[0],
            )
            return np.array([], dtype=np.float64)

        selected = values[mask > 0.5]
        selected = selected[np.isfinite(selected)]

        return selected

    strategy = _cfg_get_any(
        cfg,
        "training.scaling_strategy",
        default="standard",
    )

    strategy = str(strategy or "standard").lower()

    flows = _as_numpy_or_none(all_flows_np)
    flow_mask = _as_numpy_or_none(observed_flow_mask_np)
    od = _as_numpy_or_none(od_vector_np)
    od_mask = _as_numpy_or_none(observed_od_mask_np)

    capacity = None

    if network_params is not None:
        capacity = _as_numpy_or_none(network_params.get("capacity"))

    feature_prescaling_cfg = _cfg_get_any(
        cfg,
        "model.feature_prescaling",
        default={},
    )

    default_link_scale = _dict_get_any(
        feature_prescaling_cfg,
        "link_scale",
        default=1.0,
    )

    default_od_scale = _dict_get_any(
        feature_prescaling_cfg,
        "od_scale",
        default=1.0,
    )

    default_link_scale = float(default_link_scale or 1.0)
    default_od_scale = float(default_od_scale or 1.0)

    flows_obs = _masked_values(
        values=flows,
        mask=flow_mask,
    )

    od_obs = _masked_values(
        values=od,
        mask=od_mask,
    )

    if strategy == "standard":
        link_scale_val = float(flows_obs.mean()) if flows_obs.size else default_link_scale
        od_scale_val = float(od_obs.mean()) if od_obs.size else default_od_scale

    elif strategy == "capacity":
        if capacity is not None and capacity.size > 0:
            link_scale_val = float(np.nanmean(capacity))
            od_scale_val = link_scale_val
        else:
            logging.warning(
                "Capacity scaling requested, but capacity was not found. "
                "Falling back to configured/default scale values."
            )
            link_scale_val = default_link_scale
            od_scale_val = default_od_scale

    elif strategy == "max":
        link_scale_val = float(flows_obs.max()) if flows_obs.size else default_link_scale
        od_scale_val = float(od_obs.max()) if od_obs.size else default_od_scale

    else:
        raise ConfigurationContractError(
            f"Unknown training.scaling_strategy='{strategy}'. "
            "Supported values: standard|capacity|max."
        )

    link_scale_val = float(max(link_scale_val, 1.0e-3))
    od_scale_val = float(max(od_scale_val, 1.0e-3))

    model_params["link_scale"] = link_scale_val
    model_params["od_scale"] = od_scale_val

    logging.info(
        "Feature prescaling injected | strategy=%s | link_scale=%.6f | od_scale=%.6f",
        strategy,
        link_scale_val,
        od_scale_val,
    )

    return link_scale_val, od_scale_val

def inject_prescaling_old(
    cfg,
    model_params: dict,
    all_flows_np: Any,
    observed_flow_mask_np: Any,
    od_vector_np: Any,
    observed_od_mask_np: Any,
    network_params: dict = None,
) -> tuple:
    """
    Compute and inject link_scale and od_scale into model_params according to the
    selected scaling strategy in cfg.training.scaling_strategy.

    Strategies supported:
      - 'standard' : scale by mean of observed values (adimensionalization)
      - 'capacity' : anchor to network capacity mean (saturation-based)
      - 'max'      : scale by dataset maximum (normalization to [0,1])

    The function is defensive: accepts numpy arrays or torch tensors and will
    fall back to sensible defaults when data is missing. The computed scalers are
    stored in-place into `model_params['link_scale']` and `model_params['od_scale']`.

    Returns:
        (link_scale_val: float, od_scale_val: float)
    """
    # Resolve strategy string
    strategy = None
    try:
        strategy = cfg.training.get('scaling_strategy', None)
    except Exception:
        # cfg might be a plain dict
        logging.warning("Could not access cfg.training.scaling_strategy directly, trying fallback.")
        strategy = (cfg.get('training') or {}).get('scaling_strategy', None)

    if not strategy:
        strategy = 'standard'  # default behaviour

    # Helper: ensure numpy arrays
    def _as_np(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    flows = _as_np(all_flows_np)
    flow_mask = _as_np(observed_flow_mask_np)
    od = _as_np(od_vector_np)
    od_mask = _as_np(observed_od_mask_np)

    # Extract capacity if available in network_params
    capacity = None
    if network_params is not None:
        cap_val = network_params.get('capacity', None)
        if cap_val is not None:
            capacity = _as_np(cap_val)

    # Default scalers
    """default_link_scale = float(np.mean(capacity)) if (capacity is not None and capacity.size > 0) else 1.0
    trips_scaler_cfg = 1.0
    try:
        trips_scaler_cfg = float(cfg.training.get('trips_scaler', 1.0))
    except Exception:
        logging.warning("Could not access cfg.training.trips_scaler directly, trying fallback.")
        trips_scaler_cfg = float((cfg.get('training') or {}).get('trips_scaler', 1.0))

    default_od_scale = trips_scaler_cfg if trips_scaler_cfg != 0 else 1.0"""

    custom_link_scale = cfg.model.get('feature_prescaling', {}).get('link_scale', None)
    custom_od_scale = cfg.model.get('feature_prescaling', {}).get('od_scale', None)

    if custom_link_scale is None or custom_od_scale is None:
        default_link_scale = 1.0
        default_od_scale = 1.0

        custom_link_scale
        logging.warning("Feature prescaling values not found in model config. Using defaults of 1.0.")

    # Safe masked reductions
    def masked_vals(vals, mask):
        if vals is None or mask is None:
            return np.array([])
        m = mask.astype(bool)
        if vals.size == 0 or m.sum() == 0:
            return np.array([])
        return vals[m]

    flows_obs = masked_vals(flows, flow_mask)
    od_obs = masked_vals(od, od_mask)

    # Compute per-strategy
    if strategy == 'standard':
        link_scale_val = float(flows_obs.mean()) if flows_obs.size else default_link_scale
        od_scale_val = float(od_obs.mean()) if od_obs.size else default_od_scale

    elif strategy == 'capacity':
        # Anchor both to capacity mean if available
        if capacity is not None and capacity.size:
            link_scale_val = float(np.mean(capacity))
            od_scale_val = link_scale_val
        else:
            link_scale_val = default_link_scale
            od_scale_val = default_od_scale

    elif strategy == 'max':
        link_scale_val = float(flows_obs.max()) if flows_obs.size else default_link_scale
        od_scale_val = float(od_obs.max()) if od_obs.size else default_od_scale

    else:
        raise ValueError(f"Unknown scaling strategy: {strategy}")

    # Safety clamps
    link_scale_val = float(max(link_scale_val, 1e-3))
    od_scale_val = float(max(od_scale_val, 1e-3))

    # Inject into model_params
    model_params['link_scale'] = link_scale_val
    model_params['od_scale'] = od_scale_val

    logging.info(f"inject_prescaling: strategy={strategy} -> link_scale={link_scale_val:.2f}, od_scale={od_scale_val:.2f}")

    return link_scale_val, od_scale_val


import scipy.sparse as sp
from scipy.sparse.csgraph import shortest_path

def compute_od_cost_prior(
    link_df,
    od_indices_tensor,
    graph_obj,
    device,
    node_id_to_idx: dict | None = None,
):
    """
    Compute a free-flow shortest-path cost prior for each OD pair.

    Alignment contract
    ------------------
    The OD indices passed in od_indices_tensor are already expressed in the
    model node-index space. Therefore, this function must build the sparse graph
    using the same node_id_to_idx mapping used by the model/artifact.

    It must not reconstruct a new node order from sorted(graph.nodes()), because
    that can silently create a different index space.
    """

    logging.info("Computing t0 OD cost prior.")

    if node_id_to_idx is None:
        raise ModelInputContractError(
            "node_id_to_idx is required to compute t0 OD cost prior safely. "
            "Do not reconstruct node indices from sorted(graph.nodes())."
        )

    node_id_to_idx = {
        int(node_id): int(idx)
        for node_id, idx in dict(node_id_to_idx).items()
    }

    if not node_id_to_idx:
        raise ModelInputContractError(
            "node_id_to_idx is empty. Cannot compute t0 OD cost prior."
        )

    num_nodes = len(node_id_to_idx)
    expected_indices = set(range(num_nodes))
    found_indices = set(node_id_to_idx.values())

    if found_indices != expected_indices:
        raise ModelInputContractError(
            "node_id_to_idx must contain contiguous zero-based indices. "
            f"missing_indices={sorted(expected_indices - found_indices)[:20]} | "
            f"extra_indices={sorted(found_indices - expected_indices)[:20]}"
        )

    required_endpoint_candidates = [
        ("init_node", "term_node"),
        ("from_node", "to_node"),
    ]

    endpoint_columns = None

    for candidate in required_endpoint_candidates:
        if candidate[0] in link_df.columns and candidate[1] in link_df.columns:
            endpoint_columns = candidate
            break

    if endpoint_columns is None:
        raise ModelInputContractError(
            "link_df must contain either ('init_node', 'term_node') or "
            "('from_node', 'to_node') columns."
        )

    u_col, v_col = endpoint_columns

    links = link_df.copy()

    links[u_col] = links[u_col].astype(int)
    links[v_col] = links[v_col].astype(int)

    unknown_nodes = sorted(
        (
            set(links[u_col].astype(int))
            .union(set(links[v_col].astype(int)))
        )
        - set(node_id_to_idx.keys())
    )

    if unknown_nodes:
        raise ModelInputContractError(
            "link_df contains nodes that are absent from node_id_to_idx. "
            f"unknown_nodes_sample={unknown_nodes[:20]} | "
            f"num_unknown_nodes={len(unknown_nodes)}"
        )

    u_mapped = links[u_col].map(node_id_to_idx).to_numpy(dtype=np.int64)
    v_mapped = links[v_col].map(node_id_to_idx).to_numpy(dtype=np.int64)

    cost_col = None

    for candidate in ["t0", "free_flow_time", "cost", "length"]:
        if candidate in links.columns:
            cost_col = candidate
            break

    if cost_col is None:
        logging.warning(
            "No cost column found in link_df. Falling back to unit weights."
        )
        weights = np.ones(len(links), dtype=np.float64)
    else:
        weights = (
            links[cost_col]
            .astype(float)
            .to_numpy(dtype=np.float64)
        )

    if not np.isfinite(weights).all():
        raise ModelInputContractError(
            f"Cost column '{cost_col}' contains NaN or infinite values."
        )

    if np.any(weights < 0):
        raise ModelInputContractError(
            f"Cost column '{cost_col}' contains negative values."
        )

    graph_csr = sp.csr_matrix(
        (weights, (u_mapped, v_mapped)),
        shape=(num_nodes, num_nodes),
    )

    if torch.is_tensor(od_indices_tensor):
        od_np = od_indices_tensor.detach().cpu().numpy()
    else:
        od_np = np.asarray(od_indices_tensor)

    od_np = od_np.astype(np.int64)

    if od_np.ndim != 2 or od_np.shape[1] != 2:
        raise ModelInputContractError(
            "od_indices_tensor must have shape [num_od_pairs, 2]. "
            f"Got shape {od_np.shape}."
        )

    if od_np.size == 0:
        return torch.empty(
            0,
            dtype=torch.float32,
            device=device,
        )

    if od_np.min() < 0 or od_np.max() >= num_nodes:
        raise ModelInputContractError(
            "od_indices_tensor contains node indices outside the model node-index space. "
            f"min={int(od_np.min())}, max={int(od_np.max())}, num_nodes={num_nodes}."
        )

    unique_origins = np.unique(od_np[:, 0])

    dist_matrix = shortest_path(
        csgraph=graph_csr,
        directed=True,
        indices=unique_origins,
    )

    origin_to_row = {
        int(origin): idx
        for idx, origin in enumerate(unique_origins)
    }

    od_costs = []

    for origin_idx, destination_idx in od_np:
        row_idx = origin_to_row[int(origin_idx)]
        cost = dist_matrix[row_idx, int(destination_idx)]
        od_costs.append(float(cost))

    od_costs_np = np.asarray(
        od_costs,
        dtype=np.float32,
    )

    finite_mask = np.isfinite(od_costs_np)

    if finite_mask.any():
        fallback_value = float(od_costs_np[finite_mask].mean() * 2.0)
    else:
        fallback_value = 1.0

    od_costs_np = np.nan_to_num(
        od_costs_np,
        nan=fallback_value,
        posinf=fallback_value,
        neginf=fallback_value,
    )

    t0_costs = torch.tensor(
        od_costs_np,
        dtype=torch.float32,
        device=device,
    )

    logging.info(
        "t0 OD cost prior computed | od_pairs=%d | mean_cost=%.6f | cost_column=%s",
        int(len(od_costs_np)),
        float(t0_costs.mean().detach().cpu().item()) if t0_costs.numel() > 0 else 0.0,
        str(cost_col),
    )

    return t0_costs
