# python
# File: src/train/_pipeline_utils.py
import logging
import numpy as np
import torch
from typing import Any

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

def get_run_filenames(cfg) -> tuple:
    """
    Genera nombres de archivo descriptivos basados en la configuración del experimento.
    Recupera la lógica original de construcción de strings largos.

    Retorna:
        (model_filename, eval_filename)
    """
    # Extracción segura de parámetros para evitar KeyErrors
    epochs = cfg.training.get('epochs', 'Unknown')
    lr = cfg.training.get('lr', 'Unknown')

    # Manejo de VDF anidado (puede estar en network.cost_function o vdf.name)
    vdf_name = cfg.get('network', {}).get('cost_function',
                                          cfg.get('vdf', {}).get('name', 'UnknownVDF'))

    # Parámetros de Sampling
    sampling_cfg = cfg.get('sampling', {})
    flow_rate = sampling_cfg.get('flow_rate', 'All')
    strategy = sampling_cfg.get('strategy', 'Unknown')
    basis = sampling_cfg.get('sampling_basis', 'Unknown')

    # Construcción del nombre base
    base_name = (
        f"Epochs_{epochs}_"
        f"VDF_{vdf_name}_"
        f"Learning_Rate_{lr}_"
        f"Flow_Rate_{flow_rate}_"
        f"Strat_{strategy}_"
        f"Basis_{basis}"
    )

    model_filename = f"{base_name}.pt"
    eval_filename = f"eval_{base_name}.pt"

    return model_filename, eval_filename