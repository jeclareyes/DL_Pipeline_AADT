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

def _inject_prescaling(
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
    default_link_scale = float(np.mean(capacity)) if (capacity is not None and capacity.size > 0) else 1.0
    trips_scaler_cfg = 1.0
    try:
        trips_scaler_cfg = float(cfg.training.get('trips_scaler', 1.0))
    except Exception:
        trips_scaler_cfg = float((cfg.get('training') or {}).get('trips_scaler', 1.0))

    default_od_scale = trips_scaler_cfg if trips_scaler_cfg != 0 else 1.0

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

def compute_od_cost_prior(link_df, od_indices_tensor, graph_obj, device):
    """
    Calcula el costo t0 para cada par OD mapeando los IDs originales del DF
    a índices consecutivos (0..N) para alinear con los tensores del modelo.
    """
    logging.info("Calculando Prior Gravitacional (Matriz de Costos t0)...")

    # 1. Crear Mapeo: ID Original -> Índice Consecutivo (0..N-1)
    # Es CRÍTICO que este orden coincida con el que usó el Adapter.
    # El estándar es usar los nodos ordenados del grafo.
    sorted_nodes = sorted(list(graph_obj.nodes()))
    node_to_idx = {node: i for i, node in enumerate(sorted_nodes)}
    num_nodes = len(sorted_nodes)

    # 2. Convertir columnas del DataFrame a índices mapeados
    # Si falla aquí es porque hay nodos en el DF que no están en el grafo
    try:
        u_mapped = link_df['from_node'].map(node_to_idx).values
        v_mapped = link_df['to_node'].map(node_to_idx).values
    except Exception as e:
        logging.error("Error mapeando nodos. Verifique que link_df y graph sean consistentes.")
        raise e

    # Validar que no quedaron NaNs (nodos no encontrados)
    if np.isnan(u_mapped).any() or np.isnan(v_mapped).any():
        raise ValueError("link_df contiene nodos que no existen en el objeto graph.")

    u_mapped = u_mapped.astype(int)
    v_mapped = v_mapped.astype(int)

    # 3. Construir Grafo Esparso con índices corregidos
    cost_col = 't0' if 't0' in link_df.columns else 'free_flow_time'
    weights = link_df[cost_col].values if cost_col in link_df.columns else np.ones(len(u_mapped))

    # Ahora sí: shape=(1235, 1235) y los índices van de 0 a 1234
    graph_csr = sp.csr_matrix((weights, (u_mapped, v_mapped)), shape=(num_nodes, num_nodes))

    # 4. Calcular Shortest Paths
    # od_indices_tensor ya viene mapeado (0..N) desde el Adapter, así que lo usamos directo
    od_np = od_indices_tensor.cpu().numpy()
    unique_origins = np.unique(od_np[:, 0])

    # Dijkstra / Shortest Path
    dist_matrix = shortest_path(csgraph=graph_csr, directed=True, indices=unique_origins)

    # Mapear orígenes (índice 0..N) a fila de la matriz de resultados
    origin_to_row = {org: i for i, org in enumerate(unique_origins)}

    # 5. Extraer costos
    od_costs = []
    for i in range(len(od_np)):
        o, d = od_np[i]
        # Si el origen o destino están fuera de rango, algo anda mal con el Adapter
        if o >= num_nodes or d >= num_nodes:
            logging.warning(f"Índice OD ({o}, {d}) fuera de rango para num_nodes={num_nodes}")
            od_costs.append(1000.0)
            continue

        row_idx = origin_to_row[o]
        cost = dist_matrix[row_idx, d]
        od_costs.append(cost)

    t0_costs = torch.tensor(od_costs, dtype=torch.float32).to(device)

    # Limpieza final de infinitos
    # Usamos la media o un valor alto razonable para desconectados
    valid_mean = t0_costs[t0_costs < 1e9].mean()
    t0_costs = torch.nan_to_num(t0_costs, posinf=valid_mean * 2.0)

    logging.info(f"Prior calculado. Costo medio OD: {valid_mean:.2f}")
    return t0_costs