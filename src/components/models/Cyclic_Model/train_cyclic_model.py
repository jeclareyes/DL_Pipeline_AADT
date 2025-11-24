"""
Script de entrenamiento para Traffic Assignment con Datos Parciales.

Este script entrena el modelo CyclicODModel con:
- Muestreo de datos parciales (OD y flujos)
- Validación con múltiples métricas
- Comparación con Frank-Wolfe
- Exportación de resultados

Uso:
    python src/models/train_cyclic_model.py --od_rate 0.2 --flow_rate 0.3 --epochs 100
"""
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from typing import Dict
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Dynamic model import will be done later based on config
# from src.models.Cyclic_Model.cyclic_model import CyclicODModel, PartialDataLoss
from src.components.models.Cyclic_Model.cyclic_model_data_ingestion import LinkopingDataLoader
from src.components.models.traditional_TA.frank_wolfe_congestion import FrankWolfeAssignmentCongestion
import networkx as nx
from scipy import sparse


# =============================================================================
# DATASET
# =============================================================================

class TrafficDataset(Dataset):
    """Dataset para entrenamiento con datos parciales."""

    def __init__(self,
                 true_flows: np.ndarray,
                 true_od: np.ndarray,
                 flow_mask: np.ndarray,
                 od_mask: np.ndarray):
        """
        Args:
            true_flows: Flujos verdaderos [num_samples, num_links]
            true_od: Demandas OD verdaderas [num_samples, num_od_pairs]
            flow_mask: Máscaras de flujos [num_samples, num_links]
            od_mask: Máscaras de OD [num_samples, num_od_pairs]
        """
        self.true_flows = torch.FloatTensor(true_flows)
        self.true_od = torch.FloatTensor(true_od)
        self.flow_mask = torch.FloatTensor(flow_mask)
        self.od_mask = torch.FloatTensor(od_mask)

    def __len__(self):
        return len(self.true_flows)

    def __getitem__(self, idx):
        return {
            'true_flows': self.true_flows[idx],
            'true_od': self.true_od[idx],
            'flow_mask': self.flow_mask[idx],
            'od_mask': self.od_mask[idx]
        }


# =============================================================================
# MÉTRICAS
# =============================================================================

def compute_metrics(y_true: torch.Tensor, y_pred: torch.Tensor,
                   mask: torch.Tensor = None) -> Dict[str, float]:
    """
    Calcula métricas de evaluación.

    Args:
        y_true: Valores verdaderos
        y_pred: Valores predichos
        mask: Máscara opcional (1=evaluar, 0=ignorar)

    Returns:
        Dict con MSE, MAE, MAPE, R2
    """
    if mask is not None:
        y_true = y_true[mask.bool()]
        y_pred = y_pred[mask.bool()]

    # Convertir a numpy
    y_true_np = y_true.detach().cpu().numpy()
    y_pred_np = y_pred.detach().cpu().numpy()

    # MSE
    mse = np.mean((y_true_np - y_pred_np) ** 2)

    # MAE
    mae = np.mean(np.abs(y_true_np - y_pred_np))

    # MAPE (evitar división por cero)
    epsilon = 1e-8
    mape = np.mean(np.abs((y_true_np - y_pred_np) / (y_true_np + epsilon))) * 100

    # R2
    ss_res = np.sum((y_true_np - y_pred_np) ** 2)
    ss_tot = np.sum((y_true_np - np.mean(y_true_np)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))

    # Correlación
    correlation = np.corrcoef(y_true_np.flatten(), y_pred_np.flatten())[0, 1]

    return {
        'mse': float(mse),
        'mae': float(mae),
        'mape': float(mape),
        'r2': float(r2),
        'correlation': float(correlation)
    }


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer,
                device: str) -> Dict[str, float]:
    """Entrena una época."""
    model.train()

    total_loss = 0.0
    total_l_flow = 0.0
    total_l_od = 0.0
    total_l_reg = 0.0
    num_batches = 0

    for batch in dataloader:
        true_flows = batch['true_flows'].to(device)
        true_od = batch['true_od'].to(device)
        flow_mask = batch['flow_mask'].to(device)
        od_mask = batch['od_mask'].to(device)

        # Forward pass
        outputs = model(
            observed_flows=true_flows,
            flow_mask=flow_mask,
            true_od_demand=true_od,
            warmup=False
        )

        # Calcular pérdida
        loss_dict = loss_fn(
            predicted_flows=outputs['reconstructed_flows'],
            true_flows=true_flows,
            flow_mask=flow_mask,
            predicted_od=outputs['estimated_demand'],
            true_od=true_od,
            od_mask=od_mask,
            learned_alpha=outputs['learned_alpha'],
            learned_beta=outputs['learned_beta']
        )

        loss = loss_dict['total_loss']

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Acumular pérdidas
        total_loss += loss.item()
        total_l_flow += loss_dict['l_flow'].item()
        total_l_od += loss_dict['l_od'].item()
        total_l_reg += loss_dict['l_reg'].item()
        num_batches += 1

    return {
        'loss': total_loss / num_batches,
        'l_flow': total_l_flow / num_batches,
        'l_od': total_l_od / num_batches,
        'l_reg': total_l_reg / num_batches
    }


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   loss_fn: nn.Module,
                   device: str) -> Dict[str, float]:
    """Valida una época."""
    model.eval()

    total_loss = 0.0
    all_true_flows = []
    all_pred_flows = []
    all_flow_masks = []
    all_true_od = []
    all_pred_od = []
    all_od_masks = []
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            true_flows = batch['true_flows'].to(device)
            true_od = batch['true_od'].to(device)
            flow_mask = batch['flow_mask'].to(device)
            od_mask = batch['od_mask'].to(device)

            # Forward pass
            outputs = model(
                observed_flows=true_flows,
                flow_mask=flow_mask,
                true_od_demand=None,  # No usar ground truth en validación
                warmup=False
            )

            # Calcular pérdida
            loss_dict = loss_fn(
                predicted_flows=outputs['reconstructed_flows'],
                true_flows=true_flows,
                flow_mask=flow_mask,
                predicted_od=outputs['estimated_demand'],
                true_od=true_od,
                od_mask=od_mask,
                learned_alpha=outputs['learned_alpha'],
                learned_beta=outputs['learned_beta']
            )

            total_loss += loss_dict['total_loss'].item()
            num_batches += 1

            # Acumular para métricas
            all_true_flows.append(true_flows.cpu())
            all_pred_flows.append(outputs['reconstructed_flows'].cpu())
            all_flow_masks.append(flow_mask.cpu())
            all_true_od.append(true_od.cpu())
            all_pred_od.append(outputs['estimated_demand'].cpu())
            all_od_masks.append(od_mask.cpu())

    # Concatenar todos los batches
    all_true_flows = torch.cat(all_true_flows, dim=0)
    all_pred_flows = torch.cat(all_pred_flows, dim=0)
    all_flow_masks = torch.cat(all_flow_masks, dim=0)
    all_true_od = torch.cat(all_true_od, dim=0)
    all_pred_od = torch.cat(all_pred_od, dim=0)
    all_od_masks = torch.cat(all_od_masks, dim=0)

    # Calcular métricas en datos observados
    flow_metrics_observed = compute_metrics(all_true_flows, all_pred_flows, all_flow_masks)
    od_metrics_observed = compute_metrics(all_true_od, all_pred_od, all_od_masks)

    # Calcular métricas en datos NO observados (capacidad de generalización)
    flow_mask_unobserved = 1 - all_flow_masks
    od_mask_unobserved = 1 - all_od_masks

    flow_metrics_unobserved = compute_metrics(all_true_flows, all_pred_flows, flow_mask_unobserved)
    od_metrics_unobserved = compute_metrics(all_true_od, all_pred_od, od_mask_unobserved)

    return {
        'loss': total_loss / num_batches,
        'flow_metrics_observed': flow_metrics_observed,
        'flow_metrics_unobserved': flow_metrics_unobserved,
        'od_metrics_observed': od_metrics_observed,
        'od_metrics_unobserved': od_metrics_unobserved
    }


# =============================================================================
# COMPARACIÓN CON FRANK-WOLFE
# =============================================================================

def compare_with_frank_wolfe(model: nn.Module,
                             graph: nx.DiGraph,
                             od_matrix: sparse.spmatrix,
                             test_flows: torch.Tensor,
                             test_od: torch.Tensor,
                             flow_mask: torch.Tensor,
                             device: str,
                             output_path: str = 'outputs/tables/cyclic_vs_frankwolfe.csv'):
    """
    Compara resultados del modelo con Frank-Wolfe.

    Args:
        model: Modelo entrenado
        graph: Grafo de red
        od_matrix: Matriz OD
        test_flows: Flujos de test (ground truth observados)
        test_od: Demandas OD de test
        flow_mask: Máscara de flujos observados
        device: Device de cómputo
        output_path: Ruta para guardar resultados
    """
    print("\n" + "="*80)
    print("🔬 COMPARACIÓN: Cyclic Model vs Frank-Wolfe vs Ground Truth")
    print("="*80)

    # 1. Predicción del modelo
    print("\n📊 Generando predicciones del Cyclic Model...")
    model.eval()
    with torch.no_grad():
        # Agregar dimensión de batch si es necesario
        if test_flows.dim() == 1:
            test_flows_input = test_flows.unsqueeze(0)
            flow_mask_input = flow_mask.unsqueeze(0)
        else:
            test_flows_input = test_flows
            flow_mask_input = flow_mask

        outputs = model(
            observed_flows=test_flows_input.to(device),
            flow_mask=flow_mask_input.to(device),
            true_od_demand=None,
            warmup=False
        )
        cyclic_flows = outputs['reconstructed_flows'].cpu()

        # Quitar dimensión de batch si fue agregada
        if test_flows.dim() == 1:
            cyclic_flows = cyclic_flows.squeeze(0)

        cyclic_flows_np = cyclic_flows.numpy()

    # 2. Ejecutar Frank-Wolfe
    print("\n📊 Ejecutando Frank-Wolfe con congestión...")
    fw_assignment = FrankWolfeAssignmentCongestion(
        graph=graph,
        od_matrix=od_matrix,
        cost_function='bpr',
        solution_attr='solution_fw_comparison',
        use_cache=False,
        k_paths=10
    )

    fw_stats = fw_assignment.solve(
        max_iterations=100,
        convergence_threshold=0.01,
        verbose=False
    )

    # Extraer flujos de Frank-Wolfe (mismo orden que edge_list)
    fw_flows = np.array([
        graph[u][v].get('solution_fw_comparison', 0.0)
        for u, v in graph.edges()
    ])

    # 3. Ground truth (flujos observados del dataframe)
    true_flows_np = test_flows.numpy() if isinstance(test_flows, torch.Tensor) else test_flows
    flow_mask_np = flow_mask.numpy() if isinstance(flow_mask, torch.Tensor) else flow_mask

    # 4. Calcular métricas de comparación
    print("\n📈 Calculando métricas...")

    # Métricas del modelo vs ground truth
    cyclic_metrics = compute_metrics(
        torch.from_numpy(true_flows_np),
        torch.from_numpy(cyclic_flows_np),
        None  # Evaluar en todos los enlaces
    )

    # Métricas de Frank-Wolfe vs ground truth
    fw_metrics = compute_metrics(
        torch.from_numpy(true_flows_np),
        torch.from_numpy(fw_flows),
        None
    )

    # Correlación entre predicciones del modelo y Frank-Wolfe
    correlation_models = np.corrcoef(cyclic_flows_np.flatten(), fw_flows.flatten())[0, 1]

    print("\n📊 RESULTADOS:")
    print(f"\n   Cyclic Model vs Ground Truth:")
    print(f"      - MSE: {cyclic_metrics['mse']:.4f}")
    print(f"      - MAE: {cyclic_metrics['mae']:.4f}")
    print(f"      - MAPE: {cyclic_metrics['mape']:.2f}%")
    print(f"      - R²: {cyclic_metrics['r2']:.4f}")
    print(f"      - Correlación: {cyclic_metrics['correlation']:.4f}")

    print(f"\n   Frank-Wolfe vs Ground Truth:")
    print(f"      - MSE: {fw_metrics['mse']:.4f}")
    print(f"      - MAE: {fw_metrics['mae']:.4f}")
    print(f"      - MAPE: {fw_metrics['mape']:.2f}%")
    print(f"      - R²: {fw_metrics['r2']:.4f}")
    print(f"      - Correlación: {fw_metrics['correlation']:.4f}")

    print(f"\n   Correlación entre Cyclic y Frank-Wolfe: {correlation_models:.4f}")

    # 5. Crear tabla comparativa link-por-link
    print("\n💾 Creando tabla comparativa...")

    # Obtener atributos adicionales de los enlaces
    edge_list = list(graph.edges())
    edge_data = []

    for i, (u, v) in enumerate(edge_list):
        edge_info = graph[u][v]
        edge_data.append({
            'link_id': i,
            'from_node': u,
            'to_node': v,
            'capacity': edge_info.get('capacity', 0.0),
            'free_flow_time': edge_info.get('free_flow_time', 0.0),
            'length': edge_info.get('length', 0.0),
            'link_type': edge_info.get('link_type', 0),
            'true_flow': true_flows_np[i] if i < len(true_flows_np) else 0.0,
            'cyclic_flow': cyclic_flows_np[i] if i < len(cyclic_flows_np) else 0.0,
            'fw_flow': fw_flows[i] if i < len(fw_flows) else 0.0,
            'cyclic_error': abs(cyclic_flows_np[i] - true_flows_np[i]) if i < len(cyclic_flows_np) else 0.0,
            'fw_error': abs(fw_flows[i] - true_flows_np[i]) if i < len(fw_flows) else 0.0,
            'cyclic_rel_error': abs(cyclic_flows_np[i] - true_flows_np[i]) / (true_flows_np[i] + 1e-8) * 100 if i < len(cyclic_flows_np) else 0.0,
            'fw_rel_error': abs(fw_flows[i] - true_flows_np[i]) / (true_flows_np[i] + 1e-8) * 100 if i < len(fw_flows) else 0.0,
            'observed': int(flow_mask_np[i]) if i < len(flow_mask_np) else 0,
            'vc_ratio_true': true_flows_np[i] / (edge_info.get('capacity', 1000.0) + 1e-8) if i < len(true_flows_np) else 0.0,
            'vc_ratio_cyclic': cyclic_flows_np[i] / (edge_info.get('capacity', 1000.0) + 1e-8) if i < len(cyclic_flows_np) else 0.0,
            'vc_ratio_fw': fw_flows[i] / (edge_info.get('capacity', 1000.0) + 1e-8) if i < len(fw_flows) else 0.0
        })

    comparison_df = pd.DataFrame(edge_data)

    # Guardar
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_path, index=False)
    print(f"   ✓ Tabla guardada en: {output_path}")

    # Estadísticas adicionales
    print(f"\n📊 Estadísticas adicionales:")
    print(f"   Enlaces con flujo real > 0: {(comparison_df['true_flow'] > 0).sum()}")
    print(f"   Enlaces observados: {comparison_df['observed'].sum()}")
    print(f"   Enlaces no observados: {(1 - comparison_df['observed']).sum()}")

    # Métricas separadas por enlaces observados/no observados
    observed_mask = comparison_df['observed'] == 1
    unobserved_mask = comparison_df['observed'] == 0

    if observed_mask.any():
        print(f"\n   Métricas en enlaces OBSERVADOS:")
        print(f"      Cyclic MAE: {comparison_df[observed_mask]['cyclic_error'].mean():.2f}")
        print(f"      FW MAE: {comparison_df[observed_mask]['fw_error'].mean():.2f}")

    if unobserved_mask.any():
        print(f"\n   Métricas en enlaces NO OBSERVADOS:")
        print(f"      Cyclic MAE: {comparison_df[unobserved_mask]['cyclic_error'].mean():.2f}")
        print(f"      FW MAE: {comparison_df[unobserved_mask]['fw_error'].mean():.2f}")

    return {
        'cyclic_metrics': cyclic_metrics,
        'fw_metrics': fw_metrics,
        'correlation_models': correlation_models,
        'comparison_df': comparison_df
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train Cyclic Model for Linköping Traffic Assignment')
    parser.add_argument('--config', type=str, default='configs/linkoping.yaml',
                       help='Ruta al archivo de configuración YAML')
    parser.add_argument('--volume_year', type=int, default=None,
                       help='Año de volumen (sobrescribe config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Número de épocas (sobrescribe config)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device de cómputo (sobrescribe config)')
    parser.add_argument('--model', type=str, default=None,
                       help='Tipo de modelo (sobrescribe config)')

    args = parser.parse_args()

    # Cargar configuración
    # Resolver ruta de configuración de forma robusta
    resolved_cfg_path = _resolve_config_path(args.config)
    args.config = str(resolved_cfg_path)
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # -- Coerciones de tipos: asegurar que hiperparámetros numéricos sean del tipo correcto
    def _safe_float(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return default

    def _safe_int(x, default=0):
        try:
            return int(x)
        except Exception:
            return default

    # Training numeric fields
    if 'training' in config:
        cfg_tr = config['training']
        if 'learning_rate' in cfg_tr:
            cfg_tr['learning_rate'] = _safe_float(cfg_tr['learning_rate'], 1e-3)
        if 'weight_decay' in cfg_tr:
            cfg_tr['weight_decay'] = _safe_float(cfg_tr['weight_decay'], 0.0)
        if 'epochs' in cfg_tr:
            cfg_tr['epochs'] = _safe_int(cfg_tr['epochs'], 100)
        if 'batch_size' in cfg_tr:
            cfg_tr['batch_size'] = _safe_int(cfg_tr['batch_size'], 1)
        if 'val_frequency' in cfg_tr:
            cfg_tr['val_frequency'] = _safe_int(cfg_tr['val_frequency'], 1)
        if 'grad_clip_norm' in cfg_tr:
            cfg_tr['grad_clip_norm'] = _safe_float(cfg_tr.get('grad_clip_norm', 1.0), 1.0)
        # Scheduler params
        if 'scheduler_params' in cfg_tr and isinstance(cfg_tr['scheduler_params'], dict):
            sp = cfg_tr['scheduler_params']
            for k in ['factor', 'min_lr']:
                if k in sp:
                    sp[k] = _safe_float(sp[k], sp.get(k, 0.0))
            if 'patience' in sp:
                sp['patience'] = _safe_int(sp['patience'], sp.get('patience', 10))

    # Model loss weights
    if 'model' in config and isinstance(config['model'], dict):
        lw = config['model'].get('loss_weights', {})
        for k, v in list(lw.items()):
            lw[k] = _safe_float(v, v if isinstance(v, (int, float)) else 0.0)
        config['model']['loss_weights'] = lw

    # Seleccionar tipo de modelo
    print(f"config['model']: {config['model']}")
    model_type = args.model if args.model else config['model'].get('type', 'cyclic_model')
    print(f"   Modelo seleccionado: {model_type}")
    if model_type == 'cyclic_model':
        from src.components.models.Cyclic_Model.cyclic_model import CyclicODModel, PartialDataLoss
        ModelClass = CyclicODModel
    elif model_type == 'cyclic_model_ultra':
        from src.components.models.Cyclic_Model_2.cyclic_model_ultra import CyclicODModelUltra, PartialDataLoss
        ModelClass = CyclicODModelUltra
    else:
        raise ValueError(f"Modelo desconocido: {model_type}")

    # --- Ensure training section has sensible defaults to avoid KeyError when keys are missing ---
    config.setdefault('training', {})
    training_defaults = {
        'epochs': 100,
        'batch_size': 1,
        'learning_rate': 1e-3,
        'weight_decay': 1e-5,
        'optimizer': 'adam',
        'scheduler': 'reduce_on_plateau',
        'scheduler_params': {'factor': 0.5, 'patience': 10, 'min_lr': 1e-6},
        'grad_clip_norm': 1.0,
        'early_stopping': {'enabled': True, 'patience': 20, 'min_delta': 0.001},
        'val_frequency': 1,
        'device': 'auto'
    }
    for k, v in training_defaults.items():
        config['training'].setdefault(k, v)

    # Sobrescribir con argumentos de línea de comandos
    if args.volume_year is not None:
        config['data']['volume_year'] = args.volume_year
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.device is not None:
        config['training']['device'] = args.device
    else:
        # Use .get to be robust if device key was missing
        if config['training'].get('device', 'auto') == 'auto':
            config['training']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Semilla
    seed = config['data']['random_seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    print("="*80)
    print("🚀 ENTRENAMIENTO: Cyclic Model para Linköping Traffic Assignment")
    print("="*80)
    print(f"   Red: {config['data']['network_name']}")
    print(f"   Año de volumen: {config['data']['volume_year']}")
    print(f"   OD rate: {config['sampling']['od_rate']}")
    print(f"   Flow rate: {config['sampling']['flow_rate']}")
    print(f"   Train split: {config['data']['train_split']}")
    print(f"   Epochs: {config['training']['epochs']}")
    print(f"   K paths: {config['network']['k_paths']}")
    print(f"   Device: {config['training']['device']}")
    print(f"   Cost function: {config['network']['cost_function']}")

    # =============================================================================
    # 1. CARGAR DATOS CON LinkopingDataLoader
    # =============================================================================
    print("\n📁 Cargando datos de Linköping...")
    loader = LinkopingDataLoader(args.config)

    # Cargar todos los datos
    graph, od_matrix, link_data, routes_data = loader.load_all()

    # Preparar flujos con split train/test
    all_flows, train_flow_mask, test_flow_mask = loader.prepare_observed_flows()

    # Sincronizar máscaras con el grafo cargado (después de deduplicación)
    n_edges = graph.number_of_edges()
    if len(train_flow_mask) != n_edges:
        print(f"   ⚠️ Sincronizando máscaras: {len(train_flow_mask)} -> {n_edges}")
        train_flow_mask = train_flow_mask[:n_edges]
        test_flow_mask = test_flow_mask[:n_edges]
        all_flows = all_flows[:n_edges]

    # Preparar OD
    od_vector, od_mask = loader.prepare_od_demand_vector()

    # Preparar parámetros de red
    network_params = loader.prepare_network_parameters()

    # =============================================================================
    # 2. CREAR MÁSCARAS DE MUESTREO ADICIONAL
    # =============================================================================
    print(f"\n{'='*80}")
    print("📊 Creando máscaras de muestreo para datos parciales")
    print(f"{'='*80}")

    # Muestreo adicional sobre el conjunto de entrenamiento
    sampled_flow_mask, sampled_od_mask = loader.create_sampling_masks(
        train_flow_mask=train_flow_mask,
        od_mask=od_mask
    )

    # Estadísticas
    n_flows_sampled = sampled_flow_mask.sum()
    n_flows_train = train_flow_mask.sum()
    n_flows_test = test_flow_mask.sum()
    n_flows_total = len(all_flows)

    n_od_sampled = sampled_od_mask.sum()
    n_od_known = od_mask.sum()
    n_od_total = len(od_vector)

    print(f"   Flujos:")
    print(f"      ✓ Total de enlaces: {n_flows_total}")
    print(f"      ✓ Train disponibles: {int(n_flows_train)} ({n_flows_train/n_flows_total*100:.1f}%)")
    print(f"      ✓ Train muestreados: {int(n_flows_sampled)} ({n_flows_sampled/n_flows_train*100:.1f}% del train)")
    print(f"      ✓ Test: {int(n_flows_test)} ({n_flows_test/n_flows_total*100:.1f}%)")

    print(f"   OD:")
    print(f"      ✓ Total pares OD: {n_od_total}")
    print(f"      ✓ Conocidas: {int(n_od_known)} ({n_od_known/n_od_total*100:.1f}%)")
    print(f"      ✓ Muestreadas: {int(n_od_sampled)} ({n_od_sampled/n_od_known*100:.1f}% de conocidas)")

    # =============================================================================
    # 3. PREPARAR DATOS PARA MODELO
    # =============================================================================
    # Convertir a tensores
    true_flows = torch.FloatTensor(all_flows)
    true_od = torch.FloatTensor(od_vector)

    # Para entrenamiento: usar máscaras muestreadas
    train_flow_mask_tensor = torch.FloatTensor(sampled_flow_mask)
    train_od_mask_tensor = torch.FloatTensor(sampled_od_mask)

    # Para validación: usar máscaras de test
    test_flow_mask_tensor = torch.FloatTensor(test_flow_mask)
    test_od_mask_tensor = torch.FloatTensor(od_mask)  # Todas las OD conocidas

    print(f"\n   ✓ Flujos shape: {true_flows.shape}")
    print(f"   ✓ OD shape: {true_od.shape}")
    print(f"   ✓ Flujo total: {true_flows.sum():.2f}")
    print(f"   ✓ Demanda total conocida: {(true_od * test_od_mask_tensor).sum():.2f}")

    # =============================================================================
    # 4. CREAR DATASETS
    # =============================================================================
    # Agregar dimensión de batch
    true_flows_batch = true_flows.unsqueeze(0)
    true_od_batch = true_od.unsqueeze(0)
    train_flow_mask_batch = train_flow_mask_tensor.unsqueeze(0)
    train_od_mask_batch = train_od_mask_tensor.unsqueeze(0)
    test_flow_mask_batch = test_flow_mask_tensor.unsqueeze(0)
    test_od_mask_batch = test_od_mask_tensor.unsqueeze(0)

    # Dataset de entrenamiento
    train_dataset = TrafficDataset(
        true_flows_batch.numpy(),
        true_od_batch.numpy(),
        train_flow_mask_batch.numpy(),
        train_od_mask_batch.numpy()
    )

    # Dataset de validación
    val_dataset = TrafficDataset(
        true_flows_batch.numpy(),
        true_od_batch.numpy(),
        test_flow_mask_batch.numpy(),
        test_od_mask_batch.numpy()
    )

    train_dataloader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False)

    # =============================================================================
    # 5. CREAR MODELO
    # =============================================================================
    print(f"\n{'='*80}")
    print("⚙️ Inicializando modelo")
    print(f"{'='*80}")

    requested_device = config['training'].get('device', 'auto')
    # Resolve device
    if requested_device in (None, 'auto'):
        device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device_str = requested_device

    # If cuda was requested but not available or torch has no cuda support, fallback to cpu
    try:
        if 'cuda' in str(device_str) and not torch.cuda.is_available():
            print("\n⚠️ CUDA solicitado pero no disponible. Usando CPU en su lugar.")
            device_str = 'cpu'
    except Exception:
        # In case torch.cuda.* attributes are not present (torch built without cuda), fallback
        if 'cuda' in str(device_str):
            print("\n⚠️ CUDA solicitado pero torch no soporta CUDA. Usando CPU en su lugar.")
            device_str = 'cpu'

    device = torch.device(device_str)

    model = ModelClass(
         num_links=network_params['num_links'],
         num_od_pairs=network_params['num_od_pairs'],
         hidden_dim=config['model']['hidden_dim'],
         feature_dim=config['model']['feature_dim'],
         num_structures=config['model']['num_structures'],
         t0=network_params['t0'],
         capacity=network_params['capacity'],
         route_masks=network_params['route_masks'],
         od_pair_indices=network_params['od_pair_indices'],
         num_link_groups=network_params['num_link_groups'],
         link_group=network_params['link_group'],
         cost_function_type=config['network']['cost_function'],
         dropout=config['model']['dropout']
     ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"   ✓ Parámetros del modelo: {num_params:,}")
    print(f"   ✓ Device: {device}")

    # =============================================================================
    # 6. FUNCIÓN DE PÉRDIDA Y OPTIMIZADOR
    # =============================================================================
    loss_weights = config['model']['loss_weights']
    loss_fn = PartialDataLoss(
        w_flow=loss_weights['w_flow'],
        w_od=loss_weights['w_od'],
        w_reg=loss_weights['w_reg']
    )

    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler (opcional)
    scheduler = None
    if config['training']['scheduler'] == 'reduce_on_plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config['training']['scheduler_params']['factor'],
            patience=config['training']['scheduler_params']['patience'],
            min_lr=float(config['training']['scheduler_params']['min_lr'])
        )

    # =============================================================================
    # 7. ENTRENAMIENTO
    # =============================================================================
    print(f"\n{'='*80}")
    print(f"🎯 Iniciando entrenamiento por {config['training']['epochs']} épocas")
    print(f"{'='*80}")

    history = {
        'train_loss': [],
        'train_l_flow': [],
        'train_l_od': [],
        'val_loss': [],
        'val_metrics_observed': [],
        'val_metrics_unobserved': []
    }

    # Initialize optional results container to avoid referenced-before-assignment warnings
    comparison_results = None

    best_val_loss = float('inf')
    epochs_without_improvement = 0
    epoch = -1  # Inicializar para evitar error si no hay épocas

    for epoch in tqdm(range(config['training']['epochs']), desc="Training"):
        # Train
        train_metrics = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        history['train_loss'].append(train_metrics['loss'])
        history['train_l_flow'].append(train_metrics['l_flow'])
        history['train_l_od'].append(train_metrics['l_od'])

        # Validación
        val_frequency = config['training'].get('val_frequency', 1)
        if (epoch + 1) % val_frequency == 0 or epoch == config['training']['epochs'] - 1:
            val_metrics = validate_epoch(model, val_dataloader, loss_fn, device)

            history['val_loss'].append(val_metrics['loss'])
            history['val_metrics_observed'].append(val_metrics['flow_metrics_observed'])
            history['val_metrics_unobserved'].append(val_metrics['flow_metrics_unobserved'])

            # Scheduler step
            if scheduler is not None:
                scheduler.step(val_metrics['loss'])

            # Guardar mejor modelo
            if val_metrics['loss'] < best_val_loss:
                improvement = best_val_loss - val_metrics['loss']
                best_val_loss = val_metrics['loss']
                epochs_without_improvement = 0

                # Determinar directorio de salida (puede ser específico del run o genérico)
                output_dir = Path(config['outputs']['models_dir'])
                output_dir.mkdir(parents=True, exist_ok=True)
                model_path = output_dir / f'{model_type}_best.pt'

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'config': config,
                    'model_type': model_type,
                    'network_params': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                                     for k, v in network_params.items()}
                }, model_path)
                # Keep a concise notification when saving the best model
                # print(f"\n💾 Mejor modelo guardado (loss: {best_val_loss:.4f}, mejora: {improvement:.4f})")
            else:
                epochs_without_improvement += 1
                print(f"\n   ⏳ Sin mejora por {epochs_without_improvement} validaciones")

            # Early stopping
            if config['training']['early_stopping']['enabled']:
                patience = config['training']['early_stopping']['patience']
                if epochs_without_improvement >= patience:
                    print(f"\n   🛑 Early stopping activado (sin mejora por {patience} validaciones)")
                    break

        # Guardar checkpoint periódico
        if config['outputs']['save_checkpoints']:
            checkpoint_freq = config['outputs']['checkpoint_frequency']
            if (epoch + 1) % checkpoint_freq == 0:
                checkpoint_dir = Path(config['outputs'].get('checkpoint_dir',
                                     Path(config['outputs']['models_dir']) / 'checkpoints'))
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f'{model_type}_epoch_{epoch+1}.pt'
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'history': history,
                    'config': config,
                    'model_type': model_type
                }, checkpoint_path)
                print(f"\n   💾 Checkpoint guardado: {checkpoint_path.name}")

    # =============================================================================
    # 8. GUARDAR MODELO FINAL
    # =============================================================================
    if config['outputs']['save_final']:
        output_dir = Path(config['outputs']['models_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / f'{model_type}_final.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'history': history,
            'config': config,
            'model_type': model_type,
            'network_params': {k: v.cpu() if isinstance(v, torch.Tensor) else v
                             for k, v in network_params.items()}
        }, model_path)
        print(f"\n💾 Modelo final guardado en: {model_path}")

    # =============================================================================
    # 9. COMPARACIÓN CON FRANK-WOLFE (opcional)
    # =============================================================================
    if config['evaluation']['compare_frank_wolfe']:
        print(f"\n{'='*80}")
        print("🔬 Comparación con Frank-Wolfe")
        print(f"{'='*80}")

        # Cargar mejor modelo
        output_dir = Path(config['outputs']['models_dir'])
        best_model_path = output_dir / f'{model_type}_best.pt'

        if best_model_path.exists():
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"   ✓ Mejor modelo cargado (época {checkpoint['epoch']+1})")

        # Construir matriz OD sparse para Frank-Wolfe
        od_matrix_sparse = sparse.csr_matrix(
            (od_vector, (np.arange(len(od_vector)) // od_matrix.shape[1],
                         np.arange(len(od_vector)) % od_matrix.shape[1])),
            shape=od_matrix.shape
        )

        # Usar directorio de métricas si está disponible, sino tablas
        metrics_dir = Path(config['outputs'].get('metrics_dir',
                          Path(config['outputs']['base_dir']) / config['outputs'].get('tables_dir', 'tables')))
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # TODO esto por el momento no se va a ejecutar, pero se hará después.
        """
        comparison_results = compare_with_frank_wolfe(
            model=model,
            graph=graph,
            od_matrix=od_matrix_sparse,
            test_flows=true_flows,
            test_od=true_od,
            flow_mask=test_flow_mask_tensor,
            device=device,
            output_path=str(metrics_dir / f'{model_type}_vs_frankwolfe_{config["data"]["network_name"]}.csv')
        )"""

        print(f"\n   ✓ Tabla de comparación guardada en: {metrics_dir}")

    # =============================================================================
    # 10. RESUMEN FINAL
    # =============================================================================
    print("\n" + "="*80)
    print("✅ ENTRENAMIENTO COMPLETADO")
    print("="*80)
    print(f"\n📈 Resumen:")
    print(f"   - Modelo: {model_type}")
    print(f"   - Red: {config['data']['network_name']}")
    print(f"   - Año: {config['data']['volume_year']}")
    print(f"   - Épocas entrenadas: {epoch+1}/{config['training']['epochs']}")
    print(f"   - Mejor val loss: {best_val_loss:.4f}")

    if history['val_metrics_observed']:
        print(f"\n   📊 Métricas finales en TEST SET:")
        for k, v in history['val_metrics_observed'][-1].items():
            print(f"      {k}: {v:.4f}")

    if history['val_metrics_unobserved']:
        print(f"\n   🎯 Métricas finales en DATOS NO OBSERVADOS:")
        for k, v in history['val_metrics_unobserved'][-1].items():
            print(f"      {k}: {v:.4f}")

    if config['evaluation']['compare_frank_wolfe'] and comparison_results is not None:
        print(f"\n   🔬 Comparación con Frank-Wolfe:")
        print(f"      Correlación: {comparison_results['correlation_models']:.4f}")

    output_models_dir = Path(config['outputs']['models_dir'])
    print(f"\n💾 Modelos guardados en: {output_models_dir}")

    if 'metrics_dir' in config['outputs']:
        print(f"📊 Métricas guardadas en: {Path(config['outputs']['metrics_dir'])}")

    # Exportar resumen de métricas
    if config['evaluation']['export_metrics_summary']:
        metrics_dir = Path(config['outputs'].get('metrics_dir',
                          Path(config['outputs']['base_dir']) / config['outputs'].get('tables_dir', 'tables')))
        metrics_dir.mkdir(parents=True, exist_ok=True)
        summary_path = metrics_dir / 'training_summary.json'
        import json
        summary = {
            'model_type': model_type,
            'config': config,
            'best_val_loss': float(best_val_loss),
            'epochs_trained': epoch + 1,
            'final_metrics_observed': history['val_metrics_observed'][-1] if history['val_metrics_observed'] else {},
            'final_metrics_unobserved': history['val_metrics_unobserved'][-1] if history['val_metrics_unobserved'] else {},
            'data_summary': loader.get_summary()
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"📄 Resumen guardado en: {summary_path}")


def _resolve_config_path(config_path_str: str) -> Path:
    """Resuelve de forma robusta la ruta al archivo de configuración.

    Intentos (en orden):
      1. Path tal cual (absoluta o relativa al CWD)
      2. Relativa al project root
      3. project_root/configs/<basename>
      4. Buscar por glob dentro del project_root (primer match)

    Devuelve Path solucionado o lanza FileNotFoundError con rutas intentadas.
    """
    provided = Path(config_path_str)
    try_paths = []

    # 1) Direct
    try_paths.append(str(provided))
    if provided.exists():
        return provided

    # Determine project root (repositorio) - asumimos 'src' está dentro del repo
    project_root = Path(__file__).resolve().parents[3]

    # 2) project_root / provided
    candidate = project_root / config_path_str
    try_paths.append(str(candidate))
    if candidate.exists():
        return candidate

    # 3) project_root / configs / basename
    candidate2 = project_root / 'configs' / provided.name
    try_paths.append(str(candidate2))
    if candidate2.exists():
        return candidate2

    # 4) buscar por glob en project_root
    matches = list(project_root.glob(f"**/{provided.name}"))
    for m in matches[:20]:
        try_paths.append(str(m))
    if matches:
        return matches[0]

    # Ninguna ruta funcionó
    attempted = '\n  - '.join(try_paths)
    raise FileNotFoundError(
        f"No se encontró el archivo de configuración: '{config_path_str}'.\nSe intentaron las siguientes rutas:\n  - {attempted}"
    )


if __name__ == '__main__':
    main()
