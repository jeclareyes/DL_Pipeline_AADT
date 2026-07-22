import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Any, Optional

def calculate_regression_metrics(pred: np.ndarray, target: np.ndarray, label: str = "") -> Dict[str, Any]:
    """Calculates regression metrics (R2, MAE, RMSE, MAPE) between predictions and targets."""
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    
    if len(pred) == 0 or len(target) == 0:
        return {"subset": label, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n": 0}
        
    mae = mean_absolute_error(target, pred)
    mse = mean_squared_error(target, pred)
    rmse = np.sqrt(mse)
    
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - np.mean(target)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8)) if ss_tot > 0 else np.nan
    
    non_zero = target != 0
    mape = np.mean(np.abs((target[non_zero] - pred[non_zero]) / target[non_zero])) * 100 if np.any(non_zero) else np.nan
    
    return {
        "subset": label,
        "R2": float(r2),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape),
        "n": len(pred)
    }

def calculate_grouped_metrics(pred: np.ndarray, target: np.ndarray, groups: np.ndarray, group_labels: Optional[Dict[Any, str]] = None, label: str = "") -> pd.DataFrame:
    """Calculates metrics grouped by a specific key."""
    pred = np.asarray(pred).reshape(-1)
    target = np.asarray(target).reshape(-1)
    groups = np.asarray(groups).reshape(-1)
    
    unique_groups = np.unique(groups)
    results = []
    
    for g in unique_groups:
        mask = groups == g
        g_label = group_labels.get(g, str(g)) if group_labels else str(g)
        full_label = f"{label}_{g_label}" if label else g_label
        
        metrics = calculate_regression_metrics(pred[mask], target[mask], full_label)
        metrics["group_id"] = g
        metrics["group_name"] = g_label
        results.append(metrics)
        
    return pd.DataFrame(results)

def calculate_masked_metrics(pred: np.ndarray, target: np.ndarray, masks_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Calculates metrics for multiple named masks."""
    results = []
    
    for label, mask in masks_dict.items():
        m = np.asarray(mask).astype(bool).reshape(-1)
        if np.any(m):
            results.append(calculate_regression_metrics(pred[m], target[m], label))
        else:
            results.append({"subset": label, "R2": np.nan, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan, "n": 0})
            
    # Add full network
    results.append(calculate_regression_metrics(pred, target, "full_network"))
    
    return pd.DataFrame(results)
