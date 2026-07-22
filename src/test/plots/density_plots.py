import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Optional

def plot_density_overlay(estimated: np.ndarray, target: np.ndarray, title: str, output_path: str, log_scale: bool = False, x_label: str = "Value") -> None:
    """Plots a KDE density overlay of estimated vs target values."""
    plt.figure(figsize=(10, 6))
    
    if log_scale:
        estimated = np.log1p(np.maximum(estimated, 0))
        target = np.log1p(np.maximum(target, 0))
        x_label = f"log1p({x_label})"
        
    sns.kdeplot(target, label="Target", fill=True, alpha=0.3, color="blue")
    sns.kdeplot(estimated, label="Estimated", fill=True, alpha=0.3, color="orange")
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_density_by_group(estimated: np.ndarray, target: np.ndarray, groups: np.ndarray, group_labels: Optional[Dict[Any, str]], output_dir: str, prefix: str, log_scale: bool = False, x_label: str = "Value") -> None:
    """Plots separate density overlays for each group."""
    unique_groups = np.unique(groups)
    
    for g in unique_groups:
        mask = groups == g
        if not np.any(mask):
            continue
            
        g_label = group_labels.get(g, str(g)) if group_labels else str(g)
        title = f"{prefix} Density: {g_label}"
        output_path = os.path.join(output_dir, f"{prefix}_density_{g_label}.png")
        
        plot_density_overlay(estimated[mask], target[mask], title, output_path, log_scale, x_label)
