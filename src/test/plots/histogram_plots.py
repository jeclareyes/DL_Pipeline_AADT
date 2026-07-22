import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional

def plot_histogram_comparison(estimated: np.ndarray, target: np.ndarray, title: str, output_path: str, bins: int = 50, x_label: str = "Value") -> None:
    """Plots a histogram comparison of estimated vs target values."""
    plt.figure(figsize=(10, 6))
    
    min_val = min(np.min(estimated), np.min(target))
    max_val = max(np.max(estimated), np.max(target))
    bins_array = np.linspace(min_val, max_val, bins)
    
    plt.hist(target, bins=bins_array, alpha=0.5, label='Target', color='blue', density=True)
    plt.hist(estimated, bins=bins_array, alpha=0.5, label='Estimated', color='orange', density=True)
    
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_histogram_by_group(estimated: np.ndarray, target: np.ndarray, groups: np.ndarray, group_labels: Optional[Dict[Any, str]], output_dir: str, prefix: str, bins: int = 50, x_label: str = "Value") -> None:
    """Plots separate histogram comparisons for each group."""
    unique_groups = np.unique(groups)
    
    for g in unique_groups:
        mask = groups == g
        if not np.any(mask):
            continue
            
        g_label = group_labels.get(g, str(g)) if group_labels else str(g)
        title = f"{prefix} Histogram: {g_label}"
        output_path = os.path.join(output_dir, f"{prefix}_histogram_{g_label}.png")
        
        plot_histogram_comparison(estimated[mask], target[mask], title, output_path, bins, x_label)
