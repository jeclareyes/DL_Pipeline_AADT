# src/test/_spatial_visualizer.py

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.collections as mc
from typing import Dict, Optional

# Constante reciclada de app_trafico
OFFSET_UNIT = 0.15 

def _get_offset_coords(x1, y1, x2, y2, offset_idx, total_layers):
    """Cálculo de desplazamiento perpendicular (Lógica pura geométrica)"""
    if total_layers <= 1: return x1, y1, x2, y2
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0: return x1, y1, x2, y2
    nx, ny = -dy/length, dx/length
    shift = offset_idx - (total_layers - 1) / 2
    ox, oy = nx * shift * OFFSET_UNIT, ny * shift * OFFSET_UNIT
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy

def render_static_spatial_audit(
    pred_flows: np.ndarray,
    true_flows: np.ndarray,
    capacity: np.ndarray,
    cap_mult: Optional[np.ndarray],
    link_geometries: Dict,
    output_dir: str,
    model_name: str
):
    """
    Motor de graficación pesado aislado del core de testing.
    """
    adj_capacity = capacity * cap_mult if cap_mult is not None else capacity
    vc_ratio = pred_flows / np.clip(adj_capacity, a_min=1e-9, a_max=None)
    flow_error = pred_flows - true_flows
    
    metrics = {
        "flow_error": (flow_error, 'RdBu_r', "Error de Asignación (Estimado - Real)", -500, 500),
        "vc_ratio": (vc_ratio, 'RdYlGn_r', "Estado de Congestión (V/C Ratio)", 0.0, 1.2)
    }
    if cap_mult is not None:
        metrics["cap_mult"] = (cap_mult, 'PiYG', "Física Aprendida (Multiplicador de Capacidad)", 0.5, 1.5)

    for key, (values, cmap_name, title, vmin, vmax) in metrics.items():
        fig, ax = plt.subplots(figsize=(15, 12))
        ax.set_facecolor('#f8f9fa')
        
        lines = []
        colors = []
        cmap = plt.get_cmap(cmap_name)
        norm = plt.Normalize(vmin=vmin, vmax=vmax)

        for i in range(len(values)):
            geom = link_geometries.get(i) or link_geometries.get(str(i))
            if not geom: continue
            
            x1o, y1o, x2o, y2o = _get_offset_coords(
                geom['x1'], geom['y1'], geom['x2'], geom['y2'], 
                offset_idx=0, total_layers=1 
            )
            lines.append([(x1o, y1o), (x2o, y2o)])
            colors.append(cmap(norm(values[i])))

        lc = mc.LineCollection(lines, colors=colors, linewidths=2.5)
        ax.add_collection(lc)
        ax.autoscale()
        ax.margins(0.1)
        
        plt.title(f"{title}\nModelo: {model_name}", fontsize=16, fontweight='bold')
        plt.axis('off')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
        
        out_path = os.path.join(output_dir, f"{model_name}_map_{key}.png")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()