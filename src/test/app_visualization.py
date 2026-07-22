import streamlit as st
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import os
from glob import glob

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Traffic Dashboard - DL Pipeline", layout="wide")

# --- CONSTANTES RECICLADAS ---
OFFSET_UNIT = 0.15

def get_offset_coords(x1, y1, x2, y2, offset_idx, total_layers):
    """Cálculo de desplazamiento perpendicular para links bidireccionales."""
    if total_layers <= 1: return x1, y1, x2, y2
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0: return x1, y1, x2, y2
    nx, ny = -dy/length, dx/length
    shift = offset_idx - (total_layers - 1) / 2
    ox, oy = nx * shift * OFFSET_UNIT, ny * shift * OFFSET_UNIT
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy

# --- CARGA DE DATOS DESDE EL REGISTRY (.PT) ---
@st.cache_data
def get_available_runs(base_dir="outputs/models"):
    """Busca todas las carpetas de run_id que contengan un archivo eval."""
    if not os.path.exists(base_dir):
        return []
    
    runs = []
    for run_id in os.listdir(base_dir):
        run_path = os.path.join(base_dir, run_id)
        if os.path.isdir(run_path):
            eval_files = glob(os.path.join(run_path, "*_eval.pt"))
            if eval_files:
                runs.append((run_id, eval_files[0]))
    return runs

@st.cache_data
def load_eval_bundle(file_path):
    """Carga el archivo .pt eludiendo la restricción de PyTorch 2.6 para NumPy."""
    # weights_only=False es seguro aquí porque el archivo fue generado localmente
    bundle = torch.load(file_path, map_location='cpu', weights_only=False)
    
    # Extraer Datos Estáticos
    static = bundle.get('static_data', {})
    link_geometries = static.get('link_geometries', {})
    
    # Buscar la última época
    history = bundle.get('epochs_history', {})
    latest_epoch = sorted(history.keys(), key=lambda x: int(x))[-1]
    artifacts = history[latest_epoch].get('artifacts', {})
    
    # Procesar arrays para DataFrame
    def to_1d(t): return t.detach().cpu().numpy().flatten() if hasattr(t, 'detach') else np.array(t).flatten()
    
    pred_flows = to_1d(artifacts.get('pred_flows', []))
    true_flows = to_1d(static.get('true_flows', []))
    capacity = to_1d(static.get('capacity', []))
    cap_mult = artifacts.get('learned_capacity_multiplier')
    if cap_mult is not None:
        cap_mult = to_1d(cap_mult)
        adj_capacity = capacity * cap_mult
    else:
        adj_capacity = capacity
        
    adj_capacity = np.clip(adj_capacity, a_min=1e-9, a_max=None)
    
    # Construir DataFrame
    df_data = []
    for i in range(len(pred_flows)):
        geom = link_geometries.get(i) or link_geometries.get(str(i))
        if not geom: continue
        
        row = {
            'link_id': i,
            'u': geom.get('u'),
            'v': geom.get('v'),
            'x1': geom['x1'], 'y1': geom['y1'],
            'x2': geom['x2'], 'y2': geom['y2'],
            'estimated_flow': pred_flows[i],
            'true_flow': true_flows[i],
            'flow_error': pred_flows[i] - true_flows[i],
            'capacity': capacity[i],
            'adj_capacity': adj_capacity[i],
            'vc_ratio': pred_flows[i] / adj_capacity[i],
            'cap_mult': cap_mult[i] if cap_mult is not None else 1.0
        }
        df_data.append(row)
        
    return pd.DataFrame(df_data), latest_epoch

# --- INTERFAZ DE USUARIO (SIDEBAR) ---
st.sidebar.title("Configuración del Dashboard")

runs = get_available_runs()
if not runs:
    st.error("No se encontraron modelos evaluados en outputs/models/")
    st.stop()

# Selector de Modelo (Run ID)
run_dict = {run_id: path for run_id, path in runs}
selected_run = st.sidebar.selectbox("Seleccionar Run ID (Modelo)", list(run_dict.keys()))

# Cargar Datos
with st.spinner("Desempaquetando artefactos .pt..."):
    df, epoch = load_eval_bundle(run_dict[selected_run])

st.sidebar.success(f"Datos cargados (Época Final: {epoch})")

# Selector de Capa a Visualizar
metric_options = {
    "Error de Asignación (Est - Real)": ("flow_error", "RdBu_r", -500, 500, "Error (veh/h)"),
    "Congestión Física (V/C Ratio)": ("vc_ratio", "RdYlGn_r", 0.0, 1.2, "V/C Ratio"),
    "Flujo Estimado Absoluto": ("estimated_flow", "plasma", 0, df['estimated_flow'].max(), "Flujo (veh/h)")
}

if 'cap_mult' in df.columns and df['cap_mult'].nunique() > 1:
    metric_options["Física Aprendida (Multiplicador Capacidad)"] = ("cap_mult", "PiYG", 0.5, 1.5, "Multiplicador")

selected_metric = st.sidebar.selectbox("Métrica a Visualizar", list(metric_options.keys()))

col_metric, cmap_name, vmin, vmax, label_unit = metric_options[selected_metric]

# Filtros adicionales
st.sidebar.markdown("### Filtros")
show_nodes = st.sidebar.checkbox("Mostrar Centroides de Nodos", value=False)
error_threshold = st.sidebar.slider("Ocultar links con valor absoluto menor a:", 0.0, 500.0, 0.0)

# Aplicar Filtro
if selected_metric == "Error de Asignación (Est - Real)":
    df_plot = df[df[col_metric].abs() >= error_threshold]
else:
    df_plot = df

# --- RENDERIZADO DEL MAPA ---
st.title("Auditoría Espacial de Tráfico")
st.markdown(f"**Visualizando:** {selected_metric} | **Modelo:** `{selected_run}`")

fig = go.Figure()

cmap = plt.get_cmap(cmap_name)
norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

# Dibujar Links (Vectorizado en grupos de colores para rendimiento)
color_bins = {}
hover_texts = []
mid_x = []
mid_y = []

for idx, row in df_plot.iterrows():
    # Cálculo de offset de la App Original
    x1, y1, x2, y2 = get_offset_coords(row['x1'], row['y1'], row['x2'], row['y2'], 0, 1)
    
    val = row[col_metric]
    color = mcolors.to_hex(cmap(norm(val)))
    
    if color not in color_bins:
        color_bins[color] = {'x': [], 'y': []}
        
    color_bins[color]['x'].extend([x1, x2, None])
    color_bins[color]['y'].extend([y1, y2, None])
    
    # Datos para el Tooltip
    mid_x.append((x1 + x2) / 2)
    mid_y.append((y1 + y2) / 2)
    hover_texts.append(
        f"<b>Link:</b> {int(row['u'])} -> {int(row['v'])}<br>"
        f"<b>{label_unit}:</b> {val:.2f}<br>"
        f"<b>Flujo Real:</b> {row['true_flow']:.1f}<br>"
        f"<b>Flujo Est:</b> {row['estimated_flow']:.1f}<br>"
        f"<b>Capacidad Base:</b> {row['capacity']:.0f}"
    )

# Añadir líneas agrupadas
for color, coords in color_bins.items():
    fig.add_trace(go.Scatter(
        x=coords['x'], y=coords['y'],
        mode='lines',
        line=dict(width=3, color=color),
        hoverinfo='none',
        showlegend=False
    ))

# Añadir capa invisible para Tooltips (Interactivo)
fig.add_trace(go.Scatter(
    x=mid_x, y=mid_y,
    mode='markers',
    marker=dict(size=6, color='rgba(0,0,0,0)'),
    hovertext=hover_texts,
    hoverinfo='text',
    showlegend=False
))

# Capa opcional de Nodos
if show_nodes:
    unique_nodes = pd.concat([df_plot[['u', 'x1', 'y1']].rename(columns={'u':'id', 'x1':'x', 'y1':'y'}),
                              df_plot[['v', 'x2', 'y2']].rename(columns={'v':'id', 'x2':'x', 'y2':'y'})]).drop_duplicates(subset=['id'])
    fig.add_trace(go.Scatter(
        x=unique_nodes['x'], y=unique_nodes['y'],
        mode='markers',
        marker=dict(size=4, color='black'),
        hovertext=unique_nodes['id'].astype(int).astype(str),
        hoverinfo='text',
        name="Nodos"
    ))

# Ajustes de Layout
fig.update_layout(
    height=800,
    margin=dict(l=0, r=0, t=30, b=0),
    template="plotly_white",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
)

st.plotly_chart(fig, use_container_width=True)

# --- TABLA DE DATOS ---
with st.expander("Inspeccionar Datos Crudos"):
    st.dataframe(df_plot[['link_id', 'u', 'v', 'true_flow', 'estimated_flow', 'flow_error', 'vc_ratio', 'cap_mult']].style.format("{:.2f}"))