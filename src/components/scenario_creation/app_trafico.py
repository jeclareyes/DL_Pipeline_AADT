import streamlit as st
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# --- CONFIGURACIÓN DE PÁGINA ---
st.set_page_config(page_title="Traffic Dashboard", layout="wide")

# --- CARGA DE DATOS ---
# Como es un script independiente, necesita cargar los datos.
# Puedes leerlos de un CSV o pasarlos mediante un archivo temporal.
@st.cache_data(ttl=5) # Recarga automáticamente si los archivos cambian cada 5 seg
def load_shared_data():
    dir = Path("TEMPORAL")
    links = pd.read_parquet(f"{dir}/links_df.parquet")
    nodes = pd.read_parquet(f"{dir}/node_df.parquet")
    routes = pd.read_parquet(f"{dir}/routes_df.parquet")
    with open(f"{dir}/geom_map.pkl", "rb") as f:
        geom = pickle.load(f)
    return links, nodes, routes, geom

try:
    links_df, node_df, routes_df, link_geom_map = load_shared_data()
except FileNotFoundError:
    st.error("No se encontraron archivos de datos. Ejecuta las celdas del Notebook primero.")
    st.stop()

# --- CONFIGURATION ---
WIDTH_SCALE = 3
OFFSET_UNIT = 0.15  # Distance between parallel lines

# --- HELPER FUNCTIONS ---

def get_offset_coords(x1, y1, x2, y2, offset_idx, total_layers):
    """Calculates perpendicular offset coordinates."""
    if total_layers <= 1: return x1, y1, x2, y2
    
    dx, dy = x2 - x1, y2 - y1
    length = np.sqrt(dx**2 + dy**2)
    if length == 0: return x1, y1, x2, y2
    
    # Unit normal vector
    nx, ny = -dy/length, dx/length
    
    # Centering logic: idx 0 starts from left, idx max at right
    # Shift factor centers the group of lines on the geometric center
    shift = offset_idx - (total_layers - 1) / 2
    
    ox, oy = nx * shift * OFFSET_UNIT, ny * shift * OFFSET_UNIT
    return x1 + ox, y1 + oy, x2 + ox, y2 + oy

def get_color(value, vmin, vmax, cmap_name='RdYlGn_r'):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    return mcolors.to_hex(cmap(norm(value)))


def format_label_value(v):
    """Format numbers to at most 2 decimals, keep ints/strings as-is."""
    if pd.isna(v):
        return ""
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        s = f"{v:.2f}".rstrip('0').rstrip('.')
        return s
    return str(v)


def point_along(x1, y1, x2, y2, frac=0.5):
    """Return point coordinates a fraction along the segment from (x1,y1) to (x2,y2)."""
    return x1 + (x2 - x1) * frac, y1 + (y2 - y1) * frac

# --- STREAMLIT APP ---

st.title("🚦 Multipurpose Traffic Dashboard")

# 1. SIDEBAR CONTROLS
st.sidebar.header("Visualization Settings")

view_mode = st.sidebar.radio(
    "Analysis Mode:",
    ["Global Network", "Path Analysis (OD Pair)"]
)

# Separate controls for node and link labels
show_node_labels = st.sidebar.checkbox("Show Node Labels", value=False)
node_label_feature = st.sidebar.selectbox(
    "Node label feature:", list(node_df.columns), index=0
)

show_link_labels = st.sidebar.checkbox("Show Link Labels", value=False)
link_label_feature = st.sidebar.selectbox(
    "Link label feature:", list(links_df.columns), index=0
)

# holder for optional table shown in Global Network
selected_table_df = None

# Initialize Figure
fig = go.Figure()

# --- LOGIC: GLOBAL NETWORK VIEW ---
if view_mode == "Global Network":
    st.sidebar.subheader("Global Attributes")
    attr_option = st.sidebar.selectbox(
        "Select Attribute:",
        ["assigned_flow", "vc_ratio", "free_flow_time", "congested_time", "speed"]
    )
    
    # Dynamic Color Scale
    if attr_option == 'vc_ratio':
        c_min, c_max, cmap = 0.0, 1.2, 'RdYlGn_r' # Red is bad (high VC)
    elif attr_option == 'speed':
        c_min, c_max, cmap = 0, 60, 'RdYlGn'      # Red is bad (low speed)
    else:
        c_min, c_max, cmap = 0, links_df[attr_option].max(), 'Blues'

    # Processing Links
    # We group by u,v pair to handle bi-directionality offsets
    processed_links = set()
    
    for idx, row in links_df.iterrows():
        lid = row['link_id']
        geom = link_geom_map[lid]
        u, v = geom['u'], geom['v']
        
        # Check if reverse link exists
        is_bidirectional = not links_df[
            (links_df['init_node'] == v) & (links_df['term_node'] == u)
        ].empty
        
        # Determine offset index (0 for forward, 1 for reverse if exists)
        # We need a consistent way to assign index. 
        # Simple rule: if we found u->v first, it's 0. v->u will be handled later?
        # Better: Just check both directions.
        
        # Simplification for visualization:
        # Always offset index 0 if uni-directional
        # If bi-directional, u->v is index 0, v->u is index 1? 
        # No, let's use the explicit helper logic:
        
        offset_idx = 0
        total_layers = 1
        if is_bidirectional:
            total_layers = 2
            # Convention: Smaller node ID is "inner" or distinct index?
            # Let's use the row index or simple logic:
            # Shift right relative to direction of travel
            # My helper shifts perpendicular.
            # Let's just force a shift of 0.5 for all links to the right
            pass 

        # Drawing: compute offset so parallel opposite-direction links separate visually
        if is_bidirectional:
            total_layers = 2
            try:
                offset_idx = 0 if int(u) < int(v) else 0
            except Exception:
                offset_idx = 0
        else:
            total_layers = 1
            offset_idx = 0

        x1o, y1o, x2o, y2o = get_offset_coords(
            geom['x1'], geom['y1'], geom['x2'], geom['y2'], 
            offset_idx=offset_idx, total_layers=total_layers
        )

        color = get_color(row[attr_option], c_min, c_max, cmap)
        width = 3

        fig.add_trace(go.Scatter(
            x=[x1o, x2o], y=[y1o, y2o],
            mode='lines',
            line=dict(width=width, color=color),
            hovertext=f"Link: {u}->{v}<br>{attr_option}: {format_label_value(row[attr_option])}",
            hoverinfo="text",
            showlegend=False
        ))


        # Label placement: compute fraction along the segment.
        frac = 0.25

        if show_link_labels:
            label_val = format_label_value(row.get(link_label_feature, ''))
            tx, ty = point_along(x1o, y1o, x2o, y2o, frac=frac)
            fig.add_trace(go.Scatter(
                x=[tx], y=[ty],
                mode='text',
                text=[label_val],
                textposition='middle center',
                showlegend=False,
                hovertext=f"{link_label_feature}: {label_val}<br>Link: {u}->{v}",
                hoverinfo='text'
            ))

        # Add small directional arrow in the middle (short segment)
        # arrow between frac*0.9 and frac*1.0 to indicate direction
        #ax, ay = point_along(x1o, y1o, x2o, y2o, frac=frac - 0.06)
        #bx, by = point_along(x1o, y1o, x2o, y2o, frac=frac + 0.06)
        #fig.add_annotation(x=bx, y=by, ax=ax, ay=ay, showarrow=True, arrowhead=2, arrowsize=2, arrowwidth=2, opacity=1.0)

    # Global Node View
    fig.add_trace(go.Scatter(
        x=node_df['x'], y=node_df['y'],
        mode='markers+text' if show_node_labels else 'markers',
        marker=dict(size=10, color='gray'),
        text=node_df[node_label_feature].apply(format_label_value).tolist() if show_node_labels else None,
        textposition="top center",
        hovertext=node_df.apply(lambda r: f"Node {int(r['node_id'])}<br>Gen: {format_label_value(r['total_gen'])}<br>Attr: {format_label_value(r['total_attr'])}", axis=1),
        hoverinfo="text",
        name="Nodes"
    ))

    # Table viewer for Global Network
    st.sidebar.subheader("Table Viewer")
    table_choice = st.sidebar.selectbox("Select table to display:", ["nodes", "links", "routes"])
    if table_choice == "nodes":
        selected_table_df = node_df
    elif table_choice == "links":
        selected_table_df = links_df
    else:
        selected_table_df = routes_df


# --- LOGIC: PATH ANALYSIS VIEW ---
elif view_mode == "Path Analysis (OD Pair)":
    st.sidebar.subheader("Route Selection")
    
    # OD Selector
    available_ods = sorted(routes_df['od_pair'].unique())
    selected_od = st.sidebar.selectbox("Select OD Pair:", available_ods)
    
    # Filter Data
    subset_routes = routes_df[routes_df['od_pair'] == selected_od]
    
    # 1. Draw Faint Background Network (Context)
    for lid, geom in link_geom_map.items():
        fig.add_trace(go.Scatter(
            x=[geom['x1'], geom['x2']], y=[geom['y1'], geom['y2']],
            mode='lines',
            line=dict(width=1, color='#e0e0e0'), # Light gray
            hoverinfo='none',
            showlegend=False
        ))
        
    # 2. Draw Selected Routes
    # We iterate by route index to apply offsets
    total_routes = len(subset_routes)
    
    # Color palette for routes (using Plotly qualitative colors)
    route_colors = ['#EF553B', '#636EFA', '#00CC96', '#AB63FA', '#FFA15A']
    
    for idx, row in subset_routes.iterrows():
        route_k = row['route_idx']
        path_links = row['path_links']
        flow = row['flow']
        
        # Dynamic color per route index
        r_color = route_colors[route_k % len(route_colors)]
        
        # Iterate links in this path
        for link_id in path_links:
            if link_id not in link_geom_map: continue
            geom = link_geom_map[link_id]
            
            # Apply offset based on route index k
            # This separates Route 1 from Route 2 on the same link
            x1, y1, x2, y2 = get_offset_coords(
                geom['x1'], geom['y1'], geom['x2'], geom['y2'], 
                offset_idx=route_k, 
                total_layers=total_routes
            )

            # prepare link label (from links_df) if requested
            link_text_val = None
            if show_link_labels:
                lr = links_df[links_df['link_id'] == link_id]
                if not lr.empty:
                    link_text_val = format_label_value(lr.iloc[0].get(link_label_feature, ''))

            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(width=4, color=r_color),
                hovertext=f"Route {route_k}<br>Flow: {format_label_value(flow)}<br>Link: {link_id}",
                hoverinfo="text",
                name=f"Route {route_k}",
                legendgroup=f"Route {route_k}",
                showlegend=True if link_id == path_links[0] else False
            ))

            # place link label as separate text point to control exact position
            if show_link_labels and link_text_val is not None:
                # determine fraction to avoid overlap for reverse links; use 25%/75%
                frac = 0.5
                if 'u' in geom and 'v' in geom:
                    try:
                        frac = 0.25 if int(geom['u']) < int(geom['v']) else 0.75
                    except Exception:
                        frac = 0.5

                tx, ty = point_along(x1, y1, x2, y2, frac=frac)
                fig.add_trace(go.Scatter(
                    x=[tx], y=[ty], mode='text', text=[link_text_val], textposition='middle center', showlegend=False,
                    hovertext=f"{link_label_feature}: {link_text_val}<br>Link: {link_id}", hoverinfo='text'
                ))

    # Node Visualization for specific OD
    # Get O and D nodes
    o_id, d_id = map(int, selected_od.split('->'))
    od_nodes = node_df[node_df['node_id'].isin([o_id, d_id])]
    
    fig.add_trace(go.Scatter(
        x=od_nodes['x'], y=od_nodes['y'],
        mode='markers+text',
        marker=dict(size=15, color='black', line=dict(width=2, color='white')),
        text=od_nodes.apply(lambda r: f"{'O' if r['node_id']==o_id else 'D'}: {int(r['node_id'])}", axis=1),
        textposition="top center",
        name="OD Nodes"
    ))

# --- LAYOUT UPDATE ---
fig.update_layout(
    height=700,
    margin=dict(l=0, r=0, t=30, b=0),
    template="plotly_white",
    xaxis=dict(showgrid=False, zeroline=False, visible=False),
    yaxis=dict(showgrid=False, zeroline=False, visible=False, scaleanchor="x", scaleratio=1),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

#st.plotly_chart(fig, width=='stretch')
st.plotly_chart(fig, use_container_width=True)

# Display Data Tables below chart
if view_mode == "Path Analysis (OD Pair)":
    st.subheader(f"Detailed Metrics for {selected_od}")
    st.dataframe(subset_routes[['route_idx', 'flow', 'travel_time', 'path_links']])

if view_mode == "Global Network" and selected_table_df is not None:
    st.subheader(f"Detailed Metrics: {table_choice}")
    st.dataframe(selected_table_df)