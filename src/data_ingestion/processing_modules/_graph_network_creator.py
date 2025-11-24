"""
Construye un grafo NetworkX a partir de un DataFrame de enlaces de red.

Funciona con cualquier DataFrame que tenga columnas de nodos origen/destino.
Todos los atributos del DataFrame se guardan como atributos de aristas en el grafo.

Uso como módulo:
    from data_ingestion.processing_modules.graph_network_creator import build_graph_from_df, save_graph

    G = build_graph_from_df(unified_df, from_col='from_node', to_col='to_node')
    save_graph(G, 'data/processed/network_graph.gpickle')

Uso desde línea de comandos:
    python src/data_ingestion/_graph_network_creator.py --path data/raw/Barcelona_net.tntp
"""
from pathlib import Path
import argparse
import networkx as nx
import matplotlib.pyplot as plt
from typing import Optional, List, Union
import pickle
import pandas as pd

# Root del proyecto
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def build_graph(link_df, node_df):
    """Construye el grafo de NetworkX a partir del DataFrame unificado."""
    if link_df is None:
        raise RuntimeError("Debe cargar y fusionar la red y los flujos antes de construir el grafo.")

    # Ensure canonical join columns exist
    if 'from_node' not in link_df.columns or 'to_node' not in link_df.columns:
        raise RuntimeError("unified_df no contiene columnas 'from_node'/'to_node' para construir el grafo")

    link_df = link_df.copy()
    link_df['from_node'] = link_df['from_node'].astype(str)
    link_df['to_node'] = link_df['to_node'].astype(str)

    # Análisis de depuración para identificar discrepancias en el número de links
    print(f"Debug: DataFrame has {len(link_df)} rows.")
    null_from = link_df['from_node'].isnull().sum()
    null_to = link_df['to_node'].isnull().sum()
    print(f"Debug: Null 'from_node': {null_from}, Null 'to_node': {null_to}")
    self_loops = (link_df['from_node'] == link_df['to_node']).sum()
    print(f"Debug: Self-loops (from_node == to_node): {self_loops}")
    duplicates = link_df.duplicated(subset=['from_node', 'to_node']).sum()
    print(f"Debug: Duplicate edges (same from_node, to_node): {duplicates}")
    empty_from = (link_df['from_node'] == '').sum()
    empty_to = (link_df['to_node'] == '').sum()
    print(f"Debug: Empty 'from_node': {empty_from}, Empty 'to_node': {empty_to}")

    if duplicates > 0:
        print("ESTO AQUI YA NO PUEDE PASAR, PUES EL PROCESO DE DEDPLICACIÓN SE HA TRASLADO A OTRO SITIO")
        dup_df = link_df[link_df.duplicated(subset=['from_node', 'to_node'], keep=False)]
        print("Duplicate edges:")
        print(dup_df.to_string())

        # TODO: Implement better deduplication logic later
        # For now, prefer rows where link_type != 99
        if 'link_type' in link_df.columns:
            # Sort so that link_type != 99 comes first (assuming 99 is the highest value)
            link_df = link_df.sort_values(by='link_type', ascending=True)
            # Drop duplicates, keeping the first (which will be non-99 if available)
            link_df = link_df.drop_duplicates(subset=['from_node', 'to_node'], keep='first')
            print(f"After deduplication (preferring link_type != 99): {len(link_df)} rows.")
        else:
            print("Warning: 'link_type' column not found, skipping deduplication preference.")

    # Crear el grafo a partir del DataFrame
    graph = nx.from_pandas_edgelist(link_df, 'from_node', 'to_node', edge_attr=True, create_using=nx.DiGraph())

    # TODO this is new in case eliminate
    # Ensure numeric weight attribute 'free_flow_time' exists on edges
    sample_edge_info = []
    for u, v, data in list(graph.edges(data=True))[:5]:
        sample_edge_info.append((u, v, dict(data)))

    # Normalize common attributes to numeric when possible
    for u, v, data in graph.edges(data=True):
        # Try to coerce existing 'free_flow_time' to float
        if 'free_flow_time' in data:
            try:
                data['free_flow_time'] = float(data.get('free_flow_time', 1.0))
            except Exception:
                # fallback to 1.0
                data['free_flow_time'] = 1.0
        else:
            # Try computing from length and speed if available
            length = data.get('length') or data.get('dist') or data.get('distance')
            speed = data.get('speed')
            try:
                if length is not None and speed is not None:
                    # assume length in km and speed in km/h -> time in hours -> convert to seconds
                    lf = float(length)
                    sf = float(speed)
                    if sf > 0:
                        data['free_flow_time'] = lf / sf
                    else:
                        data['free_flow_time'] = 1.0
                else:
                    data['free_flow_time'] = 1.0
            except Exception:
                data['free_flow_time'] = 1.0

    print(f"Debug: Sample edge attributes before coercion: {sample_edge_info}")
    # ...rest of function continues

    # Agregar atributos de nodos si están disponibles
    if node_df is not None and not node_df.empty:
        # node loader may produce column 'node' or 'node_id'
        node_col = None
        for candidate in ['node', 'node_id', 'Node', 'NODE']:
            if candidate in node_df.columns:
                node_col = candidate
                break

        if node_col is None:
            # fallback: use first column
            node_col = list(node_df.columns)[0]

        for _, row in node_df.iterrows():
            nid = row[node_col]
            try:
                node_id = str(int(nid))
            except Exception:
                node_id = str(nid)

            if graph.has_node(node_id):
                graph.nodes[node_id]['x'] = row.get('x', None)
                graph.nodes[node_id]['y'] = row.get('y', None)
                graph.nodes[node_id]['pos'] = (row.get('x', None), row.get('y', None))
                graph.nodes[node_id]['type'] = row.get('type', None)
            else:
                # Node exists in node file but not in network - add it to graph
                graph.add_node(node_id)
                graph.nodes[node_id]['x'] = row.get('x', None)
                graph.nodes[node_id]['y'] = row.get('y', None)
                graph.nodes[node_id]['pos'] = (row.get('x', None), row.get('y', None))
                graph.nodes[node_id]['type'] = row.get('type', None)

    # Verificar que el número de links en el grafo coincida con el DataFrame
    num_links_df = len(link_df)
    num_links_graph = graph.number_of_edges()
    if num_links_df != num_links_graph:
        print(f"Warning: DataFrame has {num_links_df} links, but graph has {num_links_graph} edges.")
    else:
        print(f"Verification: Graph has {num_links_graph} edges, matching DataFrame.")

    return graph

def build_graph_from_df(link_df: pd.DataFrame,
                        from_col: str = 'init_node',
                        to_col: str = 'term_node',
                        exclude_cols: Optional[List[str]] = None) -> nx.DiGraph:
    G = nx.DiGraph()

    if exclude_cols is None:
        exclude_cols = []

    if '_merge' not in exclude_cols:
        exclude_cols.append('_merge')

    attr_cols = [col for col in link_df.columns
                 if col not in [from_col, to_col] + exclude_cols]

    print(f"   📊 Construyendo grafo desde DataFrame...")
    print(f"      - Columnas origen/destino: {from_col}, {to_col}")
    print(f"      - Atributos a guardar: {len(attr_cols)} columnas")
    print(f"      - Columnas excluidas: {exclude_cols}")

    for _, row in link_df.iterrows():
        u = int(row[from_col])
        v = int(row[to_col])

        attrs = {}
        for col in attr_cols:
            val = row[col]
            if pd.isna(val):
                attrs[col] = None
            elif hasattr(val, 'item'):
                try:
                    attrs[col] = val.item()
                except (ValueError, AttributeError):
                    attrs[col] = val
            else:
                attrs[col] = val

        G.add_edge(u, v, **attrs)

    print(f"      ✓ Grafo creado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    print(f"      ✓ Atributos por arista: {attr_cols}")

    return G


def save_graph(G: nx.DiGraph,
               output_path: Union[Path, str],
               format: str = 'pickle') -> Path:
    output_path = Path(output_path)

    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'pickle':
        import pickle as pkl
        with open(output_path, 'wb') as f:
            pkl.dump(G, f, protocol=pkl.HIGHEST_PROTOCOL)
    elif format == 'graphml':
        nx.write_graphml(G, output_path)
    elif format == 'gexf':
        nx.write_gexf(G, output_path)
    else:
        raise ValueError(f"Formato no soportado: {format}. Use 'pickle', 'graphml', o 'gexf'")

    print(f"      ✓ Grafo guardado: {output_path} (formato: {format})")

    return output_path


def load_graph(path: Union[Path, str], format: str = 'pickle') -> nx.DiGraph:
    path = Path(path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")

    if format == 'pickle':
        import pickle as pkl
        with open(path, 'rb') as f:
            G = pkl.load(f)
    elif format == 'graphml':
        G = nx.read_graphml(path)
    elif format == 'gexf':
        G = nx.read_gexf(path)
    else:
        raise ValueError(f"Formato no soportado: {format}")

    return G


def draw_and_save_graph(G: nx.Graph,
                        out_path: Union[Path, str],
                        dpi: int = 150,
                        show_labels: bool = False,
                        edge_attribute: Optional[str] = 'volume',
                        node_size_by_degree: bool = True) -> Path:
    out_path = Path(out_path)
    if not out_path.is_absolute():
        out_path = PROJECT_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = G.number_of_nodes()
    if n == 0:
        raise ValueError('El grafo no tiene nodos.')

    print(f"      📐 Calculando layout (esto puede tardar para grafos grandes)...")
    pos = nx.spring_layout(G, seed=42, iterations=50)

    if node_size_by_degree:
        degrees = dict(G.degree())
        node_sizes = [max(20, degrees.get(node, 0) * 20) for node in G.nodes()]
    else:
        node_sizes = 50

    if edge_attribute and edge_attribute in nx.get_edge_attributes(G, edge_attribute):
        edge_attrs = nx.get_edge_attributes(G, edge_attribute)
        edge_values = [edge_attrs.get((u, v), 0) for u, v in G.edges()]

        if edge_values:
            min_val = min(v for v in edge_values if v > 0) if any(v > 0 for v in edge_values) else 0
            max_val = max(edge_values)
            if max_val > min_val:
                widths = [0.5 + 4.5 * ((v - min_val) / (max_val - min_val)) if v > 0 else 0.1
                         for v in edge_values]
            else:
                widths = [1.0 for _ in edge_values]
        else:
            widths = 1.0
    else:
        widths = 1.0

    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10), dpi=dpi)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='tab:blue', alpha=0.7)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.5, edge_color='gray',
                          arrows=True, arrowsize=10, arrowstyle='->')

    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=6)

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_path, format='png', dpi=dpi, bbox_inches='tight')
    plt.close()

    print(f"      ✓ Imagen guardada: {out_path}")

    return out_path


def main(path: Optional[str], out: Optional[str], show_labels: bool = False):
    if path:
        path_obj = Path(path)
    else:
        path_obj = None

    print(f"Cargando DataFrame desde: {path_obj if path_obj else 'ruta por defecto'}")
    # load_network_df is colocated in processing_modules.network_loader; import relatively to avoid absolute package paths
    from ._network_loader import load_network_df
    link_df = load_network_df(path_obj)
    print(f"DataFrame cargado: {link_df.shape[0]} filas, {link_df.shape[1]} columnas")

    G = build_graph_from_df(link_df)
    print(f"Grafo creado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

    out_path = Path(out) if out else Path('outputs/figures/network_graph.png')
    print(f"Guardando imagen en: {out_path}")
    draw_and_save_graph(G, out_path, show_labels=show_labels)
    print("✓ Imagen guardada")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crear grafo NetworkX desde TNTP y exportar PNG')
    parser.add_argument('--path', type=str, default=None, help='Ruta al archivo .tntp (opcional)')
    parser.add_argument('--out', type=str, default=None, help='Ruta de salida para el PNG')
    parser.add_argument('--labels', action='store_true', help='Mostrar etiquetas de nodos en la imagen')
    args = parser.parse_args()

    main(args.path, args.out, show_labels=args.labels)
