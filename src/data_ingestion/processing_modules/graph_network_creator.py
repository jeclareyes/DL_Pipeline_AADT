"""
Construye un grafo NetworkX a partir de un DataFrame de enlaces de red.

Funciona con cualquier DataFrame que tenga columnas de nodos origen/destino.
Todos los atributos del DataFrame se guardan como atributos de aristas en el grafo.

Uso como módulo:
    from data_ingestion.processing_modules.graph_network_creator import build_graph_from_df, save_graph

    G = build_graph_from_df(unified_df, from_col='from_node', to_col='to_node')
    save_graph(G, 'data/processed/network_graph.gpickle')

Uso desde línea de comandos:
    python src/data_ingestion/graph_network_creator.py --path data/raw/Barcelona_net.tntp
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


def build_graph_from_df(df: pd.DataFrame,
                        from_col: str = 'init_node',
                        to_col: str = 'term_node',
                        exclude_cols: Optional[List[str]] = None) -> nx.DiGraph:
    G = nx.DiGraph()

    if exclude_cols is None:
        exclude_cols = []

    if '_merge' not in exclude_cols:
        exclude_cols.append('_merge')

    attr_cols = [col for col in df.columns
                 if col not in [from_col, to_col] + exclude_cols]

    print(f"   📊 Construyendo grafo desde DataFrame...")
    print(f"      - Columnas origen/destino: {from_col}, {to_col}")
    print(f"      - Atributos a guardar: {len(attr_cols)} columnas")
    print(f"      - Columnas excluidas: {exclude_cols}")

    for _, row in df.iterrows():
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
    from .network_loader import load_network_df
    df = load_network_df(path_obj)
    print(f"DataFrame cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    G = build_graph_from_df(df)
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
