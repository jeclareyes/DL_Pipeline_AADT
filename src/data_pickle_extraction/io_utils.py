from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

try:
    from .serialization_utils import make_json_serializable
except ImportError:
    from serialization_utils import make_json_serializable  # type: ignore


def as_tntp_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.suffix != ".tntp":
        path = path.with_suffix(".tntp")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe_as_tntp(df: pd.DataFrame, filepath: str | Path, sep: str = "\t") -> Path:
    path = as_tntp_path(filepath)
    df.to_csv(path, sep=sep, index=False)
    return path


def save_text_as_tntp(text: str, filepath: str | Path, encoding: str = "utf-8") -> Path:
    path = as_tntp_path(filepath)
    path.write_text(text, encoding=encoding)
    return path


def save_json_report(payload: dict[str, Any], filepath: str | Path, encoding: str = "utf-8") -> Path:
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable_payload = make_json_serializable(payload)
    path.write_text(json.dumps(serializable_payload, indent=2, ensure_ascii=False), encoding=encoding)
    return path


def ensure_output_directories(output_root: str | Path, info_dir: str | Path) -> None:
    Path(output_root).mkdir(parents=True, exist_ok=True)
    Path(info_dir).mkdir(parents=True, exist_ok=True)


def load_graph_pickle(path: str | Path) -> nx.DiGraph:
    graph_path = Path(path)
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph pickle file not found: {graph_path}")
    with graph_path.open("rb") as file:
        graph = pickle.load(file)
    if not isinstance(graph, nx.DiGraph):
        raise TypeError(f"The graph pickle must contain a networkx.DiGraph. Received: {type(graph).__name__}")
    return graph


def load_link_data(path: str | Path) -> pd.DataFrame:
    link_data_path = Path(path)
    if not link_data_path.exists():
        raise FileNotFoundError(f"Link data parquet file not found: {link_data_path}")
    return pd.read_parquet(link_data_path)


def load_od_matrix_npz(path: str | Path) -> dict[str, np.ndarray]:
    od_path = Path(path)
    if not od_path.exists():
        raise FileNotFoundError(f"OD matrix npz file not found: {od_path}")
    with np.load(od_path) as loaded:
        return {key: loaded[key] for key in loaded.files}


def build_nodes_table_from_graph(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for node_id, attrs in graph.nodes(data=True):
        row = {"ID": node_id}
        if isinstance(attrs, dict):
            row.update(attrs)
        rows.append(row)
    nodes_df = pd.DataFrame(rows)
    if "x" in nodes_df.columns and "X" not in nodes_df.columns:
        nodes_df["X"] = nodes_df["x"]
    if "y" in nodes_df.columns and "Y" not in nodes_df.columns:
        nodes_df["Y"] = nodes_df["y"]
    return nodes_df


def build_links_table_from_graph(graph: nx.DiGraph) -> pd.DataFrame:
    rows = []
    for edge_idx, (u, v, attrs) in enumerate(graph.edges(data=True), start=1):
        row = {
            "ID": attrs.get("ID", f"{u}-{v}") if isinstance(attrs, dict) else f"{u}-{v}",
            "INODE": u,
            "JNODE": v,
            "_edge_position": edge_idx,
        }
        if isinstance(attrs, dict):
            row.update(attrs)
        rows.append(row)
    return pd.DataFrame(rows)


def load_reconstruction_inputs(graph_path: str | Path, link_data_path: str | Path, od_matrix_path: str | Path) -> dict[str, Any]:
    graph = load_graph_pickle(graph_path)
    link_data = load_link_data(link_data_path)
    od_matrix = load_od_matrix_npz(od_matrix_path)
    data = {
        "_source_object_type": "multi_file_reconstruction",
        "graph": graph,
        "link_data": link_data,
        "od_matrix": od_matrix,
        "nodes_gdf": build_nodes_table_from_graph(graph),
        "links_gdf": build_links_table_from_graph(graph),
    }
    data["num_nodes"] = graph.number_of_nodes()
    data["num_links"] = graph.number_of_edges()
    return data
