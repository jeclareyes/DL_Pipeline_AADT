# src/data_handling/builders/graph_builder.py
from __future__ import annotations


"""
Graph Builder
=============

This module builds a directed NetworkX graph from the canonical link table and
the normalized node table.

Project context
---------------
In the AADT / traffic assignment pipeline, the graph is the structural
representation of the road network. It connects:

- nodes from nodes.tntp;
- directed links from the canonical link table;
- link attributes such as capacity, free-flow time, length, BPR parameters and
  link type;
- node attributes such as coordinates, node type and node class.

The graph is later used by:

- RouteModelAdapter to build route-link incidence tensors;
- validators to check route feasibility;
- visualization tools to draw the network;
- training and diagnostics to keep a stable link ordering.

This builder is intentionally limited to graph construction and graph indexing.
It does not read TNTP files, merge flow data, compute routes, create training
targets, or save artifacts.

Design principles
-----------------
- Build one directed edge per row in link_df.
- Preserve all useful link and node attributes.
- Keep edge ordering stable and explicit.
- Return indexing dictionaries for reproducibility.
- Fail early when required columns are missing.
"""

# TODO: Esto me dio un problema con el indexado y es que el graph.edges() no necesariamente 
# devuelve los edges en el mismo orden que las filas del link_df. Esto es un tema porque luego 
# asumo que el orden de los edges es el mismo que el orden de las filas del link_df para 
# construir los tensores de atributos y rutas. Para arreglar esto, necesito construir 
# explícitamente un mapeo entre los edges del graph y las filas del link_df, por ejemplo usando 
# un atributo 'link_id' que se preserve desde link_df a los atributos de los edges en el graph. 
# Luego puedo construir un diccionario link_id_to_idx que me diga qué fila del link_df corresponde 
# a cada edge del graph. Esto me asegura que puedo indexar correctamente los atributos y rutas aunque 
# el orden de los edges en graph.edges() sea diferente al orden de las filas en link_df.

from dataclasses import dataclass
import logging
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


Edge = Tuple[int, int]


@dataclass(frozen=True)
class GraphBuildResult:
    """
    Container returned by GraphBuilder.

    Attributes
    ----------
    graph : nx.DiGraph
        Directed graph built from link_df and node_df.

    edge_indexing : Dict[str, Any]
        Stable edge indexing payload, including edge_list, edge_to_idx,
        idx_to_edge, link_id_to_idx and idx_to_link_id.

    node_indexing : Dict[str, Any]
        Stable node indexing payload.

    metadata : Dict[str, Any]
        Metadata describing graph construction, including number of nodes,
        number of edges, missing node attributes and connectivity summaries.
    """

    graph: nx.DiGraph
    edge_indexing: Dict[str, Any]
    node_indexing: Dict[str, Any]
    metadata: Dict[str, Any]


class GraphBuilder:
    """
    Build a directed NetworkX graph from canonical link and node tables.

    Parameters
    ----------
    strict : bool, default=True
        If True, missing node attributes, duplicated directed links or links
        referencing unknown nodes raise errors.

    preserve_extra_attributes : bool, default=True
        If True, all columns in link_df and node_df are preserved as edge and
        node attributes.

    weight_column : str
        Edge attribute used as the route-search weight downstream.
        This builder only stores the attribute; it does not compute routes.
    """

    REQUIRED_LINK_COLUMNS = {
        "init_node",
        "term_node",
        "effective_capacity",
        "length",
        "free_flow_time",
    }

    REQUIRED_NODE_COLUMNS = {
        "node_id",
        "x",
        "y",
        "type",
    }

    DEFAULT_EDGE_ATTRIBUTE_COLUMNS = [
        "link_id",
        "reverse_link_id",
        "effective_capacity",
        "lanes",
        "length",
        "free_flow_time",
        "b",
        "power",
        "speed",
        "vdf",
        "toll",
        "link_type",
        "cost",
        "has_flow_record",
    ]

    DEFAULT_NODE_ATTRIBUTE_COLUMNS = [
        "x",
        "y",
        "type",
        "class",
    ]

    def __init__(
        self,
        weight_column: str,
        strict: bool = True,
        preserve_extra_attributes: bool = True,
        zone_node_ids: Sequence[int] | None = None,
    ) -> None:
        self.strict = bool(strict)
        self.preserve_extra_attributes = bool(preserve_extra_attributes)
        self.weight_column = str(weight_column)
        self.zone_node_ids = (
            {int(node_id) for node_id in zone_node_ids}
            if zone_node_ids is not None
            else None
        )

    def build(
        self,
        link_df: pd.DataFrame,
        node_df: pd.DataFrame,
    ) -> GraphBuildResult:
        """
        Build a directed graph and stable indexing payloads.

        Parameters
        ----------
        link_df : pd.DataFrame
            Canonical link table produced by link_table_builder.py.

        node_df : pd.DataFrame
            Normalized node table produced by tntp_node_reader.py.

        Returns
        -------
        GraphBuildResult
            Directed graph, edge indexing, node indexing and metadata.

        Raises
        ------
        ValueError
            If required columns are missing or severe consistency issues are
            detected.
        """

        logger.info("Building directed graph from link_df and node_df.")

        links = self._prepare_link_df(link_df)
        nodes = self._prepare_node_df(node_df)

        self._validate_node_references(
            link_df=links,
            node_df=nodes,
        )

        graph = self._create_graph(
            link_df=links,
            node_df=nodes,
        )

        edge_indexing = self._build_edge_indexing(graph=graph, link_df=links)
        node_indexing = self._build_node_indexing(graph)

        metadata = self._build_metadata(
            graph=graph,
            link_df=links,
            node_df=nodes,
            edge_indexing=edge_indexing,
            node_indexing=node_indexing,
        )

        logger.info(
            "Graph built successfully | nodes=%d | edges=%d",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        return GraphBuildResult(
            graph=graph,
            edge_indexing=edge_indexing,
            node_indexing=node_indexing,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Preparation
    # ------------------------------------------------------------------

    def _prepare_link_df(self, link_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the link table before graph construction.

        Parameters
        ----------
        link_df : pd.DataFrame
            Canonical link table.

        Returns
        -------
        pd.DataFrame
            Prepared link table.
        """

        self._require_columns(
            df=link_df,
            required_columns=self.REQUIRED_LINK_COLUMNS,
            df_name="link_df",
        )

        links = link_df.copy()

        links["init_node"] = pd.to_numeric(
            links["init_node"],
            errors="coerce",
        )

        links["term_node"] = pd.to_numeric(
            links["term_node"],
            errors="coerce",
        )

        before = len(links)

        links = links.dropna(
            subset=["init_node", "term_node"],
        ).copy()

        dropped = before - len(links)

        if dropped > 0:
            message = (
                f"Dropped {dropped} link rows because init_node or term_node "
                "could not be parsed."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        links["init_node"] = links["init_node"].astype(int)
        links["term_node"] = links["term_node"].astype(int)

        self._validate_unique_directed_edges(links)

        if self.weight_column not in links.columns:
            message = (
                f"weight_column='{self.weight_column}' was not found in link_df. "
                "Downstream route computations may fail if they rely on this weight."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        return links.reset_index(drop=True)

    def _prepare_node_df(self, node_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and normalize the node table before graph construction.

        Parameters
        ----------
        node_df : pd.DataFrame
            Normalized node table.

        Returns
        -------
        pd.DataFrame
            Prepared node table.
        """

        self._require_columns(
            df=node_df,
            required_columns=self.REQUIRED_NODE_COLUMNS,
            df_name="node_df",
        )

        nodes = node_df.copy()

        nodes["node_id"] = pd.to_numeric(
            nodes["node_id"],
            errors="coerce",
        )

        nodes["x"] = pd.to_numeric(
            nodes["x"],
            errors="coerce",
        )

        nodes["y"] = pd.to_numeric(
            nodes["y"],
            errors="coerce",
        )

        before = len(nodes)

        nodes = nodes.dropna(
            subset=["node_id", "x", "y"],
        ).copy()

        dropped = before - len(nodes)

        if dropped > 0:
            message = (
                f"Dropped {dropped} node rows because node_id, x or y "
                "could not be parsed."
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

        nodes["node_id"] = nodes["node_id"].astype(int)
        nodes["x"] = nodes["x"].astype(float)
        nodes["y"] = nodes["y"].astype(float)

        duplicated_node_ids = nodes["node_id"].duplicated(keep=False)

        if duplicated_node_ids.any():
            duplicated_ids = (
                nodes.loc[duplicated_node_ids, "node_id"]
                .astype(int)
                .unique()
                .tolist()
            )

            raise ValueError(
                "node_df contains duplicated node IDs: "
                f"{duplicated_ids[:20]}"
            )

        for column in ["type", "class"]:
            if column in nodes.columns:
                nodes[column] = nodes[column].astype(str).str.strip()

        return nodes.reset_index(drop=True)

    def _validate_node_references(
        self,
        link_df: pd.DataFrame,
        node_df: pd.DataFrame,
    ) -> None:
        """
        Validate that all link endpoints exist in the node table.

        Parameters
        ----------
        link_df : pd.DataFrame
            Prepared link table.

        node_df : pd.DataFrame
            Prepared node table.
        """

        link_nodes = set(link_df["init_node"].astype(int)).union(
            set(link_df["term_node"].astype(int))
        )

        known_nodes = set(node_df["node_id"].astype(int))

        missing_nodes = sorted(link_nodes - known_nodes)

        if missing_nodes:
            message = (
                "Some link endpoints are missing from node_df. "
                f"missing_nodes_sample={missing_nodes[:20]} | "
                f"num_missing_nodes={len(missing_nodes)}"
            )

            if self.strict:
                raise ValueError(message)

            logger.warning(message)

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _create_graph(
        self,
        link_df: pd.DataFrame,
        node_df: pd.DataFrame,
    ) -> nx.DiGraph:
        """
        Create a directed graph with node and edge attributes.

        Parameters
        ----------
        link_df : pd.DataFrame
            Prepared link table.

        node_df : pd.DataFrame
            Prepared node table.

        Returns
        -------
        nx.DiGraph
            Directed network graph.
        """

        graph = nx.DiGraph()

        self._add_nodes(
            graph=graph,
            node_df=node_df,
        )

        self._add_edges(
            graph=graph,
            link_df=link_df,
        )

        return graph

    def _add_nodes(
        self,
        graph: nx.DiGraph,
        node_df: pd.DataFrame,
    ) -> None:
        """
        Add nodes and node attributes to the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to populate.

        node_df : pd.DataFrame
            Prepared node table.
        """

        node_attribute_columns = self._select_node_attribute_columns(node_df)

        for row_dict in node_df.to_dict(orient="records"):
            node_id = int(row_dict["node_id"])

            attributes = {
                column: self._to_python_scalar(row_dict[column])
                for column in node_attribute_columns
                if column in row_dict
            }

            graph.add_node(node_id, **attributes)

        if "class" in node_df.columns:
            missing_class_nodes = [
                int(node)
                for node, data in graph.nodes(data=True)
                if "class" not in data
            ]

            if missing_class_nodes:
                raise ValueError(
                    "Node attribute 'class' was present in node_df but was not preserved "
                    "in the NetworkX graph. "
                    f"missing_class_nodes_sample={missing_class_nodes[:20]}"
                )

    def _add_edges(
        self,
        graph: nx.DiGraph,
        link_df: pd.DataFrame,
    ) -> None:
        """
        Add directed edges and edge attributes to the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Graph to populate.

        link_df : pd.DataFrame
            Prepared link table.
        """

        edge_attribute_columns = self._select_edge_attribute_columns(link_df)

        for row_dict in link_df.to_dict(orient="records"):

            init_node = int(row_dict["init_node"])
            term_node = int(row_dict["term_node"])

            attributes = {
                column: self._to_python_scalar(row_dict[column])
                for column in edge_attribute_columns
                if column in row_dict
            }

            # Store source and target also as attributes for easier debugging.
            # They remain redundant with the graph edge keys but are useful when
            # exporting or inspecting edge dictionaries.
            attributes["init_node"] = init_node
            attributes["term_node"] = term_node

            graph.add_edge(
                init_node,
                term_node,
                **attributes,
            )

    def _select_node_attribute_columns(
        self,
        node_df: pd.DataFrame,
    ) -> List[str]:
        """
        Select node attributes to store in the graph.

        Parameters
        ----------
        node_df : pd.DataFrame
            Prepared node table.

        Returns
        -------
        List[str]
            Node attribute columns.
        """

        canonical_present = [
            column
            for column in self.DEFAULT_NODE_ATTRIBUTE_COLUMNS
            if column in node_df.columns
        ]

        if not self.preserve_extra_attributes:
            return canonical_present

        extra_columns = [
            column
            for column in node_df.columns
            if column not in set(["node_id"] + canonical_present)
        ]

        return canonical_present + extra_columns

    def _select_edge_attribute_columns(
        self,
        link_df: pd.DataFrame,
    ) -> List[str]:
        """
        Select edge attributes to store in the graph.

        Parameters
        ----------
        link_df : pd.DataFrame
            Prepared link table.

        Returns
        -------
        List[str]
            Edge attribute columns.
        """

        canonical_present = [
            column
            for column in self.DEFAULT_EDGE_ATTRIBUTE_COLUMNS
            if column in link_df.columns
        ]

        if not self.preserve_extra_attributes:
            return canonical_present

        extra_columns = [
            column
            for column in link_df.columns
            if column not in set(["init_node", "term_node"] + canonical_present)
        ]

        return canonical_present + extra_columns

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _build_edge_indexing(
        self,
        graph: nx.DiGraph,
        link_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Build stable edge indexing dictionaries using link_df row order as the
        canonical model link order.

        The canonical order of links must not depend on graph.edges(), because
        NetworkX edge iteration order is an implementation detail of the graph
        object and should not be used as an implicit source of truth for tensors.

        The source of truth is the already validated canonical link table:
            link_df[["init_node", "term_node"]]

        This ensures that link-level arrays such as:
            - capacity
            - free_flow_time
            - observed flows
            - target vectors
            - route-link incidence rows

        can all be aligned against the same explicit edge order.
        """

        required_columns = {"init_node", "term_node"}
        missing_columns = required_columns - set(link_df.columns)

        if missing_columns:
            raise ValueError(
                "Cannot build edge indexing because link_df is missing columns: "
                f"{sorted(missing_columns)}"
            )

        # Canonical edge order: exactly the row order of link_df.
        canonical_edge_list: List[Edge] = [
            (int(row.init_node), int(row.term_node))
            for row in link_df.itertuples(index=False)
        ]

        # NetworkX graph edge order is kept only for diagnostics.
        graph_edge_list: List[Edge] = [
            (int(u), int(v))
            for u, v in graph.edges()
        ]

        canonical_edge_set = set(canonical_edge_list)
        graph_edge_set = set(graph_edge_list)

        if len(canonical_edge_list) != len(canonical_edge_set):
            duplicated_edges = self._find_duplicated_edges(canonical_edge_list)

            raise ValueError(
                "Canonical link order cannot be built because link_df contains "
                "duplicated directed edges. "
                f"Sample duplicated edges: {duplicated_edges[:20]}"
            )

        if canonical_edge_set != graph_edge_set:
            missing_in_graph = sorted(canonical_edge_set - graph_edge_set)
            extra_in_graph = sorted(graph_edge_set - canonical_edge_set)

            raise ValueError(
                "Graph edges do not match canonical link_df edges. "
                "This would break link-order alignment. "
                f"missing_in_graph_sample={missing_in_graph[:20]} | "
                f"extra_in_graph_sample={extra_in_graph[:20]} | "
                f"num_missing_in_graph={len(missing_in_graph)} | "
                f"num_extra_in_graph={len(extra_in_graph)}"
            )

        graph_order_matches_link_df_order = canonical_edge_list == graph_edge_list

        edge_to_idx = {
            edge: idx
            for idx, edge in enumerate(canonical_edge_list)
        }

        idx_to_edge = {
            idx: edge
            for edge, idx in edge_to_idx.items()
        }

        link_id_to_idx: Dict[int, int] = {}
        idx_to_link_id: Dict[int, int] = {}
        edge_to_link_id: Dict[Edge, int] = {}

        if "link_id" in link_df.columns:
            for idx, row in enumerate(link_df.itertuples(index=False)):
                row_dict = row._asdict()
                edge = (int(row_dict["init_node"]), int(row_dict["term_node"]))
                link_id_value = row_dict.get("link_id")

                if pd.notna(link_id_value):
                    link_id = int(link_id_value)
                    link_id_to_idx[link_id] = int(idx)
                    idx_to_link_id[int(idx)] = link_id
                    edge_to_link_id[edge] = link_id

        link_pair_indices = np.asarray(
            canonical_edge_list,
            dtype=np.int64,
        )

        graph_edge_order_pair_indices = np.asarray(
            graph_edge_list,
            dtype=np.int64,
        )

        return {
            "canonical_source": "link_df",
            "edge_list": canonical_edge_list,
            "edge_to_idx": edge_to_idx,
            "idx_to_edge": idx_to_edge,
            "link_id_to_idx": link_id_to_idx,
            "idx_to_link_id": idx_to_link_id,
            "edge_to_link_id": edge_to_link_id,
            "link_pair_indices": link_pair_indices,
            "graph_edge_order": graph_edge_list,
            "graph_edge_order_pair_indices": graph_edge_order_pair_indices,
            "graph_order_matches_link_df_order": bool(graph_order_matches_link_df_order),
        }

    def _build_node_indexing(
        self,
        graph: nx.DiGraph,
    ) -> Dict[str, Any]:
        """
        Build stable node indexing dictionaries.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        Returns
        -------
        Dict[str, Any]
            Node indexing payload.
        """

        node_list = [
            int(node)
            for node in graph.nodes()
        ]

        node_id_to_idx = {
            node_id: idx
            for idx, node_id in enumerate(node_list)
        }

        idx_to_node_id = {
            idx: node_id
            for node_id, idx in node_id_to_idx.items()
        }

        if self.zone_node_ids is None:
            zone_ids = [
                int(node)
                for node, data in graph.nodes(data=True)
                if self._is_zone_node(data)
            ]
        else:
            graph_node_ids = {int(node) for node in graph.nodes()}
            missing_zone_nodes = self.zone_node_ids.difference(graph_node_ids)
            if missing_zone_nodes:
                raise ValueError(
                    "Configured OD zone nodes are absent from the graph: "
                    f"{sorted(missing_zone_nodes)}"
                )
            zone_ids = [
                int(node)
                for node in graph.nodes()
                if int(node) in self.zone_node_ids
            ]

        non_zone_ids = [
            int(node)
            for node in node_list
            if node not in set(zone_ids)
        ]

        return {
            "node_list": node_list,
            "node_id_to_idx": node_id_to_idx,
            "idx_to_node_id": idx_to_node_id,
            "zone_ids": zone_ids,
            "non_zone_ids": non_zone_ids,
        }

    @staticmethod
    def _is_zone_node(node_attributes: Dict[str, Any]) -> bool:
        """
        Infer whether a node is a zone node.

        Parameters
        ----------
        node_attributes : Dict[str, Any]
            Node attributes.

        Returns
        -------
        bool
            True if the node appears to be a zone.
        """

        node_class = str(node_attributes.get("class", "")).lower()
        node_type = str(node_attributes.get("type", "")).lower()

        return (
            node_class in {"zone", "zones", "taz"}
            or node_type in {"zone", "zones", "taz"}
        )

    # ------------------------------------------------------------------
    # Validation and metadata
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        graph: nx.DiGraph,
        link_df: pd.DataFrame,
        node_df: pd.DataFrame,
        edge_indexing: Dict[str, Any],
        node_indexing: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build metadata describing graph construction.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        link_df : pd.DataFrame
            Prepared link table.

        node_df : pd.DataFrame
            Prepared node table.

        edge_indexing : Dict[str, Any]
            Edge indexing payload.

        node_indexing : Dict[str, Any]
            Node indexing payload.

        Returns
        -------
        Dict[str, Any]
            Graph metadata.
        """

        weak_components = list(nx.weakly_connected_components(graph))
        strongly_components = list(nx.strongly_connected_components(graph))

        degrees = dict(graph.degree())
        in_degrees = dict(graph.in_degree())
        out_degrees = dict(graph.out_degree())

        isolated_nodes = [
            int(node)
            for node, degree in degrees.items()
            if degree == 0
        ]

        metadata: Dict[str, Any] = {
            "num_nodes_from_node_df": int(len(node_df)),
            "num_links_from_link_df": int(len(link_df)),
            "num_graph_nodes": int(graph.number_of_nodes()),
            "num_graph_edges": int(graph.number_of_edges()),
            "num_zone_nodes": int(len(node_indexing["zone_ids"])),
            "num_non_zone_nodes": int(len(node_indexing["non_zone_ids"])),
            "num_weakly_connected_components": int(len(weak_components)),
            "num_strongly_connected_components": int(len(strongly_components)),
            "largest_weak_component_size": int(max((len(c) for c in weak_components), default=0)),
            "largest_strong_component_size": int(max((len(c) for c in strongly_components), default=0)),
            "num_isolated_nodes": int(len(isolated_nodes)),
            "isolated_nodes_sample": isolated_nodes[:20],
            "has_link_id_index": bool(len(edge_indexing["link_id_to_idx"]) > 0),
            "weight_column": self.weight_column,
        }

        if degrees:
            metadata.update(
                {
                    "min_degree": int(min(degrees.values())),
                    "max_degree": int(max(degrees.values())),
                    "mean_degree": float(np.mean(list(degrees.values()))),
                    "min_in_degree": int(min(in_degrees.values())),
                    "max_in_degree": int(max(in_degrees.values())),
                    "mean_in_degree": float(np.mean(list(in_degrees.values()))),
                    "min_out_degree": int(min(out_degrees.values())),
                    "max_out_degree": int(max(out_degrees.values())),
                    "mean_out_degree": float(np.mean(list(out_degrees.values()))),
                }
            )

        if "link_type" in link_df.columns:
            metadata["link_type_counts"] = (
                link_df["link_type"]
                .value_counts(dropna=False)
                .sort_index()
                .to_dict()
            )

        if "type" in node_df.columns:
            metadata["node_type_counts"] = (
                node_df["type"]
                .value_counts(dropna=False)
                .to_dict()
            )

        if "class" in node_df.columns:
            metadata["node_class_counts"] = (
                node_df["class"]
                .value_counts(dropna=False)
                .to_dict()
            )

        return metadata

    def _validate_unique_directed_edges(
        self,
        link_df: pd.DataFrame,
    ) -> None:
        """
        Validate that the link table has one row per directed edge.

        Parameters
        ----------
        link_df : pd.DataFrame
            Prepared link table.

        Raises
        ------
        ValueError
            If duplicated directed edges are found.
        """

        duplicated_mask = link_df.duplicated(
            subset=["init_node", "term_node"],
            keep=False,
        )

        if duplicated_mask.any():
            duplicated_edges = (
                link_df.loc[duplicated_mask, ["init_node", "term_node"]]
                .drop_duplicates()
                .apply(
                    lambda row: (
                        int(row["init_node"]),
                        int(row["term_node"]),
                    ),
                    axis=1,
                )
                .tolist()
            )

            raise ValueError(
                "link_df contains duplicated directed edges: "
                f"{duplicated_edges[:20]}"
            )

    @staticmethod
    def _require_columns(
        df: pd.DataFrame,
        required_columns: Iterable[str],
        df_name: str,
    ) -> None:
        """
        Ensure a DataFrame contains required columns.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate.

        required_columns : Iterable[str]
            Required column names.

        df_name : str
            Human-readable DataFrame name.

        Raises
        ------
        ValueError
            If required columns are missing.
        """

        missing = set(required_columns) - set(df.columns)

        if missing:
            raise ValueError(
                f"{df_name} is missing required columns: {sorted(missing)}"
            )

    @staticmethod
    def _to_python_scalar(value: Any) -> Any:
        """
        Convert pandas/numpy scalar values to plain Python values.

        This makes graph attributes easier to serialize and inspect.

        Parameters
        ----------
        value : Any
            Input value.

        Returns
        -------
        Any
            Python-native scalar or None for missing values.
        """

        if pd.isna(value):
            return None

        if isinstance(value, np.generic):
            return value.item()

        return value

    @staticmethod
    def _find_duplicated_edges(
        edge_list: List[Edge],
    ) -> List[Edge]:
        """
        Return duplicated directed edges from an edge list.

        This helper is used only for explicit error reporting. The graph builder
        should never silently accept duplicated directed links because they make
        positional link indexing ambiguous.
        """

        seen = set()
        duplicated = []

        for edge in edge_list:
            if edge in seen:
                duplicated.append(edge)
            else:
                seen.add(edge)

        return duplicated


def build_graph(
    link_df: pd.DataFrame,
    node_df: pd.DataFrame,
    weight_column: str,
    strict: bool = True,
    preserve_extra_attributes: bool = True,
    zone_node_ids: Sequence[int] | None = None,
) -> GraphBuildResult:
    """
    Convenience function to build a directed NetworkX graph.

    Parameters
    ----------
    link_df : pd.DataFrame
        Canonical link table produced by link_table_builder.py.

    node_df : pd.DataFrame
        Normalized node table produced by tntp_node_reader.py.

    strict : bool, default=True
        Whether to use strict validation behavior.

    preserve_extra_attributes : bool, default=True
        Whether to preserve extra DataFrame columns as graph attributes.

    weight_column : str
        Edge weight attribute used downstream.

    Returns
    -------
    GraphBuildResult
        Directed graph, indexing payloads and metadata.
    """

    builder = GraphBuilder(
        strict=strict,
        preserve_extra_attributes=preserve_extra_attributes,
        weight_column=weight_column,
        zone_node_ids=zone_node_ids,
    )

    return builder.build(
        link_df=link_df,
        node_df=node_df,
    )
