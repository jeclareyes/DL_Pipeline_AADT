from __future__ import annotations

from typing import Any

import networkx as nx
import pandas as pd


DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS = {
    "volume",
    "flow",
    "flows",
    "aadt",
    "AADT",
    "adt",
    "ADT",
    "from_node",
    "to_node",
    "link_pos",
    "reverse",
    "reverse_link_id",
    "forward_row",
    "forward_value",
}

DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES = (
    "Volume_",
    "volume_",
    "Flow_",
    "flow_",
    "AADT_",
    "aadt_",
    "ADT_",
    "adt_",
)

DIRECTIONAL_LINK_COLUMNS = {
    "from_node",
    "to_node",
    "INODE",
    "JNODE",
    "init_node",
    "term_node",
    "ID",
    "link_id",
    "link_pos",
    "reverse_link_id",
    "reverse_link_pos",
    "_edge_position",
    "_synthetic_reverse_link",
    "_synthetic_reverse_source_link_id",
    "_synthetic_reverse_source_from_node",
    "_synthetic_reverse_source_to_node",
}


def values_are_equivalent(left: Any, right: Any, tolerance: float = 1e-9) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return str(left) == str(right)


def should_compare_reverse_link_attribute(
    column: str,
    excluded_columns: set[str] | None = None,
    excluded_prefixes: tuple[str, ...] | None = None,
) -> bool:
    column = str(column)
    if excluded_columns is None:
        excluded_columns = DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS
    if excluded_prefixes is None:
        excluded_prefixes = DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES
    if column in excluded_columns:
        return False
    if column.startswith(excluded_prefixes):
        return False
    return True


def validate_graph_topology_and_bidirectionality(
    graph: nx.DiGraph,
    link_data: pd.DataFrame,
    attribute_tolerance: float = 1e-9,
    max_samples: int = 20,
    validate_reverse_attribute_symmetry: bool = True,
    excluded_attribute_columns: set[str] | None = None,
    excluded_attribute_prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    warnings = []
    if excluded_attribute_columns is None:
        excluded_attribute_columns = DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS
    if excluded_attribute_prefixes is None:
        excluded_attribute_prefixes = DEFAULT_REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES

    report: dict[str, Any] = {
        "passed": True,
        "num_nodes": int(graph.number_of_nodes()),
        "num_edges": int(graph.number_of_edges()),
        "is_weakly_connected": None,
        "is_strongly_connected": None,
        "num_weak_components": None,
        "num_strong_components": None,
        "weak_component_sizes": [],
        "strong_component_sizes": [],
        "isolated_nodes": [],
        "missing_reverse_links": [],
        "attribute_mismatches": [],
        "reverse_attribute_symmetry_enabled": bool(validate_reverse_attribute_symmetry),
        "excluded_attribute_columns": sorted(str(column) for column in excluded_attribute_columns),
        "excluded_attribute_prefixes": list(excluded_attribute_prefixes),
        "warnings": warnings,
    }

    if graph.number_of_nodes() == 0:
        warnings.append("Graph has no nodes.")
        report["passed"] = False
        return report

    isolated_nodes = [str(node) for node in nx.isolates(graph)]
    report["isolated_nodes"] = isolated_nodes[:max_samples]
    if isolated_nodes:
        warnings.append(f"Graph contains {len(isolated_nodes)} isolated nodes. Sample: {isolated_nodes[:max_samples]}")

    weak_components = list(nx.weakly_connected_components(graph))
    strong_components = list(nx.strongly_connected_components(graph))
    weak_component_sizes = sorted([len(component) for component in weak_components], reverse=True)
    strong_component_sizes = sorted([len(component) for component in strong_components], reverse=True)

    report["num_weak_components"] = int(len(weak_components))
    report["num_strong_components"] = int(len(strong_components))
    report["weak_component_sizes"] = weak_component_sizes[:max_samples]
    report["strong_component_sizes"] = strong_component_sizes[:max_samples]
    report["is_weakly_connected"] = len(weak_components) == 1
    report["is_strongly_connected"] = len(strong_components) == 1

    if not report["is_weakly_connected"]:
        warnings.append(
            "Graph is not weakly connected. "
            f"Number of weak components: {len(weak_components)}. Largest component sizes: {weak_component_sizes[:max_samples]}"
        )
    if not report["is_strongly_connected"]:
        warnings.append(
            "Graph is not strongly connected. "
            f"Number of strong components: {len(strong_components)}. Largest component sizes: {strong_component_sizes[:max_samples]}"
        )

    missing_reverse_links = []
    for u, v in graph.edges():
        if not graph.has_edge(str(v), str(u)):
            missing_reverse_links.append((str(u), str(v)))
    report["missing_reverse_links"] = missing_reverse_links[:max_samples]

    if missing_reverse_links:
        warnings.append(
            f"Graph contains {len(missing_reverse_links)} directed links without a reverse counterpart. Sample: {missing_reverse_links[:max_samples]}"
        )

    if not validate_reverse_attribute_symmetry:
        warnings.append("Reverse-link attribute symmetry validation is disabled by YAML.")
    else:
        required_pair_columns = {"from_node", "to_node"}
        if not required_pair_columns.issubset(link_data.columns):
            warnings.append(
                "Cannot validate reverse-link attribute symmetry because link_data does not contain both 'from_node' and 'to_node'."
            )
        else:
            pair_to_row = {}
            for row in link_data.itertuples(index=True):
                pair_to_row[(str(getattr(row, "from_node")), str(getattr(row, "to_node")))] = row.Index

            comparable_columns = [
                column
                for column in link_data.columns
                if should_compare_reverse_link_attribute(column=column, excluded_columns=excluded_attribute_columns, excluded_prefixes=excluded_attribute_prefixes)
            ]

            attribute_mismatches = []
            for (from_node, to_node), row_idx in pair_to_row.items():
                reverse_key = (to_node, from_node)
                if reverse_key not in pair_to_row:
                    continue
                if (to_node, from_node) < (from_node, to_node):
                    continue

                reverse_idx = pair_to_row[reverse_key]
                row = link_data.loc[row_idx]
                reverse_row = link_data.loc[reverse_idx]
                mismatched_columns = []
                for column in comparable_columns:
                    if not values_are_equivalent(row[column], reverse_row[column], tolerance=attribute_tolerance):
                        mismatched_columns.append({
                            "column": column,
                            "forward_value": row[column],
                            "reverse_value": reverse_row[column],
                        })

                if mismatched_columns:
                    attribute_mismatches.append({
                        "forward": (from_node, to_node),
                        "reverse": (to_node, from_node),
                        "forward_row": int(row_idx),
                        "reverse_row": int(reverse_idx),
                        "mismatched_columns": mismatched_columns[:max_samples],
                    })

            report["attribute_mismatches"] = attribute_mismatches[:max_samples]
            if attribute_mismatches:
                warnings.append(
                    f"Found {len(attribute_mismatches)} reverse-link pairs with non-identical attributes after excluding configured columns. Sample: {attribute_mismatches[:3]}"
                )

    report["passed"] = len(warnings) == 0
    return report


def validate_network_tntp_topology(
    network_tntp: pd.DataFrame,
    weight_col: str = "free_flow_time",
    validate_reverse_attribute_symmetry: bool = True,
    excluded_attribute_columns: set[str] | None = None,
    excluded_attribute_prefixes: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    try:
        from .tntp_builders import build_export_graph_from_network_tntp
    except (ImportError, ValueError):
        try:
            from src.data_pickle_extraction.tntp_builders import build_export_graph_from_network_tntp
        except ImportError:
            from tntp_builders import build_export_graph_from_network_tntp  # type: ignore

    export_graph = build_export_graph_from_network_tntp(network_tntp=network_tntp, weight_col=weight_col)
    link_data_like = network_tntp.rename(columns={"init_node": "from_node", "term_node": "to_node"})
    return validate_graph_topology_and_bidirectionality(
        graph=export_graph,
        link_data=link_data_like,
        validate_reverse_attribute_symmetry=validate_reverse_attribute_symmetry,
        excluded_attribute_columns=excluded_attribute_columns,
        excluded_attribute_prefixes=excluded_attribute_prefixes,
    )
