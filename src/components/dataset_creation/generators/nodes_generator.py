from __future__ import annotations

from itertools import count
from typing import Any

import numpy as np
import pandas as pd

from src.components.dataset_creation.config import DatasetConfig
from src.components.dataset_creation.exporters.tntp_exporter import save_dataframe_as_tntp


def _build_node_class_mapping(config: DatasetConfig) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int]]:
    quantity_cfg = config.NodeParameters.Quantity
    quantity_by_type = {
        "TAZ": int(quantity_cfg.TAZ),
        "AUX": int(quantity_cfg.AUX),
        "INTERSECTION": int(quantity_cfg.INTERSECTION),
    }
    types_cfg = config.NodeParameters.Types
    class_to_types = {
        "Zones": [str(node_type) for node_type in types_cfg.Zones],
        "Non-Zones": [str(node_type) for node_type in types_cfg.NonZones],
    }

    type_to_class: dict[str, str] = {}
    for node_class, node_types in class_to_types.items():
        for node_type in node_types:
            if node_type in type_to_class:
                raise ValueError(
                    f"Node type '{node_type}' is assigned to more than one class in NodeParameters.Types."
                )
            type_to_class[node_type] = node_class

    configured_types = set(quantity_by_type)
    partitioned_types = set(type_to_class)
    missing_types = sorted(configured_types - partitioned_types)
    extra_types = sorted(partitioned_types - configured_types)

    if missing_types or extra_types:
        problems: list[str] = []
        if missing_types:
            problems.append(f"missing from Types: {missing_types}")
        if extra_types:
            problems.append(f"not declared in Quantity: {extra_types}")
        raise ValueError(
            "NodeParameters.Quantity and NodeParameters.Types must define the same node types. "
            + "; ".join(problems)
        )

    return type_to_class, class_to_types, quantity_by_type


def build_nodes(config: DatasetConfig, data, metadata) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(config.seed)
    tntp_columns = config.NodeParameters.Tabular.Columns
    type_to_class, nodes_raw, quantity_by_type = _build_node_class_mapping(config)
    possible_node_classes = list(nodes_raw.keys())
    possible_node_types = list(quantity_by_type.keys())
    coordinate_bounds = config.NodeParameters.Coordinates

    node_id_generator = count(1)

    def _create_node_ids(quantity: int) -> list[int]:
        return [next(node_id_generator) for _ in range(quantity)]

    rows: list[pd.DataFrame] = []
    for node_type, quantity in quantity_by_type.items():
        node_class = type_to_class[node_type]
        x_min = coordinate_bounds.x_min
        x_max = coordinate_bounds.x_max
        y_min = coordinate_bounds.y_min
        y_max = coordinate_bounds.y_max
        node_ids = _create_node_ids(quantity)
        df = pd.DataFrame(
            zip(
                node_ids,
                rng.uniform(x_min, x_max, quantity),
                rng.uniform(y_min, y_max, quantity),
                [node_type] * quantity,
                [node_class] * quantity,
            ),
            columns=tntp_columns,
        )
        rows.append(df)

    nodes_df = pd.concat(rows, ignore_index=True)
    zones_ids = nodes_df[nodes_df["class"] == "Zones"]["node_id"].astype(int).to_numpy()
    nonzones_ids = nodes_df[nodes_df["class"] == "Non-Zones"]["node_id"].astype(int).to_numpy()

    metadata = {
        "nodes_raw": nodes_raw,
        "possible_node_classes": possible_node_classes,
        "possible_node_types": possible_node_types,
        "node_class_by_type": type_to_class,
        "node_quantity_by_type": quantity_by_type,
        "tntp_columns": tntp_columns,
        "num_nodes": len(nodes_df),
        "zones_ids": zones_ids,
        "num_zones": len(nodes_df[nodes_df["class"] == "Zones"]),
        "nonzones_ids": nonzones_ids,
        "num_nonzones": len(nodes_df[nodes_df["class"] == "Non-Zones"]),
        "node_counts_by_class": nodes_df["class"].value_counts().to_dict(),
        "node_counts_by_type": nodes_df["type"].value_counts().to_dict(),
        "random_seed": int(config.seed),
    }

    nodes_path = save_dataframe_as_tntp(nodes_df, config.paths.export_filepaths.nodes)
    metadata["nodes_path"] = str(nodes_path)
    return nodes_df, metadata
