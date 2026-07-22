from __future__ import annotations

import logging
from itertools import count
from typing import Any

import numpy as np
import pandas as pd
import networkx as nx

from src.components.dataset_creation.config import DatasetConfig
from src.components.dataset_creation.exporters.tntp_exporter import save_dataframe_as_tntp


def build_network(config: DatasetConfig, data, metadata) -> tuple[pd.DataFrame, dict[str, Any]]:

    nodes_df = data["nodes"]
    nodes_metadata = metadata["nodes"]
    capacity_multiplier = float(
        config.AssignmentParameters.Capacity_Adjustment_Factors["from_hourly_to_daily"]
    )

    start_id = 1 if data["nodes"].empty else int(data["nodes"]["node_id"].max()) + 1
    link_id_generator = count(start_id)
    link_catalogue_cfg = config.NetworkParameters.Catalogues.Network
    vdf_catalogue_cfg = config.NetworkParameters.Catalogues.VDF
    possible_link_types = list(link_catalogue_cfg.keys())
    connector_link_types = [k for k, v in link_catalogue_cfg.items() if v.is_connector]
    non_connector_link_types = [k for k, v in link_catalogue_cfg.items() if not v.is_connector]

    def _create_link_pair_ids(bidirectional: bool):
        main_id = next(link_id_generator)
        reverse_id = next(link_id_generator) if bidirectional else None
        return main_id, reverse_id

    def _connect_zones_to_nonzones(link_columns, bidirectional: bool = True, default_id_type: int = 99):
        zones_df = nodes_df[nodes_df["class"] == "Zones"].copy()
        nonzones_df = nodes_df[nodes_df["class"] == "Non-Zones"].copy()
        zone_xy = zones_df[["x", "y"]].to_numpy()
        nonzone_xy = nonzones_df[["x", "y"]].to_numpy()
        zone_ids = zones_df["node_id"].to_numpy()
        nonzone_ids = nonzones_df["node_id"].to_numpy()
        diff = zone_xy[:, None, :] - nonzone_xy[None, :, :]
        dist_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        nearest_idx = np.argmin(dist_matrix, axis=1)
        nearest_ids = nonzone_ids[nearest_idx]
        nearest_dist = dist_matrix[np.arange(len(zones_df)), nearest_idx]
        link_id_type = link_catalogue_cfg[default_id_type]
        rows = []
        for i in range(len(zone_ids)):
            vdf_id = int(metadata["rng"].choice(link_id_type.allowed_vdfs))
            vdf_cfg = vdf_catalogue_cfg[vdf_id]
            length = nearest_dist[i]
            free_flow_time = length / vdf_cfg.speed_limit
            capacity_per_lane = float(link_id_type.capacity)
            total_capacity = capacity_per_lane * float(link_id_type.lanes)
            effective_capacity = total_capacity * capacity_multiplier
            main_link_id, reverse_link_id = _create_link_pair_ids(bidirectional)
            rows.append(
                {
                    "link_id": main_link_id,
                    "reverse_link_id": reverse_link_id,
                    "init_node": zone_ids[i],
                    "term_node": nearest_ids[i],
                    "capacity_per_lane": capacity_per_lane,
                    "lanes": link_id_type.lanes,
                    "total_capacity": total_capacity,
                    "effective_capacity": effective_capacity,
                    "length": length,
                    "free_flow_time": free_flow_time,
                    "b": vdf_cfg.b,
                    "power": vdf_cfg.power,
                    "speed": vdf_cfg.speed_limit,
                    "vdf": vdf_id,
                    "toll": link_id_type.toll,
                    "link_type": default_id_type,
                }
            )
            if bidirectional:
                rows.append(
                    {
                        "link_id": reverse_link_id,
                        "reverse_link_id": main_link_id,
                        "init_node": nearest_ids[i],
                        "term_node": zone_ids[i],
                        "capacity_per_lane": capacity_per_lane,
                        "lanes": link_id_type.lanes,
                        "total_capacity": total_capacity,
                        "effective_capacity": effective_capacity,
                        "length": length,
                        "free_flow_time": free_flow_time,
                        "b": vdf_cfg.b,
                        "power": vdf_cfg.power,
                        "speed": vdf_cfg.speed_limit,
                        "vdf": vdf_id,
                        "toll": link_id_type.toll,
                        "link_type": default_id_type,
                    }
                )
        return pd.DataFrame(rows, columns=link_columns)

    def _connect_nonzones_to_nonzones(link_columns, bidirectional: bool = True):
        nonzones_df = nodes_df[nodes_df["class"] == "Non-Zones"].copy()
        nonzone_xy = nonzones_df[["x", "y"]].to_numpy(dtype=float)
        nonzone_ids = nonzones_df["node_id"].astype(int).to_numpy()
        rows = []
        if len(nonzones_df) < 2:
            logging.warning("Fewer than two non-zone nodes were found. No non-zone links were created.")
            return pd.DataFrame(rows, columns=link_columns)

        def _distance_between(i: int, j: int) -> float:
            return float(np.linalg.norm(nonzone_xy[i] - nonzone_xy[j]))

        def _get_delaunay_edges():
            try:
                from scipy.spatial import Delaunay

                if len(nonzones_df) < 3:
                    raise ValueError("Delaunay triangulation requires at least three points.")
                triangulation = Delaunay(nonzone_xy)
                edges = set()
                for simplex in triangulation.simplices:
                    simplex = list(simplex)
                    triangle_edges = [(simplex[0], simplex[1]), (simplex[1], simplex[2]), (simplex[2], simplex[0])]
                    for i, j in triangle_edges:
                        edges.add(tuple(sorted((int(i), int(j)))))
                return edges
            except Exception as exc:
                logging.warning(
                    "Delaunay triangulation could not be computed. Falling back to a distance-based chain. Reason: %s",
                    exc,
                )
                ordered_idx = np.lexsort((nonzone_xy[:, 1], nonzone_xy[:, 0]))
                return {
                    tuple(sorted((int(ordered_idx[i]), int(ordered_idx[i + 1]))))
                    for i in range(len(ordered_idx) - 1)
                }

        delaunay_edges = _get_delaunay_edges()
        logging.info("Generated %s undirected non-zone edges using Delaunay triangulation.", len(delaunay_edges))
        for i, j in sorted(delaunay_edges):
            init_node = int(nonzone_ids[i])
            term_node = int(nonzone_ids[j])
            distance = _distance_between(i, j)
            main_link_id, reverse_link_id = _create_link_pair_ids(bidirectional)
            non_connector_link_type = int(metadata["rng"].choice(list(non_connector_link_types)))
            link_type_cfg = link_catalogue_cfg[non_connector_link_type]
            capacity_per_lane = float(link_type_cfg.capacity)
            total_capacity = capacity_per_lane * float(link_type_cfg.lanes)
            effective_capacity = total_capacity * capacity_multiplier
            lanes = link_type_cfg.lanes
            vdf_id = int(metadata["rng"].choice(link_type_cfg.allowed_vdfs))
            vdf_cfg = vdf_catalogue_cfg[vdf_id]
            rows.append(
                {
                    "link_id": main_link_id,
                    "reverse_link_id": reverse_link_id,
                    "init_node": init_node,
                    "term_node": term_node,
                    "capacity_per_lane": capacity_per_lane,
                    "lanes": lanes,
                    "total_capacity": total_capacity,
                    "effective_capacity": effective_capacity,
                    "length": distance,
                    "free_flow_time": distance / vdf_cfg.speed_limit,
                    "b": vdf_cfg.b,
                    "power": vdf_cfg.power,
                    "speed": vdf_cfg.speed_limit,
                    "vdf": vdf_id,
                    "toll": link_type_cfg.toll,
                    "link_type": non_connector_link_type,
                }
            )
            if bidirectional:
                rows.append(
                    {
                        "link_id": reverse_link_id,
                        "reverse_link_id": main_link_id,
                        "init_node": term_node,
                        "term_node": init_node,
                        "capacity_per_lane": capacity_per_lane,
                        "lanes": lanes,
                        "total_capacity": total_capacity,
                        "effective_capacity": effective_capacity,
                        "length": distance,
                        "free_flow_time": distance / vdf_cfg.speed_limit,
                        "b": vdf_cfg.b,
                        "power": vdf_cfg.power,
                        "speed": vdf_cfg.speed_limit,
                        "vdf": vdf_id,
                        "toll": link_type_cfg.toll,
                        "link_type": non_connector_link_type,
                    }
                )
        return pd.DataFrame(rows, columns=link_columns)

    zone_connector_links = _connect_zones_to_nonzones(
        link_columns=config.NetworkParameters.Tabular.Columns,
        default_id_type=99,
    )
    nonzone_links = _connect_nonzones_to_nonzones(link_columns=config.NetworkParameters.Tabular.Columns)
    data["links"] = pd.concat([zone_connector_links, nonzone_links], ignore_index=True)

    edge_attributes = [
        "link_id",
        "reverse_link_id",
        "capacity_per_lane",
        "lanes",
        "total_capacity",
        "effective_capacity",
        "length",
        "free_flow_time",
        "b",
        "power",
        "speed",
        "vdf",
        "toll",
        "link_type",
    ]
    data["graph"] = nx.from_pandas_edgelist(
        data["links"],
        source="init_node",
        target="term_node",
        edge_attr=edge_attributes,
        create_using=nx.DiGraph(),
    )
    node_attributes = nodes_df.set_index("node_id").to_dict("index")
    nx.set_node_attributes(data["graph"], node_attributes)

    if not nx.is_strongly_connected(data["graph"]):
        logging.error("The generated graph is not strongly connected. Some OD pairs may not have valid routes.")

    network_metadata = {
        "possible_link_types": possible_link_types,
        "connector_link_types": connector_link_types,
        "non_connector_link_types": non_connector_link_types,
        "link_ids": data["links"]["link_id"].tolist(),
        "num_links": len(data["links"]),
        "link_count_by_type": data["links"]["link_type"].value_counts().to_dict(),
        "vdf_counts": data["links"]["vdf"].value_counts().to_dict(),
        "capacity_multiplier": capacity_multiplier,
        "capacity_columns": {
            "capacity_per_lane": "capacity_per_lane",
            "total_capacity": "total_capacity",
            "effective_capacity": "effective_capacity",
        },
        "effective_capacity_mean": float(data["links"]["effective_capacity"].mean()),
        "effective_capacity_total": float(data["links"]["effective_capacity"].sum()),
        "edge_attributes": edge_attributes,
        "node_attributes": list(nodes_df.columns),
    }
    network_path = save_dataframe_as_tntp(data["links"], config.paths.export_filepaths.network)
    network_metadata["network_path"] = str(network_path)
    return data["links"], network_metadata
