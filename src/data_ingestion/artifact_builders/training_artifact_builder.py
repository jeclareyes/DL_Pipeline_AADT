from __future__ import annotations
# src/data_ingestion/artifact_builders/training_artifact_builder.py

"""
Training Artifact Builder
=========================

This module builds the base artifact from TNTP-like input files.

Project context
---------------
In he AADT / traffic assignment pipeline, scenario creation produces TNTP-like
files for nodes, network, trips, routes and flows. The neural network training
pipeline should not be responsible for repeatedly parsing these files, merging
tables, building graphs, preparing targets or adapting routes into tensors.

This builder acts as the bridge between raw TNTP scenario files and the
base-artifact layer of the pipeline.

Its main responsibilities are:

1. Read raw TNTP files using dedicated readers.
2. Build processed transportation objects:
   - canonical link table;
   - NetworkX directed graph;
   - route dictionary and route table;
   - OD demand table and matrix.
3. Preserve the processed transport objects needed by downstream asset
   materialization.
4. Save everything into a single base_artifact.joblib file.
5. Save a lightweight bundle manifest for inspection and reproducibility.

Design principles
-----------------
- Readers only read raw files.
- This builder orchestrates the transformation into a base artifact.
- Training-ready tensors are materialized later by the asset pipeline.
- The artifact should preserve raw and processed layers.
- Tensors are saved on CPU by default for portability.
"""

from networkx.generators import spectral_graph_forge

from dataclasses import dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
import platform
import sys
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from scipy import sparse


from src.data_ingestion.readers.tntp_node_reader import read_tntp_nodes
from src.data_ingestion.readers.tntp_network_reader import read_tntp_network
from src.data_ingestion.readers.tntp_flow_reader import read_tntp_flows
from src.data_ingestion.readers.tntp_trips_reader import read_tntp_trips
from src.data_ingestion.readers.tntp_routes_reader import read_tntp_routes

from src.data_ingestion.adapters.route_model_adapter import RouteModelAdapter
from src.utils.serialization import dump

from src.data_ingestion.builders.link_table_builder import build_link_table
from src.data_ingestion.builders.graph_builder import build_graph
from src.data_ingestion.builders.target_builder import build_targets

from src.data_ingestion.validators.training_artifact_validator import (validate_training_artifact_or_raise,)


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifactPaths:
    """
    Resolved input and output paths used by TrainingArtifactBuilder.
    """

    node_path: Path
    network_path: Path
    flow_path: Path
    trips_path: Path
    routes_path: Path
    output_dir: Path
    artifact_path: Path
    manifest_path: Path


class TrainingArtifactBuilder:
    """
    Build a unified base artifact from TNTP files.

    Parameters
    ----------
    cfg : Union[DictConfig, Dict[str, Any]]
        Configuration object. It may be an OmegaConf DictConfig or a plain dict.

    device : str, default="cpu"
        Device used when running RouteModelAdapter. The recommended value for
        artifact generation is "cpu", because the saved artifact should be
        portable across machines.

    artifact_name : str, default="training_artifact.joblib"
        Name of the output joblib file.

    manifest_name : str, default="base_manifest.json"
        Name of the JSON manifest file.
    """

    def __init__(
        self,
        cfg: Union[DictConfig, Dict[str, Any]],
        device: str = "cpu",
        artifact_name: str = "training_artifact.joblib",
        manifest_name: str = "base_manifest.json",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.artifact_name = artifact_name
        self.manifest_name = manifest_name

        self.dataset_name = str(self._cfg_get("dataset_name"))
        self.volume_year = self._cfg_get("readers.flows.volume_year")
        self.multiday_od = self._cfg_get("readers.trips.multiday_od")

        self.max_routes_per_od = self._cfg_get("readers.routes.max_routes_per_od")
        if self.max_routes_per_od is not None:
            self.max_routes_per_od = int(self.max_routes_per_od)

        self.trips_aggregation = str(
            self._cfg_get("readers.trips.aggregation")
        )

        self.paths = self._resolve_paths()


    def _resolve_model_link_pair_indices(
        self,
        network_params: Dict[str, Any],
        processed: Dict[str, Any],
    ) -> np.ndarray:
        """
        Resolve the canonical model link order.

        Priority
        --------
        1. network_params["link_pair_indices"], if produced by RouteModelAdapter.
        2. network_params["edge_list"], if available.
        3. processed["edge_indexing"]["link_pair_indices"] as fallback.

        Returns
        -------
        np.ndarray
            Directed edge array with shape [num_links, 2].
        """

        if "link_pair_indices" in network_params:
            value = network_params["link_pair_indices"]
        elif "edge_list" in network_params:
            value = network_params["edge_list"]
        else:
            value = processed["edge_indexing"]["link_pair_indices"]

        if torch.is_tensor(value):
            edge_order = value.detach().cpu().numpy()
        else:
            edge_order = np.asarray(value)

        edge_order = edge_order.astype(np.int64)

        if edge_order.ndim != 2 or edge_order.shape[1] != 2:
            raise ValueError(
                "Model link order must have shape [num_links, 2]. "
                f"Received shape {edge_order.shape}."
            )

        expected_num_links = int(network_params["num_links"])

        if edge_order.shape[0] != expected_num_links:
            raise ValueError(
                "Model link order length does not match network_params['num_links']. "
                f"Expected {expected_num_links}, got {edge_order.shape[0]}."
            )

        return edge_order


    def _build_model_edge_indexing(
        self,
        link_df: pd.DataFrame,
        model_link_pair_indices: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Build an edge-indexing payload in the exact model link order.
        """

        edge_list = [
            (int(u), int(v))
            for u, v in np.asarray(model_link_pair_indices, dtype=np.int64)
        ]

        edge_to_idx = {
            edge: idx
            for idx, edge in enumerate(edge_list)
        }

        idx_to_edge = {
            idx: edge
            for edge, idx in edge_to_idx.items()
        }

        edge_to_link_id_lookup = {}

        if "link_id" in link_df.columns:
            for row in link_df.itertuples(index=False):
                edge = (int(row.init_node), int(row.term_node))
                edge_to_link_id_lookup[edge] = int(row.link_id)

        link_id_to_idx = {}
        idx_to_link_id = {}
        edge_to_link_id = {}

        for idx, edge in enumerate(edge_list):
            if edge not in edge_to_link_id_lookup:
                continue

            link_id = int(edge_to_link_id_lookup[edge])
            link_id_to_idx[link_id] = int(idx)
            idx_to_link_id[int(idx)] = link_id
            edge_to_link_id[edge] = link_id

        return {
            "edge_list": edge_list,
            "edge_to_idx": edge_to_idx,
            "idx_to_edge": idx_to_edge,
            "link_id_to_idx": link_id_to_idx,
            "idx_to_link_id": idx_to_link_id,
            "edge_to_link_id": edge_to_link_id,
            "link_pair_indices": np.asarray(model_link_pair_indices, dtype=np.int64),
        }


    def _build_link_table_in_model_order(
        self,
        link_df: pd.DataFrame,
        model_link_pair_indices: np.ndarray,
    ) -> pd.DataFrame:
        """
        Reorder link_df to match the model link order.

        This table should be used for diagnostics and prediction exports.
        """

        required_columns = {"init_node", "term_node"}
        missing_columns = required_columns - set(link_df.columns)

        if missing_columns:
            raise ValueError(
                "Cannot reorder link_df to model order because columns are missing: "
                f"{sorted(missing_columns)}"
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

        links = links.dropna(
            subset=["init_node", "term_node"],
        ).copy()

        links["init_node"] = links["init_node"].astype(int)
        links["term_node"] = links["term_node"].astype(int)

        duplicated_edges = links.duplicated(
            subset=["init_node", "term_node"],
            keep=False,
        )

        if duplicated_edges.any():
            duplicated_sample = (
                links.loc[duplicated_edges, ["init_node", "term_node"]]
                .head(10)
                .to_dict("records")
            )

            raise ValueError(
                "Cannot reorder link_df because duplicated directed edges exist. "
                f"Sample: {duplicated_sample}"
            )

        edge_to_row = {
            (int(row.init_node), int(row.term_node)): row._asdict()
            for row in links.itertuples(index=False)
        }

        rows = []
        missing_edges = []

        for position, (u, v) in enumerate(model_link_pair_indices):
            edge = (int(u), int(v))

            if edge not in edge_to_row:
                missing_edges.append(
                    {
                        "position": int(position),
                        "init_node": int(u),
                        "term_node": int(v),
                    }
                )
                continue

            row = dict(edge_to_row[edge])
            row["model_link_position"] = int(position)
            rows.append(row)

        if missing_edges:
            raise ValueError(
                "Some model links could not be found in link_df. "
                f"First missing edges: {missing_edges[:10]}"
            )

        return pd.DataFrame(rows).reset_index(drop=True)


    def _build_model_od_indexing(
        self,
        network_params: Dict[str, Any],
        processed: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Build OD indexing payload aligned with model OD order.
        """

        if "od_pairs" in network_params:
            od_pairs_raw = network_params["od_pairs"]
        elif "od_pair_indices" in network_params:
            od_pairs_raw = network_params["od_pair_indices"]
        else:
            return processed["od_indexing"]

        if torch.is_tensor(od_pairs_raw):
            od_array = od_pairs_raw.detach().cpu().numpy()
        else:
            od_array = np.asarray(od_pairs_raw)

        od_pairs = [
            (int(origin), int(destination))
            for origin, destination in od_array
        ]

        od_pair_to_idx = {
            od_pair: idx
            for idx, od_pair in enumerate(od_pairs)
        }

        idx_to_od_pair = {
            idx: od_pair
            for od_pair, idx in od_pair_to_idx.items()
        }

        base = dict(processed["od_indexing"])

        base.update(
            {
                "od_pairs": od_pairs,
                "od_pair_to_idx": od_pair_to_idx,
                "idx_to_od_pair": idx_to_od_pair,
            }
        )

        return base

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------


    def run(self, save: bool = True) -> Dict[str, Any]:
        """
        Build the full base artifact.

        Parameters
        ----------
        save : bool, default=True
            If True, save the artifact and manifest to disk.

        Returns
        -------
        Dict[str, Any]
            Full base artifact.
        """

        logger.info("Building base artifact for dataset: %s", self.dataset_name)

        raw = self.load_raw()
        processed = self.build_processed(raw)

        include_model_ready = bool(self._cfg_get("artifact.include_model_ready_layer"))
        if include_model_ready:
            raise ValueError(
                "Base artifact construction no longer supports a model_ready layer. "
                "Set artifact.include_model_ready_layer to false."
            )

        artifact = self.pack_artifact(
            raw=raw,
            processed=processed,
            artifact_type="base_artifact",
        )

        validation_result = validate_training_artifact_or_raise(
            artifact,
            strict=True,
        )

        artifact["metadata"]["validation"] = validation_result.summary

        if save:
            save_info = self.save_artifact(artifact)
            artifact["metadata"]["save_info"] = save_info

        logger.info("Training artifact successfully built.")

        return artifact

    def load_raw(self) -> Dict[str, Any]:
        """
        Read all raw TNTP files using the dedicated readers.

        Returns
        -------
        Dict[str, Any]
            Raw layer containing DataFrames, sparse OD matrix and reader metadata.
        """

        logger.info("Reading raw TNTP files.")

        node_result = read_tntp_nodes(self.paths.node_path)

        network_result = read_tntp_network(self.paths.network_path)

        flow_result = read_tntp_flows(
            path=self.paths.flow_path,
            volume_year=self.volume_year,
        )

        trips_result = read_tntp_trips(
            path=self.paths.trips_path,
            aggregation=self.trips_aggregation,
        )

        zone_ids = trips_result.metadata.get("zone_ids")

        if not zone_ids:
            raise ValueError(
                "Trips reader did not provide metadata['zone_ids']. "
                "Cannot read compact routes safely because OD order would be ambiguous."
            )

        routes_result = read_tntp_routes(
            path=self.paths.routes_path,
            zone_ids=zone_ids,
            max_routes_per_od=self.max_routes_per_od,
        )

        return {
            "nodes_df": node_result.nodes_df,
            "network_df": network_result.network_df,
            "flow_df": flow_result.flow_df,
            "trips_df": trips_result.trips_df,
            "od_matrix": trips_result.od_matrix,
            "routes_by_od": routes_result.routes_by_od,
            "routes_df": routes_result.routes_df,
            "metadata": {
                "nodes": node_result.metadata,
                "network": network_result.metadata,
                "flows": flow_result.metadata,
                "trips": trips_result.metadata,
                "routes": routes_result.metadata,
            },
        }

    def build_processed(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build processed transportation objects from raw reader outputs.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw layer produced by load_raw().

        Returns
        -------
        Dict[str, Any]
            Processed layer containing link_df, graph and indexing payloads.
        """

        logger.info("Building processed transportation objects.")

        link_result = build_link_table(
            network_df=raw["network_df"],
            flow_df=raw["flow_df"],
        )

        link_df = link_result.link_df
        link_metadata = link_result.metadata

        capacity_columns = {
            "capacity_per_lane": "capacity_per_lane" if "capacity_per_lane" in link_df.columns else None,
            "total_capacity": "total_capacity" if "total_capacity" in link_df.columns else None,
            "effective_capacity": "effective_capacity" if "effective_capacity" in link_df.columns else None,
        }

        capacity_summary: Dict[str, Dict[str, float]] = {}
        for column in ("capacity_per_lane", "total_capacity", "effective_capacity"):
            if column in link_df.columns:
                capacity_summary[column] = {
                    "total": float(link_df[column].sum()),
                    "mean": float(link_df[column].mean()),
                    "min": float(link_df[column].min()),
                    "max": float(link_df[column].max()),
                }

        graph_result = build_graph(
            link_df=link_df,
            node_df=raw["nodes_df"],
            strict=True,
            weight_column="free_flow_time",
        )

        graph = graph_result.graph
        edge_indexing = graph_result.edge_indexing
        node_indexing = graph_result.node_indexing
        graph_metadata = graph_result.metadata


        zone_ids = raw["metadata"]["trips"].get("zone_ids")

        if not zone_ids:
            raise ValueError(
                "Raw trips metadata does not contain 'zone_ids'. "
                "Cannot build OD indexing safely."
            )

        od_indexing = self._build_od_indexing(
            routes_by_od=raw["routes_by_od"],
            zone_ids=zone_ids,
        )

        return {
            "link_df": link_df,
            "graph": graph,
            "edge_indexing": edge_indexing,
            "node_indexing": node_indexing,
            "od_indexing": od_indexing,
            "metadata": {
                "graph": graph_metadata,
                "link": link_metadata,
                "effective_capacity": {
                    "columns": capacity_columns,
                    "summary": capacity_summary,
                    "effective_capacity_source": "effective_capacity",
                },
            },
            "routes_by_od": raw["routes_by_od"],
            "routes_df": raw["routes_df"],
            "trips_df": raw["trips_df"],
            "od_matrix": raw["od_matrix"],
        }


    def build_model_ready(
        self,
        raw: Dict[str, Any],
        processed: Dict[str, Any],
        k_paths: int,
    ) -> Dict[str, Any]:
        """
        Build model-ready objects from processed transportation objects.

        This method moves transformation logic out of the training script.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw layer.

        processed : Dict[str, Any]
            Processed layer.

        Returns
        -------
        Dict[str, Any]
            Model-ready layer with network_params, targets and visualization data.
        """

        logger.info("Building model-ready payload.")

        if int(k_paths) <= 0:
            raise ValueError("k_paths must be a positive integer.")

        adapter = RouteModelAdapter(
            device=self.device,
            k_paths=int(k_paths),
        )

        canonical_edge_order = processed["edge_indexing"].get("link_pair_indices")

        if canonical_edge_order is None:
            raise ValueError(
                "processed['edge_indexing']['link_pair_indices'] is required to build "
                "model-ready tensors with a stable canonical link order."
            )

        network_params = adapter.transform(
            graph=processed["graph"],
            routes_by_od=processed["routes_by_od"],
            edge_order=canonical_edge_order,
        )

        # TODO: corregir esta ambiguedad ante el modelo.
        if "capacity" not in network_params and "effective_capacity" in network_params:
            network_params["capacity"] = network_params["effective_capacity"]

        # ------------------------------------------------------------------
        # Resolve model link order
        # ------------------------------------------------------------------
        # The model link order is the source of truth for all link-level arrays:
        # t0, capacity, delta_matrix rows, reconstructed_flows, flow targets,
        # oracle alpha/beta and link metadata.
        model_link_pair_indices = self._resolve_model_link_pair_indices(
            network_params=network_params,
        processed=processed,
    )

        model_edge_indexing = self._build_model_edge_indexing(
            link_df=processed["link_df"],
            model_link_pair_indices=model_link_pair_indices,
        )

        network_params["link_pair_indices"] = model_link_pair_indices

        model_od_indexing = self._build_model_od_indexing(
            network_params=network_params,
            processed=processed,
        )

        # ------------------------------------------------------------------
        # Model-ordered link metadata
        # ------------------------------------------------------------------
        model_link_df = self._build_link_table_in_model_order(
            link_df=processed["link_df"],
            model_link_pair_indices=model_link_pair_indices,
        )

        target_result = build_targets(
            link_df=processed["link_df"],
            trips_df=processed["trips_df"],
            routes_by_od=processed["routes_by_od"],
            edge_indexing=model_edge_indexing,
            od_indexing=model_od_indexing,
            create_tensors=True,
            tensor_device=self.device,
        )

        targets = target_result.targets
        target_metadata = target_result.metadata

        visualization = self._build_visualization_payload(
            graph=processed["graph"],
        )

        return {
        "network_params": network_params,
        "targets": targets,
        "visualization": visualization,
        "link_metadata": model_link_df,
        "metadata": {
            "targets": target_metadata,
            # "route_model": route_model_metadata, # TODO: esto?
        },
    }

    def pack_artifact(
        self,
        raw: Dict[str, Any],
        processed: Dict[str, Any],
        artifact_type: str = "base_artifact",
        model_ready: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Pack all layers into the final training artifact.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw layer.

        processed : Dict[str, Any]
            Processed layer.

        artifact_type : str, default="base_artifact"
            Artifact flavor to store. Use ``base_artifact`` for the output of
            the data-processing pipeline and ``training_artifact`` for the
            asset-materialized training bundle.

        model_ready : Dict[str, Any] | None, default=None
            Optional model-ready payload. The base artifact omits this layer.

        Returns
        -------
        Dict[str, Any]
            Full artifact dictionary.
        """

        artifact = {
            "artifact_type": artifact_type,
            "artifact_version": "1.0",
            "dataset_name": self.dataset_name,
            "created_at": datetime.now().isoformat(),
            "config": self._to_plain_container(self.cfg),
            "environment": self._build_environment_metadata(),
            "paths": {
                "node_path": str(self.paths.node_path),
                "network_path": str(self.paths.network_path),
                "flow_path": str(self.paths.flow_path),
                "trips_path": str(self.paths.trips_path),
                "routes_path": str(self.paths.routes_path),
                "artifact_path": str(self.paths.artifact_path),
                "manifest_path": str(self.paths.manifest_path),
            },
            "raw": raw,
            "processed": processed,
            "metadata": {
                "dataset_name": self.dataset_name,
                "volume_year": self.volume_year,
                "multiday_od": self.multiday_od,
                "max_routes_per_od": self.max_routes_per_od,
                "trips_aggregation": self.trips_aggregation,
                "reader_metadata": raw["metadata"],
                "processed_summary": self._build_processed_summary(processed),
            },
        }

        if model_ready:
            artifact["model_ready"] = model_ready
            artifact["metadata"]["model_ready_summary"] = self._build_model_ready_summary(model_ready)

        return artifact

    def save_artifact(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Save the artifact as joblib and write a lightweight manifest JSON.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Full training artifact.

        Returns
        -------
        Dict[str, Any]
            Saved-file summary and paths.
        """

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

        save_joblib = bool(self._cfg_get("artifact.save_joblib"))
        save_manifest = bool(self._cfg_get("artifact.save_manifest"))

        saved_files: list[str] = []

        if save_joblib:
            dump(artifact, self.paths.artifact_path)
            saved_files.append(self.paths.artifact_path.name)

        if save_manifest:
            manifest = self._build_bundle_manifest(artifact)

            self.paths.manifest_path.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            saved_files.append(self.paths.manifest_path.name)

        if saved_files:
            logger.info(
                "Saved %s in %s",
                " and ".join(f"`{name}`" for name in saved_files),
                self.paths.output_dir,
            )
        else:
            logger.info(
                "No training artifact files were saved in %s because saving is disabled in the configuration.",
                self.paths.output_dir,
            )

        return {
            "output_dir": str(self.paths.output_dir),
            "saved_files": saved_files,
            "artifact_path": str(self.paths.artifact_path) if save_joblib else None,
            "manifest_path": str(self.paths.manifest_path) if save_manifest else None,
        }

    def _build_bundle_manifest(self, artifact: Dict[str, Any]) -> Dict[str, Any]:
        """Build the lightweight artifact-bundle manifest."""

        from src.components.artifacts.fingerprints import (
            compute_link_order_fingerprint,
            compute_network_fingerprint,
            compute_od_space_fingerprint,
            compute_zone_order_fingerprint,
        )

        processed = artifact["processed"]
        raw = artifact["raw"]
        network_fingerprint = compute_network_fingerprint(processed["link_df"], raw["nodes_df"])
        od_fingerprint = compute_od_space_fingerprint(processed["od_indexing"].get("od_pairs", []))
        link_order_fingerprint = compute_link_order_fingerprint(processed["edge_indexing"]["link_pair_indices"])
        zone_order_fingerprint = compute_zone_order_fingerprint(processed["od_indexing"].get("zone_ids", []))

        base_artifact_entry = {
            "path": str(self.paths.artifact_path),
            "artifact_type": artifact["artifact_type"],
            "artifact_version": artifact["artifact_version"],
            "fingerprints": {
                "network": network_fingerprint,
                "od_space": od_fingerprint,
                "link_order": link_order_fingerprint,
                "zone_order": zone_order_fingerprint,
            },
            "metadata": {
                "dataset_name": artifact["dataset_name"],
                "processed_summary": self._to_serializable(artifact["metadata"].get("processed_summary", {})),
                "reader_metadata": self._to_serializable(artifact["metadata"].get("reader_metadata", {})),
            },
        }

        return {
            "schema_version": "artifact_bundle.v1",
            "dataset_name": artifact["dataset_name"],
            "artifact_type": artifact["artifact_type"],
            "created_at": artifact["created_at"],
            "base_artifact": base_artifact_entry,
            "route_sets": {},
            "assignment_sets": {},
            "config": self._to_serializable(artifact["config"]),
            "environment": self._to_serializable(artifact["environment"]),
        }

    # ------------------------------------------------------------------
    # Processed object builders
    # ------------------------------------------------------------------

    def _build_od_indexing(
        self,
        routes_by_od: Dict[Tuple[int, int], List[List[int]]],
        zone_ids: Sequence[int],
    ) -> Dict[str, Any]:
        """
        Build OD indexing payload.

        Parameters
        ----------
        routes_by_od : Dict[Tuple[int, int], List[List[int]]]
            Routes dictionary.

        zone_ids : Sequence[int]
            Ordered zone IDs.

        Returns
        -------
        Dict[str, Any]
            OD indexing metadata.
        """

        od_pairs = [
            (int(origin), int(destination))
            for origin, destination in routes_by_od.keys()
        ]

        od_pair_to_idx = {
            od_pair: idx
            for idx, od_pair in enumerate(od_pairs)
        }

        idx_to_od_pair = {
            idx: od_pair
            for od_pair, idx in od_pair_to_idx.items()
        }

        zone_ids = [int(zone_id) for zone_id in zone_ids]
        zone_id_to_idx = {
            zone_id: idx
            for idx, zone_id in enumerate(zone_ids)
        }

        return {
            "zone_ids": zone_ids,
            "zone_id_to_idx": zone_id_to_idx,
            "od_pairs": od_pairs,
            "od_pair_to_idx": od_pair_to_idx,
            "idx_to_od_pair": idx_to_od_pair,
        }

    # ------------------------------------------------------------------
    # Model-ready target builders
    # ------------------------------------------------------------------

    def _build_visualization_payload(self, graph: nx.DiGraph) -> Dict[str, Any]:
        """
        Build lightweight visualization data from the graph.

        Parameters
        ----------
        graph : nx.DiGraph
            Directed graph.

        Returns
        -------
        Dict[str, Any]
            Node coordinates, link geometries and visible link types.
        """

        node_coords = {}

        for node, data in graph.nodes(data=True):
            node_coords[int(node)] = (
                float(data.get("x", 0.0)),
                float(data.get("y", 0.0)),
            )

        link_geometries = {}
        link_types_vis = {}

        for idx, (u, v, data) in enumerate(graph.edges(data=True)):
            link_id = int(data.get("link_id", idx))

            x1, y1 = node_coords.get(int(u), (0.0, 0.0))
            x2, y2 = node_coords.get(int(v), (0.0, 0.0))

            link_geometries[link_id] = {
                "u": int(u),
                "v": int(v),
                "x1": float(x1),
                "y1": float(y1),
                "x2": float(x2),
                "y2": float(y2),
            }

            link_types_vis[link_id] = int(data.get("link_type", 0))

        return {
            "node_coords": node_coords,
            "link_geometries": link_geometries,
            "link_types_vis": link_types_vis,
        }

    # ------------------------------------------------------------------
    # Metadata and serialization helpers
    # ------------------------------------------------------------------

    def _resolve_paths(self) -> TrainingArtifactPaths:
        """
        Resolve input and output paths from the configuration.

        Returns
        -------
        TrainingArtifactPaths
            Resolved paths.
        """

        node_path = self._as_path(self._cfg_get("input_routes.node_route"))
        network_path = self._as_path(self._cfg_get("input_routes.network_route"))
        flow_path = self._as_path(self._cfg_get("input_routes.flow_route"))
        trips_path = self._as_path(self._cfg_get("input_routes.trips_route"))
        routes_path = self._as_path(self._cfg_get("input_routes.routes_route"))

        output_dir = self._as_path(
            self._cfg_get("output_routes.processed_route")
        )

        artifact_path = output_dir / self.artifact_name
        manifest_path = output_dir / self.manifest_name

        return TrainingArtifactPaths(
            node_path=node_path,
            network_path=network_path,
            flow_path=flow_path,
            trips_path=trips_path,
            routes_path=routes_path,
            output_dir=output_dir,
            artifact_path=artifact_path,
            manifest_path=manifest_path,
        )

    def _extract_zone_ids(self, node_df: pd.DataFrame) -> List[int]:
        """
        Extract ordered zone IDs from the node table.

        Parameters
        ----------
        node_df : pd.DataFrame
            Normalized node table.

        Returns
        -------
        List[int]
            Ordered zone node IDs.
        """

        if "class" in node_df.columns:
            zone_mask = (
                node_df["class"]
                .astype(str)
                .str.lower()
                .isin({"zone", "zones", "taz"})
            )

            zone_ids = node_df.loc[zone_mask, "node_id"].astype(int).tolist()

            if zone_ids:
                return zone_ids

        # Fallback: if there is no class column, use nodes whose type suggests zone.
        type_mask = (
            node_df["type"]
            .astype(str)
            .str.lower()
            .isin({"zone", "zones", "taz"})
        )

        zone_ids = node_df.loc[type_mask, "node_id"].astype(int).tolist()

        if not zone_ids:
            raise ValueError(
                "Could not identify zone IDs from node_df. "
                "Expected either class='Zones' or type='TAZ'/'zone'."
            )

        return zone_ids

    def _build_environment_metadata(self) -> Dict[str, Any]:
        """
        Build environment metadata for reproducibility.

        Returns
        -------
        Dict[str, Any]
            Environment metadata.
        """

        return {
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "networkx_version": nx.__version__,
            "torch_version": torch.__version__,
        }

    def _build_processed_summary(self, processed: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a lightweight summary of the processed layer.

        Parameters
        ----------
        processed : Dict[str, Any]
            Processed artifact layer.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary.
        """

        graph = processed["graph"]
        link_df = processed["link_df"]
        routes_by_od = processed["routes_by_od"]
        od_matrix = processed["od_matrix"]

        return {
            "num_nodes": int(graph.number_of_nodes()),
            "num_links": int(graph.number_of_edges()),
            "link_df_shape": tuple(link_df.shape),
            "capacity_columns": {
                "capacity_per_lane": "capacity_per_lane" if "capacity_per_lane" in link_df.columns else None,
                "total_capacity": "total_capacity" if "total_capacity" in link_df.columns else None,
                "effective_capacity": "effective_capacity" if "effective_capacity" in link_df.columns else None,
            },
            "num_od_pairs": int(len(routes_by_od)),
            "num_routes": int(sum(len(routes) for routes in routes_by_od.values())),
            "od_matrix_shape": tuple(int(x) for x in od_matrix.shape),
            "od_matrix_nnz": int(od_matrix.nnz) if sparse.issparse(od_matrix) else None,
        }

    def _build_model_ready_summary(self, model_ready: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a lightweight summary of the model-ready layer.

        Parameters
        ----------
        model_ready : Dict[str, Any]
            Model-ready artifact layer.

        Returns
        -------
        Dict[str, Any]
            Summary dictionary.
        """

        network_params = model_ready["network_params"]
        targets = model_ready["targets"]

        summary = {
            "num_links": int(network_params["num_links"]),
            "num_od_pairs": int(network_params["num_od_pairs"]),
            "flows_target_shape": tuple(targets["flows_target_np"].shape),
            "od_target_shape": tuple(targets["od_target_np"].shape),
        }

        if "delta_matrix" in network_params:
            summary["delta_matrix_shape"] = tuple(network_params["delta_matrix"].shape)

        if "route_validity_mask" in network_params:
            summary["route_validity_mask_shape"] = tuple(
                network_params["route_validity_mask"].shape
            )

        return summary

    def _to_serializable(self, obj: Any) -> Any:
        """
        Convert complex Python objects into JSON-serializable summaries.

        Parameters
        ----------
        obj : Any
            Object to summarize.

        Returns
        -------
        Any
            JSON-serializable object.
        """

        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)

        if isinstance(obj, pd.DataFrame):
            return {
                "__type__": "DataFrame",
                "shape": tuple(obj.shape),
                "columns": obj.columns.tolist(),
            }

        if sparse.issparse(obj):
            return {
                "__type__": obj.__class__.__name__,
                "shape": tuple(int(x) for x in obj.shape),
                "nnz": int(obj.nnz),
                "dtype": str(obj.dtype),
            }

        if isinstance(obj, nx.Graph):
            return {
                "__type__": "networkx_graph",
                "num_nodes": int(obj.number_of_nodes()),
                "num_edges": int(obj.number_of_edges()),
                "is_directed": bool(obj.is_directed()),
            }

        if torch.is_tensor(obj):
            return {
                "__type__": "torch.Tensor",
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
                "device": str(obj.device),
                "is_sparse": bool(obj.is_sparse),
            }

        if isinstance(obj, np.ndarray):
            return {
                "__type__": "ndarray",
                "shape": tuple(obj.shape),
                "dtype": str(obj.dtype),
            }

        if isinstance(obj, Path):
            return str(obj)

        if isinstance(obj, dict):
            return {
                str(key): self._to_serializable(value)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [self._to_serializable(value) for value in obj]

        if isinstance(obj, tuple):
            return [self._to_serializable(value) for value in obj]

        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()

        if isinstance(obj, (int, float, str, bool)) or obj is None:
            return obj

        return str(obj)

    def _to_plain_container(self, cfg: Union[DictConfig, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Convert config to a plain Python container.

        Parameters
        ----------
        cfg : Union[DictConfig, Dict[str, Any]]
            Configuration object.

        Returns
        -------
        Dict[str, Any]
            Plain dictionary.
        """

        if isinstance(cfg, DictConfig):
            return OmegaConf.to_container(cfg, resolve=True)

        return dict(cfg)

    def _cfg_get(self, dotted_key: str) -> Any:
        """
        Get a possibly nested configuration value strictly.

        Parameters
        ----------
        dotted_key : str
            Key path such as "input_routes.node_route".

        Returns
        -------
        Any
            Configuration value.
            
        Raises
        ------
        KeyError
            If the configuration key is missing.
        """

        parts = dotted_key.split(".")
        current = self.cfg

        for part in parts:
            if isinstance(current, DictConfig):
                if part not in current:
                    raise KeyError(f"Configuration key '{dotted_key}' is missing. Bypassing with defaults is prohibited by AGENTS.md.")
                current = current[part]
            elif isinstance(current, dict):
                if part not in current:
                    raise KeyError(f"Configuration key '{dotted_key}' is missing. Bypassing with defaults is prohibited by AGENTS.md.")
                current = current[part]
            else:
                raise KeyError(f"Cannot resolve '{dotted_key}' because '{part}' is not a dict.")

        return current

    def _as_path(self, value: Any) -> Path:
        """
        Convert a path-like config value into a Path object.

        Parameters
        ----------
        value : Any
            Path-like value.

        Returns
        -------
        Path
            Resolved path.
        """

        if value is None:
            raise ValueError("Expected a path-like value, but got None.")

        return Path(str(value)).expanduser().resolve(strict=False)

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


def build_training_artifact(
    cfg: Union[DictConfig, Dict[str, Any]],
    device: str = "cpu",
    save: bool = True,
    artifact_name: str = "training_artifact.joblib",
    manifest_name: str = "base_manifest.json",
) -> Dict[str, Any]:
    """
    Convenience function to build a training artifact.

    Parameters
    ----------
    cfg : Union[DictConfig, Dict[str, Any]]
        Data-processing configuration.

    device : str, default="cpu"
        Device used for model-ready tensor construction.

    save : bool, default=True
        Whether to save the artifact and manifest.

    Returns
    -------
    Dict[str, Any]
        Full training artifact.
    """

    builder = TrainingArtifactBuilder(
        cfg=cfg,
        device=device,
        artifact_name=artifact_name,
        manifest_name=manifest_name,
    )

    return builder.run(save=save)
