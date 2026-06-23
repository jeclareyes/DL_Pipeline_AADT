from __future__ import annotations
# src/data_ingestion/artifact_builders/training_artifact_builder.py

"""
Training Artifact Builder
=========================

This module builds a unified training artifact from TNTP-like input files.

Project context
---------------
In he AADT / traffic assignment pipeline, scenario creation produces TNTP-like
files for nodes, network, trips, routes and flows. The neural network training
pipeline should not be responsible for repeatedly parsing these files, merging
tables, building graphs, preparing targets or adapting routes into tensors.

This builder acts as the bridge between raw TNTP scenario files and the model
training pipeline.

Its main responsibilities are:

1. Read raw TNTP files using dedicated readers.
2. Build processed transportation objects:
   - canonical link table;
   - NetworkX directed graph;
   - route dictionary and route table;
   - OD demand table and matrix.
3. Build model-ready objects:
   - network tensors through RouteModelAdapter;
   - flow targets and masks;
   - OD targets and masks;
   - visualization payloads.
4. Save everything into a single training_artifact.joblib file.
5. Save a lightweight JSON manifest for inspection and reproducibility.

Design principles
-----------------
- Readers only read raw files.
- This builder orchestrates the transformation into a training artifact.
- Training code should load one artifact and train.
- The artifact should preserve raw, processed and model-ready layers.
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

import joblib
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
    Build a unified model-training artifact from TNTP files.

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

    manifest_name : str, default="training_manifest.json"
        Name of the JSON manifest file.
    """

    def __init__(
        self,
        cfg: Union[DictConfig, Dict[str, Any]],
        device: str = "cpu",
        artifact_name: str = "training_artifact.joblib",
        manifest_name: str = "training_manifest.json",
    ) -> None:
        self.cfg = cfg
        self.device = device
        self.artifact_name = artifact_name
        self.manifest_name = manifest_name

        self.dataset_name = str(self._cfg_get("dataset"))
        self.volume_year = self._cfg_get("volume_year")
        self.multiday_od = bool(self._cfg_get("multiday_od"))

        self.k_paths = int(
            self._cfg_get("model.k_paths")
        )

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

    def _recompute_flows_from_current_routes(self) -> Dict[str, Any]:
        """
        Recompute the flow TNTP file from the current trips, network and routes.

        This is intentionally executed after optional route recomputation. Flow
        targets produced from a different route set are not compatible with the
        delta matrix used by route-based training models.
        """

        flow_cfg = self._cfg_get("flow_recompute", default={}) or {}

        if not bool(flow_cfg.get("enabled", False)):
            return {"enabled": False, "skipped": True}

        if self.paths.flow_path.exists() and not bool(flow_cfg.get("overwrite_existing", True)):
            return {
                "enabled": True,
                "skipped": True,
                "reason": "exists",
                "flow_path": str(self.paths.flow_path),
            }

        logger.info("Recomputing flow targets from the active route set.")

        network_result = read_tntp_network(self.paths.network_path)
        trips_result = read_tntp_trips(
            path=self.paths.trips_path,
            aggregation=self.trips_aggregation,
        )

        zone_ids = trips_result.metadata.get("zone_ids")
        if not zone_ids:
            raise ValueError(
                "Flow recompute failed: trips metadata does not contain zone_ids."
            )

        routes_result = read_tntp_routes(
            path=self.paths.routes_path,
            zone_ids=zone_ids,
            max_routes_per_od=self.k_paths,
        )

        od_matrix = trips_result.od_matrix
        if sparse.issparse(od_matrix):
            assignment_matrix = od_matrix.toarray()
        else:
            assignment_matrix = np.asarray(od_matrix, dtype=float)

        from src.components.assignment_motors import (
            BehaviorModelName,
            SolverName,
            build_assignment_composition,
            build_assignment_config_from_mapping,
        )

        assignment_yaml_path = (
            Path(__file__).resolve().parents[3]
            / "configs"
            / "assignment"
            / "assignment.yaml"
        )
        assign_cfg = OmegaConf.to_container(
            OmegaConf.load(assignment_yaml_path),
            resolve=True,
        )

        behavior_name = str(
            flow_cfg.get("behavior_model", BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value)
        )
        solver_name = str(flow_cfg.get("solver", SolverName.MSA.value))

        if behavior_name not in {
            BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value,
            BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM.value,
        }:
            raise ValueError(
                "flow_recompute.behavior_model must be "
                "'stochastic_user_equilibrium' or 'route_based_user_equilibrium'. "
                f"Received {behavior_name!r}."
            )

        if solver_name not in {
            SolverName.MSA.value,
            SolverName.FRANK_WOLFE.value,
            SolverName.GRADIENT_PROJECTION.value,
        }:
            raise ValueError(
                "flow_recompute.solver must be one of 'msa', 'frank_wolfe', "
                f"or 'gradient_projection'. Received {solver_name!r}."
            )

        if (
            behavior_name == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value
            and solver_name != SolverName.MSA.value
        ):
            raise ValueError("flow_recompute supports SUE only with solver='msa'.")

        assign_cfg["behavior_model"]["source"] = "explicit"
        assign_cfg["behavior_model"]["name"] = behavior_name
        assign_cfg["common"]["max_iterations"] = int(flow_cfg.get("max_iterations", 10000))
        assign_cfg["common"]["capacity_scaling"]["source"] = "explicit"
        assign_cfg["common"]["capacity_scaling"]["value"] = 1.0
        assign_cfg["common"]["capacity_scaling"]["training_config_path"] = None
        assign_cfg["solvers"]["active_solver"] = solver_name

        if solver_name == SolverName.MSA.value:
            assign_cfg["solvers"]["msa"]["step_rule"] = str(
                flow_cfg.get("msa_step_rule")
            )

        assign_cfg["route_based_user_equilibrium"]["policy"]["expected_solver"] = solver_name
        assign_cfg["stochastic_user_equilibrium"]["policy"]["expected_solver"] = solver_name
        assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_source"] = "explicit"
        assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_value"] = float(
            flow_cfg.get("theta", 1.0)
        )
        assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_artifact_key"] = None

        convergence_cfg = flow_cfg.get("convergence", {}) or {}
        if behavior_name == BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value:
            sue_conv = assign_cfg["stochastic_user_equilibrium"]["convergence"]
            for key in (
                "equilibrium_l1_threshold",
                "max_absolute_gap_threshold",
                "max_relative_gap_threshold",
                "min_flow_for_relative_gap",
            ):
                if key in convergence_cfg:
                    sue_conv[key] = float(convergence_cfg[key])
        elif "max_relative_gap_threshold" in convergence_cfg:
            assign_cfg["route_based_user_equilibrium"]["convergence"]["relative_gap_threshold"] = float(
                convergence_cfg["max_relative_gap_threshold"]
            )

        assignment_config = build_assignment_config_from_mapping(assign_cfg)
        zone_id_to_idx = {
            int(zone_id): int(idx)
            for zone_id, idx in trips_result.metadata["zone_id_to_idx"].items()
        }

        composition = build_assignment_composition(
            links_df=network_result.network_df,
            routes_by_od=routes_result.routes_by_od,
            zone_id_to_idx=zone_id_to_idx,
            assignment_config=assignment_config,
            training_config={},
            artifacts={},
        )

        result = composition.behavior_model.solve(
            od_matrix=assignment_matrix,
            config=composition.runtime_config,
        )

        flows_df = network_result.network_df[["init_node", "term_node"]].copy()
        flows_df["Volume"] = np.asarray(result.final_link_flows, dtype=float)
        flows_df = flows_df.rename(columns={"init_node": "From", "term_node": "To"})

        self.paths.flow_path.parent.mkdir(parents=True, exist_ok=True)
        flows_df.to_csv(self.paths.flow_path, sep="\t", index=False)

        metadata = {
            "enabled": True,
            "skipped": False,
            "flow_path": str(self.paths.flow_path),
            "behavior_model": behavior_name,
            "solver": solver_name,
            "capacity_scaling_factor": 1.0,
            "iterations_run": result.metadata.get("iterations_run"),
            "converged": result.metadata.get("converged"),
            "total_assigned_flow": float(np.sum(result.final_link_flows)),
        }
        logger.info(
            "Flow targets recomputed | behavior=%s | solver=%s | iterations=%s | converged=%s | path=%s",
            metadata["behavior_model"],
            metadata["solver"],
            metadata["iterations_run"],
            metadata["converged"],
            metadata["flow_path"],
        )
        return metadata

    def run(self, save: bool = True) -> Dict[str, Any]:
        """
        Build the full training artifact.

        Parameters
        ----------
        save : bool, default=True
            If True, save the artifact and manifest to disk.

        Returns
        -------
        Dict[str, Any]
            Full training artifact.
        """

        logger.info("Building training artifact for dataset: %s", self.dataset_name)
        
        # Ensure updated routes if requested
        from src.data_ingestion.builders.routes_builder import recompute_routes_from_tntp
        route_recompute_metadata = recompute_routes_from_tntp(self.cfg)
        flow_recompute_metadata = self._recompute_flows_from_current_routes()

        raw = self.load_raw()
        processed = self.build_processed(raw)
        model_ready = self.build_model_ready(raw=raw, processed=processed)

        artifact = self.pack_artifact(
            raw=raw,
            processed=processed,
            model_ready=model_ready,
            route_recompute_metadata=route_recompute_metadata,
            flow_recompute_metadata=flow_recompute_metadata,
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
            max_routes_per_od=self.k_paths,
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

        adapter = RouteModelAdapter(
            device=self.device,
            k_paths=self.k_paths,
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
        model_ready: Dict[str, Any],
        route_recompute_metadata: Dict[str, Any] | None = None,
        flow_recompute_metadata: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Pack all layers into the final training artifact.

        Parameters
        ----------
        raw : Dict[str, Any]
            Raw layer.

        processed : Dict[str, Any]
            Processed layer.

        model_ready : Dict[str, Any]
            Model-ready layer.

        Returns
        -------
        Dict[str, Any]
            Full artifact dictionary.
        """

        artifact = {
            "artifact_type": "training_artifact",
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
            "model_ready": model_ready,
            "metadata": {
                "dataset_name": self.dataset_name,
                "volume_year": self.volume_year,
                "multiday_od": self.multiday_od,
                "k_paths": self.k_paths,
                "trips_aggregation": self.trips_aggregation,
                "reader_metadata": raw["metadata"],
                "route_recompute": route_recompute_metadata or {},
                "flow_recompute": flow_recompute_metadata or {},
                "processed_summary": self._build_processed_summary(processed),
                "model_ready_summary": self._build_model_ready_summary(model_ready),
            },
        }

        return artifact

    def save_artifact(self, artifact: Dict[str, Any]) -> Dict[str, str]:
        """
        Save the artifact as joblib and write a lightweight manifest JSON.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Full training artifact.

        Returns
        -------
        Dict[str, str]
            Paths to the saved artifact and manifest.
        """

        logger.info("Saving training artifact to: %s", self.paths.artifact_path)

        self.paths.output_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(artifact, self.paths.artifact_path)

        manifest = self._to_serializable(artifact)
        manifest["artifact_path"] = str(self.paths.artifact_path)

        self.paths.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        return {
            "artifact_path": str(self.paths.artifact_path),
            "manifest_path": str(self.paths.manifest_path),
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

    def _cfg_get(self, dotted_key: str, default: Any = None) -> Any:
        """
        Get a possibly nested configuration value.

        Parameters
        ----------
        dotted_key : str
            Key path such as "input_routes.node_route".

        default : Any, default=None
            Fallback value.

        Returns
        -------
        Any
            Configuration value or default.
        """

        parts = dotted_key.split(".")
        current = self.cfg

        for part in parts:
            if isinstance(current, DictConfig):
                if part not in current:
                    return default
                current = current[part]
            elif isinstance(current, dict):
                if part not in current:
                    return default
                current = current[part]
            else:
                if not hasattr(current, part):
                    return default
                current = getattr(current, part)

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
    )

    return builder.run(save=save)
