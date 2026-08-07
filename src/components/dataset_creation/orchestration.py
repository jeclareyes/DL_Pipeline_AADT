from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .adapters.assignment_adapter import build_flows
from .adapters.route_generation_adapter import build_routes
from .config import DatasetConfig
from .exporters.artifact_writer import save_master_artifact
from .generators.demand_generator import build_trips
from .generators.network_generator import build_network
from .generators.nodes_generator import build_nodes
from .reports.assignment_control_reporter import run_assignment_control
from .reports.network_visualizer import export_network_previsualization
from .reports.tables_reporter import export_all_tables

LOGGER = logging.getLogger(__name__)


def _enabled(section: Any, default: bool = True) -> bool:
    if section is None:
        return default
    if isinstance(section, bool):
        return section
    if hasattr(section, "enabled"):
        return bool(section.enabled)
    return default


def dataset_creation_orchestration(config: DatasetConfig) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    rng = np.random.default_rng(config.seed)
    data: dict[str, Any] = {}
    metadata: dict[str, Any] = {"rng": rng}

    LOGGER.info("Step 1: Generating nodes...")
    nodes_df, nodes_metadata = build_nodes(config, data, metadata)
    data["nodes"] = nodes_df
    metadata["nodes"] = nodes_metadata
    LOGGER.info("Completed node generation.")

    LOGGER.info("Step 2: Generating network...")
    network_df, network_metadata = build_network(config, data, metadata)
    data["network"] = network_df
    metadata["network"] = network_metadata
    LOGGER.info("Completed network generation.")

    LOGGER.info("Step 3: Generating trips...")
    trips_array, trips_metadata = build_trips(config, data, metadata)
    data["trips"] = trips_array
    metadata["trips"] = trips_metadata
    LOGGER.info("Completed trip generation.")

    LOGGER.info("Step 4: Generating routes...")
    routes_by_od, routes_metadata = build_routes(config, data, metadata)
    data["routes"] = routes_by_od
    metadata["routes"] = routes_metadata
    LOGGER.info("Completed route generation.")

    LOGGER.info("Step 5: Generating flows...")
    flows_df, flows_metadata = build_flows(config, data, metadata)
    data["flows"] = flows_df
    metadata["flows"] = flows_metadata
    metadata["flow_columns"] = {
        "traffic_counts": [],
        "reference_assignment": flows_metadata["flow_column"],
        "estimated_flows": None,
    }
    LOGGER.info("Completed flow generation.")

    reports: dict[str, Any] = {}
    validations: dict[str, Any] = {}

    reports_cfg = getattr(config, "Reports", None)
    if _enabled(getattr(reports_cfg, "Tables", None), default=True):
        reports["tables"] = export_all_tables(data, config, metadata)
    if _enabled(getattr(reports_cfg, "NetworkVisualization", None), default=True):
        reports["network_visualization"] = export_network_previsualization(data, config)
    if _enabled(getattr(reports_cfg, "AssignmentControl", None), default=True):
        reports["assignment_control"] = run_assignment_control(data, config, metadata)

    artifact_paths = save_master_artifact(
        dataset_name=config.name,
        config=config,
        data=data,
        metadata=metadata,
        reports=reports,
        validations=validations,
        info_dir=config.paths.export_dirs.info_dataset,
    )
    LOGGER.info("Saved master artifact: %s", artifact_paths)
    return data, metadata, artifact_paths
