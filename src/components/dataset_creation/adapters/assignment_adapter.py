from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.components.dataset_creation.config import DatasetConfig
from src.components.assignment_motors import (
    BehaviorModelName,
    SolverName,
    build_assignment_composition,
    build_assignment_config_from_mapping,
)

from ..exporters.tntp_exporter import save_dataframe_as_tntp

LOGGER = logging.getLogger(__name__)


def _build_average_daily_matrix(trips_array: np.ndarray) -> np.ndarray:
    daily_matrices = trips_array.sum(axis=1)
    return daily_matrices.mean(axis=0)


def build_flows(config: DatasetConfig, data: dict[str, Any], metadata: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if data["trips"] is None:
        raise ValueError("Trips array is missing. Run demand generation before assignment.")
    if data["network"].empty:
        raise ValueError("Links dataframe is missing. Run network generation before assignment.")
    if not data["routes"]:
        raise ValueError("Routes are missing. Run route generation before assignment.")

    assign_cfg_path = Path(__file__).resolve().parents[4] / "configs" / "assignment" / "assignment.yaml"
    assign_cfg = OmegaConf.to_container(OmegaConf.load(assign_cfg_path), resolve=True)

    assignment_params = config.AssignmentParameters
    paradigm = str(assignment_params.Paradigm)
    method = str(assignment_params.Method)
    theta = float(assignment_params.SUE_Parameters.theta)
    max_iterations = int(assignment_params.Max_Iterations)
    convergence_params = assignment_params.ConvergenceParameters
    capacity_multiplier = float(
        assignment_params.Capacity_Adjustment_Factors["from_hourly_to_daily"]
    )

    paradigm_normalized = paradigm.strip().upper()
    method_normalized = method.strip().lower()

    if paradigm_normalized == "SUE":
        behavior_model_name = BehaviorModelName.STOCHASTIC_USER_EQUILIBRIUM.value
    elif paradigm_normalized == "UE":
        behavior_model_name = BehaviorModelName.ROUTE_BASED_USER_EQUILIBRIUM.value
    else:
        raise ValueError(
            "Unsupported AssignmentParameters.Paradigm. "
            f"Expected 'SUE' or 'UE', received {paradigm!r}."
        )

    if method_normalized == "msa":
        solver_name = SolverName.MSA.value
    elif method_normalized == "frank_wolfe":
        solver_name = SolverName.FRANK_WOLFE.value
    else:
        raise ValueError(
            "Unsupported AssignmentParameters.Method. "
            f"Expected 'MSA' or 'frank_wolfe', received {method!r}."
        )

    if paradigm_normalized == "SUE" and solver_name != SolverName.MSA.value:
        raise ValueError("SUE scenario assignment currently supports only Method='MSA'.")

    assign_cfg["behavior_model"]["source"] = "explicit"
    assign_cfg["behavior_model"]["name"] = behavior_model_name
    assign_cfg["common"]["max_iterations"] = max_iterations
    assign_cfg["common"]["capacity_scaling"]["source"] = "explicit"
    assign_cfg["common"]["capacity_scaling"]["value"] = 1.0
    assign_cfg["common"]["capacity_scaling"]["training_config_path"] = None
    assign_cfg["solvers"]["active_solver"] = solver_name

    if solver_name == SolverName.MSA.value:
        assign_cfg["solvers"]["msa"]["step_rule"] = str(assignment_params.MSA.msa_step_rule)

    assign_cfg["route_based_user_equilibrium"]["policy"]["expected_solver"] = solver_name
    assign_cfg["stochastic_user_equilibrium"]["policy"]["expected_solver"] = solver_name
    assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_source"] = "explicit"
    assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_value"] = theta
    assign_cfg["stochastic_user_equilibrium"]["logit"]["theta_artifact_key"] = None

    if paradigm_normalized == "SUE":
        assign_cfg["stochastic_user_equilibrium"]["convergence"]["equilibrium_l1_threshold"] = float(
            convergence_params.equilibrium_l1_threshold
        )
        assign_cfg["stochastic_user_equilibrium"]["convergence"]["max_absolute_gap_threshold"] = float(
            convergence_params.max_absolute_gap_threshold
        )
        assign_cfg["stochastic_user_equilibrium"]["convergence"]["max_relative_gap_threshold"] = float(
            convergence_params.max_relative_gap_threshold
        )
        assign_cfg["stochastic_user_equilibrium"]["convergence"]["min_flow_for_relative_gap"] = float(
            convergence_params.min_flow_for_relative_gap
        )
    else:
        assign_cfg["route_based_user_equilibrium"]["convergence"]["relative_gap_threshold"] = float(
            convergence_params.max_relative_gap_threshold
        )

    assignment_config = build_assignment_config_from_mapping(assign_cfg)
    zone_id_to_idx = {
        int(zone_id): int(idx)
        for zone_id, idx in metadata["trips"]["zone_id_to_idx"].items()
    }

    assignment_matrix = _build_average_daily_matrix(data["trips"])

    composition = build_assignment_composition(
        links_df=data["network"],
        routes_by_od=data["routes"],
        zone_id_to_idx=zone_id_to_idx,
        assignment_config=assignment_config,
        training_config={},
        artifacts={},
    )

    assignment_result = composition.behavior_model.solve(
        od_matrix=assignment_matrix,
        config=composition.runtime_config,
    )

    assigned_flows = assignment_result.final_link_flows
    assignment_metadata = dict(assignment_result.metadata)
    assignment_metadata["final_travel_times"] = assignment_result.final_link_costs
    assignment_metadata["behavior_model"] = behavior_model_name
    assignment_metadata["solver"] = solver_name

    data["network"] = data["network"].copy()
    data["network"]["assigned_flow"] = assigned_flows
    data["network"]["travel_time"] = assignment_metadata["final_travel_times"]
    data["network"]["v/c"] = (
        data["network"]["assigned_flow"] / data["network"]["effective_capacity"].astype(float)
    )

    flows_tntp = data["network"][["init_node", "term_node"]].copy()
    flows_tntp["Volume"] = np.asarray(assigned_flows, dtype=float)
    flows_tntp = flows_tntp.rename(
        columns={
            "init_node": "From",
            "term_node": "To",
        }
    )

    flows_path = save_dataframe_as_tntp(flows_tntp, config.paths.export_filepaths.flows)
    data["flows"] = flows_tntp

    data["assigned_matrix"] = assignment_matrix
    data["assignment_config"] = assign_cfg
    data["zone_id_to_idx"] = zone_id_to_idx

    metadata: dict[str, Any] = {
        "flows_path": str(flows_path),
        "flows_columns": flows_tntp.columns.tolist(),
        "assignment_method": method,
        "assignment_paradigm": paradigm,
        "assignment_theta": theta,
        "effective_capacity_scaling_factor": 1.0,
        "capacity_multiplier": capacity_multiplier,
        "capacity_columns": {
            "capacity_per_lane": "capacity_per_lane",
            "total_capacity": "total_capacity",
            "effective_capacity": "effective_capacity",
        },
        "assignment_metadata": assignment_metadata,
    }
    return flows_tntp, metadata
