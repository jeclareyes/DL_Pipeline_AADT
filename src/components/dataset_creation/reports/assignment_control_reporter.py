from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def run_assignment_control(data: dict[str, Any], config, metadata: dict[str, Any]) -> dict[str, Any]:
    if data.get("flows") is None or data["flows"].empty or not metadata.get("flows"):
        raise ValueError("Flows metadata is missing. Run assignment before control checks.")

    control_cfg = config.AssignmentParameters.AssignmentControl
    demand_abs_tol = float(control_cfg.DemandBalance.absolute_tolerance)
    demand_rel_tol = float(control_cfg.DemandBalance.relative_tolerance)
    zone_abs_tol = float(control_cfg.ZoneNodeBalance.absolute_tolerance)
    zone_rel_tol = float(control_cfg.ZoneNodeBalance.relative_tolerance)
    nonzone_abs_tol = float(control_cfg.NonZoneNodeBalance.absolute_tolerance)
    nonzone_rel_tol = float(control_cfg.NonZoneNodeBalance.relative_tolerance)
    eps = 1e-9
    alerts: list[dict[str, Any]] = []

    nodes_df = data["nodes"].copy()
    assignment_matrix = data["assigned_matrix"]
    zone_ids = [int(zone_id) for zone_id in metadata["trips"]["zone_ids"]]
    idx_to_zone_id = {
        int(idx): int(zone_id)
        for idx, zone_id in metadata["trips"]["idx_to_zone_id"].items()
    }

    generated_by_zone = np.nansum(assignment_matrix, axis=1)
    attracted_by_zone = np.nansum(assignment_matrix, axis=0)
    total_generated = float(np.nansum(generated_by_zone))
    total_attracted = float(np.nansum(attracted_by_zone))
    demand_abs_gap = abs(total_generated - total_attracted)
    demand_rel_gap = demand_abs_gap / max(abs(total_generated), abs(total_attracted), eps)

    if demand_abs_gap > demand_abs_tol or demand_rel_gap > demand_rel_tol:
        alerts.append(
            {
                "criterion": "GLOBAL_DEMAND_BALANCE",
                "node_id": "ALL_ZONES",
                "node_type": "ZONE_SET",
                "node_class": "Zones",
                "absolute_gap": demand_abs_gap,
                "relative_gap": demand_rel_gap,
                "absolute_threshold": demand_abs_tol,
                "relative_threshold": demand_rel_tol,
                "details": (
                    f"Total generated demand ({total_generated:.6f}) differs from "
                    f"total attracted demand ({total_attracted:.6f})."
                ),
            }
        )

    flows = data["flows"].copy()
    flows["From"] = flows["From"].astype(int)
    flows["To"] = flows["To"].astype(int)
    flows["Volume"] = flows["Volume"].astype(float)
    outflow_by_node = flows.groupby("From")["Volume"].sum()
    inflow_by_node = flows.groupby("To")["Volume"].sum()

    for _, node in nodes_df.iterrows():
        node_id = int(node["node_id"])
        node_type = node["type"]
        node_class = node["class"]
        assigned_outflow = float(outflow_by_node.get(node_id, 0.0))
        assigned_inflow = float(inflow_by_node.get(node_id, 0.0))
        assigned_net_outflow = assigned_outflow - assigned_inflow

        if node_class == "Zones":
            if node_id not in zone_ids:
                continue
            zone_idx = metadata["trips"]["zone_id_to_idx"][node_id]
            generated = float(generated_by_zone[zone_idx])
            attracted = float(attracted_by_zone[zone_idx])
            expected_net_outflow = generated - attracted
            absolute_gap = abs(assigned_net_outflow - expected_net_outflow)
            relative_gap = absolute_gap / max(generated + attracted, eps)
            if absolute_gap > zone_abs_tol or relative_gap > zone_rel_tol:
                alerts.append(
                    {
                        "criterion": "ZONE_NODE_BALANCE",
                        "node_id": node_id,
                        "node_type": node_type,
                        "node_class": node_class,
                        "absolute_gap": absolute_gap,
                        "relative_gap": relative_gap,
                        "absolute_threshold": zone_abs_tol,
                        "relative_threshold": zone_rel_tol,
                        "details": (
                            f"assigned_net_outflow={assigned_net_outflow:.6f}, "
                            f"expected_net_outflow={expected_net_outflow:.6f}, "
                            f"generated={generated:.6f}, attracted={attracted:.6f}"
                        ),
                    }
                )
        else:
            absolute_gap = abs(assigned_net_outflow)
            throughput = assigned_inflow + assigned_outflow
            relative_gap = absolute_gap / max(throughput, eps)
            if absolute_gap > nonzone_abs_tol or relative_gap > nonzone_rel_tol:
                alerts.append(
                    {
                        "criterion": "NONZONE_NODE_BALANCE",
                        "node_id": node_id,
                        "node_type": node_type,
                        "node_class": node_class,
                        "absolute_gap": absolute_gap,
                        "relative_gap": relative_gap,
                        "absolute_threshold": nonzone_abs_tol,
                        "relative_threshold": nonzone_rel_tol,
                        "details": (
                            f"assigned_inflow={assigned_inflow:.6f}, "
                            f"assigned_outflow={assigned_outflow:.6f}, "
                            f"assigned_net_outflow={assigned_net_outflow:.6f}"
                        ),
                    }
                )

    report_path = Path(config.paths.export_dirs.info_dataset) / "assignment_control_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ASSIGNMENT CONTROL REPORT",
        "=" * 80,
        f"Scenario: {config.name}",
        f"Number of alerts: {len(alerts)}",
        "",
        "Control thresholds:",
        f"- Global demand balance: abs <= {demand_abs_tol}, rel <= {demand_rel_tol}",
        f"- Zone node balance: abs <= {zone_abs_tol}, rel <= {zone_rel_tol}",
        f"- Non-zone node balance: abs <= {nonzone_abs_tol}, rel <= {nonzone_rel_tol}",
        "",
    ]
    if not alerts:
        lines.append("All assignment control checks passed.")
    else:
        lines.append("FAILED CHECKS")
        lines.append("-" * 80)
        for i, alert in enumerate(alerts, start=1):
            lines.extend(
                [
                    f"[{i}] Criterion: {alert['criterion']}",
                    f"    Node ID: {alert['node_id']}",
                    f"    Node type: {alert['node_type']}",
                    f"    Node class: {alert['node_class']}",
                    f"    Absolute gap: {alert['absolute_gap']:.6f}",
                    f"    Absolute threshold: {alert['absolute_threshold']:.6f}",
                    f"    Relative gap: {alert['relative_gap']:.6f}",
                    f"    Relative threshold: {alert['relative_threshold']:.6f}",
                    f"    Details: {alert['details']}",
                    "",
                ]
            )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "passed": len(alerts) == 0,
        "num_alerts": len(alerts),
        "alerts": alerts,
        "report_path": str(report_path),
    }
