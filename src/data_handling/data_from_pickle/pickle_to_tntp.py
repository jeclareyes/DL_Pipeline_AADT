from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
import sys

if __package__ is None or __package__ == "":
    PROJECT_ROOT_BOOTSTRAP = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT_BOOTSTRAP) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT_BOOTSTRAP))

from data_handling.data_from_pickle.config_utils import find_project_root, load_config, resolve_project_path
from data_handling.data_from_pickle.graph_validation_utils import (
    validate_graph_topology_and_bidirectionality,
    validate_network_tntp_topology,
)
from data_handling.data_from_pickle.io_utils import (
    ensure_output_directories,
    load_reconstruction_inputs,
    save_dataframe_as_tntp,
    save_json_report,
    save_text_as_tntp,
)
from data_handling.data_from_pickle.route_generation_utils import build_routes_tntp_text
from data_handling.data_from_pickle.tntp_builders import (
    MISSING_REVERSE_LINK_ID,
    REQUIRED_FLOW_COLUMNS_PREFIX,
    build_flows_tntp,
    build_network_tntp,
    build_nodes_tntp,
    build_trips_tntp_text,
    repair_missing_reverse_links_in_link_data,
    select_network_export_columns,
    select_flows_export_columns,
)

logger = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
CONFIG = load_config(CONFIG_PATH)

SCENARIO_NAME = CONFIG["scenario"]["name"]
START_DATE = CONFIG["scenario"]["start_date"]
ENTRIES_PER_LINE = int(CONFIG["scenario"]["entries_per_line"])

GRAPH_PATH = resolve_project_path(CONFIG["paths"]["input"]["graph_path"], PROJECT_ROOT)
LINK_DATA_PATH = resolve_project_path(CONFIG["paths"]["input"]["link_data_path"], PROJECT_ROOT)
OD_MATRIX_PATH = resolve_project_path(CONFIG["paths"]["input"]["od_matrix_path"], PROJECT_ROOT)

OUTPUT_ROOT = resolve_project_path(CONFIG["paths"]["output"]["root"], PROJECT_ROOT)
INFO_DIR = resolve_project_path(CONFIG["paths"]["output"]["info_dir"], PROJECT_ROOT)

NODES_PATH = resolve_project_path(CONFIG["paths"]["output"]["nodes_filepath"], PROJECT_ROOT)
NETWORK_PATH = resolve_project_path(CONFIG["paths"]["output"]["network_filepath"], PROJECT_ROOT)
TRIPS_PATH = resolve_project_path(CONFIG["paths"]["output"]["trips_filepath"], PROJECT_ROOT)
ROUTES_PATH = resolve_project_path(CONFIG["paths"]["output"]["routes_filepath"], PROJECT_ROOT)
FLOWS_PATH = resolve_project_path(CONFIG["paths"]["output"]["flows_filepath"], PROJECT_ROOT)
MANIFEST_PATH = resolve_project_path(CONFIG["paths"]["output"]["manifest_filepath"], PROJECT_ROOT)

ROUTE_EXPORT = bool(CONFIG.get("routes", {}).get("export", True))
ROUTE_K_PATHS = int(CONFIG.get("routes", {}).get("k_routes", 10))
ROUTE_WEIGHT = str(CONFIG.get("routes", {}).get("weight", "free_flow_time"))
ROUTE_ALLOW_INTRAZONAL = bool(CONFIG.get("routes", {}).get("allow_intrazonal", False))
ROUTE_INTRAZONAL_POLICY = str(CONFIG.get("routes", {}).get("intrazonal_policy", "cycle"))
ROUTE_ALLOW_LOOPS = bool(CONFIG.get("routes", {}).get("allow_loops", False))
ROUTE_REQUIRE_EXACT_K = bool(CONFIG.get("routes", {}).get("require_exact_k_routes", True))
ROUTE_SHOW_PROGRESS = bool(CONFIG.get("routes", {}).get("show_progress", True))
ROUTE_PARALLEL = bool(CONFIG.get("routes", {}).get("parallel", False))
ROUTE_PARALLEL_WORKERS_RAW = CONFIG.get("routes", {}).get("parallel_workers", None)
ROUTE_PARALLEL_WORKERS = None if ROUTE_PARALLEL_WORKERS_RAW is None else int(ROUTE_PARALLEL_WORKERS_RAW)
ROUTE_OD_BATCH_SIZE = int(CONFIG.get("routes", {}).get("od_batch_size", 100))

VALIDATE_REVERSE_ATTRIBUTE_SYMMETRY = bool(CONFIG.get("validation", {}).get("reverse_attribute_symmetry", {}).get("enabled", True))
REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS = set(
    CONFIG.get("validation", {}).get("reverse_attribute_symmetry", {}).get("excluded_columns", [])
)
REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES = tuple(
    CONFIG.get("validation", {}).get("reverse_attribute_symmetry", {}).get("excluded_prefixes", [])
)

REQUIRED_LINK_DATA_COLUMNS = [
    "from_node",
    "to_node",
    "capacity",
    "lanes",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "VDF",
    "toll",
    "link_type",
]
REQUIRED_OD_MATRIX_KEYS = ["indices", "indptr", "format", "shape", "data"]


def print_reconstruction_input_summary(data: dict[str, Any]) -> None:
    graph = data["graph"]
    link_data = data["link_data"]
    od_matrix = data["od_matrix"]

    print("\nRECONSTRUCTION INPUT SUMMARY")
    print("=" * 80)
    print(f"Source object type: {data['_source_object_type']}")
    print("\nGraph:")
    print(f"  - Nodes: {graph.number_of_nodes()}")
    print(f"  - Edges: {graph.number_of_edges()}")
    print("\nNode table extracted from graph:")
    print(f"  - Shape: {list(data['nodes_gdf'].shape)}")
    print(f"  - Columns: {list(data['nodes_gdf'].columns)}")
    print("\nLink table extracted from graph:")
    print(f"  - Shape: {list(data['links_gdf'].shape)}")
    print(f"  - Columns: {list(data['links_gdf'].columns)}")
    print("\nLink parquet data:")
    print(f"  - Shape: {list(link_data.shape)}")
    print(f"  - Columns: {list(link_data.columns)}")
    volume_columns = [col for col in link_data.columns if str(col).startswith(REQUIRED_FLOW_COLUMNS_PREFIX)]
    print(f"  - Volume columns: {volume_columns}")
    print("\nOD matrix:")
    print(f"  - Keys: {list(od_matrix.keys())}")

    if "shape" in od_matrix:
        print(f"  - Shape: {tuple(od_matrix['shape'])}")

    if "data" in od_matrix:
        print(f"  - Stored values: {len(od_matrix['data'])}")

    topology_report = data.get("graph_topology_validation_report")
    if topology_report is not None:
        print("\nGraph topology validation:")
        print(f"  - Passed: {topology_report['passed']}")
        print(f"  - Weakly connected: {topology_report['is_weakly_connected']}")
        print(f"  - Strongly connected: {topology_report['is_strongly_connected']}")
        print(f"  - Weak components: {topology_report['num_weak_components']}")
        print(f"  - Strong components: {topology_report['num_strong_components']}")
        print(f"  - Isolated nodes sample: {topology_report['isolated_nodes']}")
        print(f"  - Missing reverse links sample: {topology_report['missing_reverse_links']}")
        print(f"  - Attribute mismatch sample count: {len(topology_report['attribute_mismatches'])}")

    print("=" * 80)


def validate_reconstruction_inputs(data: dict[str, Any]) -> dict[str, Any]:
    errors = []
    warnings = []

    graph = data["graph"]
    link_data = data["link_data"]
    od_matrix = data["od_matrix"]

    topology_report = validate_graph_topology_and_bidirectionality(
        graph=graph,
        link_data=link_data,
        validate_reverse_attribute_symmetry=VALIDATE_REVERSE_ATTRIBUTE_SYMMETRY,
        excluded_attribute_columns=REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS,
        excluded_attribute_prefixes=REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES,
    )
    data["graph_topology_validation_report"] = topology_report

    if topology_report["warnings"]:
        warnings.extend(topology_report["warnings"])
        for warning in topology_report["warnings"]:
            logging.warning(f"Graph topology validation warning: {warning}")

    if graph.number_of_edges() != len(link_data):
        errors.append(f"Graph edges ({graph.number_of_edges()}) and link_data rows ({len(link_data)}) do not match.")

    missing_link_cols = [col for col in REQUIRED_LINK_DATA_COLUMNS if col not in link_data.columns]
    if missing_link_cols:
        errors.append(f"Missing required link_data columns: {missing_link_cols}")

    volume_columns = [col for col in link_data.columns if str(col).startswith(REQUIRED_FLOW_COLUMNS_PREFIX)]
    if not volume_columns:
        errors.append("No Volume_{year} columns found in link_data.")

    missing_od_keys = [key for key in REQUIRED_OD_MATRIX_KEYS if key not in od_matrix]
    if missing_od_keys:
        errors.append(f"Missing OD matrix keys: {missing_od_keys}")

    if "shape" in od_matrix:
        shape = tuple(int(x) for x in od_matrix["shape"])
        if len(shape) != 2:
            errors.append(f"OD matrix shape must have length 2. Received: {shape}")
        if shape[0] != shape[1]:
            warnings.append(f"OD matrix is not square. Received shape: {shape}")

    report = {
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "graph_topology_validation_report": topology_report,
    }

    if errors:
        raise ValueError("Reconstruction input validation failed:\n" + "\n".join(f"- {error}" for error in errors))

    return report


def build_reconstruction_manifest(
    data: dict[str, Any],
    validation_report: dict[str, Any],
    reverse_link_repair_report: dict[str, Any],
    nodes_tntp,
    network_tntp,
    flows_tntp,
    nodes_path: Path,
    network_path: Path,
    flows_path: Path,
    trips_path: Path,
    routes_path: Path | None,
) -> dict[str, Any]:
    volume_columns = [column for column in flows_tntp.columns if str(column).startswith(REQUIRED_FLOW_COLUMNS_PREFIX)]

    return {
        "scenario": {
            "name": SCENARIO_NAME,
            "start_date": START_DATE,
        },
        "config": CONFIG,
        "input_files": {
            "graph_path": GRAPH_PATH,
            "link_data_path": LINK_DATA_PATH,
            "od_matrix_path": OD_MATRIX_PATH,
        },
        "output_files": {
            "nodes_path": nodes_path,
            "network_path": network_path,
            "flows_path": flows_path,
            "trips_path": trips_path,
            "routes_path": routes_path,
            "manifest_path": MANIFEST_PATH,
        },
        "input_validation_report": validation_report,
        "graph_topology_validation_report": data.get("graph_topology_validation_report"),
        "reverse_link_repair_report": reverse_link_repair_report,
        "export_network_topology_validation_report": data.get("export_network_topology_validation_report"),
        "nodes_summary": {
            "shape": list(nodes_tntp.shape),
            "columns": list(nodes_tntp.columns),
            "node_types": nodes_tntp["type"].value_counts().to_dict(),
            "node_classes": nodes_tntp["class"].value_counts().to_dict(),
        },
        "network_summary": {
            "shape": list(network_tntp.shape),
            "columns": list(network_tntp.columns),
            "missing_reverse_links": int((network_tntp["reverse_link_id"] == MISSING_REVERSE_LINK_ID).sum()),
            "link_types": network_tntp["link_type"].value_counts(dropna=False).to_dict(),
            "vdf_values": network_tntp["vdf"].value_counts(dropna=False).to_dict(),
        },
        "flows_summary": {
            "shape": list(flows_tntp.shape),
            "columns": list(flows_tntp.columns),
            "volume_columns": volume_columns,
            "volume_totals": {column: float(flows_tntp[column].sum(skipna=True)) for column in volume_columns},
        },
        "trips_reconstruction_metadata": data.get("trips_reconstruction_metadata"),
        "routes_reconstruction_metadata": data.get("routes_reconstruction_metadata"),
    }


def main() -> None:
    ensure_output_directories(OUTPUT_ROOT, INFO_DIR)

    data = load_reconstruction_inputs(
        graph_path=GRAPH_PATH,
        link_data_path=LINK_DATA_PATH,
        od_matrix_path=OD_MATRIX_PATH,
    )

    validation_report = validate_reconstruction_inputs(data)
    reverse_link_repair_report = repair_missing_reverse_links_in_link_data(data)

    print_reconstruction_input_summary(data)
    print("\nValidation report:")
    print(f"  - Passed: {validation_report['passed']}")

    if validation_report["warnings"]:
        print("  - Warnings:")
        for warning in validation_report["warnings"]:
            print(f"    * {warning}")

    logger.info("Building nodes.tntp...")
    nodes_tntp = build_nodes_tntp(data)
    nodes_path = save_dataframe_as_tntp(nodes_tntp, NODES_PATH)
    logger.info(f"Exported nodes.tntp to {nodes_path}")

    network_tntp = build_network_tntp(data)
    logger.info("Building network.tntp with repaired reverse links...")
    export_network_topology_report = validate_network_tntp_topology(
        network_tntp=network_tntp,
        weight_col=ROUTE_WEIGHT,
        validate_reverse_attribute_symmetry=VALIDATE_REVERSE_ATTRIBUTE_SYMMETRY,
        excluded_attribute_columns=REVERSE_ATTRIBUTE_EXCLUDED_COLUMNS,
        excluded_attribute_prefixes=REVERSE_ATTRIBUTE_EXCLUDED_PREFIXES,
    )
    data["export_network_topology_validation_report"] = export_network_topology_report

    if export_network_topology_report["warnings"]:
        print("\nExport network topology warnings:")
        for warning in export_network_topology_report["warnings"]:
            print(f"  * {warning}")
            logger.warning(f"Export network topology warning: {warning}")

    network_export_tntp = select_network_export_columns(network_tntp)
    network_path = save_dataframe_as_tntp(network_export_tntp, NETWORK_PATH)
    logger.info(f"Exported network.tntp to {network_path}")

    logger.info("Building flows.tntp from link_data flow columns...")
    flows_tntp = build_flows_tntp(data)
    flows_export_tntp = select_flows_export_columns(flows_tntp)
    flows_path = save_dataframe_as_tntp(flows_export_tntp, FLOWS_PATH)
    logger.info(f"Exported flows.tntp to {flows_path}")

    logger.info("Building trips.tntp...")
    trips_tntp_text = build_trips_tntp_text(
        data=data,
        nodes_tntp=nodes_tntp,
        # start_date=START_DATE,
        # entries_per_line=ENTRIES_PER_LINE,
    )
    trips_path = save_text_as_tntp(trips_tntp_text, TRIPS_PATH)
    logger.info(f"Exported trips.tntp to {trips_path}")

    routes_path = None
    if ROUTE_EXPORT:
        logger.info("Building routes.tntp from repaired network...")
        routes_tntp_text = build_routes_tntp_text(
            data=data,
            nodes_tntp=nodes_tntp,
            network_tntp=network_tntp,
            k_routes=ROUTE_K_PATHS,
            weight=ROUTE_WEIGHT,
            allow_intrazonal=ROUTE_ALLOW_INTRAZONAL,
            allow_loops=ROUTE_ALLOW_LOOPS,
            intrazonal_policy=ROUTE_INTRAZONAL_POLICY,
            require_exact_k_routes=ROUTE_REQUIRE_EXACT_K,
            show_progress=ROUTE_SHOW_PROGRESS,
            parallel=ROUTE_PARALLEL,
            parallel_workers=ROUTE_PARALLEL_WORKERS,
            od_batch_size=ROUTE_OD_BATCH_SIZE,
        )
        routes_path = save_text_as_tntp(routes_tntp_text, ROUTES_PATH)
        logger.info(f"Exported routes.tntp to {routes_path}")

    manifest = build_reconstruction_manifest(
        data=data,
        validation_report=validation_report,
        reverse_link_repair_report=reverse_link_repair_report,
        nodes_tntp=nodes_tntp,
        network_tntp=network_export_tntp,
        flows_tntp=flows_export_tntp,
        nodes_path=nodes_path,
        network_path=network_path,
        flows_path=flows_path,
        trips_path=trips_path,
        routes_path=routes_path,
        )
    manifest_path = save_json_report(payload=manifest, filepath=MANIFEST_PATH)
    logger.info(f"Exported reconstruction manifest to {manifest_path}")

    print("\nExported TNTP files:")
    print(f"  - Nodes: {nodes_path}")
    print(f"  - Network: {network_path}")
    print(f"  - Flows: {flows_path}")
    print(f"  - Trips: {trips_path}")
    print(f"  - Routes: {routes_path}")
    print(f"  - Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
