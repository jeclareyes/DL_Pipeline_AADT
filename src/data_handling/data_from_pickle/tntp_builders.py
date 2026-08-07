from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

try:
    from .od_utils import (
        get_zone_ids_from_nodes_tntp,
        reconstruct_dense_od_matrix_from_csr_npz,
        reconstruct_od_array_from_npz,
        compute_average_day_od_matrices,
    )
    from .config_utils import load_config
except (ImportError, ValueError):
    try:
        # pyrefly: ignore [missing-import]
        from data_handling.data_from_pickle.od_utils import (
            get_zone_ids_from_nodes_tntp,
            reconstruct_dense_od_matrix_from_csr_npz,
            reconstruct_od_array_from_npz,
            compute_average_day_od_matrices,
        )
        from data_handling.data_from_pickle.config_utils import load_config
    except ImportError:
        from od_utils import (  # type: ignore
            get_zone_ids_from_nodes_tntp,
            reconstruct_dense_od_matrix_from_csr_npz,
            reconstruct_od_array_from_npz,
            compute_average_day_od_matrices,
        )
        from config_utils import load_config  # type: ignore


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
CONFIG = load_config(CONFIG_PATH)
START_DATE = CONFIG["scenario"]["start_date"]
ENTRIES_PER_LINE = int(CONFIG["scenario"]["entries_per_line"])

TRIPS_EXPORT_FORMAT = str(
    CONFIG.get("trips", {}).get("export_format", "average_day_hourly")
)

TRIPS_AGGREGATION = str(
    CONFIG.get("trips", {}).get("aggregation", "mean_over_days")
)

TRIPS_START_DATE = str(
    CONFIG.get("trips", {}).get("start_date", START_DATE)
)

TRIPS_HOURS_PER_DAY = int(
    CONFIG.get("trips", {}).get("hours_per_day", 24)
)

BASE_NODE_COLUMNS = ["node_id", "x", "y", "type", "class"]


INTERNAL_NETWORK_COLUMNS = [
    "link_pos",
    "link_id",
    "reverse_link_pos",
    "reverse_link_id",
    "init_node",
    "term_node",
    "capacity",
    "lanes",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "vdf",
    "toll",
    "link_type",
]

EXPORT_NETWORK_COLUMNS = [
    "link_id",
    "reverse_link_id",
    "init_node",
    "term_node",
    "capacity",
    "lanes",
    "length",
    "free_flow_time",
    "b",
    "power",
    "speed",
    "vdf",
    "toll",
    "link_type",
]

EXPORT_FLOW_BASE_COLUMNS = ["From", "To"]

BASE_NODE_COLUMNS = ["node_id", "x", "y", "type", "class"]

REQUIRED_FLOW_COLUMNS_PREFIX = "Volume_"
DEFAULT_BPR_B = -1.0
DEFAULT_BPR_POWER = -1.0
DEFAULT_TOLL = 0.0
MISSING_REVERSE_LINK_ID = -1


def normalize_text_token(value: Any) -> str:
    if value is None:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip().upper()


def normalize_node_type(value: Any) -> str:
    token = normalize_text_token(value)
    if token in {"TAZ", "ZAT"}:
        return "TAZ"
    if token in {"AUX", "AUXILIAR", "AUXILIARY"}:
        return "AUX"
    if token in {"INT", "INTERSECTION"}:
        return "INTERSECTION"
    if token:
        return token
    return "INTERSECTION"


def infer_node_class(normalized_type: str) -> str:
    normalized_type = normalize_node_type(normalized_type)
    if normalized_type in {"TAZ", "AUX"}:
        return "Zones"
    return "Non-Zones"


def get_first_existing_column(df: pd.DataFrame, candidate_columns: list[str], target_name: str) -> pd.Series:
    for column in candidate_columns:
        if column in df.columns:
            return df[column]
    raise KeyError(
        f"Could not build '{target_name}'. None of these columns exist: {candidate_columns}. Available columns: {list(df.columns)}"
    )


def reorder_columns_with_base_first(df: pd.DataFrame, base_columns: list[str]) -> pd.DataFrame:
    existing_base_columns = [column for column in base_columns if column in df.columns]
    extra_columns = [column for column in df.columns if column not in existing_base_columns]
    return df[existing_base_columns + extra_columns]


def build_nodes_tntp(data: dict[str, Any]) -> pd.DataFrame:
    nodes_raw = data["nodes_gdf"].copy()
    node_id = get_first_existing_column(nodes_raw, ["node_id", "ID", "id"], "node_id")
    x_coord = get_first_existing_column(nodes_raw, ["x", "X"], "x")
    y_coord = get_first_existing_column(nodes_raw, ["y", "Y"], "y")

    if "type" in nodes_raw.columns:
        raw_type = nodes_raw["type"]
    else:
        raw_type = pd.Series(["INTERSECTION"] * len(nodes_raw), index=nodes_raw.index)

    nodes_tntp = pd.DataFrame({
        "node_id": node_id.astype(str),
        "x": pd.to_numeric(x_coord, errors="coerce"),
        "y": pd.to_numeric(y_coord, errors="coerce"),
        "type": raw_type.apply(normalize_node_type),
    })
    nodes_tntp["class"] = nodes_tntp["type"].apply(infer_node_class)

    for column in nodes_raw.columns:
        if column in {"ID", "id", "node_id", "x", "X", "y", "Y", "type"}:
            continue
        if column not in nodes_tntp.columns:
            nodes_tntp[column] = nodes_raw[column].values

    nodes_tntp.drop(columns=nodes_tntp.columns[~nodes_tntp.columns.isin(BASE_NODE_COLUMNS)], inplace=True)

    return reorder_columns_with_base_first(nodes_tntp, BASE_NODE_COLUMNS)


def repair_missing_reverse_links_in_link_data(data: dict[str, Any]) -> dict[str, Any]:
    if "link_data_repaired" in data and "reverse_link_repair_report" in data:
        return data["reverse_link_repair_report"]

    link_data = data["link_data"].copy()
    required_columns = {"from_node", "to_node"}
    missing_columns = required_columns - set(link_data.columns)
    if missing_columns:
        raise KeyError(f"Cannot repair reverse links. Missing columns: {sorted(missing_columns)}")

    pair_set = {(str(row["from_node"]), str(row["to_node"])) for _, row in link_data.iterrows()}
    original_num_links = len(link_data)
    synthetic_rows = []
    repaired_links = []

    for row_position, row in link_data.iterrows():
        from_node = str(row["from_node"])
        to_node = str(row["to_node"])
        reverse_pair = (to_node, from_node)

        if reverse_pair in pair_set:
            continue

        synthetic_row = row.copy()
        synthetic_row["from_node"] = row["to_node"]
        synthetic_row["to_node"] = row["from_node"]
        synthetic_row["_synthetic_reverse_link"] = True
        synthetic_row["_synthetic_reverse_source_link_id"] = int(row_position) + 1
        synthetic_row["_synthetic_reverse_source_from_node"] = from_node
        synthetic_row["_synthetic_reverse_source_to_node"] = to_node

        created_link_id = original_num_links + len(synthetic_rows) + 1
        synthetic_rows.append(synthetic_row)
        repaired_links.append(
            {
                "source_link_id": int(row_position) + 1,
                "source_from_node": from_node,
                "source_to_node": to_node,
                "created_link_id": int(created_link_id),
                "created_from_node": to_node,
                "created_to_node": from_node,
            }
        )
        pair_set.add(reverse_pair)

    link_data["_synthetic_reverse_link"] = False
    if synthetic_rows:
        repaired_link_data = pd.concat([link_data, pd.DataFrame(synthetic_rows)], ignore_index=True)
    else:
        repaired_link_data = link_data

    report = {
        "num_original_links": int(original_num_links),
        "num_created_reverse_links": int(len(synthetic_rows)),
        "num_repaired_links": int(len(repaired_link_data)),
        "repaired_links": repaired_links,
    }

    data["link_data_repaired"] = repaired_link_data
    data["reverse_link_repair_report"] = report
    return report


def get_link_data_for_export(data: dict[str, Any]) -> pd.DataFrame:
    return data.get("link_data_repaired", data["link_data"])


def infer_reverse_link_positions_and_ids(
    network_df: pd.DataFrame,
    init_col: str = "init_node",
    term_col: str = "term_node",
    link_pos_col: str = "link_pos",
    link_id_col: str = "link_id",
) -> tuple[pd.Series, pd.Series]:
    pair_to_link_refs: dict[tuple[str, str], list[tuple[int, int]]] = {}

    for row in network_df.itertuples(index=False):
        pair = (str(getattr(row, init_col)), str(getattr(row, term_col)))
        pair_to_link_refs.setdefault(pair, []).append((int(getattr(row, link_pos_col)), int(getattr(row, link_id_col))))

    reverse_positions = []
    reverse_ids = []

    for row in network_df.itertuples(index=False):
        reverse_pair = (str(getattr(row, term_col)), str(getattr(row, init_col)))
        candidates = pair_to_link_refs.get(reverse_pair, [])

        if candidates:
            reverse_pos, reverse_id = candidates[0]
            reverse_positions.append(reverse_pos)
            reverse_ids.append(reverse_id)
        else:
            reverse_positions.append(MISSING_REVERSE_LINK_ID)
            reverse_ids.append(MISSING_REVERSE_LINK_ID)

    return (
        pd.Series(reverse_positions, index=network_df.index, dtype="int64"),
        pd.Series(reverse_ids, index=network_df.index, dtype="int64"),
    )


def build_network_tntp(data: dict[str, Any]) -> pd.DataFrame:
    link_data = get_link_data_for_export(data).copy()
    network_tntp = pd.DataFrame(index=link_data.index)

    network_tntp["link_pos"] = np.arange(len(link_data), dtype=int)
    network_tntp["link_id"] = network_tntp["link_pos"] + 1
    network_tntp["init_node"] = link_data["from_node"].astype(str)
    network_tntp["term_node"] = link_data["to_node"].astype(str)
    network_tntp["capacity"] = pd.to_numeric(link_data["capacity"], errors="coerce")
    network_tntp["lanes"] = pd.to_numeric(link_data["lanes"], errors="coerce")
    network_tntp["length"] = pd.to_numeric(link_data["length"], errors="coerce")
    network_tntp["free_flow_time"] = pd.to_numeric(link_data["free_flow_time"], errors="coerce")
    network_tntp["b"] = pd.to_numeric(link_data["b"], errors="coerce").fillna(DEFAULT_BPR_B)
    network_tntp["power"] = pd.to_numeric(link_data["power"], errors="coerce").fillna(DEFAULT_BPR_POWER)
    network_tntp["speed"] = pd.to_numeric(link_data["speed"], errors="coerce")
    network_tntp["vdf"] = pd.to_numeric(link_data["VDF"], errors="coerce").astype("Int64")
    network_tntp["toll"] = pd.to_numeric(link_data["toll"], errors="coerce").fillna(DEFAULT_TOLL)
    network_tntp["link_type"] = pd.to_numeric(link_data["link_type"], errors="coerce").astype("Int64")
    network_tntp["reverse_link_pos"], network_tntp["reverse_link_id"] = infer_reverse_link_positions_and_ids(network_tntp)

    return network_tntp[INTERNAL_NETWORK_COLUMNS]


def build_flows_tntp(data: dict[str, Any]) -> pd.DataFrame:
    link_data = get_link_data_for_export(data).copy()

    volume_columns = [column for column in link_data.columns if str(column).startswith(REQUIRED_FLOW_COLUMNS_PREFIX)]
    if not volume_columns:
        raise ValueError(f"No flow columns starting with '{REQUIRED_FLOW_COLUMNS_PREFIX}' were found in link_data.")

    def _volume_year_key(column: str) -> int:
        try:
            return int(str(column).replace(REQUIRED_FLOW_COLUMNS_PREFIX, ""))
        except ValueError:
            return 10**9

    volume_columns = sorted(volume_columns, key=_volume_year_key)

    flows_tntp = pd.DataFrame(
        {
            "link_pos": np.arange(len(link_data), dtype=int),
            "link_id": np.arange(1, len(link_data) + 1, dtype=int),
            "From": link_data["from_node"].astype(str),
            "To": link_data["to_node"].astype(str),
        }
    )

    for column in volume_columns:
        flows_tntp[column] = pd.to_numeric(link_data[column], errors="coerce")

    return reorder_columns_with_base_first(flows_tntp, EXPORT_FLOW_BASE_COLUMNS)

def select_network_export_columns(network_tntp: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in EXPORT_NETWORK_COLUMNS
        if column not in network_tntp.columns
    ]
    if missing_columns:
        raise KeyError(
            f"Cannot export network.tntp. Missing columns: {missing_columns}"
        )

    return network_tntp[EXPORT_NETWORK_COLUMNS].copy()


def select_flows_export_columns(flows_tntp: pd.DataFrame) -> pd.DataFrame:
    volume_columns = [
        column for column in flows_tntp.columns
        if str(column).startswith(REQUIRED_FLOW_COLUMNS_PREFIX)
    ]

    if not volume_columns:
        raise ValueError(
            f"Cannot export flows.tntp. No columns starting with "
            f"'{REQUIRED_FLOW_COLUMNS_PREFIX}' were found."
        )

    return flows_tntp[EXPORT_FLOW_BASE_COLUMNS + volume_columns].copy()

def format_static_od_matrix_as_tntp_deprecated(
        od_matrix: np.ndarray,
        zone_ids: list[str],
        start_date: str,
        entries_per_line: int = 6,
    ) -> str:
    if od_matrix.shape[0] != len(zone_ids) or od_matrix.shape[1] != len(zone_ids):
        raise ValueError(
            "OD matrix shape and number of zone IDs do not match. "
            f"OD shape={od_matrix.shape}, num_zone_ids={len(zone_ids)}"
        )

    parsed_date = datetime.strptime(start_date, "%Y-%m-%d")
    matrix_date = parsed_date.strftime("%Y-%m-%d")

    lines = [
        f"<NUMBER OF ZONES> {len(zone_ids)}",
        "<END OF METADATA>",
        "",
        f"<MATRIX DATE> {matrix_date} 00:00",
    ]

    for origin_idx, origin_id in enumerate(zone_ids):
        lines.append(f"Origin {origin_id}")
        entries = []

        for destination_idx, destination_id in enumerate(zone_ids):
            flow = float(od_matrix[origin_idx, destination_idx])
            entries.append(f"{destination_id} : {flow:.2f};")

        for i in range(0, len(entries), entries_per_line):
            lines.append("    " + " ".join(entries[i:i + entries_per_line]))

    lines.append("")
    return "\n".join(lines)

def format_od_matrices_as_tntp(
    od_matrices: np.ndarray,
    zone_ids: list[str],
    matrix_date: str,
    entries_per_line: int = 6,
    start_hour: int = 0,
) -> str:
    """
    Format one or more OD matrices as TNTP-like trips text.

    Expected input:
        od_matrices shape = [num_matrices, zones, zones]

    For an average day, num_matrices is usually 24.
    """
    od_matrices = np.asarray(od_matrices, dtype=float)

    if od_matrices.ndim != 3:
        raise ValueError(
            "od_matrices must have shape [num_matrices, zones, zones]. "
            f"Received shape={od_matrices.shape}."
        )

    num_matrices, num_origins, num_destinations = od_matrices.shape

    if num_origins != len(zone_ids) or num_destinations != len(zone_ids):
        raise ValueError(
            "OD matrix shape and number of zone IDs do not match. "
            f"OD shape={od_matrices.shape}, num_zone_ids={len(zone_ids)}."
        )

    lines = [
        f"<NUMBER OF ZONES> {len(zone_ids)}",
        "<END OF METADATA>",
        "",
    ]

    for matrix_idx in range(num_matrices):
        hour = (start_hour + matrix_idx) % 24
        matrix_time = f"{hour:02d}:00"

        lines.append(f"<MATRIX DATE> {matrix_date} {matrix_time}")

        od_matrix = od_matrices[matrix_idx]

        for origin_idx, origin_id in enumerate(zone_ids):
            lines.append(f"Origin {origin_id}")

            entries = []

            for destination_idx, destination_id in enumerate(zone_ids):
                flow = float(od_matrix[origin_idx, destination_idx])
                entries.append(f"{destination_id} : {flow:.2f};")

            for i in range(0, len(entries), entries_per_line):
                lines.append("    " + " ".join(entries[i:i + entries_per_line]))

        lines.append("")

    return "\n".join(lines)

def build_trips_tntp_text(
        data: dict[str, Any],
        nodes_tntp: pd.DataFrame,
    ) -> str:
    """
    Build trips.tntp text from the OD matrix/tensor .npz.

    The preferred output is an average-day hourly TNTP file:
        24 matrices, one per hour, averaged across all available days.
    """
    od_array = reconstruct_od_array_from_npz(data["od_matrix"])
    zone_ids = get_zone_ids_from_nodes_tntp(nodes_tntp)

    if TRIPS_EXPORT_FORMAT == "average_day_hourly":
        od_matrices = compute_average_day_od_matrices(
            od_array=od_array,
            hours_per_day=TRIPS_HOURS_PER_DAY,
            aggregation=TRIPS_AGGREGATION,
        )

        if TRIPS_EXPORT_FORMAT == "average_day_hourly" and od_matrices.shape[0] != TRIPS_HOURS_PER_DAY:
            raise ValueError(
                "trips.export_format='average_day_hourly' requires the exported OD "
                f"array to contain {TRIPS_HOURS_PER_DAY} hourly matrices. "
                f"Received shape={od_matrices.shape}. This usually means the input "
                "OD .npz does not contain temporal daily/hourly matrices."
            )

        trips_text = format_od_matrices_as_tntp(
            od_matrices=od_matrices,
            zone_ids=zone_ids,
            matrix_date=TRIPS_START_DATE,
            entries_per_line=ENTRIES_PER_LINE,
            start_hour=0,
        )

        data["trips_reconstruction_metadata"] = {
            "source": "od_matrix_npz",
            "format": "average_day_hourly",
            "aggregation": TRIPS_AGGREGATION,
            "matrix_date": TRIPS_START_DATE,
            "raw_od_array_shape": list(od_array.shape),
            "exported_od_array_shape": list(od_matrices.shape),
            "hours_per_day": int(TRIPS_HOURS_PER_DAY),
            "num_exported_matrices": int(od_matrices.shape[0]),
            "num_zones": len(zone_ids),
            "num_nonzero_od_values": int(np.count_nonzero(od_matrices)),
            "total_average_day_demand": float(np.nansum(od_matrices)),
            "zone_ids": zone_ids,
        }

        return trips_text

    if TRIPS_EXPORT_FORMAT == "single_static_matrix":
        if od_array.ndim != 2:
            raise ValueError(
                "TRIPS_EXPORT_FORMAT='single_static_matrix' requires a 2D OD "
                f"matrix. Received shape={od_array.shape}."
            )

        od_matrices = od_array[None, :, :]

        trips_text = format_od_matrices_as_tntp(
            od_matrices=od_matrices,
            zone_ids=zone_ids,
            matrix_date=TRIPS_START_DATE,
            entries_per_line=ENTRIES_PER_LINE,
            start_hour=0,
        )

        data["trips_reconstruction_metadata"] = {
            "source": "od_matrix_npz",
            "format": "single_static_matrix",
            "matrix_date": f"{TRIPS_START_DATE} 00:00",
            "raw_od_array_shape": list(od_array.shape),
            "exported_od_array_shape": list(od_matrices.shape),
            "num_exported_matrices": 1,
            "num_zones": len(zone_ids),
            "num_nonzero_od_values": int(np.count_nonzero(od_matrices)),
            "total_demand": float(np.nansum(od_matrices)),
            "zone_ids": zone_ids,
        }

        return trips_text

    raise ValueError(
        f"Unsupported trips.export_format='{TRIPS_EXPORT_FORMAT}'. "
        "Use 'average_day_hourly' or 'single_static_matrix'."
    )

def build_export_graph_from_network_tntp(network_tntp: pd.DataFrame, weight_col: str = "free_flow_time") -> nx.DiGraph:
    graph = nx.DiGraph()

    for row in network_tntp.itertuples(index=False):
        edge_attrs = {
            "link_pos": int(row.link_pos),
            "link_id": int(row.link_id),
            "reverse_link_pos": int(row.reverse_link_pos),
            "reverse_link_id": int(row.reverse_link_id),
            "capacity": float(row.capacity),
            "lanes": float(row.lanes),
            "length": float(row.length),
            "free_flow_time": float(row.free_flow_time),
            "b": float(row.b),
            "power": float(row.power),
            "speed": float(row.speed),
            "vdf": int(row.vdf) if not pd.isna(row.vdf) else -1,
            "toll": float(row.toll),
            "link_type": int(row.link_type) if not pd.isna(row.link_type) else -1,
        }

        if weight_col not in edge_attrs:
            raise KeyError(f"Weight column '{weight_col}' is not available in edge attributes.")

        graph.add_edge(str(row.init_node), str(row.term_node), **edge_attrs)

    return graph
