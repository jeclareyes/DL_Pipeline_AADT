from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .serialization_utils import to_numpy
except ImportError:
    from serialization_utils import to_numpy  # type: ignore



# This to be deprecated
def reconstruct_dense_od_matrix_from_csr_npz(od_matrix_data: dict[str, np.ndarray]) -> np.ndarray:
    required_keys = {"data", "indices", "indptr", "shape"}
    missing_keys = required_keys - set(od_matrix_data.keys())
    if missing_keys:
        raise KeyError(f"Cannot reconstruct OD matrix. Missing keys: {sorted(missing_keys)}")

    values = to_numpy(od_matrix_data["data"]).astype(float)
    indices = to_numpy(od_matrix_data["indices"]).astype(int)
    indptr = to_numpy(od_matrix_data["indptr"]).astype(int)
    shape = tuple(int(item) for item in to_numpy(od_matrix_data["shape"]))

    if len(shape) != 2:
        raise ValueError(f"OD matrix shape must be 2D. Received: {shape}")

    dense_matrix = np.zeros(shape, dtype=float)
    for row_idx in range(shape[0]):
        start = indptr[row_idx]
        end = indptr[row_idx + 1]
        dense_matrix[row_idx, indices[start:end]] = values[start:end]
    return dense_matrix

def reconstruct_od_array_from_npz(
    od_matrix_data: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Reconstruct an OD array from .npz data.

    Supported cases:
    1. CSR-like 2D matrix:
       keys = data, indices, indptr, shape

    2. Dense array stored directly:
       key = od_matrix, matrix, matrices, demand, or data
    """
    # Case 1: CSR matrix
    csr_keys = {"data", "indices", "indptr", "shape"}

    if csr_keys.issubset(od_matrix_data.keys()):
        shape = tuple(int(item) for item in to_numpy(od_matrix_data["shape"]))

        # Standard 2D CSR
        if len(shape) == 2:
            return reconstruct_dense_od_matrix_from_csr_npz(od_matrix_data)

        raise ValueError(
            "CSR-like OD reconstruction currently supports only 2D matrices. "
            f"Received shape={shape}. If this file contains temporal OD data, "
            "it should be stored as a dense 3D or 4D array."
        )

    # Case 2: Dense array under common keys
    candidate_keys = [
        "od_matrix",
        "matrix",
        "matrices",
        "demand",
        "od_matrices",
        "od_array",
        "data",
    ]

    for key in candidate_keys:
        if key in od_matrix_data:
            return to_numpy(od_matrix_data[key]).astype(float)

    raise KeyError(
        "Could not reconstruct OD array from .npz. Expected CSR keys "
        "{'data', 'indices', 'indptr', 'shape'} or one dense OD key among: "
        f"{candidate_keys}. Available keys: {list(od_matrix_data.keys())}"
    )

def compute_average_day_od_matrices(
    od_array: np.ndarray,
    hours_per_day: int = 24,
    aggregation: str = "mean_over_days",
) -> np.ndarray:
    """
    Compute average-day hourly OD matrices.

    Supported input shapes
    ----------------------
    2D:
        [zones, zones]
        Returns shape [1, zones, zones].

    3D:
        [time_steps, zones, zones]
        If time_steps is divisible by hours_per_day, reshapes to:
        [days, hours, zones, zones]
        and averages over days.

    4D:
        [days, hours, zones, zones]
        Averages over days.

    Returns
    -------
    np.ndarray
        Array with shape [num_matrices, zones, zones].
        Usually [24, zones, zones] for an average day.
    """
    od_array = np.asarray(od_array, dtype=float)

    if aggregation != "mean_over_days":
        raise ValueError(
            f"Unsupported trips aggregation='{aggregation}'. "
            "Currently supported: 'mean_over_days'."
        )

    if od_array.ndim == 2:
        return od_array[None, :, :]

    if od_array.ndim == 3:
        time_steps, num_origins, num_destinations = od_array.shape

        if num_origins != num_destinations:
            raise ValueError(
                "OD array must be square in its last two dimensions. "
                f"Received shape={od_array.shape}."
            )

        if time_steps % hours_per_day != 0:
            raise ValueError(
                "Cannot compute average day from 3D OD array because the "
                f"number of time steps ({time_steps}) is not divisible by "
                f"hours_per_day ({hours_per_day})."
            )

        num_days = time_steps // hours_per_day

        od_by_day_hour = od_array.reshape(
            num_days,
            hours_per_day,
            num_origins,
            num_destinations,
        )

        return np.nanmean(od_by_day_hour, axis=0)

    if od_array.ndim == 4:
        num_days, num_hours, num_origins, num_destinations = od_array.shape

        if num_hours != hours_per_day:
            raise ValueError(
                f"Expected {hours_per_day} hours per day, but received "
                f"{num_hours} in OD array shape={od_array.shape}."
            )

        if num_origins != num_destinations:
            raise ValueError(
                "OD array must be square in its last two dimensions. "
                f"Received shape={od_array.shape}."
            )

        return np.nanmean(od_array, axis=0)

    raise ValueError(
        "Unsupported OD array dimensionality. Expected 2D, 3D or 4D array. "
        f"Received shape={od_array.shape}."
    )

def get_zone_ids_from_nodes_tntp(nodes_tntp: pd.DataFrame) -> list[str]:
    if "class" not in nodes_tntp.columns:
        raise KeyError("nodes_tntp must contain a 'class' column.")
    if "node_id" not in nodes_tntp.columns:
        raise KeyError("nodes_tntp must contain a 'node_id' column.")

    zone_ids = nodes_tntp.loc[nodes_tntp["class"] == "Zones", "node_id"].astype(str).tolist()
    if not zone_ids:
        raise ValueError("No zone nodes were found in nodes_tntp. Cannot build trips.tntp without zone IDs.")
    return zone_ids
