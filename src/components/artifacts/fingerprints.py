"""Deterministic fingerprint helpers for artifact bundles and assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import sparse


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("Fingerprints cannot be computed from non-finite float values.")
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        if not np.isfinite(number):
            raise ValueError("Fingerprints cannot be computed from non-finite float values.")
        return number
    if isinstance(value, np.ndarray):
        return {
            "__type__": "ndarray",
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype),
            "values": value.tolist(),
        }
    if sparse.issparse(value):
        matrix = value.tocoo()
        return {
            "__type__": value.__class__.__name__,
            "shape": tuple(int(dim) for dim in value.shape),
            "dtype": str(value.dtype),
            "row": matrix.row.tolist(),
            "col": matrix.col.tolist(),
            "data": matrix.data.tolist(),
        }
    if isinstance(value, pd.DataFrame):
        return _canonicalize_dataframe(value)
    if isinstance(value, pd.Series):
        return {
            "__type__": "Series",
            "name": value.name,
            "index": [_canonicalize(item) for item in value.index.tolist()],
            "values": [_canonicalize(item) for item in value.tolist()],
        }
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set):
        return sorted(_canonicalize(item) for item in value)
    return str(value)


def _canonicalize_dataframe(frame: pd.DataFrame) -> Dict[str, Any]:
    columns = list(frame.columns)
    records = []
    for row in frame.itertuples(index=False, name=None):
        records.append([_canonicalize(value) for value in row])
    return {
        "__type__": "DataFrame",
        "shape": (int(frame.shape[0]), int(frame.shape[1])),
        "columns": [str(column) for column in columns],
        "records": records,
    }


def compute_network_fingerprint(link_df: Any, node_df: Any) -> str:
    """Fingerprint a network from node and link tables."""

    payload = {
        "links": _canonicalize(link_df),
        "nodes": _canonicalize(node_df),
    }
    return _hash_payload(payload)


def compute_od_space_fingerprint(od_matrix: Any) -> str:
    """Fingerprint the OD space and demand structure."""

    payload = {"od_space": _canonicalize(od_matrix)}
    return _hash_payload(payload)


def compute_link_order_fingerprint(link_order: Sequence[Sequence[int]]) -> str:
    """Fingerprint a canonical link order."""

    payload = {"link_order": _canonicalize(list(link_order))}
    return _hash_payload(payload)


def compute_zone_order_fingerprint(zone_ids: Sequence[int]) -> str:
    """Fingerprint a canonical zone order."""

    payload = {"zone_order": _canonicalize([int(zone_id) for zone_id in zone_ids])}
    return _hash_payload(payload)


def compute_route_set_signature(spec: Mapping[str, Any]) -> str:
    """Fingerprint a route-set recipe without binding it to a dataset."""

    return _hash_payload(_canonicalize(spec))


def compute_route_set_fingerprint(signature: Dict[str, Any], network_fp: str, od_fp: str) -> str:
    """Fingerprint a route-set asset bound to a base artifact."""

    combined = {
        "signature": _canonicalize(signature),
        "network_fingerprint": network_fp,
        "od_space_fingerprint": od_fp,
    }
    return _hash_payload(combined)


def compute_assignment_set_signature(spec: Mapping[str, Any]) -> str:
    """Fingerprint an assignment-set recipe without binding it to a dataset."""

    return _hash_payload(_canonicalize(spec))


def compute_assignment_set_fingerprint(signature: Dict[str, Any], base_fingerprint: str, route_set_fingerprint: str) -> str:
    """Fingerprint an assignment-set recipe bound to a base artifact and route set."""

    combined = {
        "signature": _canonicalize(signature),
        "base_fingerprint": base_fingerprint,
        "route_set_fingerprint": route_set_fingerprint,
    }
    return _hash_payload(combined)
