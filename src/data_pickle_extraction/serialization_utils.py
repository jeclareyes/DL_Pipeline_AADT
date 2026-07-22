from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, pd.Series):
        return value.to_numpy()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy()
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return np.asarray(value)


def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.DataFrame):
        return {"__type__": "DataFrame", "shape": list(obj.shape), "columns": obj.columns.tolist()}
    if isinstance(obj, np.ndarray):
        return {"__type__": "ndarray", "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, dict):
        return {str(make_json_serializable(key)): make_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, tuple):
        return [make_json_serializable(item) for item in obj]
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    return obj
