"""Serialization helpers with a joblib-compatible fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:  # Prefer joblib when it is installed.
    import joblib as _joblib  # type: ignore
except Exception:  # pragma: no cover - fallback for minimal environments.
    _joblib = None

import pickle


def dump(obj: Any, path: str | Path) -> None:
    """Persist an object to disk using joblib when available, otherwise pickle."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _joblib is not None:
        _joblib.dump(obj, target)
        return
    with target.open("wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: str | Path) -> Any:
    """Load an object from disk using joblib when available, otherwise pickle."""

    source = Path(path)
    if _joblib is not None:
        return _joblib.load(source)
    with source.open("rb") as handle:
        return pickle.load(handle)

