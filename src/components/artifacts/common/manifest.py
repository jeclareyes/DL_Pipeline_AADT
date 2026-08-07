"""Common manifest access helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .manifest_store import ManifestStore


def read_manifest(path: str | Path) -> dict[str, Any]:
    data = ManifestStore(path).read()
    if not isinstance(data, dict):
        raise TypeError("Artifact manifest must contain a JSON object.")
    return data


def get_manifest_entry(
    manifest: Mapping[str, Any],
    entry_name: str,
) -> dict[str, Any]:
    entry = manifest.get(entry_name)
    if not isinstance(entry, dict):
        raise KeyError(f"Manifest does not contain a {entry_name!r} entry.")
    return entry


__all__ = ["get_manifest_entry", "read_manifest"]
