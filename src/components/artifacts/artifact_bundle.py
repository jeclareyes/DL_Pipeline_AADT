"""Convenience accessors for the artifact-bundle manifest."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .manifest_store import ManifestStore
from src.utils.serialization import load


class ArtifactBundle:
    """Load bundle-level metadata and the base artifact."""

    def __init__(self, manifest_path: str | Path):
        self.manifest_store = ManifestStore(manifest_path)

    @classmethod
    def load(cls, manifest_path: str | Path) -> "ArtifactBundle":
        return cls(manifest_path)

    def manifest(self) -> Dict[str, Any]:
        return self.manifest_store.read()

    def get_base_artifact(self) -> Dict[str, Any]:
        manifest = self.manifest()
        base_entry = manifest.get("base_artifact")
        if not isinstance(base_entry, dict):
            raise KeyError("Manifest does not contain a base_artifact entry.")

        base_path = base_entry.get("path")
        if not base_path:
            raise KeyError("Manifest base_artifact entry does not contain a path.")

        return load(Path(base_path))

    def get_base_artifact_entry(self) -> Dict[str, Any]:
        manifest = self.manifest()
        base_entry = manifest.get("base_artifact")
        if not isinstance(base_entry, dict):
            raise KeyError("Manifest does not contain a base_artifact entry.")
        return base_entry
