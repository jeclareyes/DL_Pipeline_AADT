"""Base-artifact manifest accessors."""

from src.components.artifacts.common.manifest import get_manifest_entry, read_manifest


def get_base_artifact_entry(manifest: dict):
    return get_manifest_entry(manifest, "base_artifact")


__all__ = ["get_base_artifact_entry", "read_manifest"]
