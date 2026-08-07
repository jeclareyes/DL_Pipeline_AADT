"""Experiment-artifact manifest accessors."""

from src.components.artifacts.common.manifest import get_manifest_entry, read_manifest


def get_experiment_artifact_entry(manifest: dict):
    return get_manifest_entry(manifest, "experiment_artifact")


__all__ = ["get_experiment_artifact_entry", "read_manifest"]
