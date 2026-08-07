"""Post-training-artifact manifest accessors."""

from src.components.artifacts.common.manifest import get_manifest_entry, read_manifest


def get_post_training_artifact_entry(manifest: dict):
    return get_manifest_entry(manifest, "post_training_artifact")


__all__ = ["get_post_training_artifact_entry", "read_manifest"]
