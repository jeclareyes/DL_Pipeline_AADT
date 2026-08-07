"""Base-artifact builder facade.

The implementation still lives in the data-processing package during phase 1.
This module establishes the stage-specific import boundary without duplicating
the builder or changing the data-processing entrypoint.
"""

from __future__ import annotations

from src.data_handling.data_processing.artifact_builders.training_artifact_builder import (
    TrainingArtifactBuilder,
    build_training_artifact,
)

BaseArtifactBuilder = TrainingArtifactBuilder


def build_base_artifact(*args, **kwargs):
    """Build a base artifact through the existing implementation."""

    return build_training_artifact(*args, **kwargs)


__all__ = ["BaseArtifactBuilder", "build_base_artifact"]
