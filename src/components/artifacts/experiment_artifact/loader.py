"""Loader facade for experiment artifacts.

The legacy loader accepts ``training_artifact`` because that is the persisted
type currently emitted by AssetPipeline. The alias is intentionally isolated
here and will be removed when the stage rename is completed.
"""

from src.data_handling.data_processing.artifact_loaders.training_artifact_loader import (
    TrainingArtifactLoadResult,
    TrainingArtifactLoader,
)

ExperimentArtifactLoader = TrainingArtifactLoader
ExperimentArtifactLoadResult = TrainingArtifactLoadResult

__all__ = ["ExperimentArtifactLoadResult", "ExperimentArtifactLoader"]
