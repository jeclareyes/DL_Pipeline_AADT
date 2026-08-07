"""Loader for the base-artifact stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from src.components.artifacts.common.contracts import ArtifactStage, require_artifact_stage
from src.utils.serialization import load


@dataclass(frozen=True)
class BaseArtifactLoadResult:
    artifact: Dict[str, Any]
    artifact_path: Path
    metadata: Dict[str, Any]


class BaseArtifactLoader:
    """Load and minimally identify a persisted base artifact."""

    def __init__(self, artifact_path: str | Path):
        self.artifact_path = Path(artifact_path)

    def load(self) -> BaseArtifactLoadResult:
        artifact = load(self.artifact_path)
        if not isinstance(artifact, dict):
            raise TypeError("Base artifact payload must be a dictionary.")
        require_artifact_stage(artifact, ArtifactStage.BASE)
        metadata = artifact.get("metadata", {})
        if not isinstance(metadata, dict):
            raise TypeError("Base artifact metadata must be a dictionary.")
        return BaseArtifactLoadResult(
            artifact=artifact,
            artifact_path=self.artifact_path,
            metadata=metadata,
        )

    def load_artifact(self) -> Dict[str, Any]:
        return self.load().artifact


__all__ = ["BaseArtifactLoadResult", "BaseArtifactLoader"]
