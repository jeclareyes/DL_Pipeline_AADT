"""Stage-independent artifact contracts.

These contracts intentionally describe lifecycle identity and provenance only.
The payload schema remains owned by each artifact stage while the migration is
completed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class ArtifactStage(StrEnum):
    """Canonical lifecycle stages for persisted artifacts."""

    BASE = "base_artifact"
    EXPERIMENT = "experiment_artifact"
    POST_TRAINING = "post_training_artifact"


@dataclass(frozen=True)
class ArtifactReference:
    """Reference from a downstream artifact to an upstream artifact."""

    stage: ArtifactStage | str
    path: str
    manifest_path: str | None = None
    artifact_version: str | None = None
    fingerprint: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": str(self.stage),
            "path": self.path,
            "manifest_path": self.manifest_path,
            "artifact_version": self.artifact_version,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True)
class ArtifactManifestContract:
    """Minimal common shape shared by all stage manifests."""

    schema_version: str
    artifact_stage: ArtifactStage | str
    dataset_name: str
    artifact_path: str
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "artifact_stage": str(self.artifact_stage),
            "dataset_name": self.dataset_name,
            "artifact_path": self.artifact_path,
            "created_at": self.created_at,
        }


def require_artifact_stage(
    artifact: Mapping[str, Any],
    expected: ArtifactStage,
    *,
    allow_legacy_training_artifact: bool = False,
) -> None:
    """Validate a stage marker while allowing the current migration alias."""

    actual = artifact.get("artifact_type")
    accepted = {expected.value}
    if allow_legacy_training_artifact and expected is ArtifactStage.EXPERIMENT:
        accepted.add("training_artifact")
    if actual not in accepted:
        raise ValueError(
            f"Expected artifact stage {expected.value!r}; received {actual!r}."
        )


def resolve_reference_path(reference: Mapping[str, Any]) -> Path:
    path = reference.get("path")
    if not path:
        raise KeyError("Artifact reference must contain a non-empty 'path'.")
    return Path(str(path))
