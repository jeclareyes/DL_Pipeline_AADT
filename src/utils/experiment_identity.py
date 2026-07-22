"""Experiment identity helpers.

This module computes stable experiment identifiers and artifact locations from
the resolved experiment overlay and the current base artifact manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from src.utils.paths import resolve_path


@dataclass(frozen=True)
class ExperimentArtifactIdentity:
    """Resolved experiment identity and artifact paths."""

    name: str
    selector: str
    source: str | None
    hash: str
    artifact_dir: str
    artifact_path: str
    manifest_path: str
    base_manifest_path: str


def build_experiment_artifact_identity(
    cfg: DictConfig | Mapping[str, Any],
    *,
    experiment_name: str,
    experiment_selector: str,
    experiment_source: str | None,
    experiment_overlay: Mapping[str, Any],
) -> ExperimentArtifactIdentity:
    """Build a stable experiment identity from config and base manifest data."""

    cfg_container = _to_plain_container(cfg)
    dataset_name = _require_nested_value(cfg_container, ("dataset", "name"))
    base_manifest_path = resolve_path(
        _require_nested_value(cfg_container, ("dataset", "manifests", "processed_default"))
    )

    if not base_manifest_path.exists():
        raise FileNotFoundError(
            f"Base manifest not found: {base_manifest_path}. Run data processing first."
        )

    with base_manifest_path.open("r", encoding="utf-8") as handle:
        base_manifest = json.load(handle)

    base_artifact = base_manifest.get("base_artifact", {})
    if not isinstance(base_artifact, dict):
        raise ValueError("base_manifest.json must contain a base_artifact object.")

    signature = {
        "dataset_name": dataset_name,
        "base_artifact_fingerprints": base_artifact.get("fingerprints", {}),
        "experiment_overlay": _canonicalize(experiment_overlay),
    }
    experiment_hash = _hash_payload(signature)[:16]

    processed_root = resolve_path(
        _require_nested_value(cfg_container, ("paths", "data_processed"))
    )
    artifact_dir = (
        processed_root
        / str(dataset_name)
        / "experiments"
        / experiment_name
        / experiment_hash
    )
    artifact_path = artifact_dir / "experiment_artifact.joblib"
    manifest_path = artifact_dir / "experiment_artifact_manifest.json"

    return ExperimentArtifactIdentity(
        name=experiment_name,
        selector=experiment_selector,
        source=experiment_source,
        hash=experiment_hash,
        artifact_dir=str(artifact_dir),
        artifact_path=str(artifact_path),
        manifest_path=str(manifest_path),
        base_manifest_path=str(base_manifest_path),
    )


def _hash_payload(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _canonicalize(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if value != value:
            raise ValueError("Experiment identity cannot be computed from NaN values.")
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _canonicalize(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return str(value)


def _to_plain_container(cfg: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(cfg, DictConfig):
        container = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(container, dict):
            raise TypeError("Config must resolve to a dictionary.")
        return container
    return dict(cfg)


def _require_nested_value(container: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = container
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            raise KeyError(f"Missing configuration key: {'.'.join(path)}")
        current = current[key]
    return current
