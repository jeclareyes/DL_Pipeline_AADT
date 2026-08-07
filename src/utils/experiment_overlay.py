"""Experiment overlay loading shared by asset and training entrypoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.experiment_identity import build_experiment_artifact_identity


def resolve_experiment_overlay(cfg: DictConfig) -> DictConfig:
    """Merge the selected experiment overlay and materialize its identity."""

    root_container = OmegaConf.to_container(cfg, resolve=False)
    if not isinstance(root_container, dict):
        return cfg

    experiment_name = "default"
    experiment_selector = "default"
    experiment_source: str | None = None
    overlay_container: dict[str, Any] = {}

    if "experiment" in cfg:
        experiment_ref = cfg.experiment
        if isinstance(experiment_ref, DictConfig):
            overlay = _strip_hydra_defaults(experiment_ref)
            overlay_candidate = OmegaConf.to_container(overlay, resolve=False)
            if isinstance(overlay_candidate, dict):
                overlay_container = _normalize_experiment_overlay_container(overlay_candidate)
                if _is_experiment_overlay_payload(overlay_candidate):
                    root_container.pop("experiment", None)
                experiment_section = overlay_container.get("experiment")
                if isinstance(experiment_section, dict) and "name" in experiment_section:
                    experiment_name = str(experiment_section["name"])
                    experiment_selector = experiment_name
        elif isinstance(experiment_ref, (str, Path)):
            experiment_selector = str(experiment_ref)
            experiment_name = Path(experiment_selector).stem
            project_root = Path(__file__).resolve().parents[2]
            candidate_path = Path(experiment_selector)
            if not candidate_path.suffix:
                candidate_path = project_root / "configs" / "experiments" / f"{experiment_selector}.yaml"
            elif not candidate_path.is_absolute():
                candidate_path = (project_root / candidate_path).resolve(strict=False)
            if not candidate_path.exists():
                raise FileNotFoundError(f"Experiment overlay not found: {candidate_path}")
            overlay = _strip_hydra_defaults(OmegaConf.load(candidate_path))
            overlay_candidate = OmegaConf.to_container(overlay, resolve=False)
            if isinstance(overlay_candidate, dict):
                overlay_container = overlay_candidate
            experiment_source = str(candidate_path)
        else:
            raise TypeError("cfg.experiment must be a DictConfig, string selector, or Path.")

        overlay_container = _resolve_dataset_overlay(overlay_container)

    merged_container = _deep_merge_dicts(root_container, overlay_container)
    merged_cfg = OmegaConf.create(merged_container)
    if "experiment" not in merged_container or not isinstance(merged_container["experiment"], dict):
        merged_container["experiment"] = {}

    identity = build_experiment_artifact_identity(
        merged_cfg,
        experiment_name=experiment_name,
        experiment_selector=experiment_selector,
        experiment_source=experiment_source,
        experiment_overlay=overlay_container,
    )
    experiment_section = dict(merged_container["experiment"])
    experiment_section.update(
        {
            "name": identity.name,
            "selector": identity.selector,
            "source": identity.source,
            "identity": {
                "hash": identity.hash,
                "artifact_dir": identity.artifact_dir,
                "artifact_path": identity.artifact_path,
                "manifest_path": identity.manifest_path,
                "base_manifest_path": identity.base_manifest_path,
            },
        }
    )
    merged_container["experiment"] = experiment_section
    return OmegaConf.create(merged_container)


def _strip_hydra_defaults(config: DictConfig) -> DictConfig:
    container = OmegaConf.to_container(config, resolve=False)
    if not isinstance(container, dict):
        return config
    container.pop("defaults", None)
    return OmegaConf.create(container)


def _is_experiment_overlay_payload(candidate: dict[str, Any]) -> bool:
    return bool(
        {
            "assets", "assignment_settings", "dataset", "k_active", "model",
            "od_timeframe", "route_settings", "volume_year", "weight_column",
        }.intersection(candidate)
    )


def _normalize_experiment_overlay_container(candidate: dict[str, Any]) -> dict[str, Any]:
    if _is_experiment_overlay_payload(candidate):
        return candidate
    if "name" in candidate:
        return {"experiment": candidate}
    return candidate


def _resolve_dataset_overlay(overlay_container: dict[str, Any]) -> dict[str, Any]:
    raw_dataset = overlay_container.get("dataset")
    if not isinstance(raw_dataset, str):
        return overlay_container

    project_root = Path(__file__).resolve().parents[2]
    dataset_config_path = project_root / "configs" / "dataset" / f"{raw_dataset}.yaml"
    if not dataset_config_path.exists():
        raise FileNotFoundError(
            f"Dataset config group not found for dataset '{raw_dataset}': {dataset_config_path}"
        )
    dataset_cfg = OmegaConf.load(dataset_config_path)
    dataset_container = OmegaConf.to_container(
        _strip_hydra_defaults(dataset_cfg), resolve=False
    )
    if not isinstance(dataset_container, dict):
        raise ValueError(f"Dataset config at '{dataset_config_path}' must contain a mapping.")
    result = dict(overlay_container)
    result["dataset"] = dataset_container
    return result


def _deep_merge_dicts(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
