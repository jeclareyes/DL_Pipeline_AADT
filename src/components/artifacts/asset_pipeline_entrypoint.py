"""Hydra entrypoint for materializing assets without starting training."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Allow direct execution from the repository checkout.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import hydra
from omegaconf import DictConfig, OmegaConf

from src.components.artifacts.asset_pipeline import AssetPipeline
from src.utils.experiment_overlay import resolve_experiment_overlay
from src.utils.paths import resolve_path

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Materialize the selected experiment assets and training artifact."""

    cfg = resolve_experiment_overlay(cfg)
    OmegaConf.resolve(cfg)

    manifest_path = resolve_path(cfg.dataset.paths.manifests.base)
    base_artifact_path = resolve_path(cfg.dataset.paths.artifacts.base)

    logger.info(
        "Starting standalone asset materialization | experiment=%s | manifest=%s | base_artifact=%s",
        cfg.experiment.name,
        manifest_path,
        base_artifact_path,
    )

    result = AssetPipeline(
        experiment_config=cfg,
        dataset_config=cfg.dataset,
        manifest_path=manifest_path,
        base_artifact_path=base_artifact_path,
    ).run()

    training_artifact = result.training_artifact
    if training_artifact is None:
        raise RuntimeError(
            "Asset materialization completed without a training_artifact. "
            "Check that the selected experiment contains a route_set requirement."
        )

    logger.info(
        "Standalone asset materialization completed | training_artifact=%s",
        training_artifact["path"],
    )


if __name__ == "__main__":
    main()
