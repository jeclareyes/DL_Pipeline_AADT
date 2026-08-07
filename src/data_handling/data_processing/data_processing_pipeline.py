#!/usr/bin/env python3
# src/data_handling/data_processing.py

"""
Data Processing Entrypoint
==========================

This script is the Hydra entrypoint for building a unified training artifact
from TNTP files.

It intentionally delegates all processing logic to TrainingArtifactBuilder.
"""

import logging
import os
import sys
from pathlib import Path
# Add project root to sys.path to enable absolute imports starting with 'src'
_project_root = str(Path(__file__).resolve().parents[3])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
# Hydra's runtime working directory is invocation-dependent.  The dataset and
# output paths must remain anchored to this repository when the entrypoint is
# launched directly, from another directory, or through an IDE.
os.environ.setdefault("PROJECT_ROOT", _project_root)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_handling.data_processing.artifact_builders.training_artifact_builder import (
    build_training_artifact,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Build the base artifact from the configured TNTP inputs.
    """

    # Resolve the entire configuration first to prevent InterpolationResolutionError
    # when sub-configurations contain interpolations referencing outer scope parameters.
    OmegaConf.resolve(cfg)

    dm_cfg = cfg.data_handling.data_processing
    dataset_cfg = cfg.dataset

    artifact = build_training_artifact(
        cfg=dm_cfg,
        dataset_cfg=dataset_cfg,
        device=str(cfg.device),
        save=bool(dm_cfg.artifact.save_joblib or dm_cfg.artifact.save_manifest),
        artifact_name="base_artifact.joblib",
        manifest_name="base_manifest.json",
    )

    logger.info("Base artifact creation completed successfully.")


if __name__ == "__main__":
    main()
