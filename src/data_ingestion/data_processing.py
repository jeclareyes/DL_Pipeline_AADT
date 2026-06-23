# src/data_ingestion/data_processing.py

"""
Data Processing Entrypoint
==========================

This script is the Hydra entrypoint for building a unified training artifact
from TNTP files.

It intentionally delegates all processing logic to TrainingArtifactBuilder.
"""

import logging
import sys
from pathlib import Path
# Add project root to sys.path to enable absolute imports starting with 'src'
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import hydra
from omegaconf import DictConfig, OmegaConf

from src.data_ingestion.artifact_builders.training_artifact_builder import (
    build_training_artifact,
)

logger = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Build the training artifact from the configured TNTP inputs.
    """

    # Resolve the entire configuration first to prevent InterpolationResolutionError
    # when sub-configurations contain interpolations referencing outer scope parameters.
    OmegaConf.resolve(cfg)

    dm_cfg = cfg.data_ingestion.data_processing

    artifact = build_training_artifact(
        cfg=dm_cfg,
        device=str(dm_cfg.device),
        save=bool(dm_cfg.artifact.save_joblib),
    )

    logger.info(
        "Training artifact created successfully: %s",
        artifact["paths"]["artifact_path"],
    )


if __name__ == "__main__":
    main()