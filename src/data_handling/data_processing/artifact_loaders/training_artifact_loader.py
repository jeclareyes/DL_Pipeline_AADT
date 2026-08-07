# src/data_handling/artifact_loaders/training_artifact_loader.py

"""
Training Artifact Loader
========================

This module loads the final training artifact produced after the base
artifact has been materialized and the asset pipeline has added the
model-ready layer.

Project context
---------------
In the AADT / traffic assignment pipeline, the training pipeline should not
repeatedly parse TNTP files, rebuild graphs, merge link tables, reconstruct
OD targets, or adapt routes into PyTorch tensors.

The data-processing stage builds a base artifact. The asset pipeline then
materializes the training artifact, which is the file loaded by this module.

- raw data;
- processed transportation objects;
- model-ready tensors;
- targets and masks;
- visualization payloads;
- metadata and reproducibility information.

This loader is intentionally lightweight. Its responsibilities are:

1. Locate and load the training artifact from disk.
2. Validate that the artifact has the expected high-level structure.
3. Optionally move PyTorch tensors to a target device.
4. Provide convenience accessors for the most commonly used artifact sections.

It does not process TNTP files, build graphs, compute routes, or create tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

from ..validators.training_artifact_validator import validate_training_artifact_or_raise
from src.utils.serialization import load
from src.utils.paths import resolve_path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingArtifactLoadResult:
    """
    Container returned by TrainingArtifactLoader.

    Attributes
    ----------
    artifact : Dict[str, Any]
        Full loaded training artifact.

    artifact_path : Path
        Resolved path to the loaded artifact.

    metadata : Dict[str, Any]
        Shortcut to artifact-level metadata.
    """

    artifact: Dict[str, Any]
    artifact_path: Path
    metadata: Dict[str, Any]


class TrainingArtifactLoader:
    """
    Load a unified training artifact from a joblib file.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to the training artifact file or to a directory containing
        "training_artifact.joblib".

    device : Optional[str], default=None
        If provided, all PyTorch tensors inside the artifact are moved to this
        device after loading. Example values: "cpu", "cuda", "cuda:0".


    artifact_filename : str, default="training_artifact.joblib"
        File name used when artifact_path points to a directory.
    """

    def __init__(
        self,
        artifact_path: Union[str, Path],
        device: Optional[str] = None,
        validate: bool = True,
        artifact_filename: str = "training_artifact.joblib",
    ) -> None:
        self.artifact_path = self._resolve_artifact_path(
            artifact_path=artifact_path,
            artifact_filename=artifact_filename,
        )
        self.device = device
        self.validate = bool(validate)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> TrainingArtifactLoadResult:
        """
        Load the training artifact from disk.

        Returns
        -------
        TrainingArtifactLoadResult
            Loaded artifact, path and metadata.

        Raises
        ------
        FileNotFoundError
            If the artifact file does not exist.

        ValueError
            If the loaded object is not a valid training artifact.

        RuntimeError
            If joblib loading fails.
        """

        logger.info("Loading training artifact from: %s", self.artifact_path)

        try:
            artifact = load(self.artifact_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load training artifact from {self.artifact_path}"
            ) from exc

        if self.validate:
            validation_result = validate_training_artifact_or_raise(artifact, strict=True)
            artifact["metadata"]["validation"] = validation_result.summary

        if self.device is not None:
            artifact = self.move_to_device(artifact, self.device)

        metadata = artifact.get("metadata", {})

        logger.info(
            "Training artifact loaded successfully | dataset=%s | version=%s",
            artifact.get("dataset_name", "unknown"),
            artifact.get("artifact_version", "unknown"),
        )

        return TrainingArtifactLoadResult(
            artifact=artifact,
            artifact_path=self.artifact_path,
            metadata=metadata,
        )

    def load_artifact(self) -> Dict[str, Any]:
        """
        Load and return only the artifact dictionary.

        This is a convenience method for training scripts that do not need the
        wrapper result object.

        Returns
        -------
        Dict[str, Any]
            Full training artifact.
        """

        return self.load().artifact

    def load_model_ready(self) -> Dict[str, Any]:
        """
        Load and return the model-ready section.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing network_params, targets and visualization.
        """

        artifact = self.load_artifact()
        model_ready = artifact.get("model_ready")
        if not model_ready:
            raise ValueError(
                "Loaded artifact does not contain a model_ready section. "
                "The current base_artifact contract delegates model-ready materialization to the asset pipeline."
            )
        return model_ready

    def load_training_inputs(self) -> Dict[str, Any]:
        """
        Load the artifact and return the main objects needed by training.

        Returns
        -------
        Dict[str, Any]
            Dictionary with network_params, targets, visualization and metadata.

        Example
        -------
        >>> inputs = loader.load_training_inputs()
        >>> network_params = inputs["network_params"]
        >>> targets = inputs["targets"]
        """

        artifact = self.load_artifact()
        model_ready = artifact.get("model_ready")
        if not model_ready:
            raise ValueError(
                "Loaded artifact does not contain a model_ready section. "
                "Use the asset pipeline to materialize the active assets before preparing training inputs."
            )

        return {
            "network_params": model_ready["network_params"],
            "targets": model_ready["targets"],
            "visualization": model_ready["visualization"],
            "metadata": artifact["metadata"],
            "config": artifact["config"],
        }

    def load_processed(self) -> Dict[str, Any]:
        """
        Load and return the processed section.

        This is useful for diagnostics, graph inspection, route analysis or
        validation outside the training loop.

        Returns
        -------
        Dict[str, Any]
            Processed layer containing link_df, graph, routes and indexing data.
        """

        artifact = self.load_artifact()
        return artifact["processed"]

    def load_raw(self) -> Dict[str, Any]:
        """
        Load and return the raw section.

        This is mainly useful for debugging or auditing the data ingestion stage.

        Returns
        -------
        Dict[str, Any]
            Raw layer containing reader outputs and reader metadata.
        """

        artifact = self.load_artifact()
        return artifact["raw"]

    # ------------------------------------------------------------------
    # Device handling
    # ------------------------------------------------------------------

    def move_to_device(self, obj: Any, device: str) -> Any:
        """
        Recursively move PyTorch tensors to a target device.

        Non-tensor objects are returned unchanged.

        Parameters
        ----------
        obj : Any
            Object to move.

        device : str
            Target device, for example "cpu", "cuda" or "cuda:0".

        Returns
        -------
        Any
            Object with all nested tensors moved to the target device.
        """

        if torch.is_tensor(obj):
            return obj.to(device)

        if isinstance(obj, dict):
            return {
                key: self.move_to_device(value, device)
                for key, value in obj.items()
            }

        if isinstance(obj, list):
            return [
                self.move_to_device(value, device)
                for value in obj
            ]

        if isinstance(obj, tuple):
            return tuple(
                self.move_to_device(value, device)
                for value in obj
            )

        return obj

    # ------------------------------------------------------------------
    # Convenience static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def get_network_params(artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract network parameters from a loaded artifact.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Loaded training artifact.

        Returns
        -------
        Dict[str, Any]
            Model-ready network parameters.
        """

        return artifact["model_ready"]["network_params"]

    @staticmethod
    def get_targets(artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract training targets from a loaded artifact.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Loaded training artifact.

        Returns
        -------
        Dict[str, Any]
            Model-ready targets.
        """

        return artifact["model_ready"]["targets"]

    @staticmethod
    def get_visualization(artifact: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract visualization payload from a loaded artifact.

        Parameters
        ----------
        artifact : Dict[str, Any]
            Loaded training artifact.

        Returns
        -------
        Dict[str, Any]
            Visualization payload.
        """

        return artifact["model_ready"]["visualization"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_artifact_path(
        artifact_path: Union[str, Path],
        artifact_filename: str,
    ) -> Path:
        """
        Resolve an artifact path.

        If artifact_path is a directory, artifact_filename is appended.

        Parameters
        ----------
        artifact_path : Union[str, Path]
            Path to either the artifact file or containing directory.

        artifact_filename : str
            Default artifact filename.

        Returns
        -------
        Path
            Resolved artifact file path.

        Raises
        ------
        FileNotFoundError
            If the final artifact file does not exist.
        """

        path = resolve_path(artifact_path)

        if path.is_dir():
            path = path / artifact_filename

        if not path.exists():
            raise FileNotFoundError(f"Training artifact file not found: {path}")

        if path.suffix.lower() not in {".joblib", ".pkl", ".pickle"}:
            logger.warning(
                "Training artifact path does not use a typical serialized extension: %s",
                path,
            )

        return path

    @staticmethod
    def _require_keys(
        obj: Dict[str, Any],
        required_keys: set[str],
        object_name: str,
    ) -> None:
        """
        Ensure that a dictionary contains required keys.

        Parameters
        ----------
        obj : Dict[str, Any]
            Dictionary to validate.

        required_keys : set[str]
            Required keys.

        object_name : str
            Human-readable object name used in error messages.

        Raises
        ------
        ValueError
            If required keys are missing.
        """

        missing = required_keys - set(obj.keys())

        if missing:
            raise ValueError(
                f"{object_name} is missing required keys: {sorted(missing)}"
            )


def load_training_artifact(
    artifact_path: Union[str, Path],
    device: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to load a training artifact.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to the training artifact file or containing directory.

    device : Optional[str], default=None
        If provided, move all nested PyTorch tensors to this device.

    validate : bool, default=True
        Whether to validate the artifact structure.

    Returns
    -------
    Dict[str, Any]
        Loaded training artifact.
    """

    loader = TrainingArtifactLoader(
        artifact_path=artifact_path,
        device=device,
        validate=validate,
    )

    return loader.load_artifact()


def load_training_inputs(
    artifact_path: Union[str, Path],
    device: Optional[str] = None,
    validate: bool = True,
) -> Dict[str, Any]:
    """
    Convenience function to load only the main training inputs.

    Parameters
    ----------
    artifact_path : Union[str, Path]
        Path to the training artifact file or containing directory.

    device : Optional[str], default=None
        If provided, move all nested PyTorch tensors to this device.

    validate : bool, default=True
        Whether to validate the artifact structure.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing network_params, targets, visualization, metadata
        and config.
    """

    loader = TrainingArtifactLoader(
        artifact_path=artifact_path,
        device=device,
        validate=validate,
    )

    return loader.load_training_inputs()
