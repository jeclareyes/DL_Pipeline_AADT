# src/train/run_context.py

"""
Training Run Context
====================

This module builds and manages the runtime context for one training pipeline run.

Project context
---------------
The training pipeline needs a consistent runtime environment before it can:

- load the training artifact;
- prepare model inputs;
- run standard or k-fold training;
- save model checkpoints;
- save evaluation outputs;
- save logs and summaries;
- update experiment ledgers.

Those responsibilities require repeated setup code such as:

- generating a run hash;
- resolving the project root;
- resolving output directories;
- configuring file logging;
- selecting CPU/GPU device;
- generating model/evaluation filenames.

This module centralizes that setup so training_pipeline.py can remain a clean
orchestrator.

This module should not:

- load training artifacts;
- prepare tensors;
- instantiate models;
- train models;
- evaluate predictions;
- save model checkpoints.

Those responsibilities belong to:

- TrainingArtifactLoader;
- TrainingInputPreparer;
- training_pipeline.py;
- GeneralTrainer;
- evaluator.py.

Design principles
-----------------
- Create a reproducible run context.
- Keep path resolution explicit.
- Keep logging setup centralized.
- Avoid model-specific logic.
- Return a structured dataclass that the pipeline can pass around.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
import platform
import socket
import sys
from typing import Any, Dict, Optional
import uuid
import json
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from src.contracts.runtime_contracts import ConfigurationContractError
from src.train._pipeline_utils import get_run_filenames, to_json_serializable
from src.utils.paths import get_project_root, resolve_path


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingRunContext:
    """
    Runtime context for one training pipeline execution.

    Attributes
    ----------
    run_hash : str
        Short unique identifier for the run.

    device : str
        Torch device selected for the run.

    project_root : Path
        Resolved project root.

    output_root : Path
        Root output directory.

    run_dir : Path
        Main directory for this specific run.

    models_dir : Path
        Directory where model checkpoints are saved.

    diagnostics_dir : Path
        Directory where diagnostics are saved.

    evaluation_dir : Path
        Directory where evaluation metrics and prediction tables are saved.

    logs_dir : Path
        Directory where logs are saved.

    summaries_dir : Path
        Directory where JSON summaries are saved.

    model_filename : str
        Main model checkpoint filename.

    eval_filename : str
        Main trainer evaluation filename.

    log_path : Path
        File path for the pipeline log.

    is_multirun : bool
        Whether this run is part of a Hydra multirun.

    metadata : Dict[str, Any]
        Runtime metadata.
    """

    run_hash: str
    device: str

    project_root: Path
    output_root: Path
    run_dir: Path
    models_dir: Path
    diagnostics_dir: Path
    evaluation_dir: Path
    logs_dir: Path
    summaries_dir: Path

    model_filename: str
    eval_filename: str
    log_path: Path

    is_multirun: bool
    metadata: Dict[str, Any]


class TrainingRunContextBuilder:
    """
    Build a TrainingRunContext from Hydra configuration.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    is_multirun : bool, default=False
        Whether the current execution is part of a Hydra multirun.

    run_hash : Optional[str], default=None
        Optional externally provided run hash. If None, a new short UUID is used.

    configure_logging : bool, default=True
        If True, attach a file handler to the root logger.
    """

    def __init__(
        self,
        cfg: DictConfig,
        is_multirun: bool = False,
        run_hash: Optional[str] = None,
        configure_logging: bool = True,
    ) -> None:
        self.cfg = cfg
        self.is_multirun = bool(is_multirun)
        self.run_hash = run_hash or self._generate_run_hash()
        self.configure_logging = bool(configure_logging)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> TrainingRunContext:
        """
        Build the full training run context.

        Returns
        -------
        TrainingRunContext
            Runtime context for the training pipeline.
        """

        project_root = self._resolve_project_root()
        output_root = self._resolve_output_root(project_root=project_root)
        device = self._resolve_device()

        configure_training_runtime(
            cfg=self.cfg,
            device=device,
        )

        model_filename, eval_filename = get_run_filenames(
            cfg=self.cfg,
            run_hash=self.run_hash,
        )

        run_dir = self._build_run_dir(
            output_root=output_root,
            run_hash=self.run_hash,
        )

        models_dir = run_dir / "models"
        diagnostics_dir = run_dir / "diagnostics"
        evaluation_dir = run_dir / "evaluation"
        logs_dir = run_dir / "logs"
        summaries_dir = run_dir / "summaries"

        self._create_directories(
            run_dir=run_dir,
            models_dir=models_dir,
            diagnostics_dir=diagnostics_dir,
            evaluation_dir=evaluation_dir,
            logs_dir=logs_dir,
            summaries_dir=summaries_dir,
        )

        log_path = logs_dir / f"training_pipeline_{self.run_hash}.log"

        if self.configure_logging:
            configure_file_logging(
                log_path=log_path,
                level=self._resolve_logging_level(),
            )

        metadata = self._build_metadata(
            project_root=project_root,
            output_root=output_root,
            run_dir=run_dir,
            device=device,
            model_filename=model_filename,
            eval_filename=eval_filename,
            log_path=log_path,
        )

        context = TrainingRunContext(
            run_hash=self.run_hash,
            device=device,
            project_root=project_root,
            output_root=output_root,
            run_dir=run_dir,
            models_dir=models_dir,
            diagnostics_dir=diagnostics_dir,
            evaluation_dir=evaluation_dir,
            logs_dir=logs_dir,
            summaries_dir=summaries_dir,
            model_filename=model_filename,
            eval_filename=eval_filename,
            log_path=log_path,
            is_multirun=self.is_multirun,
            metadata=metadata,
        )

        logger.info(
            "Training run context created | run_hash=%s | device=%s | run_dir=%s",
            context.run_hash,
            context.device,
            context.run_dir,
        )

        return context

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _resolve_project_root(self) -> Path:
        """
        Resolve the project root directory.

        The project root is centralized in src.utils.paths so all scripts
        resolve against the same repository anchor regardless of launch
        directory.

        Returns
        -------
        Path
            Project root directory.
        """

        return get_project_root()

    def _resolve_output_root(
        self,
        project_root: Path,
    ) -> Path:
        """
        Resolve root output directory.

        Parameters
        ----------
        project_root : Path
            Project root.

        Returns
        -------
        Path
            Output root directory.
        """

        configured_output = _safe_get(
            self.cfg,
            "training.output_root",
            default=None,
        )

        if configured_output is None:
            configured_output = _safe_get(
                self.cfg,
                "paths.outputs",
                default="outputs",
            )

        return resolve_path(configured_output, relative_to=project_root)

    def _build_run_dir(
        self,
        output_root: Path,
        run_hash: str,
    ) -> Path:
        """
        Build the run-specific output directory.

        Parameters
        ----------
        output_root : Path
            Output root.

        run_hash : str
            Run hash.

        Returns
        -------
        Path
            Run directory.
        """

        run_dir_template = _safe_get(
            self.cfg,
            "training.run_dir_template",
            default=None,
        )

        model_name = _safe_get(
            self.cfg,
            "model.model_name",
            default="model",
        )

        dataset_name = _safe_get(
            self.cfg,
            "dataset.name",
            default="dataset",
        )

        if run_dir_template:
            rendered = str(run_dir_template).format(
                run_hash=run_hash,
                model_name=model_name,
                dataset_name=dataset_name,
            )
            return resolve_path(rendered, relative_to=output_root)

        return resolve_path(
            Path("runs") / str(model_name) / f"{str(dataset_name)}_{run_hash}",
            relative_to=output_root,
        )

    @staticmethod
    def _create_directories(
        run_dir: Path,
        models_dir: Path,
        diagnostics_dir: Path,
        evaluation_dir: Path,
        logs_dir: Path,
        summaries_dir: Path,
    ) -> None:
        """
        Create all required run directories.

        Parameters
        ----------
        run_dir : Path
            Run directory.

        models_dir : Path
            Models directory.

        diagnostics_dir : Path
            Diagnostics directory.

        evaluation_dir : Path
            Evaluation directory.

        logs_dir : Path
            Logs directory.

        summaries_dir : Path
            Summaries directory.
        """

        for path in [
            run_dir,
            models_dir,
            diagnostics_dir,
            evaluation_dir,
            logs_dir,
            summaries_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Device and logging
    # ------------------------------------------------------------------

    def _resolve_device(self) -> str:
        """
        Resolve torch device from configuration.

        Supported values
        ----------------
        - auto
        - cpu
        - cuda
        - cuda:0
        - cuda:1

        Returns
        -------
        str
            Torch device string.
        """

        requested = str(
            _safe_get(
                self.cfg,
                "training.device",
                default="auto",
            )
        ).lower()

        if requested == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"

        if requested == "cpu":
            return "cpu"

        if requested.startswith("cuda"):
            if not torch.cuda.is_available():
                fallback = str(
                    _safe_get(
                        self.cfg,
                        "training.device_fallback",
                        default="cpu",
                    )
                )

                if fallback == "error":
                    raise ConfigurationContractError(
                        f"Requested device '{requested}', but CUDA is not available."
                    )

                logger.warning(
                    "Requested device '%s', but CUDA is not available. Falling back to '%s'.",
                    requested,
                    fallback,
                )

                return fallback

            return requested

        raise ConfigurationContractError(
            f"Unsupported training.device='{requested}'. "
            "Supported values: auto|cpu|cuda|cuda:N."
        )

    def _resolve_logging_level(self) -> int:
        """
        Resolve logging level from configuration.

        Returns
        -------
        int
            Logging level.
        """

        level_name = str(
            _safe_get(
                self.cfg,
                "logging.level",
                default=_safe_get(
                    self.cfg,
                    "training.logging.level",
                    default="INFO",
                ),
            )
        ).upper()

        return getattr(logging, level_name, logging.INFO)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        project_root: Path,
        output_root: Path,
        run_dir: Path,
        device: str,
        model_filename: str,
        eval_filename: str,
        log_path: Path,
    ) -> Dict[str, Any]:
        """
        Build runtime metadata.

        Parameters
        ----------
        project_root : Path
            Project root.

        output_root : Path
            Output root.

        run_dir : Path
            Run directory.

        device : str
            Torch device.

        model_filename : str
            Model filename.

        eval_filename : str
            Eval filename.

        log_path : Path
            Log path.

        Returns
        -------
        Dict[str, Any]
            Runtime metadata.
        """

        metadata = {
            "run_hash": self.run_hash,
            "is_multirun": self.is_multirun,
            "project_root": str(project_root),
            "output_root": str(output_root),
            "run_dir": str(run_dir),
            "device": str(device),
            "model_filename": str(model_filename),
            "eval_filename": str(eval_filename),
            "log_path": str(log_path),
            "environment": {
                "python_version": sys.version,
                "platform": platform.platform(),
                "hostname": socket.gethostname(),
                "numpy_version": np.__version__,
                "torch_version": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_device_count": int(torch.cuda.device_count())
                if torch.cuda.is_available()
                else 0,
            },
        }

        if torch.cuda.is_available():
            metadata["environment"]["cuda_device_name"] = torch.cuda.get_device_name(0)

        return metadata

    @staticmethod
    def _generate_run_hash() -> str:
        """
        Generate a short unique run hash.

        Returns
        -------
        str
            Short UUID hash.
        """

        return str(uuid.uuid4())[:8]


# =============================================================================
# Public convenience functions
# =============================================================================

def _dict_get(
        obj: Any,
        key: str,
        default: Any = None,
    ) -> Any:
    """
    Safely read a key from dict-like or DictConfig objects.
    """

    if obj is None:
        return default

    if isinstance(obj, DictConfig):
        if key not in obj:
            return default
        return obj[key]

    if isinstance(obj, dict):
        return obj.get(key, default)

    if hasattr(obj, key):
        return getattr(obj, key)

    return default


def build_run_context(
    cfg: DictConfig,
    is_multirun: bool = False,
    run_hash: Optional[str] = None,
    configure_logging: bool = True,
) -> TrainingRunContext:
    """
    Build a training run context.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    is_multirun : bool, default=False
        Whether the current execution is part of a Hydra multirun.

    run_hash : Optional[str], default=None
        Optional run hash.

    configure_logging : bool, default=True
        Whether to configure file logging.

    Returns
    -------
    TrainingRunContext
        Runtime context.
    """

    builder = TrainingRunContextBuilder(
        cfg=cfg,
        is_multirun=is_multirun,
        run_hash=run_hash,
        configure_logging=configure_logging,
    )

    return builder.build()


def configure_training_runtime(
    cfg: DictConfig,
    device: str,
) -> None:
    """
    Configure reproducibility and backend behavior for the training run.

    This function belongs to run_context.py because it affects the runtime
    environment, not the model, the data artifact, or the training loop logic.

    YAML block
    ----------
    training.runtime:
    seed_everything: true
    deterministic: false
    cudnn_benchmark: true
    cudnn_deterministic: false
    matmul_precision: high
    """

    runtime_cfg = _safe_get(
        cfg,
        "training.runtime",
        default={},
    )

    seed = int(
        _safe_get(
            cfg,
            "training.random_seed",
            default=_safe_get(cfg, "random_seed", default=42),
        )
    )

    seed_everything = bool(
        _dict_get(
            runtime_cfg,
            "seed_everything",
            default=True,
        )
    )

    if seed_everything:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    deterministic = bool(
        _dict_get(
            runtime_cfg,
            "deterministic",
            default=False,
        )
    )

    cudnn_benchmark = bool(
        _dict_get(
            runtime_cfg,
            "cudnn_benchmark",
            default=not deterministic,
        )
    )

    cudnn_deterministic = bool(
        _dict_get(
            runtime_cfg,
            "cudnn_deterministic",
            default=deterministic,
        )
    )

    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = cudnn_deterministic

    if deterministic:
        torch.use_deterministic_algorithms(
            True,
            warn_only=bool(
                _dict_get(
                    runtime_cfg,
                    "deterministic_warn_only",
                    default=True,
                )
            ),
        )

    matmul_precision = _dict_get(
        runtime_cfg,
        "matmul_precision",
        default=None,
    )

    if matmul_precision is not None:
        matmul_precision = str(matmul_precision).lower()

        if matmul_precision not in {"highest", "high", "medium"}:
            raise ConfigurationContractError(
                "training.runtime.matmul_precision must be one of: "
                "highest|high|medium."
            )

        torch.set_float32_matmul_precision(matmul_precision)

    logger.info(
        "Training runtime configured | seed=%d | seed_everything=%s | deterministic=%s | "
        "cudnn_benchmark=%s | cudnn_deterministic=%s | matmul_precision=%s | device=%s",
        seed,
        seed_everything,
        deterministic,
        cudnn_benchmark,
        cudnn_deterministic,
        matmul_precision,
        device,
    )



def configure_file_logging(
    log_path: str | Path,
    level: int = logging.INFO,
    formatter: Optional[logging.Formatter] = None,
) -> None:
    """
    Attach a file handler to the root logger.

    The function avoids adding duplicate handlers pointing to the same log file.

    Parameters
    ----------
    log_path : str | Path
        Path to log file.

    level : int, default=logging.INFO
        Logging level.

    formatter : Optional[logging.Formatter], default=None
        Optional custom formatter.
    """

    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    resolved_log_path = str(log_path.resolve())

    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            existing_path = str(Path(handler.baseFilename).resolve())

            if existing_path == resolved_log_path:
                return

    if formatter is None:
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
        )

    file_handler = logging.FileHandler(
        resolved_log_path,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)


def save_run_context_metadata(
    context: TrainingRunContext,
    filename: str = "run_context.json",
) -> str:
    """
    Save run context metadata to JSON.

    Parameters
    ----------
    context : TrainingRunContext
        Run context.

    filename : str, default="run_context.json"
        Output filename.

    Returns
    -------
    str
        Saved metadata path.
    """

    output_path = context.summaries_dir / filename

    payload = {
        "run_hash": context.run_hash,
        "device": context.device,
        "project_root": str(context.project_root),
        "output_root": str(context.output_root),
        "run_dir": str(context.run_dir),
        "models_dir": str(context.models_dir),
        "diagnostics_dir": str(context.diagnostics_dir),
        "evaluation_dir": str(context.evaluation_dir),
        "logs_dir": str(context.logs_dir),
        "summaries_dir": str(context.summaries_dir),
        "model_filename": context.model_filename,
        "eval_filename": context.eval_filename,
        "log_path": str(context.log_path),
        "is_multirun": context.is_multirun,
        "metadata": context.metadata,
    }

    output_path.write_text(
        json.dumps(
            to_json_serializable(payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return str(output_path)


# =============================================================================
# Internal helpers
# =============================================================================


def _safe_get(
    cfg: Any,
    dotted_key: str,
    default: Any = None,
) -> Any:
    """
    Safely read a nested value from DictConfig, dict or object attributes.

    Parameters
    ----------
    cfg : Any
        Config object.

    dotted_key : str
        Dotted key path.

    default : Any, default=None
        Fallback value.

    Returns
    -------
    Any
        Retrieved value or default.
    """

    current = cfg

    for part in dotted_key.split("."):
        if current is None:
            return default

        if isinstance(current, DictConfig):
            if part not in current:
                return default
            current = current[part]
            continue

        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
            continue

        if hasattr(current, part):
            current = getattr(current, part)
            continue

        return default

    return current
