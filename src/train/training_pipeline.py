# src/train/training_pipeline.py
from __future__ import annotations

"""
Training Pipeline
=================

This module is the high-level orchestration entrypoint for model training.

Project context
---------------
After the data-processing refactor, the training pipeline consumes a training
artifact materialized from the base artifact by the asset pipeline.

Main responsibilities
---------------------
- Build runtime context.
- Load the training artifact.
- Prepare global and task-specific training inputs.
- Generate standard or k-fold training tasks.
- Instantiate one model per task.
- Delegate training to GeneralTrainer.
- Run post-training evaluation.
- Save pipeline summaries.
- Optionally update the experiment ledger.

Design principles
-----------------
- Keep the pipeline as an orchestrator.
- Keep data processing outside the training pipeline.
- Keep training loop inside GeneralTrainer.
- Keep evaluation inside simple_evaluation/evaluator.py.
- Keep runtime setup inside run_context.py.
"""

import copy
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "src"
for import_root in (PROJECT_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

# Import centralized resolver for pathing
from src.utils.paths import resolve_path

from src.contracts.runtime_contracts import (
    ConfigurationContractError,
    ModelInputContractError,
)
from src.components.artifacts.asset_pipeline import AssetPipeline
from data_handling.data_processing.artifact_loaders.training_artifact_loader import (
    TrainingArtifactLoader,
)
from src.utils.experiment_overlay import resolve_experiment_overlay
from src.train._pipeline_utils import (
    generate_training_tasks,
    persist_pipeline_summary,
    summarize_training_task,
    to_json_serializable,
)
from src.train.run_context import (
    TrainingRunContext,
    build_run_context,
    save_run_context_metadata,
)
from src.train.simple_evaluation.evaluator import (
    build_task_evaluation_summary,
    evaluate_full_reconstruction,
    evaluate_holdout_metrics,
    evaluate_validation_metrics,
    save_evaluation_result,
    save_task_evaluation_summary,
)
from src.train.trainer import GeneralTrainer
from src.train.training_input_preparer import (
    GlobalTrainingInputs,
    TaskTrainingInputs,
    TrainingInputPreparer,
)

try:
    from src.utils.ledger import update_experiments_ledger
except Exception:  # pragma: no cover - optional project utility
    update_experiments_ledger = None


logger = logging.getLogger(__name__)


# =============================================================================
# Public pipeline API
# =============================================================================


def run_pipeline(
    cfg: DictConfig,
    is_multirun: bool = False,
) -> Dict[str, Any]:
    """
    Run the full training pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    is_multirun : bool, default=False
        Whether the current run is part of a Hydra multirun.

    Returns
    -------
    Dict[str, Any]
        Pipeline summary.
    """

    cfg = resolve_experiment_overlay(cfg)
    OmegaConf.resolve(cfg)

    materialize_experiment_assets(cfg=cfg)

    run_context = build_run_context(
        cfg=cfg,
        is_multirun=is_multirun,
    )

    save_run_context_metadata(run_context)

    logger.info(
        "Training pipeline started | run_hash=%s | device=%s",
        run_context.run_hash,
        run_context.device,
    )

    artifact, artifact_inputs = load_training_artifact_for_pipeline(
        cfg=cfg,
        device=run_context.device,
    )

    sampled_flow_mask_np, sampled_od_mask_np = build_sampling_masks(
        cfg=cfg,
        targets=artifact_inputs["targets"],
    )

    input_preparer = TrainingInputPreparer(
        cfg=cfg,
        device=run_context.device,
        strict=bool(_cfg_get(cfg, "training.strict_data")),
    )

    global_inputs = input_preparer.prepare_global_inputs(
        artifact_inputs=artifact_inputs,
        sampled_flow_mask_np=sampled_flow_mask_np,
        sampled_od_mask_np=sampled_od_mask_np,
    )

    training_tasks = list(
        generate_training_tasks(
            train_mask_global=global_inputs.flow_train_global_mask_np,
            k_folds=int(_cfg_get(cfg, "training.k_folds")),
            random_seed=int(_cfg_get(cfg, "training.random_seed")),
        )
    )

    if not training_tasks:
        raise ConfigurationContractError(
            "No training tasks were generated."
        )

    task_summaries = []
    task_scores = []

    model_ready_layer = artifact.get("model_ready", {})
    link_metadata = model_ready_layer.get("link_metadata")

    _validate_link_metadata_alignment(
        link_metadata=link_metadata,
        network_params=global_inputs.network_params,
    )

    for task_idx, task in enumerate(training_tasks, start=1):
        logger.info(
            "Starting training task %d/%d: %s",
            task_idx,
            len(training_tasks),
            task.get("name", "Unnamed Task"),
        )

        task_result = run_training_task(
            cfg=cfg,
            run_context=run_context,
            input_preparer=input_preparer,
            global_inputs=global_inputs,
            task=task,
            link_metadata=link_metadata,
        )

        task_summaries.append(task_result)
        task_scores.append(float(task_result["training"]["score"]))

    pipeline_summary = build_pipeline_summary(
        cfg=cfg,
        run_context=run_context,
        artifact=artifact,
        global_inputs=global_inputs,
        training_tasks=training_tasks,
        task_summaries=task_summaries,
        task_scores=task_scores,
    )

    summary_path = persist_pipeline_summary(
        output_dir=run_context.summaries_dir,
        summary=pipeline_summary,
        filename="pipeline_summary.json",
    )

    pipeline_summary["paths"]["pipeline_summary_path"] = summary_path

    maybe_update_ledger(
        cfg=cfg,
        run_context=run_context,
        pipeline_summary=pipeline_summary,
    )

    logger.info(
        "Training pipeline completed | run_hash=%s | best_score=%.6f | summary= %s",
        run_context.run_hash,
        float(pipeline_summary["results"]["best_score"]),
        summary_path,
    )

    return pipeline_summary


def run_training_task(
    cfg: DictConfig,
    run_context: TrainingRunContext,
    input_preparer: TrainingInputPreparer,
    global_inputs: GlobalTrainingInputs,
    task: Dict[str, Any],
    link_metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run one training task.

    A task is either:

    - the standard training run; or
    - one fold in k-fold cross-validation.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    run_context : TrainingRunContext
        Runtime context.

    input_preparer : TrainingInputPreparer
        Input preparer.

    global_inputs : GlobalTrainingInputs
        Global training inputs.

    task : Dict[str, Any]
        Task dictionary generated by generate_training_tasks().

    link_metadata : Optional[Any], default=None
        Optional link table used to enrich prediction outputs.

    Returns
    -------
    Dict[str, Any]
        Task summary.
    """

    task_inputs = input_preparer.prepare_task_tensors(
        global_inputs=global_inputs,
        task=task,
    )

    model_preparation = input_preparer.prepare_model_params(
        global_inputs=global_inputs,
        task=task,
    )

    model = instantiate_model(
        cfg=cfg,
        model_params=model_preparation.model_params,
        device=run_context.device,
    )

    model_filename, eval_filename = build_task_filenames(
        base_model_filename=run_context.model_filename,
        base_eval_filename=run_context.eval_filename,
        task=task,
    )

    trainer = GeneralTrainer(
        cfg=cfg,
        device=run_context.device,
        output_dir=str(run_context.models_dir),
        model_filename=model_filename,
        eval_filename=eval_filename,
        diagnostics_dir=str(run_context.diagnostics_dir),
        is_multirun=run_context.is_multirun,
    )

    fit_result = trainer.fit(
        model=model,
        network_params=global_inputs.network_params,
        train_tensors=task_inputs.train_tensors,
        val_tensors=task_inputs.val_tensors,
    )

    # Reload best model weights for final bundle enrichment and subsequent evaluation tasks
    model_checkpoint = torch.load(fit_result.model_path, map_location=run_context.device, weights_only=False)
    model.load_state_dict(model_checkpoint["model_state_dict"])
    


    evaluation_outputs = run_task_evaluations(
        cfg=cfg,
        run_context=run_context,
        model=model,
        global_inputs=global_inputs,
        task=task,
        task_inputs=task_inputs,
        link_metadata=link_metadata,
    )

    task_summary = build_task_evaluation_summary(
        task_name=str(task.get("name", "Unnamed Task")),
        fit_result=fit_result,
        holdout_result=evaluation_outputs.get("holdout"),
        validation_result=evaluation_outputs.get("validation"),
        full_result=evaluation_outputs.get("full_observed"),
        extra_metadata={
            "task": summarize_training_task(task),
            "task_inputs": task_inputs.task_metadata,
            "model_preparation": model_preparation.metadata,
            "saved_evaluations": evaluation_outputs.get("saved_paths", {}),
        },
    )

    safe_task_name = safe_name(str(task.get("name", "task")))

    save_task_evaluation_summary(
        summary=task_summary,
        output_dir=run_context.summaries_dir,
        filename=f"{safe_task_name}_summary.json",
    )

    return task_summary


# =============================================================================
# Artifact loading
# =============================================================================


def load_training_artifact_for_pipeline(
    cfg: DictConfig,
    device: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load the unified training artifact for the pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    device : str
        Target device.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        Full artifact and artifact inputs.
    """

    artifact_path = resolve_training_artifact_path(cfg)

    validate_on_load = bool(
        _cfg_get(
            cfg,
            "training.artifact.validate_on_load",
        )
    )

    move_to_device = bool(
        _cfg_get(
            cfg,
            "training.artifact.move_to_device_on_load",
        )
    )

    loader_device = device if move_to_device else None

    loader = TrainingArtifactLoader(
        artifact_path=artifact_path,
        device=loader_device,
        validate=validate_on_load,
        artifact_filename=str(
            _cfg_get(
                cfg,
                "training.artifact.filename",
            )
        ),
    )

    load_result = loader.load()
    artifact = load_result.artifact

    model_ready = artifact.get("model_ready")
    if not model_ready:
        raise ConfigurationContractError(
            "Loaded training artifact does not contain a model_ready section. "
            "The asset pipeline must materialize the training artifact before training starts."
        )

    artifact_inputs = {
        "network_params": model_ready["network_params"],
        "targets": model_ready["targets"],
        "visualization": model_ready["visualization"],
        "metadata": artifact.get("metadata", {}),
        "config": artifact.get("config", {}),
    }

    logger.info(
        "Training artifact loaded | dataset=%s | path=%s",
        artifact.get("dataset_name", "unknown"),
        load_result.artifact_path,
    )

    return artifact, artifact_inputs


def materialize_experiment_assets(cfg: DictConfig) -> None:
    """
    Materialize declared assets before training starts.

    The asset pipeline is a separate responsibility from training. This helper
    bridges the current experiment orchestration so the training pipeline can
    consume the manifest entries required by the experiment.
    """

    if "assets" not in cfg:
        return

    assets_cfg = _cfg_get(cfg, "assets")
    requirements_cfg = _dict_get(assets_cfg, "requirements", default=None)
    if requirements_cfg is None:
        return

    route_set_requirement = _dict_get(requirements_cfg, "route_set", default=None)
    assignment_set_requirement = _dict_get(requirements_cfg, "assignment_set", default=None)
    if route_set_requirement is None and assignment_set_requirement is None:
        return

    manifest_path = resolve_path(_cfg_get(cfg, "dataset.paths.manifests.base"))
    base_artifact_path = resolve_base_artifact_path(cfg)
    experiment_name = _cfg_get(cfg, "experiment.name")
    experiment_hash = _cfg_get(cfg, "experiment.identity.hash")

    logger.info(
        "Materializing experiment assets | experiment=%s | hash=%s | manifest=%s | base_artifact=%s",
        experiment_name,
        experiment_hash,
        manifest_path,
        base_artifact_path,
    )

    pipeline = AssetPipeline(
        experiment_config=cfg,
        dataset_config=cfg.dataset,
        manifest_path=manifest_path,
        base_artifact_path=base_artifact_path,
    )
    pipeline.run()


def resolve_base_artifact_path(cfg: DictConfig) -> Path:
    """
    Resolve the base artifact path from the data-processing configuration.
    """

    artifact_path = _cfg_get(cfg, "dataset.paths.artifacts.base")
    if not artifact_path:
        raise ConfigurationContractError(
            "Could not resolve base artifact path. Please define dataset.paths.artifacts.base."
        )

    return resolve_path(artifact_path)


def resolve_training_artifact_path(cfg: DictConfig) -> Path:
    """
    Resolve the training artifact path from configuration.
    """
    artifact_path = _cfg_get(cfg, "artifact.path")
    if artifact_path:
        return resolve_path(artifact_path)

    experiment_artifact_path = _cfg_get(cfg, "experiment.identity.artifact_path")
    if experiment_artifact_path:
        return resolve_path(experiment_artifact_path)

    raise ConfigurationContractError(
        "Could not resolve experiment artifact path. Please define 'training.artifact.path' or ensure the experiment identity is materialized."
    )

# =============================================================================
# Sampling
# =============================================================================


def build_sampling_masks(
    cfg: DictConfig,
    targets: Dict[str, Any],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build optional sampling masks for flow and OD supervision.

    The data-processing artifact already contains observed masks. This function
    only selects which observed entries are used for training.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    targets : Dict[str, Any]
        Artifact target payload.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        sampled_flow_mask_np and sampled_od_mask_np.
    """

    sampling_enabled = bool(
        _cfg_get(
            cfg,
            "sampling.enabled",
        )
    )

    if not sampling_enabled:
        return None, None

    flows_observed_mask_np = _as_binary_mask(
        targets["flows_observed_mask_np"],
        name="flows_observed_mask_np",
    )

    od_observed_mask_np = _as_binary_mask(
        targets["od_observed_mask_np"],
        name="od_observed_mask_np",
    )

    flow_rate = float(
        _cfg_get(
            cfg,
            "sampling.flow_rate",
        )
    )

    od_rate = float(
        _cfg_get(
            cfg,
            "sampling.od_rate",
        )
    )

    random_seed = int(
        _cfg_get(
            cfg,
            "sampling.random_seed",
        )
    )

    flow_strategy = str(
        _cfg_get(
            cfg,
            "sampling.flow_strategy",
        )
    ).lower()

    od_strategy = str(
        _cfg_get(
            cfg,
            "sampling.od_strategy",
        )
    ).lower()

    sampled_flow_mask_np = build_single_sampling_mask(
        observed_mask_np=flows_observed_mask_np,
        rate=flow_rate,
        strategy=flow_strategy,
        random_seed=random_seed,
        name="flow",
    )

    sampled_od_mask_np = build_single_sampling_mask(
        observed_mask_np=od_observed_mask_np,
        rate=od_rate,
        strategy=od_strategy,
        random_seed=random_seed + 17,
        name="od",
    )

    logger.info(
        "Sampling masks built | observed_flows=%d | sampled_flows=%d | observed_od=%d | sampled_od=%d",
        int(flows_observed_mask_np.sum()),
        int(sampled_flow_mask_np.sum()),
        int(od_observed_mask_np.sum()),
        int(sampled_od_mask_np.sum()),
    )

    return sampled_flow_mask_np, sampled_od_mask_np


def build_single_sampling_mask(
    observed_mask_np: np.ndarray,
    rate: float,
    strategy: str,
    random_seed: int,
    name: str,
) -> np.ndarray:
    """
    Build one binary sampling mask from an observed mask.

    Parameters
    ----------
    observed_mask_np : np.ndarray
        Binary observed mask.

    rate : float
        Sampling rate in [0, 1].

    strategy : str
        Sampling strategy.

        Supported values:
        - all
        - all_observed
        - random
        - observed_random

    random_seed : int
        Random seed.

    name : str
        Human-readable mask name.

    Returns
    -------
    np.ndarray
        Binary sampling mask.
    """

    observed_mask_np = _as_binary_mask(
        observed_mask_np,
        name=f"{name}_observed_mask_np",
    )

    if rate < 0.0 or rate > 1.0:
        raise ConfigurationContractError(
            f"{name} sampling rate must be in [0, 1]. Got {rate}."
        )

    strategy = str(strategy).lower()

    if strategy in {"all", "all_observed"} or rate >= 1.0:
        return observed_mask_np.copy().astype(np.float32)

    if strategy in {"random", "observed_random"}:
        observed_indices = np.where(observed_mask_np > 0)[0]

        if observed_indices.size == 0:
            raise ModelInputContractError(
                f"Cannot sample {name} mask because observed mask is empty."
            )

        sample_size = int(round(float(rate) * observed_indices.size))
        sample_size = max(1, min(sample_size, observed_indices.size))

        rng = np.random.default_rng(seed=int(random_seed))
        selected = rng.choice(
            observed_indices,
            size=sample_size,
            replace=False,
        )

        mask = np.zeros_like(observed_mask_np, dtype=np.float32)
        mask[selected] = 1.0

        return mask

    raise ConfigurationContractError(
        f"Unsupported {name} sampling strategy '{strategy}'. "
        "Supported values: all|all_observed|random|observed_random."
    )


# =============================================================================
# Model instantiation
# =============================================================================

def instantiate_model(
    cfg: DictConfig,
    model_params: Dict[str, Any],
    device: str,
) -> torch.nn.Module:
    """
    Instantiate the configured model without passing model_params through
    OmegaConf/Hydra merge.

    Why:
    Hydra/OmegaConf cannot merge dictionaries that contain tuple keys, such as:
        edge_to_idx = {(u, v): idx}

    Those mappings are valid Python objects and are part of the model alignment
    contract, so they must reach the model unchanged.
    """

    if not hasattr(cfg, "model"):
        raise ConfigurationContractError(
            "cfg.model is required to instantiate the training model."
        )

    model_cfg = copy.deepcopy(cfg.model)

    target_path = _dict_get(
        model_cfg,
        "_target_",
        default=None,
    )

    if target_path is None:
        raise ConfigurationContractError(
            "cfg.model must define a '_target_' entry."
        )

    try:
        target_cls = hydra.utils.get_class(str(target_path))
    except Exception as exc:
        raise ConfigurationContractError(
            f"Could not resolve model target class: {target_path}"
        ) from exc

    model_cfg_kwargs = OmegaConf.to_container(
        model_cfg,
        resolve=True,
    )

    if not isinstance(model_cfg_kwargs, dict):
        raise ConfigurationContractError(
            f"cfg.model must resolve to a dictionary. Got {type(model_cfg_kwargs)}."
        )

    # Remove Hydra-only keys. These are configuration directives, not model
    # constructor arguments.
    model_cfg_kwargs.pop("_target_", None)
    model_cfg_kwargs.pop("_recursive_", None)
    model_cfg_kwargs.pop("_convert_", None)
    model_cfg_kwargs.pop("_partial_", None)

    # model_params comes from the artifact and must remain raw Python.
    # Do NOT pass it through OmegaConf.create(), OmegaConf.merge(), or any
    # sanitizer that converts tuple keys to strings.
    constructor_kwargs = {
        **model_cfg_kwargs,
        **model_params,
    }

    try:
        model = target_cls(**constructor_kwargs)
    except Exception as exc:
        raise ConfigurationContractError(
            "Failed to instantiate model directly from cfg.model['_target_']. "
            "Check that the model constructor accepts the provided config keys "
            "and artifact-derived network_params."
        ) from exc

    if not isinstance(model, torch.nn.Module):
        raise ConfigurationContractError(
            f"Instantiated model must be a torch.nn.Module. Got {type(model)}."
        )

    return model.to(device)

def instantiate_model_old(
    cfg: DictConfig,
    model_params: Dict[str, Any],
    device: str,
) -> torch.nn.Module:
    """
    Instantiate the configured model.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    model_params : Dict[str, Any]
        Model parameters prepared from network_params.

    device : str
        Target device.

    Returns
    -------
    torch.nn.Module
        Instantiated model.
    """

    if not hasattr(cfg, "model"):
        raise ConfigurationContractError(
            "cfg.model is required to instantiate the training model."
        )

    model_cfg = copy.deepcopy(cfg.model)

    # Sanitization
    # Avoids that OmegaConfg collapses when findin tuples as keys
    def sanitize_for_omegaconf(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k) if isinstance(k, tuple) else k: sanitize_for_omegaconf(v)
                    for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [sanitize_for_omegaconf(v) for v in obj]
        return obj
    
    safe_model_params = sanitize_for_omegaconf(model_params)

    try:
        model = hydra.utils.instantiate(
            model_cfg,
            **safe_model_params,
            _recursive_=False,
        )
    except Exception as exc:
        raise ConfigurationContractError(
            "Failed to instantiate model from cfg.model. "
            "Check that the model YAML defines a valid _target_ and accepts "
            "the keys provided by network_params."
        ) from exc

    if not isinstance(model, torch.nn.Module):
        raise ConfigurationContractError(
            f"Instantiated model must be a torch.nn.Module. Got {type(model)}."
        )

    return model.to(device)


# =============================================================================
# Evaluation orchestration
# =============================================================================


def run_task_evaluations(
    cfg: DictConfig,
    run_context: TrainingRunContext,
    model: torch.nn.Module,
    global_inputs: GlobalTrainingInputs,
    task: Dict[str, Any],
    task_inputs: TaskTrainingInputs,
    link_metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run configured evaluations for one task.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    run_context : TrainingRunContext
        Runtime context.

    model : torch.nn.Module
        Trained model.

    global_inputs : GlobalTrainingInputs
        Global inputs.

    task : Dict[str, Any]
        Training task.

    task_inputs : TaskTrainingInputs
        Prepared task tensors.

    link_metadata : Optional[Any], default=None
        Optional link metadata.

    Returns
    -------
    Dict[str, Any]
        Evaluation outputs.
    """

    evaluation_cfg = _cfg_get(
        cfg,
        "training.evaluation",
    )

    enabled = bool(
        _dict_get(
            evaluation_cfg,
            "enabled",
        )
    )

    if not enabled:
        return {
            "holdout": None,
            "validation": None,
            "full_observed": None,
            "saved_paths": {},
        }

    return_predictions = bool(
        _dict_get(
            evaluation_cfg,
            "return_predictions",
        )
    )

    save_predictions = bool(
        _dict_get(
            evaluation_cfg,
            "save_predictions",
        )
    )

    safe_task = safe_name(str(task.get("name", "task")))
    saved_paths = {}

    holdout_result = None
    validation_result = None
    full_result = None

    od_eval_t = task_inputs.train_tensors.get("od")
    od_eval_mask_t = task_inputs.train_tensors.get("od_mask")

    if bool(_dict_get(evaluation_cfg, "evaluate_validation")):
        if np.asarray(task["val_mask"]).sum() > 0:
            validation_result = evaluate_validation_metrics(
                model=model,
                true_flows_t=global_inputs.flows_target_t,
                train_mask_np=task["train_mask"],
                val_mask_np=task["val_mask"],
                device=run_context.device,
                return_predictions=return_predictions,
                link_metadata=link_metadata,
                true_od_t=od_eval_t,
                od_mask_t=od_eval_mask_t,
            )

            saved_paths["validation"] = save_evaluation_result(
                result=validation_result,
                output_dir=run_context.evaluation_dir,
                prefix=f"{safe_task}_validation",
                save_predictions=save_predictions,
            )

    if bool(_dict_get(evaluation_cfg, "evaluate_holdout")):
        if global_inputs.flow_holdout_global_mask_np.sum() > 0:
            holdout_result = evaluate_holdout_metrics(
                model=model,
                true_flows_t=global_inputs.flows_target_t,
                train_mask_np=task["train_mask"],
                holdout_mask_np=global_inputs.flow_holdout_global_mask_np,
                device=run_context.device,
                return_predictions=return_predictions,
                link_metadata=link_metadata,
                true_od_t=od_eval_t,
                od_mask_t=od_eval_mask_t,
            )

            saved_paths["holdout"] = save_evaluation_result(
                result=holdout_result,
                output_dir=run_context.evaluation_dir,
                prefix=f"{safe_task}_holdout",
                save_predictions=save_predictions,
            )

    if bool(_dict_get(evaluation_cfg, "evaluate_full_observed")):
        full_result = evaluate_full_reconstruction(
            model=model,
            true_flows_t=global_inputs.flows_target_t,
            input_mask_np=task["train_mask"],
            observed_mask_np=global_inputs.flows_observed_mask_np,
            device=run_context.device,
            return_predictions=return_predictions,
            link_metadata=link_metadata,
            true_od_t=od_eval_t,
            od_mask_t=od_eval_mask_t,
        )

        saved_paths["full_observed"] = save_evaluation_result(
            result=full_result,
            output_dir=run_context.evaluation_dir,
            prefix=f"{safe_task}_full_observed",
            save_predictions=save_predictions,
        )

    return {
        "holdout": holdout_result,
        "validation": validation_result,
        "full_observed": full_result,
        "saved_paths": saved_paths,
    }


# =============================================================================
# Pipeline summary and ledger
# =============================================================================


def build_pipeline_summary(
    cfg: DictConfig,
    run_context: TrainingRunContext,
    artifact: Dict[str, Any],
    global_inputs: GlobalTrainingInputs,
    training_tasks: List[Dict[str, Any]],
    task_summaries: List[Dict[str, Any]],
    task_scores: List[float],
) -> Dict[str, Any]:
    """
    Build final pipeline summary.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    run_context : TrainingRunContext
        Runtime context.

    artifact : Dict[str, Any]
        Loaded training artifact.

    global_inputs : GlobalTrainingInputs
        Global training inputs.

    training_tasks : List[Dict[str, Any]]
        Training tasks.

    task_summaries : List[Dict[str, Any]]
        Task summaries.

    task_scores : List[float]
        Task scores.

    Returns
    -------
    Dict[str, Any]
        Pipeline summary.
    """

    best_score = float(np.min(task_scores)) if task_scores else float("inf")
    mean_score = float(np.mean(task_scores)) if task_scores else float("inf")

    return {
        "schema_version": 1,
        "run_context": run_context.metadata,
        "dataset": {
            "dataset_name": artifact.get("dataset_name", "unknown"),
            "artifact_version": artifact.get("artifact_version", "unknown"),
            "artifact_created_at": artifact.get("created_at", None),
        },
        "model": {
            "model_name": _cfg_get(cfg, "model.model_name"),
        },
        "training": {
            "num_tasks": int(len(training_tasks)),
            "tasks": [
                summarize_training_task(task)
                for task in training_tasks
            ],
            "num_observed_flows": int(global_inputs.flows_observed_mask_np.sum()),
            "num_train_global_flows": int(global_inputs.flow_train_global_mask_np.sum()),
            "num_holdout_flows": int(global_inputs.flow_holdout_global_mask_np.sum()),
            "num_supervised_od": int(global_inputs.od_train_supervision_mask_np.sum()),
        },
        "results": {
            "best_score": best_score,
            "mean_score": mean_score,
            "scores": [float(score) for score in task_scores],
        },
        "task_summaries": task_summaries,
        "paths": {
            "run_dir": str(run_context.run_dir),
            "models_dir": str(run_context.models_dir),
            "evaluation_dir": str(run_context.evaluation_dir),
            "logs_dir": str(run_context.logs_dir),
            "summaries_dir": str(run_context.summaries_dir),
            "pipeline_summary_path": None,
        },
        "config_snapshot": to_json_serializable(cfg),
    }


def maybe_update_ledger(
    cfg: DictConfig,
    run_context: TrainingRunContext,
    pipeline_summary: Dict[str, Any],
) -> None:
    """
    Optionally update experiment ledger.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    run_context : TrainingRunContext
        Runtime context.

    pipeline_summary : Dict[str, Any]
        Final pipeline summary.
    """

    ledger_enabled = bool(
        _cfg_get(
            cfg,
            "training.ledger.enabled",
        )
    )

    if not ledger_enabled:
        return

    if update_experiments_ledger is None:
        logger.warning(
            "training.ledger.enabled=True, but update_experiments_ledger could not be imported."
        )
        return

    try:
        update_experiments_ledger(
            cfg=cfg,
            run_hash=run_context.run_hash,
            summary=pipeline_summary,
        )
    except TypeError:
        try:
            update_experiments_ledger(cfg=cfg, 
                                      run_hash=run_context.run_hash
                                      )
        except Exception as exc:
            logger.warning(
                "Failed to update experiments ledger: %s",
                exc,
            )
    except Exception as exc:
        logger.warning(
            "Failed to update experiments ledger: %s",
            exc,
        )


# =============================================================================
# Filename helpers
# =============================================================================


def build_task_filenames(
    base_model_filename: str,
    base_eval_filename: str,
    task: Dict[str, Any],
) -> Tuple[str, str]:
    """
    Build task-specific model and evaluation filenames.

    Parameters
    ----------
    base_model_filename : str
        Base model filename.

    base_eval_filename : str
        Base evaluation filename.

    task : Dict[str, Any]
        Training task.

    Returns
    -------
    Tuple[str, str]
        Task model filename and eval filename.
    """

    suffix = str(task.get("suffix", ".pt"))

    model_filename = apply_suffix_to_filename(
        filename=base_model_filename,
        suffix=suffix,
    )

    eval_suffix = suffix.replace(".pt", "_eval.pt")

    eval_filename = apply_suffix_to_filename(
        filename=base_eval_filename,
        suffix=eval_suffix,
    )

    return model_filename, eval_filename


def apply_suffix_to_filename(
    filename: str,
    suffix: str,
) -> str:
    """
    Apply a suffix before the file extension.

    Parameters
    ----------
    filename : str
        Base filename.

    suffix : str
        Suffix.

    Returns
    -------
    str
        Filename with suffix.
    """

    path = Path(filename)

    if suffix == ".pt":
        return filename

    if path.suffix:
        clean_suffix = suffix

        if clean_suffix.endswith(path.suffix):
            clean_suffix = clean_suffix[: -len(path.suffix)]

        return f"{path.stem}{clean_suffix}{path.suffix}"

    return f"{filename}{suffix}"


def safe_name(
    value: str,
) -> str:
    """
    Convert a human-readable name to a safe file prefix.

    Parameters
    ----------
    value : str
        Input name.

    Returns
    -------
    str
        Safe filename component.
    """

    safe = str(value).strip().lower()
    safe = safe.replace(" ", "_")
    safe = safe.replace("/", "_")
    safe = safe.replace("\\", "_")
    safe = safe.replace(":", "_")

    return safe or "task"


# =============================================================================
# Generic helpers
# =============================================================================

def _validate_link_metadata_alignment(
    link_metadata: Optional[Any],
    network_params: Dict[str, Any],
) -> None:
    """
    Validate that link_metadata is aligned with the model link order.

    The evaluator attaches link_metadata to prediction rows by row position.
    Therefore, row i in link_metadata must describe the same directed link as
    network_params['link_pair_indices'][i].
    """

    if link_metadata is None:
        return

    if "link_pair_indices" not in network_params:
        raise ModelInputContractError(
            "network_params is missing 'link_pair_indices'. "
            "Cannot validate link_metadata alignment."
        )

    if not hasattr(link_metadata, "columns"):
        raise ModelInputContractError(
            "link_metadata must be a pandas-like DataFrame when provided."
        )

    required_columns = {"init_node", "term_node"}
    missing_columns = required_columns - set(link_metadata.columns)

    if missing_columns:
        raise ModelInputContractError(
            "link_metadata is missing columns required for alignment validation: "
            f"{sorted(missing_columns)}"
        )

    metadata_pairs = (
        link_metadata[["init_node", "term_node"]]
        .astype(np.int64)
        .to_numpy()
    )

    if torch.is_tensor(network_params["link_pair_indices"]):
        model_pairs = (
            network_params["link_pair_indices"]
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
    else:
        model_pairs = np.asarray(
            network_params["link_pair_indices"],
            dtype=np.int64,
        )

    if model_pairs.ndim != 2 or model_pairs.shape[1] != 2:
        raise ModelInputContractError(
            "network_params['link_pair_indices'] must have shape [num_links, 2]. "
            f"Got shape {model_pairs.shape}."
        )

    if metadata_pairs.shape != model_pairs.shape:
        raise ModelInputContractError(
            "link_metadata shape does not match model link order shape. "
            f"link_metadata={metadata_pairs.shape}, "
            f"network_params['link_pair_indices']={model_pairs.shape}."
        )

    if not np.array_equal(metadata_pairs, model_pairs):
        mismatch_idx = np.where(
            np.any(metadata_pairs != model_pairs, axis=1)
        )[0]

        sample = [
            {
                "position": int(idx),
                "metadata_edge": tuple(map(int, metadata_pairs[idx])),
                "model_edge": tuple(map(int, model_pairs[idx])),
            }
            for idx in mismatch_idx[:20]
        ]

        raise ModelInputContractError(
            "link_metadata is not aligned with network_params link order. "
            f"num_mismatches={len(mismatch_idx)} | sample={sample}"
        )

    if "model_link_position" in link_metadata.columns:
        positions = (
            link_metadata["model_link_position"]
            .astype(np.int64)
            .to_numpy()
        )

        expected_positions = np.arange(len(link_metadata), dtype=np.int64)

        if not np.array_equal(positions, expected_positions):
            raise ModelInputContractError(
                "link_metadata['model_link_position'] must equal row position. "
                "This is required because prediction tables attach metadata by row order."
            )

def _as_binary_mask(
    value: Any,
    name: str,
) -> np.ndarray:
    """
    Convert input to a one-dimensional binary mask.

    Parameters
    ----------
    value : Any
        Input value.

    name : str
        Mask name.

    Returns
    -------
    np.ndarray
        Binary mask.
    """

    if torch.is_tensor(value):
        arr = value.detach().cpu().numpy()
    else:
        arr = np.asarray(value)

    arr = arr.astype(np.float32).reshape(-1)

    unique_values = set(np.unique(arr).tolist())

    if not unique_values.issubset({0.0, 1.0}):
        raise ModelInputContractError(
            f"{name} must be binary. Found values: {sorted(unique_values)}."
        )

    return arr


def _cfg_get(
    cfg: Any,
    dotted_key: str,
    default: Any = None,
) -> Any:
    """
    Safely read nested configuration values.

    Parameters
    ----------
    cfg : Any
        Config object.

    dotted_key : str
        Dotted key path.

    default : Any

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


def _dict_get(
    obj: Any,
    key: str,
    default: Any= None,
) -> Any:
    """
    Safely read a key from a dictionary-like object.

    Parameters
    ----------
    obj : Any
        Dictionary-like object.

    key : str
        Key.

    default : Any

    Returns
    -------
    Any
        Retrieved value or default.
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


# =============================================================================
# Hydra entrypoint
# =============================================================================

# =============================================================================
# Hydra entrypoint
# =============================================================================


@hydra.main(
    config_path="../../configs",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint for the training pipeline.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.
    """
    run_pipeline(
        cfg=cfg,
    )


if __name__ == "__main__":
    main()
