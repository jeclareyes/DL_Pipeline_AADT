# src/train/training_input_preparer.py

"""
Training Input Preparer
=======================

This module prepares model-training inputs from a unified training artifact.

Project context
---------------
After the data-processing refactor, the training pipeline should no longer:

- read graph/link/route files separately;
- call RouteModelAdapter;
- search for flow columns;
- flatten OD matrices;
- build flow or OD targets manually;
- create observed masks manually;
- extract visualization payloads from the graph.

Those responsibilities are handled by the data-processing block, which produces
a unified training artifact containing:

- network_params;
- targets;
- visualization payload;
- metadata.

This module acts as the bridge between the loaded training artifact and the
training loop. It prepares:

- global target tensors;
- global observed masks;
- sampled train masks;
- hold-out masks;
- task-specific train/validation tensors;
- model parameter dictionaries.

It does not instantiate models, train models, evaluate metrics, or save
artifacts. Those responsibilities belong to training_pipeline.py, trainer.py
and evaluator.py.

Design principles
-----------------
- Consume the training artifact as the source of truth.
- Keep tensors and masks aligned with artifact ordering.
- Preserve the trainer input contract.
- Avoid data-processing logic in the training pipeline.
- Fail early when required targets or masks are missing.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import DictConfig

from src.contracts.runtime_contracts import (
    ModelInputContractError,
)
from src.train._pipeline_utils import (
    apply_model_pre_instantiate_hook,
    inject_prescaling,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GlobalTrainingInputs:
    """
    Container with global training inputs derived from the training artifact.

    Attributes
    ----------
    network_params : Dict[str, Any]
        Model-ready network parameters.

    targets : Dict[str, Any]
        Artifact target payload.

    visualization : Dict[str, Any]
        Visualization payload from the artifact.

    flows_target_np : np.ndarray
        Link-flow target vector.

    flows_observed_mask_np : np.ndarray
        Binary mask identifying observed link-flow targets.

    od_target_np : np.ndarray
        OD target vector aligned with route OD space.

    od_observed_mask_np : np.ndarray
        Binary mask identifying observed OD targets.

    flow_train_global_mask_np : np.ndarray
        Global flow training mask after applying sampling and observed-mask
        constraints.

    flow_holdout_global_mask_np : np.ndarray
        Global hold-out mask defined as observed flows not used for training.

    od_train_supervision_mask_np : np.ndarray
        OD supervision mask used during training.

    flows_target_t : torch.Tensor
        Flow target tensor.

    od_target_t : torch.Tensor
        OD target tensor.

    od_train_supervision_mask_t : torch.Tensor
        OD supervision mask tensor.
    """

    network_params: Dict[str, Any]
    targets: Dict[str, Any]
    visualization: Dict[str, Any]

    flows_target_np: np.ndarray
    flows_observed_mask_np: np.ndarray
    od_target_np: np.ndarray
    od_observed_mask_np: np.ndarray

    flow_train_global_mask_np: np.ndarray
    flow_holdout_global_mask_np: np.ndarray
    od_train_supervision_mask_np: np.ndarray

    flows_target_t: torch.Tensor
    od_target_t: torch.Tensor
    od_train_supervision_mask_t: torch.Tensor


@dataclass(frozen=True)
class TaskTrainingInputs:
    """
    Container with tensors for one training task.

    Attributes
    ----------
    train_tensors : Dict[str, torch.Tensor]
        Dictionary passed to GeneralTrainer.fit() as train_tensors.

    val_tensors : Dict[str, torch.Tensor]
        Dictionary passed to GeneralTrainer.fit() as val_tensors.

    task_metadata : Dict[str, Any]
        Lightweight task summary useful for logging and experiment metadata.
    """

    train_tensors: Dict[str, torch.Tensor]
    val_tensors: Dict[str, torch.Tensor]
    task_metadata: Dict[str, Any]


@dataclass(frozen=True)
class ModelInputPreparationResult:
    """
    Container with model parameters and scaling metadata.

    Attributes
    ----------
    model_params : Dict[str, Any]
        Parameters passed to Hydra model instantiation.

    metadata : Dict[str, Any]
        Scaling values and hook updates.
    """

    model_params: Dict[str, Any]
    metadata: Dict[str, Any]


class TrainingInputPreparer:
    """
    Prepare training inputs from a loaded training artifact.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    device : str
        Target torch device.

    strict : bool, default=True
        If True, missing keys and incompatible dimensions raise errors.
    """

    REQUIRED_INPUT_KEYS = {
        "network_params",
        "targets",
        "visualization",
    }

    REQUIRED_TARGET_KEYS = {
    "flows_target_np",
    "flows_observed_mask_np",
    "od_target_np",
    "od_observed_mask_np",
    "flow_target_link_pair_indices",
    "od_pairs",
    }

    def __init__(
        self,
        cfg: DictConfig,
        device: str,
        strict: bool = True,
    ) -> None:
        self.cfg = cfg
        self.device = str(device)
        self.strict = bool(strict)

    # ------------------------------------------------------------------
    # Global preparation
    # ------------------------------------------------------------------

    def prepare_global_inputs(
        self,
        artifact_inputs: Dict[str, Any],
        sampled_flow_mask_np: Optional[np.ndarray] = None,
        sampled_od_mask_np: Optional[np.ndarray] = None,
    ) -> GlobalTrainingInputs:
        """
        Prepare global tensors and masks from loaded artifact inputs.

        Parameters
        ----------
        artifact_inputs : Dict[str, Any]
            Dictionary returned by TrainingArtifactLoader.load_training_inputs().
            It must contain network_params, targets and visualization.

        sampled_flow_mask_np : Optional[np.ndarray], default=None
            Optional flow sampling mask. If None, all observed flow targets are
            used for training.

        sampled_od_mask_np : Optional[np.ndarray], default=None
            Optional OD sampling mask. If None, all observed OD targets are used
            for OD supervision.

        Returns
        -------
        GlobalTrainingInputs
            Prepared global inputs.
        """

        self._require_keys(
            artifact_inputs,
            self.REQUIRED_INPUT_KEYS,
            object_name="artifact_inputs",
        )

        self._validate_artifact_alignment(
            network_params=artifact_inputs["network_params"],
            targets=artifact_inputs["targets"],
        )   

        network_params = self.move_to_device(
            artifact_inputs["network_params"],
            self.device,
        )

        targets = self.move_to_device(
            artifact_inputs["targets"],
            self.device,
        )

        visualization = artifact_inputs.get("visualization", {})

        self._require_keys(
            targets,
            self.REQUIRED_TARGET_KEYS,
            object_name="targets",
        )

        flows_target_np = self._as_1d_float_np(
            targets["flows_target_np"],
            name="flows_target_np",
        )

        flows_observed_mask_np = self._as_binary_1d_mask(
            targets["flows_observed_mask_np"],
            name="flows_observed_mask_np",
        )

        od_target_np = self._as_1d_float_np(
            targets["od_target_np"],
            name="od_target_np",
        )

        od_observed_mask_np = self._as_binary_1d_mask(
            targets["od_observed_mask_np"],
            name="od_observed_mask_np",
        )

        self._validate_same_length(
            flows_target_np,
            flows_observed_mask_np,
            "flows_target_np",
            "flows_observed_mask_np",
        )

        self._validate_same_length(
            od_target_np,
            od_observed_mask_np,
            "od_target_np",
            "od_observed_mask_np",
        )

        flow_train_global_mask_np = self._build_flow_train_mask(
            flows_observed_mask_np=flows_observed_mask_np,
            sampled_flow_mask_np=sampled_flow_mask_np,
        )

        flow_holdout_global_mask_np = self._build_flow_holdout_mask(
            flows_observed_mask_np=flows_observed_mask_np,
            flow_train_global_mask_np=flow_train_global_mask_np,
        )

        od_train_supervision_mask_np = self._build_od_train_mask(
            od_observed_mask_np=od_observed_mask_np,
            sampled_od_mask_np=sampled_od_mask_np,
        )

        flows_target_t = self._get_or_create_tensor(
            targets=targets,
            tensor_key="flows_target_t",
            array=flows_target_np,
        )

        od_target_t = self._get_or_create_tensor(
            targets=targets,
            tensor_key="od_target_t",
            array=od_target_np,
        )

        od_train_supervision_mask_t = self._to_float_tensor(
            od_train_supervision_mask_np,
        )

        logger.info(
            "Global training inputs prepared | observed_flows=%d | train_flows=%d | holdout_flows=%d | supervised_od=%d",
            int(flows_observed_mask_np.sum()),
            int(flow_train_global_mask_np.sum()),
            int(flow_holdout_global_mask_np.sum()),
            int(od_train_supervision_mask_np.sum()),
        )

        return GlobalTrainingInputs(
            network_params=network_params,
            targets=targets,
            visualization=visualization,
            flows_target_np=flows_target_np,
            flows_observed_mask_np=flows_observed_mask_np,
            od_target_np=od_target_np,
            od_observed_mask_np=od_observed_mask_np,
            flow_train_global_mask_np=flow_train_global_mask_np,
            flow_holdout_global_mask_np=flow_holdout_global_mask_np,
            od_train_supervision_mask_np=od_train_supervision_mask_np,
            flows_target_t=flows_target_t,
            od_target_t=od_target_t,
            od_train_supervision_mask_t=od_train_supervision_mask_t,
        )

    def _validate_artifact_alignment(
        self,
        network_params: Dict[str, Any],
        targets: Dict[str, Any],
    ) -> None:
        """
        Validate that training targets are aligned with model network parameters.

        This check protects the training pipeline from consuming an artifact where:
            - flow targets follow one link order;
            - network tensors / delta_matrix follow another link order;
            - OD targets follow one OD-pair order;
            - route tensors follow another OD-pair order.

        The data-processing validator should already catch this, but the training
        pipeline must also fail early if validation-on-load is disabled or bypassed.
        """

        if "link_pair_indices" not in network_params:
            raise ModelInputContractError(
                "network_params is missing 'link_pair_indices'. "
                "Cannot verify link-flow target alignment."
            )

        if "od_pairs" not in network_params:
            raise ModelInputContractError(
                "network_params is missing 'od_pairs'. "
                "Cannot verify OD target alignment."
            )

        model_link_pairs = self._as_int_pair_array(
            network_params["link_pair_indices"],
            name="network_params['link_pair_indices']",
        )

        target_link_pairs = self._as_int_pair_array(
            targets["flow_target_link_pair_indices"],
            name="targets['flow_target_link_pair_indices']",
        )

        if model_link_pairs.shape != target_link_pairs.shape:
            raise ModelInputContractError(
                "Link-order shape mismatch between network_params and targets. "
                f"network_params['link_pair_indices'] shape={model_link_pairs.shape}, "
                f"targets['flow_target_link_pair_indices'] shape={target_link_pairs.shape}."
            )

        if not np.array_equal(model_link_pairs, target_link_pairs):
            mismatch_idx = np.where(
                np.any(model_link_pairs != target_link_pairs, axis=1)
            )[0]

            sample = [
                {
                    "position": int(idx),
                    "network_edge": tuple(map(int, model_link_pairs[idx])),
                    "target_edge": tuple(map(int, target_link_pairs[idx])),
                }
                for idx in mismatch_idx[:20]
            ]

            raise ModelInputContractError(
                "Flow targets are not aligned with network_params link order. "
                f"num_mismatches={len(mismatch_idx)} | sample={sample}"
            )

        model_od_pairs = self._as_od_pair_list(
            network_params["od_pairs"],
            name="network_params['od_pairs']",
        )

        target_od_pairs = self._as_od_pair_list(
            targets["od_pairs"],
            name="targets['od_pairs']",
        )

        if model_od_pairs != target_od_pairs:
            mismatch_positions = [
                idx
                for idx, (model_pair, target_pair) in enumerate(
                    zip(model_od_pairs, target_od_pairs)
                )
                if model_pair != target_pair
            ]

            length_mismatch = len(model_od_pairs) != len(target_od_pairs)

            sample = [
                {
                    "position": int(idx),
                    "network_od": model_od_pairs[idx],
                    "target_od": target_od_pairs[idx],
                }
                for idx in mismatch_positions[:20]
            ]

            raise ModelInputContractError(
                "OD targets are not aligned with network_params OD order. "
                f"length_mismatch={length_mismatch} | "
                f"network_od_count={len(model_od_pairs)} | "
                f"target_od_count={len(target_od_pairs)} | "
                f"num_position_mismatches={len(mismatch_positions)} | "
                f"sample={sample}"
            )

    def _build_flow_train_mask(
        self,
        flows_observed_mask_np: np.ndarray,
        sampled_flow_mask_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build the global flow training mask.

        Parameters
        ----------
        flows_observed_mask_np : np.ndarray
            Binary observed-flow mask.

        sampled_flow_mask_np : Optional[np.ndarray]
            Optional sampling mask.

        Returns
        -------
        np.ndarray
            Binary global flow training mask.
        """

        if sampled_flow_mask_np is None:
            train_mask = flows_observed_mask_np.copy()
        else:
            sampled = self._as_binary_1d_mask(
                sampled_flow_mask_np,
                name="sampled_flow_mask_np",
            )

            self._validate_same_length(
                sampled,
                flows_observed_mask_np,
                "sampled_flow_mask_np",
                "flows_observed_mask_np",
            )

            # A link can be used for training only if it was sampled and observed.
            train_mask = sampled * flows_observed_mask_np

        train_mask = self._clip_binary_mask(train_mask)

        if self.strict and train_mask.sum() <= 0:
            raise ModelInputContractError(
                "Global flow training mask is empty after applying observed-mask constraints."
            )

        return train_mask.astype(np.float32)

    def _build_flow_holdout_mask(
        self,
        flows_observed_mask_np: np.ndarray,
        flow_train_global_mask_np: np.ndarray,
    ) -> np.ndarray:
        """
        Build the global flow hold-out mask.

        Hold-out is defined as:

            observed flows - training flows

        Parameters
        ----------
        flows_observed_mask_np : np.ndarray
            Binary observed-flow mask.

        flow_train_global_mask_np : np.ndarray
            Binary global training mask.

        Returns
        -------
        np.ndarray
            Binary hold-out mask.
        """

        holdout = flows_observed_mask_np - flow_train_global_mask_np
        holdout = self._clip_binary_mask(holdout)

        return holdout.astype(np.float32)

    def _build_od_train_mask(
        self,
        od_observed_mask_np: np.ndarray,
        sampled_od_mask_np: Optional[np.ndarray],
    ) -> np.ndarray:
        """
        Build the OD supervision mask.

        Parameters
        ----------
        od_observed_mask_np : np.ndarray
            Binary observed-OD mask.

        sampled_od_mask_np : Optional[np.ndarray]
            Optional OD sampling mask.

        Returns
        -------
        np.ndarray
            Binary OD supervision mask.
        """

        if sampled_od_mask_np is None:
            od_mask = od_observed_mask_np.copy()
        else:
            sampled = self._as_binary_1d_mask(
                sampled_od_mask_np,
                name="sampled_od_mask_np",
            )

            if sampled.shape[0] != od_observed_mask_np.shape[0]:
                message = (
                    "Sampled OD mask size does not match artifact OD target space. "
                    "Falling back to observed OD mask."
                )

                if self.strict:
                    raise ModelInputContractError(
                        f"{message} sampled={sampled.shape[0]}, "
                        f"expected={od_observed_mask_np.shape[0]}"
                    )

                logger.warning(message)
                od_mask = od_observed_mask_np.copy()
            else:
                od_mask = sampled * od_observed_mask_np

        od_mask = self._clip_binary_mask(od_mask)

        if self.strict and od_mask.sum() <= 0:
            logger.warning(
                "OD supervision mask is empty. Training may still work for models "
                "that do not require supervised OD targets."
            )

        return od_mask.astype(np.float32)

    # ------------------------------------------------------------------
    # Task-specific preparation
    # ------------------------------------------------------------------

    def prepare_task_tensors(
        self,
        global_inputs: GlobalTrainingInputs,
        task: Dict[str, Any],
    ) -> TaskTrainingInputs:
        """
        Prepare train and validation tensors for one training task.

        Parameters
        ----------
        global_inputs : GlobalTrainingInputs
            Global prepared inputs.

        task : Dict[str, Any]
            Training task generated by generate_training_tasks().

            Preferred keys:
            - train_mask
            - val_mask

            Backward-compatible key:
            - flow_train_input_mask

        Returns
        -------
        TaskTrainingInputs
            Train tensors, validation tensors and task metadata.
        """

        train_mask_np = self._resolve_task_train_mask(task)
        val_mask_np = self._resolve_task_val_mask(task)

        train_mask_np = self._as_binary_1d_mask(
            train_mask_np,
            name="task_train_mask",
        )

        val_mask_np = self._as_binary_1d_mask(
            val_mask_np,
            name="task_val_mask",
        )

        self._validate_same_length(
            train_mask_np,
            global_inputs.flows_target_np,
            "task_train_mask",
            "flows_target_np",
        )

        self._validate_same_length(
            val_mask_np,
            global_inputs.flows_target_np,
            "task_val_mask",
            "flows_target_np",
        )

        # Ensure task masks cannot supervise unobserved flow targets.
        train_mask_np = self._clip_binary_mask(
            train_mask_np * global_inputs.flows_observed_mask_np
        )

        val_mask_np = self._clip_binary_mask(
            val_mask_np * global_inputs.flows_observed_mask_np
        )

        if self.strict and train_mask_np.sum() <= 0:
            raise ModelInputContractError(
                f"Task '{task.get('name', 'unknown')}' has an empty train mask."
            )

        train_tensors = {
            "flows": global_inputs.flows_target_t,
            "od": global_inputs.od_target_t,
            "mask": self._to_float_tensor(train_mask_np),
            "od_mask": global_inputs.od_train_supervision_mask_t,
        }

        val_tensors = {
            "mask": self._to_float_tensor(val_mask_np),
        }

        task_metadata = {
            "name": task.get("name", "Unnamed Task"),
            "suffix": task.get("suffix", ".pt"),
            "num_train_links": int(train_mask_np.sum()),
            "num_val_links": int(val_mask_np.sum()),
            "num_supervised_od": int(
                global_inputs.od_train_supervision_mask_np.sum()
            ),
        }

        return TaskTrainingInputs(
            train_tensors=train_tensors,
            val_tensors=val_tensors,
            task_metadata=task_metadata,
        )

    @staticmethod
    def _resolve_task_train_mask(task: Dict[str, Any]) -> Any:
        """
        Resolve the task training mask with backward compatibility.

        Parameters
        ----------
        task : Dict[str, Any]
            Training task.

        Returns
        -------
        Any
            Training mask.
        """

        if "train_mask" in task:
            return task["train_mask"]

        if "flow_train_input_mask" in task:
            return task["flow_train_input_mask"]

        raise ModelInputContractError(
            "Task is missing a training mask. Expected 'train_mask' or "
            "legacy 'flow_train_input_mask'."
        )

    @staticmethod
    def _resolve_task_val_mask(task: Dict[str, Any]) -> Any:
        """
        Resolve the task validation mask.

        Parameters
        ----------
        task : Dict[str, Any]
            Training task.

        Returns
        -------
        Any
            Validation mask.
        """

        if "val_mask" not in task:
            raise ModelInputContractError(
                "Task is missing validation mask 'val_mask'."
            )

        return task["val_mask"]

    # ------------------------------------------------------------------
    # Model parameter preparation
    # ------------------------------------------------------------------

    def prepare_model_params(
        self,
        global_inputs: GlobalTrainingInputs,
        task: Dict[str, Any],
        extra_context: Optional[Dict[str, Any]] = None,
    ) -> ModelInputPreparationResult:
        """
        Prepare model parameters for Hydra instantiation.

        This method copies network_params, injects OD initialization metadata,
        applies feature pre-scaling and executes optional model hooks.

        Parameters
        ----------
        global_inputs : GlobalTrainingInputs
            Global prepared training inputs.

        task : Dict[str, Any]
            Current training task.

        extra_context : Optional[Dict[str, Any]], default=None
            Extra context passed to optional model hooks.

        Returns
        -------
        ModelInputPreparationResult
            Model parameters and preparation metadata.
        """

        model_params = copy.copy(global_inputs.network_params)

        od_known_mean = self._compute_observed_mean(
            values=global_inputs.od_target_np,
            mask=global_inputs.od_observed_mask_np,
            fallback_value=1.0,
        )

        model_params["od_known_mean"] = od_known_mean

        link_scale_val, od_scale_val = inject_prescaling(
            cfg=self.cfg,
            model_params=model_params,
            all_flows_np=global_inputs.flows_target_np,
            observed_flow_mask_np=global_inputs.flows_observed_mask_np,
            od_vector_np=global_inputs.od_target_np,
            observed_od_mask_np=global_inputs.od_observed_mask_np,
            network_params=global_inputs.network_params,
        )

        context = {
            "task": task,
            "network_params": global_inputs.network_params,
            "targets": global_inputs.targets,
            "visualization": global_inputs.visualization,
            "link_scale_val": link_scale_val,
            "od_scale_val": od_scale_val,
        }

        if extra_context:
            context.update(extra_context)

        hook_updates = apply_model_pre_instantiate_hook(
            cfg=self.cfg,
            model_params=model_params,
            context=context,
        )

        metadata = {
            "od_known_mean": float(od_known_mean),
            "link_scale": float(link_scale_val),
            "od_scale": float(od_scale_val),
            "hook_updates": hook_updates or {},
        }

        return ModelInputPreparationResult(
            model_params=model_params,
            metadata=metadata,
        )

    @staticmethod
    def _compute_observed_mean(
        values: np.ndarray,
        mask: np.ndarray,
        fallback_value: float = 1.0,
    ) -> float:
        """
        Compute mean value over observed entries.

        Parameters
        ----------
        values : np.ndarray
            Target vector.

        mask : np.ndarray
            Binary observed mask.

        fallback_value : float, default=1.0
            Value used when no observed entries exist.

        Returns
        -------
        float
            Observed mean or fallback value.
        """

        observed = values[mask.astype(bool)]

        if observed.size == 0:
            logger.warning(
                "No observed values available to compute mean. Using fallback %.4f.",
                fallback_value,
            )
            return float(fallback_value)

        return float(observed.mean())

    # ------------------------------------------------------------------
    # Generic helpers
    # ------------------------------------------------------------------

    def _get_or_create_tensor(
        self,
        targets: Dict[str, Any],
        tensor_key: str,
        array: np.ndarray,
    ) -> torch.Tensor:
        """
        Get an existing tensor from targets or create one from a NumPy array.

        Parameters
        ----------
        targets : Dict[str, Any]
            Target payload.

        tensor_key : str
            Tensor key to retrieve.

        array : np.ndarray
            Fallback NumPy array.

        Returns
        -------
        torch.Tensor
            Tensor on the configured device.
        """

        if tensor_key in targets and torch.is_tensor(targets[tensor_key]):
            return targets[tensor_key].to(self.device)

        return self._to_float_tensor(array)

    def _to_float_tensor(self, array: Any) -> torch.Tensor:
        """
        Convert input to float tensor on the configured device.

        Parameters
        ----------
        array : Any
            Input array-like object.

        Returns
        -------
        torch.Tensor
            Float tensor.
        """

        if torch.is_tensor(array):
            return array.detach().to(
                device=self.device,
                dtype=torch.float32,
            )

        return torch.tensor(
            np.asarray(array, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        )

    @staticmethod
    def _as_1d_float_np(
        value: Any,
        name: str,
    ) -> np.ndarray:
        """
        Convert an input to a one-dimensional float NumPy array.

        Parameters
        ----------
        value : Any
            Input value.

        name : str
            Human-readable name used in errors.

        Returns
        -------
        np.ndarray
            One-dimensional float32 array.
        """

        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)

        arr = arr.astype(np.float32).reshape(-1)

        if not np.isfinite(arr).all():
            raise ModelInputContractError(
                f"{name} contains NaN or infinite values."
            )

        return arr

    @staticmethod
    def _as_int_pair_array(
        value: Any,
        name: str,
    ) -> np.ndarray:
        """
        Convert an input object to an integer array with shape [N, 2].
        """

        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
        else:
            arr = np.asarray(value)

        arr = arr.astype(np.int64)

        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ModelInputContractError(
                f"{name} must have shape [N, 2]. Got shape {arr.shape}."
            )

        return arr


    @staticmethod
    def _as_od_pair_list(
        value: Any,
        name: str,
    ) -> list[tuple[int, int]]:
        """
        Convert an OD-pair payload to a list of integer tuples.
        """

        if torch.is_tensor(value):
            arr = value.detach().cpu().numpy()
            raw_pairs = arr.tolist()
        else:
            raw_pairs = value

        try:
            od_pairs = [
                (int(origin), int(destination))
                for origin, destination in raw_pairs
            ]
        except Exception as exc:
            raise ModelInputContractError(
                f"{name} could not be interpreted as a sequence of OD pairs."
            ) from exc

        return od_pairs

    def _as_binary_1d_mask(
        self,
        value: Any,
        name: str,
    ) -> np.ndarray:
        """
        Convert an input to a one-dimensional binary float mask.

        Parameters
        ----------
        value : Any
            Input value.

        name : str
            Human-readable name used in errors.

        Returns
        -------
        np.ndarray
            One-dimensional binary float32 mask.
        """

        arr = self._as_1d_float_np(value, name=name)
        arr = self._clip_binary_mask(arr)

        unique_values = set(np.unique(arr).tolist())

        if not unique_values.issubset({0.0, 1.0}):
            raise ModelInputContractError(
                f"{name} must be binary after clipping. "
                f"Found values: {sorted(unique_values)}"
            )

        return arr.astype(np.float32)

    @staticmethod
    def _clip_binary_mask(mask: np.ndarray) -> np.ndarray:
        """
        Clip a numeric mask to binary range.

        Parameters
        ----------
        mask : np.ndarray
            Numeric mask.

        Returns
        -------
        np.ndarray
            Binary-like mask in {0.0, 1.0}.
        """

        return np.clip(
            np.asarray(mask, dtype=np.float32),
            0.0,
            1.0,
        )

    @staticmethod
    def _validate_same_length(
        left: np.ndarray,
        right: np.ndarray,
        left_name: str,
        right_name: str,
    ) -> None:
        """
        Validate that two one-dimensional arrays have the same length.

        Parameters
        ----------
        left : np.ndarray
            First array.

        right : np.ndarray
            Second array.

        left_name : str
            First array name.

        right_name : str
            Second array name.
        """

        if left.shape[0] != right.shape[0]:
            raise ModelInputContractError(
                f"{left_name} and {right_name} must have the same length. "
                f"Got {left.shape[0]} and {right.shape[0]}."
            )

    @staticmethod
    def _require_keys(
        obj: Dict[str, Any],
        required_keys: set[str],
        object_name: str,
    ) -> None:
        """
        Validate that a dictionary contains required keys.

        Parameters
        ----------
        obj : Dict[str, Any]
            Dictionary to validate.

        required_keys : set[str]
            Required keys.

        object_name : str
            Human-readable object name.
        """

        missing = required_keys - set(obj.keys())

        if missing:
            raise ModelInputContractError(
                f"{object_name} is missing required keys: {sorted(missing)}"
            )

    def move_to_device(
        self,
        obj: Any,
        device: str,
    ) -> Any:
        """
        Recursively move torch tensors to a target device.

        Parameters
        ----------
        obj : Any
            Object possibly containing tensors.

        device : str
            Target device.

        Returns
        -------
        Any
            Object with tensors moved to device.
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


def prepare_global_training_inputs(
    cfg: DictConfig,
    device: str,
    artifact_inputs: Dict[str, Any],
    sampled_flow_mask_np: Optional[np.ndarray] = None,
    sampled_od_mask_np: Optional[np.ndarray] = None,
    strict: bool = True,
) -> GlobalTrainingInputs:
    """
    Convenience function to prepare global training inputs.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    device : str
        Target torch device.

    artifact_inputs : Dict[str, Any]
        Inputs returned by TrainingArtifactLoader.load_training_inputs().

    sampled_flow_mask_np : Optional[np.ndarray], default=None
        Optional flow sampling mask.

    sampled_od_mask_np : Optional[np.ndarray], default=None
        Optional OD sampling mask.

    strict : bool, default=True
        Whether to use strict validation.

    Returns
    -------
    GlobalTrainingInputs
        Prepared global inputs.
    """

    preparer = TrainingInputPreparer(
        cfg=cfg,
        device=device,
        strict=strict,
    )

    return preparer.prepare_global_inputs(
        artifact_inputs=artifact_inputs,
        sampled_flow_mask_np=sampled_flow_mask_np,
        sampled_od_mask_np=sampled_od_mask_np,
    )


def prepare_task_training_inputs(
    cfg: DictConfig,
    device: str,
    global_inputs: GlobalTrainingInputs,
    task: Dict[str, Any],
    strict: bool = True,
) -> TaskTrainingInputs:
    """
    Convenience function to prepare one task's train and validation tensors.

    Parameters
    ----------
    cfg : DictConfig
        Full Hydra configuration.

    device : str
        Target torch device.

    global_inputs : GlobalTrainingInputs
        Prepared global inputs.

    task : Dict[str, Any]
        Training task.

    strict : bool, default=True
        Whether to use strict validation.

    Returns
    -------
    TaskTrainingInputs
        Task-specific training tensors.
    """

    preparer = TrainingInputPreparer(
        cfg=cfg,
        device=device,
        strict=strict,
    )

    return preparer.prepare_task_tensors(
        global_inputs=global_inputs,
        task=task,
    )