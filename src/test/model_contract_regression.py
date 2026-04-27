"""Minimal contract regression checks for train/test runtime validators.

Run:
    .venv/bin/python src/test/model_contract_regression.py
"""

from __future__ import annotations

import torch

from src.contracts.runtime_contracts import (
    EvalBundleContractError,
    ModelInputContractError,
    ModelOutputContractError,
    validate_artifacts_contract,
    validate_eval_bundle_contract,
    validate_model_input_contract,
    validate_model_output_contract,
)


def _expect_raises(exc_type: type[Exception], fn) -> None:
    try:
        fn()
    except exc_type:
        return
    except Exception as exc:  # pragma: no cover - defensive path for clearer failures
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(exc).__name__}") from exc
    raise AssertionError(f"Expected {exc_type.__name__}, but no exception was raised")


def _test_model_input_contract() -> None:
    train_tensors = {
        "flows": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "mask": torch.tensor([1.0, 1.0], dtype=torch.float32),
        "od": torch.tensor([3.0], dtype=torch.float32),
        "od_mask": torch.tensor([1.0], dtype=torch.float32),
    }
    val_tensors = {"mask": torch.tensor([0.0, 1.0], dtype=torch.float32)}

    validate_model_input_contract(train_tensors, val_tensors)

    _expect_raises(
        ModelInputContractError,
        lambda: validate_model_input_contract({"flows": train_tensors["flows"]}, val_tensors),
    )
    _expect_raises(
        ModelInputContractError,
        lambda: validate_model_input_contract(train_tensors, {}),
    )


def _test_model_output_contract() -> None:
    valid_outputs = {
        "reconstructed_flows": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "estimated_demand": torch.tensor([3.0], dtype=torch.float32),
        "loss": {
            "total_loss": torch.tensor(1.0, dtype=torch.float32),
            "l_flow": torch.tensor(1.0, dtype=torch.float32),
        },
    }

    outputs_map = validate_model_output_contract(valid_outputs)
    assert torch.is_tensor(outputs_map["loss"]["total_loss"]), "total_loss must remain tensor"

    _expect_raises(
        ModelOutputContractError,
        lambda: validate_model_output_contract(
            {
                "reconstructed_flows": valid_outputs["reconstructed_flows"],
                "estimated_demand": valid_outputs["estimated_demand"],
            }
        ),
    )
    _expect_raises(
        ModelOutputContractError,
        lambda: validate_model_output_contract(
            {
                "reconstructed_flows": valid_outputs["reconstructed_flows"],
                "estimated_demand": valid_outputs["estimated_demand"],
                "loss": {"total_loss": 1.0},
            }
        ),
    )


def _test_eval_bundle_and_artifacts_contract() -> None:
    valid_bundle = {
        "schema_version": 1,
        "static_data": {"dummy": 1},
        "epochs_history": {
            "1": {
                "artifacts": {
                    "pred_flows": torch.tensor([1.0], dtype=torch.float32),
                    "pred_od": torch.tensor([1.0], dtype=torch.float32),
                }
            }
        },
    }

    validate_eval_bundle_contract(valid_bundle)

    artifacts = valid_bundle["epochs_history"]["1"]["artifacts"]
    validate_artifacts_contract(artifacts)

    _expect_raises(
        EvalBundleContractError,
        lambda: validate_eval_bundle_contract(
            {
                "schema_version": 0,
                "static_data": {},
                "epochs_history": {"1": {}},
            }
        ),
    )
    _expect_raises(
        EvalBundleContractError,
        lambda: validate_eval_bundle_contract(
            {
                "schema_version": 1,
                "static_data": {},
                "epochs_history": {},
            }
        ),
    )


def main() -> None:
    _test_model_input_contract()
    _test_model_output_contract()
    _test_eval_bundle_and_artifacts_contract()

    print("[OK] Model contract regression checks passed.")


if __name__ == "__main__":
    main()
