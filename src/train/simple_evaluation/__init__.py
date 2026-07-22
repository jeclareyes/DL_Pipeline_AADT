# src/train/simple_evaluation/__init__.py

"""
Simple evaluation package.

This package exposes the model-agnostic post-training evaluator used by the
training pipeline.
"""

from src.train.simple_evaluation.evaluator import (
    EvaluationResult,
    build_task_evaluation_summary,
    compute_flow_conservation_summary,
    compute_masked_regression_metrics,
    evaluate_full_reconstruction,
    evaluate_holdout_metrics,
    evaluate_masked_reconstruction,
    evaluate_validation_metrics,
    run_flow_reconstruction,
    save_evaluation_result,
    save_task_evaluation_summary,
)

__all__ = [
    "EvaluationResult",
    "build_task_evaluation_summary",
    "compute_flow_conservation_summary",
    "compute_masked_regression_metrics",
    "evaluate_full_reconstruction",
    "evaluate_holdout_metrics",
    "evaluate_masked_reconstruction",
    "evaluate_validation_metrics",
    "run_flow_reconstruction",
    "save_evaluation_result",
    "save_task_evaluation_summary",
]