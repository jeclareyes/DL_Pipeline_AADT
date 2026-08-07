"""Implementation of immutable post-training artifact materialization."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import numpy as np
import pandas as pd

from src.data_handling.data_processing.artifact_builders.training_artifact_builder import (
    TrainingArtifactBuilder,
)
from src.utils.flow_columns import FlowColumnContract
from src.utils.serialization import dump, load


def materialize_post_trained_artifact(
    source_artifact_path: str | Path,
    estimated_flows: Sequence[float] | np.ndarray,
    *,
    year: int | str,
    output_artifact_path: str | Path,
    output_manifest_path: str | Path | None = None,
) -> dict[str, str]:
    """Copy an experiment artifact and add its model-estimated flow column."""

    source_path = Path(source_artifact_path)
    output_path = Path(output_artifact_path)
    artifact = deepcopy(load(source_path))
    values = np.asarray(estimated_flows, dtype=float).reshape(-1)
    column = f"EF_{year}"

    processed = artifact.get("processed")
    if not isinstance(processed, dict) or not isinstance(processed.get("link_df"), pd.DataFrame):
        raise TypeError("Source artifact must contain processed.link_df as a DataFrame.")
    link_df = processed["link_df"].copy()
    if len(link_df) != len(values):
        raise ValueError(
            f"Estimated flow length {len(values)} does not match link count {len(link_df)}."
        )
    link_df[column] = values
    processed["link_df"] = link_df

    raw = artifact.get("raw")
    if isinstance(raw, dict) and isinstance(raw.get("flow_df"), pd.DataFrame):
        raw_flow = raw["flow_df"].copy()
        if len(raw_flow) == len(values):
            raw_flow[column] = values
            raw["flow_df"] = raw_flow

    contract = FlowColumnContract.from_mapping(
        artifact.get("metadata", {}).get("flow_columns", {})
    )
    updated_flow_columns = FlowColumnContract(
        traffic_counts=contract.traffic_counts,
        reference_assignment=contract.reference_assignment,
        estimated_flows=column,
    )
    updated_flow_columns.validate()

    artifact["artifact_type"] = "post_training_artifact"
    artifact["created_at"] = datetime.now().isoformat()
    artifact["metadata"] = dict(artifact.get("metadata", {}))
    artifact["metadata"]["flow_columns"] = updated_flow_columns.as_dict()
    artifact["metadata"]["post_training"] = {
        "source_artifact_path": str(source_path),
        "estimated_flow_column": column,
        "estimated_flow_year": str(year),
    }
    artifact["processed"] = processed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dump(artifact, output_path)

    manifest_path = Path(output_manifest_path) if output_manifest_path else output_path.with_name(
        "post_trained_artifact_manifest.json"
    )
    builder = object.__new__(TrainingArtifactBuilder)
    builder.paths = SimpleNamespace(artifact_path=output_path)
    manifest = builder.build_manifest(
        artifact,
        entry_name="post_training_artifact",
        artifact_path=output_path,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {
        "artifact_path": str(output_path),
        "manifest_path": str(manifest_path),
        "estimated_flow_column": column,
    }


__all__ = ["materialize_post_trained_artifact"]
