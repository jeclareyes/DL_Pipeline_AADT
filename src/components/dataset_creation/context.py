from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.utils.paths import get_project_root, resolve_path


@dataclass
class DatasetState:
    """Shared mutable state for scenario creation."""

    config: DictConfig
    scenario_name: str
    rng: np.random.Generator

    nodes_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    links_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    graph: Any = None

    base_demand: np.ndarray | None = None
    trips_array: np.ndarray | None = None
    routes_by_od: dict[tuple[int, int], list[list[int]]] = field(default_factory=dict)
    flows_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    nodes_metadata: dict[str, Any] = field(default_factory=dict)
    links_metadata: dict[str, Any] = field(default_factory=dict)
    trips_metadata: dict[str, Any] = field(default_factory=dict)
    routes_metadata: dict[str, Any] = field(default_factory=dict)
    flows_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def output_dir(self):
        return resolve_path(
            self.config.paths.export_dirs.generated_dataset,
            relative_to=self.config.paths.root_dir or get_project_root(),
        )
