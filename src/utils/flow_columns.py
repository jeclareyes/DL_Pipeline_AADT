"""Canonical flow-column contract shared by datasets and artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Any, Mapping
import json
from pathlib import Path

import pandas as pd


FLOW_COLUMN_TYPES = ("traffic_counts", "reference_assignment", "estimated_flows")
_CANONICAL_PATTERNS = {
    "traffic_counts": re.compile(r"^TC_.+$"),
    "reference_assignment": re.compile(r"^RA_.+$"),
    "estimated_flows": re.compile(r"^EF_.+$"),
}


@dataclass(frozen=True)
class FlowColumnContract:
    """Declared names for the flow columns available at an artifact stage."""

    traffic_counts: tuple[str, ...]
    reference_assignment: str | None
    estimated_flows: str | None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FlowColumnContract":
        unknown = set(value) - set(FLOW_COLUMN_TYPES)
        if unknown:
            raise KeyError(f"Unknown flow_columns keys: {sorted(unknown)}")

        traffic_counts_value = value.get("traffic_counts", [])
        if traffic_counts_value is None:
            traffic_counts_value = []
        if isinstance(traffic_counts_value, str):
            traffic_counts_value = [traffic_counts_value]
        if not isinstance(traffic_counts_value, (list, tuple)):
            raise TypeError("flow_columns.traffic_counts must be a list of names.")

        traffic_counts = tuple(str(item) for item in traffic_counts_value)
        reference_assignment = _optional_name(value.get("reference_assignment"))
        estimated_flows = _optional_name(value.get("estimated_flows"))

        contract = cls(
            traffic_counts=traffic_counts,
            reference_assignment=reference_assignment,
            estimated_flows=estimated_flows,
        )
        contract.validate()
        return contract

    def validate(self) -> None:
        names = list(self.traffic_counts)
        if self.reference_assignment is not None:
            names.append(self.reference_assignment)
        if self.estimated_flows is not None:
            names.append(self.estimated_flows)
        if len(names) != len(set(names)):
            raise ValueError(f"flow_columns contains duplicate names: {names}")

        for subtype, subtype_names in (
            ("traffic_counts", self.traffic_counts),
            ("reference_assignment", (self.reference_assignment,) if self.reference_assignment else ()),
            ("estimated_flows", (self.estimated_flows,) if self.estimated_flows else ()),
        ):
            pattern = _CANONICAL_PATTERNS[subtype]
            for name in subtype_names:
                if not pattern.match(name):
                    raise ValueError(
                        f"flow_columns.{subtype} name {name!r} must match "
                        f"{pattern.pattern}."
                    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "traffic_counts": list(self.traffic_counts),
            "reference_assignment": self.reference_assignment,
            "estimated_flows": self.estimated_flows,
        }

    def all_declared(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in (
                *self.traffic_counts,
                self.reference_assignment,
                self.estimated_flows,
            )
            if name is not None
        )

    def first_traffic_count(self) -> str:
        if not self.traffic_counts:
            raise ValueError("No traffic-count column is declared for this dataset.")
        return self.traffic_counts[0]

    def default_training_column(self) -> str:
        if self.traffic_counts:
            return self.traffic_counts[0]
        if self.reference_assignment is not None:
            return self.reference_assignment
        raise ValueError("No flow column is available for training.")

    def rename_legacy_columns(
        self,
        frame: pd.DataFrame,
        *,
        logger: logging.Logger | None = None,
    ) -> pd.DataFrame:
        """Rename legacy flow names to the declared canonical names.

        Matching is centralized here so readers and exporters do not each
        implement their own ``Volume_``/``assigned_flow`` heuristics.
        """

        log = logger or logging.getLogger(__name__)
        columns = {str(column): column for column in frame.columns}
        rename_map: dict[Any, str] = {}
        used_sources: set[Any] = set()

        for target in self.all_declared():
            if target in columns:
                continue
            source = _find_legacy_source(target, columns, used_sources)
            if source is not None:
                rename_map[source] = target
                used_sources.add(source)
                log.info("Renamed flow column [%s -> %s]", source, target)

        if not rename_map:
            return frame.copy()
        return frame.rename(columns=rename_map).copy()


def _optional_name(value: Any) -> str | None:
    if value in (None, "", False):
        return None
    return str(value)


def _find_legacy_source(
    target: str,
    columns: Mapping[str, Any],
    used_sources: set[Any],
) -> Any | None:
    suffix = target.split("_", 1)[1]
    candidates = (
        target,
        f"Volume_{suffix}",
        f"volume_{suffix}",
        f"Flow_{suffix}",
        f"flow_{suffix}",
        "Volume" if target.startswith("TC_") else None,
        "volume" if target.startswith("TC_") else None,
        "assigned_flow" if target.startswith("RA_") else None,
    )
    lower_columns = {name.lower(): original for name, original in columns.items()}
    for candidate in candidates:
        if candidate is not None and candidate.lower() in lower_columns:
            original = lower_columns[candidate.lower()]
            if original not in used_sources:
                return original
    return None


def flow_columns_from_config(config: Mapping[str, Any]) -> FlowColumnContract:
    value = config.get("flow_columns")
    if not isinstance(value, Mapping):
        raise KeyError("Dataset configuration must define a flow_columns mapping.")
    return FlowColumnContract.from_mapping(value)


def flow_columns_from_dataset_config(
    config: Mapping[str, Any],
    *,
    creation_manifest_path: str | Path | None = None,
) -> FlowColumnContract:
    """Resolve the declared contract, optionally completing it from creation."""

    declared = dict(config.get("flow_columns", {}))
    if creation_manifest_path not in (None, ""):
        path = Path(str(creation_manifest_path))
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
            generated = manifest.get("flow_columns") or manifest.get("metadata", {}).get(
                "flow_columns", {}
            )
            if isinstance(generated, Mapping):
                for key in FLOW_COLUMN_TYPES:
                    current = declared.get(key)
                    if current in (None, "", []):
                        declared[key] = generated.get(key, current)
    return FlowColumnContract.from_mapping(declared)
