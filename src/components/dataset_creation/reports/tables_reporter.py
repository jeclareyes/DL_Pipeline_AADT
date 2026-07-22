from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def _write_csv(df: pd.DataFrame, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=True)
    return output


def export_all_tables(data: dict[str, Any], config, metadata: dict[str, Any]) -> dict[str, Any]:
    info_dir = Path(config.paths.export_dirs.info_dataset)
    info_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    if data.get("nodes") is not None and not data["nodes"].empty:
        outputs["nodes_csv"] = str(_write_csv(data["nodes"], info_dir / "nodes_df.csv"))
    if data.get("network") is not None and not data["network"].empty:
        outputs["network_csv"] = str(_write_csv(data["network"], info_dir / "network_df.csv"))
    if metadata.get("routes") and metadata["routes"].get("routes_df") is not None:
        outputs["routes_csv"] = str(_write_csv(metadata["routes"]["routes_df"], info_dir / "routes_df.csv"))
    if data.get("flows") is not None and not data["flows"].empty:
        outputs["flows_csv"] = str(_write_csv(data["flows"], info_dir / "flows_df.csv"))
    return outputs
