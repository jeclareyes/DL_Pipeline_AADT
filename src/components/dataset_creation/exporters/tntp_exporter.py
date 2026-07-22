from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


def tntp_path(filepath: str | Path) -> Path:
    path = Path(filepath)
    if path.suffix != ".tntp":
        path = path.with_suffix(".tntp")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe_as_tntp(df: pd.DataFrame, filepath: str | Path, sep: str = "\t") -> Path:
    path = tntp_path(filepath)
    df.to_csv(path, sep=sep, index=False)
    return path


def save_text_as_tntp(text: str, filepath: str | Path) -> Path:
    path = tntp_path(filepath)
    path.write_text(text, encoding="utf-8")
    return path


def save_routes_as_tntp(
    routes_by_od: Mapping[tuple[int, int], list[list[int]]],
    routes_path: str | Path,
) -> Path:
    path = tntp_path(routes_path)
    lines = [str(routes) if routes else "[]" for routes in routes_by_od.values()]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def format_trips_as_tntp(
    trips_array: np.ndarray,
    zone_ids: np.ndarray,
    start_date: datetime,
    entries_per_line: int = 6,
) -> str:
    zone_id_to_idx = {int(zone_id): idx for idx, zone_id in enumerate(zone_ids)}
    lines = [
        f"<NUMBER OF ZONES> {len(zone_ids)}",
        "<END OF METADATA>",
        "",
    ]

    num_days = trips_array.shape[0]
    for day_idx in range(num_days):
        current_date = (start_date + timedelta(days=day_idx)).strftime("%Y-%m-%d")
        for hour in range(24):
            lines.append(f"<MATRIX DATE> {current_date} {hour:02d}:00")
            for origin_id in zone_ids:
                origin_id = int(origin_id)
                origin_idx = zone_id_to_idx[origin_id]
                lines.append(f"Origin {origin_id}")
                entries = []
                for destination_id in zone_ids:
                    destination_id = int(destination_id)
                    destination_idx = zone_id_to_idx[destination_id]
                    flow = trips_array[day_idx, hour, origin_idx, destination_idx]
                    entries.append(f"{destination_id} : {flow:.2f};")
                for i in range(0, len(entries), entries_per_line):
                    lines.append("    " + " ".join(entries[i : i + entries_per_line]))
            lines.append("")
    return "\n".join(lines)

