from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from src.components.dataset_creation.config import DatasetConfig
from src.components.dataset_creation.exporters.tntp_exporter import format_trips_as_tntp, save_text_as_tntp


def _generate_base_demand(
    *,
    zone_ids: np.ndarray,
    min_trips_od: float,
    max_trips_od: float,
    include_intrazonal: bool,
    rng: np.random.Generator,
) -> np.ndarray:
    base_demand = rng.uniform(min_trips_od, max_trips_od, size=(len(zone_ids), len(zone_ids)))
    if not include_intrazonal:
        np.fill_diagonal(base_demand, np.nan)
    return base_demand


def _generate_trips_array(
    *,
    base_demand: np.ndarray,
    hourly_profile: np.ndarray,
    num_days: int,
    rng: np.random.Generator,
    noise_low: float = 0.95,
    noise_high: float = 1.05,
) -> np.ndarray:
    noise = rng.uniform(noise_low, noise_high, size=(num_days, 24, *base_demand.shape))
    return base_demand[None, None, :, :] * hourly_profile[None, :, None, None] * noise


def build_trips(config: DatasetConfig, data: dict[str, Any], metadata: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    trips_cfg = config.DemandParameters
    num_days = int(trips_cfg.Num_Days)
    start_date = datetime.strptime(trips_cfg.Start_Date, "%Y-%m-%d")
    include_intrazonal = bool(trips_cfg.Accept_IntraZonal_Demand)
    min_trips_od = float(trips_cfg.Magnitude.min_trips_od)
    max_trips_od = float(trips_cfg.Magnitude.max_trips_od)
    noise_low = float(getattr(trips_cfg.Magnitude, "noise_low", 0.95))
    noise_high = float(getattr(trips_cfg.Magnitude, "noise_high", 1.05))

    zones_df = data["nodes"][data["nodes"]["class"] == "Zones"].copy()
    if zones_df.empty:
        raise ValueError("No zone nodes were generated. Demand generation requires at least one node classified as 'Zones'.")

    zone_ids = zones_df["node_id"].astype(int).values
    zone_id_to_idx = {int(zone_id): idx for idx, zone_id in enumerate(zone_ids)}
    idx_to_zone_id = {idx: int(zone_id) for idx, zone_id in enumerate(zone_ids)}
    hourly_profile = np.array([float(trips_cfg.Profile[f"{hour:02d}:00"]) for hour in range(24)])

    data["base_demand"] = _generate_base_demand(
        zone_ids=zone_ids,
        min_trips_od=min_trips_od,
        max_trips_od=max_trips_od,
        include_intrazonal=include_intrazonal,
        rng=metadata["rng"],
    )
    data["trips_array"] = _generate_trips_array(
        base_demand=data["base_demand"],
        hourly_profile=hourly_profile,
        num_days=num_days,
        rng=metadata["rng"],
        noise_low=noise_low,
        noise_high=noise_high,
    )

    tntp_text = format_trips_as_tntp(
        trips_array=data["trips_array"],
        zone_ids=zone_ids,
        start_date=start_date,
    )
    trips_path = save_text_as_tntp(tntp_text, config.paths.export_filepaths.trips)

    metadata = {
        "zone_ids": zone_ids,
        "zone_id_to_idx": zone_id_to_idx,
        "idx_to_zone_id": idx_to_zone_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "num_days": num_days,
        "hours": list(range(24)),
        # Esta metadata desde aquí, hasta donde menciono abajo, la he comentado puesto que no creo que sea de relevancia
        #"min_trips_od": min_trips_od,
        #"max_trips_od": max_trips_od,
        #"hourly_profile": hourly_profile,
        #"noise_low": noise_low,
        #"noise_high": noise_high,
        # Hasta aquí.
        "seed": config.seed,
        "hourly_totals": data["trips_array"].sum(axis=(2, 3)),
        "daily_totals": data["trips_array"].sum(axis=(1, 2, 3)),
        "total_demand": float(data["trips_array"].sum()),
        "base_demand_shape": data["base_demand"].shape,
        "trips_array_shape": data["trips_array"].shape,
        "trips_path": str(trips_path),
    }
    trips_array = data["trips_array"]
    return trips_array, metadata 
