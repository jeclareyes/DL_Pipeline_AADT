from __future__ import annotations

import logging
from typing import Any

import pandas as pd


def export_spatial_audit(
    df_links: pd.DataFrame,
    geometries: Any,
    output_path: str,
    source_crs: str = "EPSG:3006",
    target_crs: str = "EPSG:4977",
) -> None:
    """Export link-level diagnostics as a GeoPackage with LineString geometry."""

    try:
        import geopandas as gpd
        from shapely.geometry import LineString
    except ImportError:
        logging.warning("geopandas/shapely not installed. Skipping GeoPackage export.")
        return

    if geometries is None:
        raise ValueError("geometries is None.")

    geometry_list = []

    if isinstance(geometries, dict):
        for _, geom in geometries.items():
            geometry_list.append(
                LineString(
                    [
                        (float(geom["x1"]), float(geom["y1"])),
                        (float(geom["x2"]), float(geom["y2"])),
                    ]
                )
            )
    else:
        geometry_list = list(geometries)

    if len(geometry_list) != len(df_links):
        raise ValueError(
            f"Geometry length mismatch: {len(geometry_list)} geometries "
            f"for {len(df_links)} link rows."
        )

    source_crs = "EPSG:3006"
    target_crs = "EPSG:4977" # Also could be "EPSG:3006"

    gdf = gpd.GeoDataFrame(df_links.copy(),
                            geometry=geometry_list,
                            crs=source_crs,
                            ).to_crs(target_crs)

    for col in gdf.columns:
        if col != "geometry" and gdf[col].dtype == "object":
            gdf[col] = gdf[col].astype(str)

    gdf.to_file(output_path, driver="GPKG")
    logging.info(f"Spatial audit exported to: {output_path}")