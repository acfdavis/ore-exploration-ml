"""Geospatial helper utilities for the streamlined app version.

This module isolates geometry/grid related logic so that Streamlit caching
is simpler and the main UI stays small.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import geopandas as gpd
import numpy as np
from shapely.geometry import box


def build_square_grid(base_grid: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Return a GeoDataFrame of square polygons (one per point) using median dx, dy.

    If geometry is already polygonal, it's returned unchanged.
    """
    if base_grid.empty:
        return base_grid
    if base_grid.geometry.iloc[0].geom_type != "Point":
        return base_grid
    xs = base_grid.geometry.x.to_numpy()
    ys = base_grid.geometry.y.to_numpy()
    ux = np.unique(np.sort(xs))
    uy = np.unique(np.sort(ys))
    dx = float(np.median(np.diff(ux))) if len(ux) > 1 else 0.1
    dy = float(np.median(np.diff(uy))) if len(uy) > 1 else dx
    if dx <= 0:  # safety net
        dx = 0.1
    if dy <= 0:
        dy = dx
    halfx, halfy = dx / 2.0, dy / 2.0
    polys = [box(x - halfx, y - halfy, x + halfx, y + halfy) for x, y in zip(xs, ys)]
    gpoly = base_grid.copy()
    gpoly.geometry = polys
    return gpoly


def load_state_boundary() -> Optional[gpd.GeoDataFrame]:
    """Load dissolved state boundary if available; returns single-row GeoDataFrame.

    Returns None if missing or on error.
    """
    path = Path("data/raw/state_geology.gpkg")
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
        gdf["__one"] = 1
        boundary = gdf.dissolve("__one").geometry.iloc[0]
        return gpd.GeoDataFrame({"geometry": [boundary]}, crs=gdf.crs)
    except Exception:
        return None


def clip_to_boundary(gdf: gpd.GeoDataFrame, boundary_gdf: Optional[gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    if boundary_gdf is None:
        return gdf
    try:
        if gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(gdf.crs)
        return gpd.clip(gdf, boundary_gdf)
    except Exception:
        return gdf
