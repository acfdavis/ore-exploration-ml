"""
Utilities to fetch and aggregate USGS NGS geochemical data
from the Missouri DNR ArcGIS REST service into grid-aligned features.

Outputs are median and 90th percentile per grid cell for each element,
optionally log10 + z-score transformed.

Author: you
"""

from __future__ import annotations
import time
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import io, json, time, requests, geopandas as gpd, pandas as pd

# ArcGIS REST endpoint (Layer 0 = NGS points)
NGS_BASE = "https://gis.dnr.mo.gov/host/rest/services/soils/national_geochemical_survey/MapServer/0/query"
MAX_REC = 2000  # service paging


CORE_FIELDS = ["OBJECTID", "LONGITUDE", "LATITUDE"]


def _read_geojson_bytes(content: bytes) -> gpd.GeoDataFrame:
    """Read GeoJSON bytes into a GeoDataFrame via file-like buffer."""
    return gpd.read_file(io.BytesIO(content))  # works for f="geojson"

def _read_esrijson_points(content: bytes) -> gpd.GeoDataFrame:
    """
    Minimal ESRI JSON -> GeoDataFrame converter for POINT layers.
    Expects {'features':[{'attributes':{}, 'geometry':{'x':..., 'y':...}}, ...]}
    """
    obj = json.loads(content.decode("utf-8"))
    feats = obj.get("features", [])
    if not feats:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

    # Flatten attributes and extract x,y
    rows = []
    xs, ys = [], []
    for f in feats:
        attrs = f.get("attributes", {}) or {}
        geom  = f.get("geometry", {}) or {}
        x = geom.get("x", None)
        y = geom.get("y", None)
        xs.append(x); ys.append(y)
        rows.append(attrs)

    df = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(xs, ys, crs=4326)
    )
    return gdf

def _page_query(params_base: dict, use_geojson_first: bool = True) -> gpd.GeoDataFrame:
    """Iterate pages; try geojson first then ESRI JSON on each page."""
    out = []
    offset = 0
    while True:
        common = dict(resultOffset=offset, resultRecordCount=MAX_REC)
        # Try GeoJSON
        if use_geojson_first:
            params = {**params_base, **common, "f": "geojson"}
            r = requests.get(NGS_BASE, params=params, timeout=60)
            if r.ok:
                try:
                    gdf = _read_geojson_bytes(r.content)
                    if len(gdf) == 0:
                        break
                    out.append(gdf)
                    if len(gdf) < MAX_REC:
                        break
                    offset += MAX_REC
                    continue
                except Exception:
                    pass  # fall through to ESRI JSON

        # ESRI JSON fallback
        params = {**params_base, **common, "f": "json"}
        r = requests.get(NGS_BASE, params=params, timeout=60)
        if not r.ok:
            break
        try:
            err_probe = r.json()
            if isinstance(err_probe, dict) and "error" in err_probe:
                break
        except Exception:
            pass
        gdf = _read_esrijson_points(r.content)
        if len(gdf) == 0:
            break
        out.append(gdf)
        if len(gdf) < MAX_REC:
            break
        offset += MAX_REC

    if not out:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    res = pd.concat(out, ignore_index=True)
    if not isinstance(res, gpd.GeoDataFrame):
        if "LONGITUDE" in res and "LATITUDE" in res:
            res = gpd.GeoDataFrame(
                res, geometry=gpd.points_from_xy(res["LONGITUDE"], res["LATITUDE"]), crs=4326
            )
        else:
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    if res.crs is None:
        res = res.set_crs(4326)
    return res

def fetch_ngs_geojson_by_bbox(
    bbox: Tuple[float, float, float, float],
    fields: List[str] | str = "*",
) -> gpd.GeoDataFrame:
    """Fetch NGS points in bbox. Tries geometry filter first; if 0 rows, falls back to LONG/LAT attribute filter."""
    minx, miny, maxx, maxy = map(float, bbox)
    if isinstance(fields, str):
        out_fields = fields
    else:
        out_fields = ",".join(fields)

    # --- A/B: geometry envelope path ---
    params_geom = dict(
        where="1=1",
        geometry=f"{minx},{miny},{maxx},{maxy}",
        geometryType="esriGeometryEnvelope",
        inSR=4326,
        spatialRel="esriSpatialRelIntersects",
        outFields=out_fields,
        returnGeometry="true",
        outSR=4326,
    )
    gdf = _page_query(params_geom, use_geojson_first=True)
    if len(gdf) > 0:
        return gdf

    # --- C/D: attribute fallback on lon/lat (no geometry param) ---
    where_attr = f"(LONGITUDE >= {minx}) AND (LONGITUDE <= {maxx}) AND (LATITUDE >= {miny}) AND (LATITUDE <= {maxy})"
    params_attr = dict(
        where=where_attr,
        outFields=out_fields,
        returnGeometry="true",
        outSR=4326,
    )
    gdf2 = _page_query(params_attr, use_geojson_first=True)
    return gdf2

ELEMENTS_DEFAULT = {
    "CU": ["CU_ICP40", "CU_ICP10", "CU_INAA", "CU_NURE", "CU_AA"],
    "NI": ["NI_ICP40", "NI_ICP10", "NI_INAA", "NI_NURE", "NI_AA"],
    "CO": ["CO_ICP40", "CO_ICP10", "CO_INAA", "CO_NURE", "CO_AA"],
    "LI": ["LI_ICP40", "LI_ICP10", "LI_INAA", "LI_NURE", "LI_AA"],
}

def _best_available_row(row: pd.Series, candidates: list[str]) -> float:
    """Return the first valid, positive float value from a list of candidate columns."""
    for c in candidates:
        if c in row and pd.notna(row[c]):
            try:
                val = float(row[c])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass
    return np.nan

def process_geochem_data(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    elements: dict[str, list[str]] = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Processes raw geochemical point data and aggregates it onto a grid.

    This function performs the following steps:
    1. For each element, it selects the best available measurement from a prioritized list of columns.
    2. It filters out points that do not have any of the target elements measured.
    3. It spatially joins the points to the grid cells.
    4. It aggregates the data for each cell, calculating the median and 90th percentile for each element.
    5. It applies a log10 + z-score transformation to the aggregated data.

    Args:
        grid_gdf: A GeoDataFrame representing the grid to which the data will be aggregated.
        points_gdf: A GeoDataFrame containing the raw geochemical point data.
        elements: A dictionary defining the elements to process and their corresponding column candidates.

    Returns:
        A tuple containing:
        - A NumPy array with the processed geochemical features.
        - A list of the feature names.
    """
    if elements is None:
        elements = ELEMENTS_DEFAULT

    # Compute best-available value for each element
    processed_points = points_gdf.copy()
    for el, candidates in elements.items():
        processed_points[el] = processed_points.apply(
            lambda r: _best_available_row(r, candidates), axis=1
        )

    # Filter for points with at least one measured element
    value_cols = list(elements.keys())
    processed_points = processed_points.dropna(subset=value_cols, how="all").reset_index(drop=True)

    # Spatial join
    joined = gpd.sjoin(
        processed_points,
        grid_gdf[["geometry"]].reset_index(),
        how="inner",
        predicate="intersects",
    )

    # Aggregate per cell
    agg_parts = []
    feat_names = []
    for el in value_cols:
        sub = joined[["index", el]].dropna()
        if not sub.empty:
            gb = sub.groupby("index")[el]
            med = gb.median().reindex(range(len(grid_gdf)))
            p90 = gb.quantile(0.90).reindex(range(len(grid_gdf)))
            agg_parts.extend([med, p90])
            feat_names.extend([f"{el}_med", f"{el}_p90"])

    if not agg_parts:
        return np.array([]), []

    agg_df = pd.concat(agg_parts, axis=1)

    # Transform and save
    for c in agg_df.columns:
        agg_df[c] = _log_z_transform(agg_df[c].values)

    return agg_df.values.astype(np.float32), feat_names

def _log_z_transform(x: np.ndarray) -> np.ndarray:
    """
    Apply log10 to positive values, then z-score (μ=0, σ=1). NaNs → 0.
    """
    x = x.astype(float)
    mask = np.isfinite(x) & (x > 0)
    x_log = np.full_like(x, np.nan, dtype=float)
    x_log[mask] = np.log10(x[mask])
    mu = np.nanmean(x_log)
    sd = np.nanstd(x_log) + 1e-9
    z = (x_log - mu) / sd
    return np.nan_to_num(z, nan=0.0)

def points_to_grid_geochem_features(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    elements: Dict[str, List[str]] = ELEMENTS_DEFAULT,
    stats: Tuple[str, ...] = ("median", "p90"),
    log_z_transform: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    """
    Aggregate point geochemistry to grid cells.
    For each element, compute per-cell statistics (e.g., median, 90th percentile).
    Optionally apply log10 + z-score to each statistic column.

    Returns:
        X: (N_cells × N_features) numpy array
        feature_names: list of column names
    """
    assert "geometry" in grid_gdf.columns, "grid_gdf must have 'geometry'."
    assert grid_gdf.crs is not None and grid_gdf.crs.to_epsg() == 4326, "grid CRS must be EPSG:4326"
    assert points_gdf.crs is not None and points_gdf.crs.to_epsg() == 4326, "points CRS must be EPSG:4326"

    # Spatial join: assign each point to a grid cell id
    join_left = gpd.sjoin(
        points_gdf,
        grid_gdf[["geometry"]].reset_index(),  # 'index' will be the cell id
        how="inner",
        predicate="intersects",
    )

    value_cols = list(elements.keys())
    N = len(grid_gdf)

    # Prepare aggregation per element
    parts = []
    names: List[str] = []

    for el in value_cols:
        col = join_left[[el, "index"]].dropna()
        if col.empty:
            # All-NaN columns if no data for this element in the area
            agg_med = pd.Series(np.nan, index=range(N), name=f"{el}_med") if "median" in stats else None
            agg_p90 = pd.Series(np.nan, index=range(N), name=f"{el}_p90") if "p90" in stats else None
        else:
            gb = col.groupby("index")[el]
            agg_med = gb.median().reindex(range(N), fill_value=np.nan) if "median" in stats else None
            agg_p90 = gb.quantile(0.90).reindex(range(N), fill_value=np.nan) if "p90" in stats else None

        for s, series in (("med", agg_med), ("p90", agg_p90)):
            if series is not None:
                arr = series.values.astype(float)
                if log_z_transform:
                    arr = _log_z_transform(arr)
                parts.append(arr)
                names.append(f"{el}_{s}")

    if not parts:
        # no features generated
        return np.zeros((N, 0), dtype=np.float32), []

    X = np.column_stack(parts)
    return X, names
