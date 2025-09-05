from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box

EQUAL_AREA_CRS = "EPSG:5070"  # NAD83 / Conus Albers (good for US area calcs)

UNIT_CANDIDATE_COLUMNS = [
    "MAP_UNIT","UNIT","UNIT_NAME","UNITNAME","UNITDESC",
    "LABEL","NAME","UNITCODE","STRAT_UNIT"
]

def choose_unit_column(gdf: gpd.GeoDataFrame, provided: Optional[str] = None) -> str:
    if provided and provided in gdf.columns:
        return provided
    for c in UNIT_CANDIDATE_COLUMNS:
        if c in gdf.columns:
            return c
    # Fallback: first non-geometry object column
    for c in gdf.columns:
        if c != "geometry" and pd.api.types.is_object_dtype(gdf[c]):
            return c
    raise ValueError("Could not infer a geology unit column — please provide 'unit_col'.")

def clip_to_bbox(gdf: gpd.GeoDataFrame, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
    """Clip geology polygons to a lon/lat bbox (EPSG:4326)."""
    if gdf.crs is None:
        gdf = gdf.set_crs(4326, allow_override=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    minx, miny, maxx, maxy = bbox
    return gpd.clip(gdf, gpd.GeoDataFrame(geometry=[box(minx, miny, maxx, maxy)], crs=4326))

@dataclass
class GeologyEncodingResult:
    features: pd.DataFrame
    feature_names: List[str]
    unit_code_map: Dict[str, int]

def encode_geology_to_grid(grid_gdf: gpd.GeoDataFrame,
                           geol_gdf: gpd.GeoDataFrame,
                           unit_col: str | None = None,
                           top_n_units: int = 15):
    """
    One-hot encodes bedrock units into the grid using area-weighted overlap.

    - Projects both layers to an equal-area CRS (EPSG:5070) before area math.
    - If unit_col is None, tries to infer a reasonable unit/name field.
    - Returns a SimpleNamespace with .features (DataFrame) and .feature_names (list).
    """
    from types import SimpleNamespace

    # Ensure both have a CRS
    if grid_gdf.crs is None:
        grid_gdf = grid_gdf.set_crs("EPSG:4326")
    if geol_gdf.crs is None:
        geol_gdf = geol_gdf.set_crs("EPSG:4326")

    # Filter to polygonal geoms (optional but safer)
    geol_gdf = geol_gdf[geol_gdf.geometry.geom_type.isin(["Polygon","MultiPolygon"])].copy()

    # Pick unit column if not provided
    unit_col = choose_unit_column(geol_gdf, unit_col)

    # Reproject both to equal-area CRS for area computations
    grid_proj = grid_gdf.to_crs(EQUAL_AREA_CRS).copy()
    geol_proj = geol_gdf.to_crs(EQUAL_AREA_CRS).copy()

    # Optional: fix slight invalidities to avoid overlay errors
    try:
        geol_proj["geometry"] = geol_proj.buffer(0)
        grid_proj["geometry"] = grid_proj.buffer(0)
    except Exception:
        pass
    
    # Keep a stable id
    if "cell_id" not in grid_proj.columns:
        grid_proj = grid_proj.reset_index(drop=False).rename(columns={"index":"cell_id"})

    # Geometry clip is assumed already done upstream; intersect + compute area
    inter = gpd.overlay(
        grid_proj[["cell_id","geometry"]],
        geol_proj[["geometry", unit_col]].rename(columns={unit_col: "unit"}),
        how="intersection",
    )
    if inter.empty:
        raise ValueError("No grid/geology intersections found – check bbox/CRS alignment.")

    # Area in m² (valid because of equal-area CRS)
    inter["area"] = inter.geometry.area

    # For each cell, find total area to normalize
    cell_area = inter.groupby("cell_id")["area"].sum().rename("cell_area")
    inter = inter.merge(cell_area, on="cell_id", how="left")
    inter["frac"] = inter["area"] / inter["cell_area"]

    # Aggregate by (cell, unit): sum of area fraction
    agg = (inter.groupby(["cell_id","unit"])["frac"]
                 .sum()
                 .reset_index())

    # Choose top N most common units to one-hot (others collapsed to "OTHER")
    unit_counts = agg.groupby("unit")["frac"].sum().sort_values(ascending=False)
    top_units = unit_counts.head(top_n_units).index.tolist()
    agg["unit_slim"] = np.where(agg["unit"].isin(top_units), agg["unit"], "OTHER")

    # --- when pivoting, optionally sanitize column labels: ---
    def _safe_col(s):
        s = "UNKNOWN" if s is None or (isinstance(s, float) and pd.isna(s)) else str(s)
        return s.strip().replace("/", "_").replace(" ", "_")

    agg["unit_slim"] = agg["unit_slim"].map(_safe_col)

    # Pivot to wide: one column per unit_slim (values are area fractions)
    wide = (agg.pivot_table(index="cell_id", columns="unit_slim", values="frac", fill_value=0.0)
                .reset_index())

    # Ensure every top unit column exists (even if absent in slice)
    for u in top_units + ["OTHER"]:
        if u not in wide.columns:
            wide[u] = 0.0

    # Reattach to original grid index order
    out = grid_proj[["cell_id"]].merge(wide, on="cell_id", how="left").fillna(0.0)
    # Back to original grid order/length
    out = out.set_index("cell_id").reindex(range(len(grid_proj))).fillna(0.0).reset_index()

    feature_cols = [c for c in out.columns if c not in ("cell_id")]
    features = out[feature_cols]

    return SimpleNamespace(
        features=features,            # DataFrame (n_cells x (top_n_units+1))
        feature_names=feature_cols,   # list of unit names used as columns
        unit_col=unit_col,
        crs_used=EQUAL_AREA_CRS,
    )
