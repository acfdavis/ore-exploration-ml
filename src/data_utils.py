from __future__ import annotations
import io
import zipfile
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import geopandas as gpd
import requests
from shapely.geometry import Point, box

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
RAW_DIR = PROJECT_ROOT / "data/raw"
PROC_DIR = PROJECT_ROOT / "data/processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

MRDS_ZIP_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"

def download_mrds_csv(dest_zip: Path = RAW_DIR / "mrds-csv.zip") -> Path:
    if dest_zip.exists():
        return dest_zip
    r = requests.get(MRDS_ZIP_URL, timeout=60)
    r.raise_for_status()
    dest_zip.write_bytes(r.content)
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        zf.extractall(RAW_DIR)
    return dest_zip

def load_mrds_geodata(state_abbrev: Optional[str] = None) -> gpd.GeoDataFrame:
    """Load MRDS CSV and return a GeoDataFrame filtered by state if provided."""
    csv_path = RAW_DIR / "mrds.csv"
    if not csv_path.exists():
        download_mrds_csv()
    df = pd.read_csv(csv_path, low_memory=False)
    # Normalize column names
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)
    # Drop rows without coordinates
    df = df.dropna(subset=["longitude", "latitude"])
    gdf = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )
    if state_abbrev:
        state_abbrev = state_abbrev.strip().upper()
        if "state" in gdf.columns:
            gdf = gdf[gdf["state"].astype(str).str.upper() == state_abbrev]

    # Build a commodity column from commod1/2/3 if present
    commod_cols = [c for c in ["commod1", "commod2", "commod3"] if c in gdf.columns]
    if commod_cols:
        gdf["commodity"] = gdf[commod_cols].astype(str).replace("nan", "").agg(
            lambda x: ";".join([c for c in x if c.strip()]), axis=1
        )
    else:
        gdf["commodity"] = ""

    keep_cols = ["geometry", "commodity", "dep_id", "site_name", "state"]
    return gdf[[c for c in keep_cols if c in gdf.columns]]

def save_gdf(gdf: gpd.GeoDataFrame, name: str) -> Path:
    out = PROC_DIR / f"{name}.gpkg"
    gdf.to_file(out, driver="GPKG")
    return out

def bounding_box_for_state(state_abbrev: str) -> tuple[float, float, float, float]:
    """
    Quick-and-dirty bounding box for a state.
    Uses MRDS extents after filtering, so it's rectangular and dependency-light.
    """
    gdf = load_mrds_geodata(state_abbrev)
    return tuple(gdf.total_bounds)  # (minx, miny, maxx, maxy)



def state_polygon(state_abbrev: str, prefer_geology: bool = True):
    """
    Returns a GeoSeries (single polygon) for the state.
    - If prefer_geology=True and data/raw/state_geology.gpkg exists, uses that.
    - Otherwise falls back to bounding box.
    """
    geo_path = Path(PROJECT_ROOT / "data/raw/state_geology.gpkg")
    if prefer_geology and geo_path.exists():
        gdf = gpd.read_file(geo_path)
        if gdf.crs is None or gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(4326)
        return gdf.unary_union.buffer(0)  # shapely Polygon/MultiPolygon
    else:
        minx, miny, maxx, maxy = bounding_box_for_state(state_abbrev)
        return box(minx, miny, maxx, maxy)

def get_grid_bbox(grid_gdf: gpd.GeoDataFrame) -> tuple[float, float, float, float]:
    """Return (minx, miny, maxx, maxy) for an EPSG:4326 grid GeoDataFrame."""
    if grid_gdf.crs is None or grid_gdf.crs.to_epsg() != 4326:
        grid_gdf = grid_gdf.to_crs(4326)
    return tuple(map(float, grid_gdf.total_bounds))
