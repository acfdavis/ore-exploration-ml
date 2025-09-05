from __future__ import annotations
"""
USGS SGMC (State Geologic Map Compilation) fetch helpers.

These helpers target **ArcGIS FeatureServer/MapServer** style endpoints that return GeoJSON.
Default endpoint is a best-effort guess; feel free to swap to a state dataset or a known SGMC URL.

Typical ArcGIS query params (GeoJSON):
- where: SQL filter (e.g., "1=1")
- geometry: xmin,ymin,xmax,ymax (in WGS84)
- geometryType: "esriGeometryEnvelope"
- inSR: 4326
- spatialRel: "esriSpatialRelIntersects"
- outFields: "*" (or a subset)
- f: "geojson"

Example base_url (verify/update as needed):
https://mrdata.usgs.gov/arcgis/rest/services/sgmc2/MapServer/0/query
"""
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import requests
import geopandas as gpd
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

DEFAULT_SGMC_QUERY_URL = "https://gis.dnr.mo.gov/host/rest/services/geology/surficial_geology/MapServer/0/query"

@dataclass
class SGMCResult:
    gdf: gpd.GeoDataFrame
    url: str
    params: Dict[str, Any]

def fetch_sgmc_by_bbox(
    bbox: Tuple[float, float, float, float],
    out_path: Path = PROJECT_ROOT / "data/raw/state_geology.gpkg",
    base_url: str = DEFAULT_SGMC_QUERY_URL,
    where: str = "1=1",
    out_fields: str = "*",
    verify: bool = True,
) -> SGMCResult:
    """Query an ArcGIS FeatureServer layer by bbox; save GeoPackage; return GeoDataFrame.

    Args:
        bbox: (minx, miny, maxx, maxy) in WGS84 (EPSG:4326)
        out_path: where to save geopackage
        base_url: ArcGIS query endpoint (layer path ending with /query)
        where: SQL where-clause
        out_fields: columns to return ("*" for all)
        verify: TLS verification for requests

    Returns:
        SGMCResult with gdf, url, and params used.
    """
    minx, miny, maxx, maxy = bbox
    params = {
        "where": where,
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": out_fields,
        "returnGeometry": "true",
        "f": "geojson",
    }
    r = requests.get(base_url, params=params, timeout=60, verify=verify)
    r.raise_for_status()
    gdf = gpd.read_file(r.text)  # GeoJSON string
    # Normalize CRS
    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    elif gdf.crs.to_string() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(out_path, driver="GPKG")
    return SGMCResult(gdf=gdf, url=r.url, params=params)
