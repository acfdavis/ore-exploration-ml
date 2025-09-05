from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box
import rasterio

@dataclass
class GridSpec:
    cell_deg: float = 0.05  # ~5 km at mid-lat; tweak based on region

def build_grid_from_bbox(bbox: Tuple[float, float, float, float], spec: GridSpec = GridSpec()) -> gpd.GeoDataFrame:
    minx, miny, maxx, maxy = bbox
    xs = np.arange(minx, maxx, spec.cell_deg)
    ys = np.arange(miny, maxy, spec.cell_deg)
    cells = []
    for x in xs:
        for y in ys:
            cells.append(box(x, y, x + spec.cell_deg, y + spec.cell_deg))
    return gpd.GeoDataFrame(geometry=cells, crs="EPSG:4326")

def rasterize_occurrence_counts(grid: gpd.GeoDataFrame, occurrences: gpd.GeoDataFrame) -> pd.DataFrame:
    # Spatial join to count occurrences per cell
    joined = gpd.sjoin(occurrences, grid, predicate="within", how="left")
    counts = joined.groupby(joined.index_right).size()
    # index_right is index into grid
    df = pd.DataFrame({"occ_count": counts}).reindex(range(len(grid))).fillna(0).astype(int)
    return df

def grid_centroids(grid: gpd.GeoDataFrame) -> np.ndarray:
    centroids = grid.geometry.centroid
    return np.vstack([centroids.x.values, centroids.y.values]).T


def sample_raster_to_grid(grid: gpd.GeoDataFrame, raster_path: str, band: int = 1) -> np.ndarray:
    """
    Sample mean raster values for each polygon cell in the grid.
    """
    values = []
    with rasterio.open(raster_path) as src:
        for geom in grid.geometry:
            try:
                # mask raster to grid cell
                out_image, out_transform = rasterio.mask.mask(src, [geom], crop=True)
                data = out_image[band-1]
                # filter no-data
                valid = data[data != src.nodata]
                if len(valid) == 0:
                    values.append(np.nan)
                else:
                    values.append(float(np.nanmean(valid)))
            except Exception:
                values.append(np.nan)
    return np.array(values)
