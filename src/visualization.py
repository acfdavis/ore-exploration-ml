from __future__ import annotations
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt

def plot_probability_map(grid: gpd.GeoDataFrame, probs: np.ndarray, title: str = "Predicted Ore Likelihood"):
    fig, ax = plt.subplots(figsize=(10, 8))
    gdf = grid.copy()
    gdf["prob"] = probs
    gdf.plot(column="prob", ax=ax, legend=True)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    return fig, ax
