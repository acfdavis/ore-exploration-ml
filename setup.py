import os
import sys
import subprocess
from pathlib import Path
import requests
import zipfile
import io

def download_processed_data():
    """
    Downloads pre-processed data files for the demo.
    """
    print("Downloading pre-processed data...")

    # Create directories
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # For now, let's create a simple demo setup
    # In a real deployment, you would host the processed data somewhere
    print("Setting up demo data...")

    # Create a simple grid for demonstration
    import numpy as np
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import Point

    # Create a simple grid
    lons = np.linspace(-95, -89, 20)  # Missouri longitude range
    lats = np.linspace(36, 40, 20)    # Missouri latitude range

    points = []
    for lat in lats:
        for lon in lons:
            points.append(Point(lon, lat))

    grid = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')

    # Save the grid
    import joblib
    joblib.dump(grid, processed_dir / "grid_gdf.joblib")

    # Create dummy feature data
    n_points = len(grid)
    np.random.seed(42)

    # Coordinates
    coords = np.column_stack([np.array(grid.geometry.x.values), np.array(grid.geometry.y.values)])
    np.save(processed_dir / "X_coords.npy", coords)

    # Dummy features
    features = {
        'X_geo': np.random.randn(n_points, 5),
        'X_gravity': np.random.randn(n_points, 1),
        'X_geochem': np.random.randn(n_points, 10),
        'X_mag': np.random.randn(n_points, 3)
    }

    for name, data in features.items():
        np.save(processed_dir / f"{name}.npy", data)

    # Create dummy labels
    y = np.random.binomial(1, 0.1, n_points)  # 10% positive labels
    np.save(processed_dir / "y_labels.npy", y)

    # Create dummy model outputs
    probs = np.random.beta(2, 5, n_points)  # Skewed toward low probabilities
    np.save(processed_dir / "mean_probs.npy", probs)
    np.save(processed_dir / "std_probs.npy", probs * 0.1)

    print("Demo data created successfully!")

def main():
    """
    Sets up the data for the app.
    """
    print("Starting data setup process...")

    try:
        download_processed_data()
        print("Data setup process complete.")
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Demo setup failed. Please check the app requirements.")

if __name__ == "__main__":
    main()
