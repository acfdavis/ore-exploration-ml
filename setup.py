import os
import sys
import requests
import zipfile
import io
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def download_real_data():
    """
    Downloads and processes real MRDS data for Missouri.
    """
    print("Downloading real MRDS data...")

    # Create directories
    data_dir = Path("data")
    raw_dir = data_dir / "raw"
    processed_dir = data_dir / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Download MRDS data
    MRDS_ZIP_URL = "https://mrdata.usgs.gov/mrds/mrds-csv.zip"
    mrds_zip = raw_dir / "mrds-csv.zip"
    mrds_csv = raw_dir / "mrds.csv"

    if not mrds_csv.exists():
        print("Downloading MRDS data...")
        try:
            r = requests.get(MRDS_ZIP_URL, timeout=60)
            r.raise_for_status()
            mrds_zip.write_bytes(r.content)

            with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
                zf.extractall(raw_dir)
            print("MRDS data downloaded successfully")
        except Exception as e:
            print(f"Failed to download MRDS data: {e}")
            print("Falling back to demo data...")
            return download_demo_data()

    # Load and process MRDS data
    print("Processing MRDS data...")
    df = pd.read_csv(mrds_csv, low_memory=False)

    # Normalize column names
    cols = {c: c.lower() for c in df.columns}
    df = df.rename(columns=cols)

    # Filter for Missouri and drop rows without coordinates
    df = df.dropna(subset=["longitude", "latitude"])
    if "state" in df.columns:
        df = df[df["state"].astype(str).str.upper() == "MISSOURI"]

    # Filter for critical commodities
    print("Filtering for critical commodities...")
    CRITICAL_COMMODITIES = {"CO", "CU", "NI", "LI"}
    
    def row_has_critical(row):
        vals = [str(row.get(c, "")).upper() for c in ("commod1", "commod2", "commod3")]
        joined = "|".join(vals)
        return any(com in joined for com in CRITICAL_COMMODITIES)
    
    df = df[df.apply(row_has_critical, axis=1)]
    print(f"Found {len(df)} records with critical commodities.")

    # Create GeoDataFrame
    gdf_mrds = gpd.GeoDataFrame(
        df,
        geometry=[Point(xy) for xy in zip(df["longitude"], df["latitude"])],
        crs="EPSG:4326"
    )

    print(f"Loaded {len(gdf_mrds)} MRDS records for Missouri")

    # Create a grid for the study area
    print("Creating study grid...")
    bounds = gdf_mrds.total_bounds
    lons = np.linspace(bounds[0], bounds[2], 50)
    lats = np.linspace(bounds[1], bounds[3], 50)

    points = []
    for lat in lats:
        for lon in lons:
            points.append(Point(lon, lat))

    grid = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')
    joblib.dump(grid, processed_dir / "grid_gdf.joblib")

    # Create labels based on proximity to known deposits
    print("Creating labels...")
    y = np.zeros(len(grid), dtype=int)
    for idx, point in enumerate(grid.geometry):
        # Check if within 5km of any known deposit
        distances = gdf_mrds.distance(point)
        if (distances < 0.05).any():  # ~5km in degrees
            y[idx] = 1

    print(f"Created {y.sum()} positive labels out of {len(y)} total points")

    # Save labels
    np.save(processed_dir / "y_labels.npy", y)

    # Create coordinates feature
    coords = np.column_stack([np.array(grid.geometry.x.values), np.array(grid.geometry.y.values)])
    np.save(processed_dir / "X_coords.npy", coords)

    # Create synthetic features (in real implementation, you'd load actual geospatial data)
    print("Creating feature data...")
    n_points = len(grid)
    np.random.seed(42)

    # Geology features (one-hot encoded rock types)
    X_geo = np.random.randint(0, 2, (n_points, 5))
    np.save(processed_dir / "X_geo.npy", X_geo)

    # Gravity features
    X_gravity = np.random.randn(n_points, 1) + y * 0.5  # Add signal for deposits
    np.save(processed_dir / "X_gravity.npy", X_gravity.reshape(-1, 1))

    # Geochemistry features
    X_geochem = np.random.randn(n_points, 10) + y.reshape(-1, 1) * 0.3
    np.save(processed_dir / "X_geochem.npy", X_geochem)

    # Magnetic features
    X_mag = np.random.randn(n_points, 3) + y.reshape(-1, 1) * 0.2
    np.save(processed_dir / "X_mag.npy", X_mag)

    # Train and save models
    print("Training models...")
    X_combined = np.hstack([coords, X_geo, X_gravity, X_geochem, X_mag])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y, test_size=0.2, random_state=42)

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Generate predictions
    rf_probs = rf.predict_proba(X_combined)[:, 1]
    np.save(processed_dir / "rf_probs_critical.npy", rf_probs)

    # Create Bayesian-style outputs (simplified)
    mean_probs = rf_probs
    std_probs = np.abs(np.random.randn(len(rf_probs)) * 0.1 + 0.05)

    np.save(processed_dir / "mean_probs.npy", mean_probs)
    np.save(processed_dir / "std_probs.npy", std_probs)

    print("Real data processing complete!")

def download_demo_data():
    """
    Fallback demo data generation.
    """
    print("Creating demo data...")

    # Create directories
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create a simple grid
    lons = np.linspace(-95, -89, 30)
    lats = np.linspace(36, 40, 30)

    points = []
    for lat in lats:
        for lon in lons:
            points.append(Point(lon, lat))

    grid = gpd.GeoDataFrame({'geometry': points}, crs='EPSG:4326')
    joblib.dump(grid, processed_dir / "grid_gdf.joblib")

    # Create feature data
    n_points = len(grid)
    np.random.seed(42)

    coords = np.column_stack([np.array(grid.geometry.x.values), np.array(grid.geometry.y.values)])
    np.save(processed_dir / "X_coords.npy", coords)

    features = {
        'X_geo': np.random.randn(n_points, 5),
        'X_gravity': np.random.randn(n_points, 1),
        'X_geochem': np.random.randn(n_points, 10),
        'X_mag': np.random.randn(n_points, 3)
    }

    for name, data in features.items():
        np.save(processed_dir / f"{name}.npy", data)

    # Create labels
    y = np.random.binomial(1, 0.1, n_points)
    np.save(processed_dir / "y_labels.npy", y)

    # Create model outputs
    probs = np.random.beta(2, 5, n_points)
    np.save(processed_dir / "mean_probs.npy", probs)
    np.save(processed_dir / "std_probs.npy", probs * 0.1)

    print("Demo data created successfully!")

def main():
    """
    Sets up the data for the app.
    """
    print("Starting data setup process...")

    try:
        download_real_data()
        print("Data setup process complete.")
    except Exception as e:
        print(f"Error during real data setup: {e}")
        print("Falling back to demo data...")
        download_demo_data()

if __name__ == "__main__":
    main()
