# Critical Mineral Prospectivity (Geospatial ML Pipeline)

This repository demonstrates an end‑to‑end geospatial machine learning workflow for critical mineral prospectivity (Ni / Co / Li / Cu + combined “Crit”). It includes: reproducible data assembly, feature engineering over a unified grid, model training (Random Forest + Bayesian Logistic Regression), precomputation scripts, and a streamlined Streamlit demo app (`app_v2.py`).

## Highlighted Capabilities

- **Automated Data Pipeline** (notebooks `00`–`02x`): downloads, cleans, and grids raw geology / geochem / gravity / magnetics + MRDS occurrences.
- **Unified Grid Feature Stack**: geology categorical fractions, gravity & gradient, magnetics bands, geochemical signal features, spatial coordinates.
- **Models**:
   - Random Forest (balanced, calibrated offline) with feature importances.
   - Bayesian Logistic Regression (PyMC) for posterior mean & uncertainty.
- **Target Shortlisting**: Ranking filtered by Bayesian uncertainty quantile → top high‑confidence cells per mineral.
- **Precomputation Script**: `scripts/precompute_predictions.py` produces ready‑to‑serve arrays (RF probs, Bayesian mean/std, importances, targets).
- **Demo App**: `app_v2.py` fast interface (precomputed layers + optional on‑the‑fly RF retrain, feature panels, targets table).
- **Reproducibility**: Deterministic seeds; complete provenance in notebooks & scripts.

## Running the Streamlit App

Two paths:

### 1. Use Precomputed Artifacts (FAST – recommended)
Clone & install, then run the lean app:

```bash
pip install -r requirements.txt
streamlit run app_v2.py
```

Works if required processed files already exist under `data/processed/` (see “Required Artifacts”).

### 2. Regenerate Everything (FULL PIPELINE)
Execute the master notebook which orchestrates the others and finally calls the precompute script:

```bash
jupyter lab  # or jupyter notebook
# Open and run: notebooks/00_run_pipeline.ipynb (runs 01..02x + precompute)
```

This repopulates `data/processed/` then you can launch:

```bash
streamlit run app_v2.py
```

### What `app_v2.py` Provides

| Mode | Layers | Notes |
|------|--------|-------|
| Model | Precomputed RF, Bayesian Mean, Bayesian Uncertainty, On‑the‑fly RF | Precomputed ignore feature toggles; on‑the‑fly can subset feature groups |
| Feature Panel | Geology, Gravity (+ gradient), Geochemistry, Magnetics | Static derived maps; optional target overlay |

Additional Sidebar Items: target overlay, boundary clipping, feature importances (live + precomputed), histogram (log‑transformed values), data source references.

## Quickstart (Concise)

```bash
git clone https://github.com/acfdavis/ore-exploration-ml.git
cd ore-exploration-ml
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app_v2.py
```

If missing processed artifacts, run the pipeline notebook or manually run:

```bash
python scripts/precompute_predictions.py --force   # after generating grid + features + labels
```

## Repository Layout (Simplified)

```text
ore-exploration-ml/
├── app_v2.py                # Streamlined demo app (primary)
├── app.py                   # Original prototype app (legacy)
├── scripts/
│   └── precompute_predictions.py  # Batch RF + Bayesian + shortlist + importances
├── notebooks/               # 00 master + 01..06 thematic notebooks
├── src/                     # app_* helpers, feature logic, utilities
├── data/
│   ├── raw/                 # Large source datasets (ignored in Git)
│   └── processed/           # Grid, features, labels, predictions (whitelisted subset)
├── figures/                 # Exported static figures
├── tests/                   # Unit tests (core utilities & models)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Required Artifacts for `app_v2.py`

Minimum (per mineral or combined crit):

- `data/processed/grid_gdf.joblib`
- `data/processed/X_coords.npy`
- Feature arrays: `X_geo.npy`, `X_geochem.npy`, `X_gravity.npy`, `X_gravity_grad.npy`, `X_mag.npy`
- Feature name JSON: `feature_names_geo.json`, `feature_names_geochem.json`, `feature_names_mag.json`
- Labels: `y_labels_crit.npy` (or `y_labels_<mineral>.npy` for ni / co / li / cu)

Optional (enables richer panels / metrics):

- RF probs: `rf_probs_<mineral>.npy`
- Bayesian: `bayes_mean_<mineral>.npy`, `bayes_std_<mineral>.npy`
- RF importances: `rf_importances_<mineral>.csv`
- Shortlist: `targets_<mineral>.csv`
- Boundary clipping (if desired): `data/raw/state_geology.gpkg`

If any optional file is absent, the app degrades gracefully (layer or table simply omitted).

## Precomputation Script Usage

```bash
python scripts/precompute_predictions.py            # Normal (includes Bayesian)
python scripts/precompute_predictions.py --fast-bayes  # Skip Bayesian + fewer RF trees
python scripts/precompute_predictions.py --force       # Recompute even if outputs exist
```

Artifacts written to `data/processed/`.

## Data Sources (Public / Open)

| Domain | Source | Link |
|--------|--------|------|
| Mineral Occurrences | USGS MRDS | <https://mrdata.usgs.gov/mrds/> |
| Geology | State Geologic Map Compilation (SGMC) | <https://pubs.usgs.gov/ds/1052/> |
| Geochemistry | National Geochemical Database | <https://mrdata.usgs.gov/geochemistry/> |
| Gravity | USGS / NOAA (Isostatic / Bouguer Grids) | <https://www.ngdc.noaa.gov/mgg/gravity/> |
| Magnetics | North American Magnetic Anomaly Grid | <https://pubs.usgs.gov/of/2009/1258/> |

All preprocessing standardizes coordinate reference systems and raster/vector alignment to the project grid.

## Outputs

- Probability maps (RF + Bayesian mean) per mineral
- Uncertainty maps (Bayesian std)
- RF feature importances (global, impurity-based)
- Target shortlist CSVs (rank + RF prob + Bayesian stats)
- Static geology / gravity / geochem / magnetics panels
- Summary metadata (`labels_meta.txt`)

## Git & Large Files

Raw data are excluded via `.gitignore`. If you intend to share processed predictions, consider using **Git LFS** for: `*.npy`, `*.joblib`, `*.gpkg`, large rasters, and figures. Example `.gitattributes` entries:

```text
data/processed/*.npy filter=lfs diff=lfs merge=lfs -text
data/processed/*.joblib filter=lfs diff=lfs merge=lfs -text
data/**/*.gpkg filter=lfs diff=lfs merge=lfs -text
figures/*.png filter=lfs diff=lfs merge=lfs -text
```

## Testing

Run minimal unit tests:
 
```bash
pytest -q
```

## License

MIT License – see `LICENSE` file.

## Acknowledgements

All underlying datasets are credited to their respective USGS / NOAA publishers. Bayesian modeling uses PyMC; geospatial operations use GeoPandas/Shapely; visualization layers via Folium/Matplotlib/Streamlit.

---
Questions or improvements? Open an issue or submit a PR.


