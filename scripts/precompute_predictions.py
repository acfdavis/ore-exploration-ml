"""Precompute mineral-specific RF and (optionally) Bayesian logistic regression predictions.

This script is designed to be run after the data generation notebooks have been
executed. It assembles a feature matrix from grid-aligned data, trains models
for each specified mineral, and saves the resulting probability maps.

Outputs written to data/processed/ as:
  rf_probs_<mineral>.npy
  bayes_mean_<mineral>.npy
  bayes_std_<mineral>.npy

Usage (examples):
  python scripts/precompute_predictions.py
  python scripts/precompute_predictions.py --fast-bayes
  python scripts/precompute_predictions.py --force
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys
import platform
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score
import joblib

# ------------------------------------------------------------------
# Configuration & Pathing
# ------------------------------------------------------------------
# Make path relative to the project root, not the current working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED = PROJECT_ROOT / "data/processed"
MINERALS_DEFAULT = ["ni", "co", "li","cu", "crit"]  # "crit" for general critical minerals


def _train_rf(X, y, fast=False):
    """Trains a calibrated Random Forest and returns predictions, metrics, and feature importances."""
    train_idx = np.where(y > -1)[0]
    if len(train_idx) == 0:
        return np.zeros(len(X)), 0, 0, 0, None

    X_train, y_train = X[train_idx], y[train_idx]
    n_est = 20 if fast else 100
    
    # Check for minimum number of samples for calibration
    n_positives = np.sum(y_train)
    cv = 3 # The number of folds in CalibratedClassifierCV

    # Always fit a base RF first so we can extract feature importances reliably
    base_rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced', n_jobs=-1)
    base_rf.fit(X_train, y_train)
    feat_imp = getattr(base_rf, 'feature_importances_', None)

    # Optional calibration for probabilities (use a fresh estimator clone)
    if n_positives < cv:
        print(f"[warn] Too few positive samples ({int(n_positives)}) for calibrated CV. Using uncalibrated RF probabilities.")
        calibrated_model = base_rf
    else:
        calib_rf = RandomForestClassifier(n_estimators=n_est, random_state=42, class_weight='balanced', n_jobs=-1)
        calibrated_model = CalibratedClassifierCV(calib_rf, method='isotonic', cv=cv)
        calibrated_model.fit(X_train, y_train)
    
    # If the model has only seen one class, predict_proba will have only one column.
    # We need to handle this to avoid an IndexError.
    if len(getattr(calibrated_model, 'classes_', [])) < 2:
        print(f"[warn] Model for was trained on only one class. Predicting all zeros.")
        probs = np.zeros(len(X))
        train_probs = np.zeros(len(X_train))
    else:
        probs = calibrated_model.predict_proba(X)[:, 1]
        train_probs = calibrated_model.predict_proba(X_train)[:, 1]

    # roc_auc_score requires at least one sample from each class.
    if len(np.unique(y_train)) < 2:
        auc = 0
        pr = 0
    else:
        auc = roc_auc_score(y_train, train_probs)
        pr = average_precision_score(y_train, train_probs)
    
    return probs, auc, pr, len(train_idx), feat_imp


def _bayes_logreg(X, y, fast=False):
    """Trains a Bayesian logistic regression using modern PyMC, with preprocessing."""
    try:
        import pymc as pm
        import pytensor.tensor as pt
        from arviz import InferenceData
        from scipy.special import logit
    except ImportError:
        print("[warn] PyMC/PyTensor not found, skipping Bayesian modeling.")
        return np.full(len(X), np.nan), np.full(len(X), np.nan), 0, 0, 0

    train_idx = np.where(y > -1)[0]
    if len(train_idx) < 10 or len(np.unique(y[train_idx])) < 2:
        print("[warn] Not enough training data or only one class present. Skipping Bayesian model.")
        return np.full(len(X), np.nan), np.full(len(X), np.nan), 0, 0, 0

    # --- Preprocessing (from notebook 04) ---
    # 1. Remove near-constant columns for stability
    std_devs = X.std(axis=0)
    keep_cols = std_devs > 1e-6
    X_filtered = X[:, keep_cols]
    print(f"[info] Bayes preprocessing: removed {X.shape[1] - X_filtered.shape[1]} constant columns.")

    # 2. Standardize features based on the training set
    X_train_filtered = X_filtered[train_idx]
    mu = X_train_filtered.mean(axis=0)
    sd = X_train_filtered.std(axis=0)
    sd[sd < 1e-6] = 1.0 # Avoid division by zero if a column is constant in the training set

    X_scaled = (X_filtered - mu) / sd
    X_train_scaled = X_scaled[train_idx]
    y_train = y[train_idx]
    
    # 3. Set intercept prior based on training set prevalence
    p0 = float(np.clip(y_train.mean(), 1e-4, 1 - 1e-4))
    mu_intercept = logit(p0)
    # --- End Preprocessing ---

    with pm.Model() as logreg_model:
        X_data = pm.Data("X_data", X_train_scaled)
        
        b = pm.Normal("b", mu=0, sigma=1.0, shape=X_train_scaled.shape[1])
        a = pm.Normal("a", mu=mu_intercept, sigma=1.5)
        
        logits = a + pt.dot(X_data, b)
        p = pm.Deterministic("p", pm.invlogit(logits))
        
        pm.Bernoulli("obs", logit_p=logits, observed=y_train)

    draws = 500 if fast else 1500
    tune = 500 if fast else 1000
    # Determine a safe number of cores. On Windows (especially when executed via
    # `%run` inside Jupyter/IPython) multiprocessing with PyMC can fail with:
    # AttributeError: module '__main__' has no attribute '__spec__'
    # Force single-core in that scenario.
    requested_cores = 8
    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    if platform.system() == "Windows" and main_spec is None:
        safe_cores = 1
    else:
        safe_cores = min(requested_cores, max(1, (os.cpu_count() or 1))) if 'os' in globals() else 1

    with logreg_model:
        try:
            trace = pm.sample(draws=draws, tune=tune, cores=safe_cores, random_seed=42, progressbar=True)
        except AttributeError as e:
            # Fallback: retry single core if multiprocessing triggers spec issue
            if safe_cores > 1:
                print(f"[warn] PyMC multiprocessing failed ({e}). Retrying with cores=1.")
                trace = pm.sample(draws=draws, tune=tune, cores=1, random_seed=42, progressbar=True)
            else:
                raise

    with logreg_model:
        pm.set_data({"X_data": X_scaled}) # Use the full, scaled dataset for prediction
        post_pred: InferenceData = pm.sample_posterior_predictive(trace, model=logreg_model, var_names=["p"], random_seed=42)

    pred_samples = post_pred.posterior_predictive["p"]  # type: ignore[attr-defined]
    mean_probs = pred_samples.mean(dim=("chain", "draw")).values
    std_probs = pred_samples.std(dim=("chain", "draw")).values

    train_probs = mean_probs[train_idx]
    auc = roc_auc_score(y_train, train_probs)
    pr = average_precision_score(y_train, train_probs)
    
    return mean_probs, std_probs, auc, pr, len(train_idx)


def _generate_shortlist(grid, mineral: str, rf_probs: np.ndarray, bayes_mean: np.ndarray | None, bayes_std: np.ndarray | None, out_dir: Path, top_n: int = 10, quantile: float = 0.6, use_bayes: bool = True):
    """Create a shortlist of top-N target locations and save to CSV.

    Logic (when Bayesian outputs available and use_bayes=True):
      1. Compute uncertainty threshold as the given quantile of bayes_std.
      2. Keep rows with bayes_std <= threshold.
      3. Rank remaining by RF probability (descending) and take top N.
    If Bayesian outputs are not available or fully NaN, fallback to ranking by RF only.
    If after filtering fewer than N candidates remain, fallback to RF-only ranking (no uncertainty filter).
    """
    try:
        shortlist_path = out_dir / f"targets_{mineral}.csv"

        # Extract lon/lat from grid geometry (robust to Points or Polygons)
        lons = lats = None
        if hasattr(grid, "geometry"):
            try:
                centroids = grid.geometry.centroid
                lons = centroids.x.values
                lats = centroids.y.values
            except Exception:
                pass
        if lons is None or lats is None:
            # Fallback to common column names if geometry failed/missing
            if "lon" in grid.columns and "lat" in grid.columns:
                lons = grid["lon"].values
                lats = grid["lat"].values
            else:
                # Last resort: create ordinal placeholders
                lons = np.arange(len(rf_probs))
                lats = np.zeros(len(rf_probs))

        df = pd.DataFrame({
            "mineral": mineral,
            "lon": lons,
            "lat": lats,
            "rf_prob": rf_probs,
        })

        have_bayes = (
            use_bayes
            and bayes_mean is not None
            and bayes_std is not None
            and not np.isnan(bayes_mean).all()
            and not np.isnan(bayes_std).all()
        )

        if have_bayes:
            df["bayes_mean"] = bayes_mean
            df["bayes_std"] = bayes_std
            try:
                arr_std = np.asarray(bayes_std)
                thresh = np.nanquantile(arr_std, quantile)
            except Exception:
                thresh = np.nan
            if not np.isnan(thresh):
                subset = df[df["bayes_std"] <= thresh]
            else:
                subset = df.copy()
            # If filtering removed too many, fallback to full RF ranking
            if len(subset) < top_n:
                subset = df
        else:
            subset = df

        shortlist = subset.sort_values("rf_prob", ascending=False).head(top_n).copy()
        shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1))

        # Ensure consistent columns
        expected_cols = ["rank", "mineral", "lon", "lat", "rf_prob", "bayes_mean", "bayes_std"]
        for col in ["bayes_mean", "bayes_std"]:
            if col not in shortlist.columns:
                shortlist[col] = np.nan
        shortlist = shortlist[expected_cols]

        shortlist.to_csv(shortlist_path, index=False)
        print(f"[info] Shortlist saved: {shortlist_path.name} ({len(shortlist)} rows)")
    except Exception as e:
        print(f"[warn] Failed to generate shortlist for {mineral}: {e}")


def main(args):
    """Main function to precompute model predictions."""
    results = []

    grid_path = PROCESSED / "grid_gdf.joblib"
    if not grid_path.exists():
        print(f"[error] Grid file not found at {grid_path}.")
        print("Please run the '02_feature_engineering.ipynb' notebook first to create the grid.")
        return
        
    grid = joblib.load(grid_path)
    grid_shape = len(grid)
    print(f"[info] Loaded grid with {grid_shape} points.")

    feature_files = sorted(PROCESSED.glob("X_*.npy"))
    feature_arrays = []
    feature_names_order = []  # flattened feature names aligned to columns of X
    
    print("[info] Assembling feature matrix...")
    for f in feature_files:
        try:
            feature = np.load(f)
            if feature.shape[0] == grid_shape:
                if feature.ndim == 1:
                    feature = feature.reshape(-1, 1)
                # Derive column names
                base = f.stem  # e.g. X_geo
                if feature.shape[1] == 1:
                    feature_names_order.append(base)
                else:
                    feature_names_order.extend([f"{base}_{i}" for i in range(feature.shape[1])])
                feature_arrays.append(feature)
                print(f"[feat] Loaded {f.name}: {feature.shape}")
            else:
                print(f"[warn] Skipping {f.name} due to shape mismatch: {feature.shape[0]} rows vs grid's {grid_shape} rows.")
        except Exception as e:
            print(f"[error] Could not load or process {f.name}: {e}")

    if not feature_arrays:
        print("\n[error] No grid-aligned features were found. Cannot proceed.")
        print("Please run the data generation notebooks (02x) to create features.")
        return

    X = np.hstack(feature_arrays)
    print(f"\n[info] Assembled feature matrix with shape: {X.shape}")

    for mineral in MINERALS_DEFAULT:
        y_labels_file = PROCESSED / f"y_labels_{mineral}.npy"
        if not y_labels_file.exists():
            print(f"[warn] Missing labels for {mineral} ({y_labels_file.name}). Skipping.")
            continue

        y = np.load(y_labels_file)
        
        if y.shape[0] != grid_shape:
            print(f"[warn] Label shape mismatch for {mineral}: {y.shape[0]} labels vs {grid_shape} grid points. Skipping.")
            continue

        start_time = time.time()
        print(f"\n--- Processing: {mineral.upper()} ---")

        rf_probs, rf_auc, rf_pr, n_samp_rf, rf_importances = _train_rf(X, y, fast=args.fast_bayes)

        rf_out = PROCESSED / f"rf_probs_{mineral}.npy"
        np.save(rf_out, rf_probs)
        print(f"[info] RF predictions saved: {rf_out.name}")
        if rf_importances is not None and rf_importances.shape[0] == len(feature_names_order):
            try:
                import pandas as _pd
                imp_df = _pd.DataFrame({"feature": feature_names_order, "importance": rf_importances})
                imp_df.sort_values("importance", ascending=False).to_csv(PROCESSED / f"rf_importances_{mineral}.csv", index=False)
                print(f"[info] RF importances saved: rf_importances_{mineral}.csv")
            except Exception as e:
                print(f"[warn] Failed to save RF importances for {mineral}: {e}")
        else:
            print(f"[warn] RF importances not saved for {mineral} (None or length mismatch: got {None if rf_importances is None else rf_importances.shape[0]}, expected {len(feature_names_order)})")

        bayes_auc, bayes_pr, n_samp_bayes = np.nan, np.nan, 0
        bayes_mean = bayes_std = None
        if not args.fast_bayes:
            bayes_mean, bayes_std, bayes_auc, bayes_pr, n_samp_bayes = _bayes_logreg(X, y, fast=args.fast_bayes)

            if not np.isnan(bayes_mean).all():
                np.save(PROCESSED / f"bayes_mean_{mineral}.npy", bayes_mean)
                np.save(PROCESSED / f"bayes_std_{mineral}.npy", bayes_std)
                print(f"[info] Bayesian predictions saved for {mineral}.")
        else:
            print("[info] Skipping Bayesian model (--fast-bayes mode).")

        # --- Shortlist generation ---
        _generate_shortlist(
            grid=grid,
            mineral=mineral,
            rf_probs=rf_probs,
            bayes_mean=bayes_mean,
            bayes_std=bayes_std,
            out_dir=PROCESSED,
            top_n=10,
            quantile=0.6,
            use_bayes=not args.fast_bayes,
        )

        elapsed_time = time.time() - start_time
        results.append({
            "mineral": mineral,
            "rf_auc": rf_auc,
            "rf_pr": rf_pr,
            "bayes_auc": bayes_auc,
            "bayes_pr": bayes_pr,
            "samples": n_samp_rf,
            "seconds": round(elapsed_time, 3),
        })

    summary_df = pd.DataFrame(results)
    if not summary_df.empty:
        print("\n--- Summary of Results ---")
        print(summary_df.to_string())
    else:
        # Fallback: generate shortlist for existing 'critical' probability maps if present
        crit_rf = PROCESSED / "rf_probs_critical.npy"
        if crit_rf.exists():
            print("[info] No mineral-specific labels processed. Falling back to existing 'critical' probabilities for shortlist.")
            rf_probs = np.load(crit_rf)
            bayes_mean_path = PROCESSED / "mean_probs.npy"
            bayes_std_path = PROCESSED / "std_probs.npy"
            bayes_mean = np.load(bayes_mean_path) if bayes_mean_path.exists() else None
            bayes_std = np.load(bayes_std_path) if bayes_std_path.exists() else None
            _generate_shortlist(
                grid=grid,
                mineral="critical",
                rf_probs=rf_probs,
                bayes_mean=bayes_mean,
                bayes_std=bayes_std,
                out_dir=PROCESSED,
                top_n=10,
                quantile=0.6,
                use_bayes=(bayes_mean is not None and bayes_std is not None),
            )
        else:
            print("[warn] No results and no fallback probability files found; nothing to shortlist.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Precompute model predictions for the Streamlit app."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-computation even if output files exist.",
    )
    parser.add_argument(
        "--fast-bayes",
        action="store_true",
        help="Skip Bayesian modeling and use fewer estimators for RF for a faster run.",
    )
    args = parser.parse_args()
    main(args)
