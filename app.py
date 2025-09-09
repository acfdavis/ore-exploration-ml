import os
import numpy as np
import joblib
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import matplotlib.pyplot as plt
from pathlib import Path
import json

st.set_page_config(page_title="Interactive Ore Exploration", layout="wide")

st.title("Interactive Ore Exploration Dashboard")
st.markdown(
    """
    An interactive tool to explore critical mineral (Cu, Ni, Co, Li) prospectivity models. 
    - **Select features** in the sidebar to train a model on-the-fly.
    - **View model outputs** or raw data layers on the map.
    - **Analyze results** with feature importance and probability histograms.
    """
)

# --- Data Loading ---
def run_setup():
    """Runs the data setup script."""
    import subprocess
    import sys
    
    st.info("First-time setup: Running data processing scripts. This may take several minutes...")
    with st.spinner("Downloading data and generating features... Please wait."):
        try:
            process = subprocess.run(
                [sys.executable, "setup.py"],
                capture_output=True,
                text=True,
                check=True
            )
            st.code(process.stdout)
            st.success("Data setup complete! The app will now load.")
            # Clear the cache and rerun to load the new data
            st.cache_data.clear()
        except subprocess.CalledProcessError as e:
            st.error("An error occurred during the data setup process.")
            st.code(e.stderr)
            st.stop()

@st.cache_data
def load_data():
    """Loads all necessary data from disk, caching the result."""
    root = Path("data/processed")
    if not root.exists() or not (root / "grid_gdf.joblib").exists():
        run_setup()
        # After setup, we need to rerun the script to enter the main app logic
        st.rerun()

    grid = joblib.load(root / "grid_gdf.joblib")

    features = {}
    feature_names = {}
    for f_path in root.glob("X_*.npy"):
        key = f_path.stem
        features[key] = np.load(f_path)
        json_path = f_path.with_name(f"feature_names_{key.split('_')[1]}.json")
        if json_path.exists():
            with open(json_path, 'r') as f:
                feature_names[key] = json.load(f)
        else:
            # Fallback for simple features like coords
            if 'coords' in key:
                feature_names[key] = ['lon', 'lat']
            else:
                feature_names[key] = [key]


    outputs = {f.stem: np.load(f) for f in root.glob("*_probs.npy")}
    if (root / f"rf_probs_critical.npy").exists():
        outputs['rf_probs_critical'] = np.load(root / f"rf_probs_critical.npy")

    labels = np.load(root / "y_labels_crit.npy") if (root / "y_labels_crit.npy").exists() else np.load(root / "y_labels.npy")
    
    targets = None
    if (root / "targets.gpkg").exists():
        targets = gpd.read_file(root / "targets.gpkg")

    return grid, features, feature_names, outputs, labels, targets

grid, features, feature_names, outputs, y_labels, targets_gdf = load_data()

# --- Helpers for polygon grid + clipping ---
def build_square_grid(_base_grid):
    """One square polygon per point (median dx,dy)."""
    base_grid = _base_grid
    if base_grid.empty:
        return base_grid
    if base_grid.geometry.iloc[0].geom_type != 'Point':
        return base_grid
    import numpy as _np
    xs = base_grid.geometry.x.to_numpy(); ys = base_grid.geometry.y.to_numpy()
    ux = _np.unique(_np.sort(xs)); uy = _np.unique(_np.sort(ys))
    dx = float(_np.median(_np.diff(ux))) if len(ux) > 1 else 0.1
    dy = float(_np.median(_np.diff(uy))) if len(uy) > 1 else dx
    if dx <= 0: dx = 0.1
    if dy <= 0: dy = dx
    halfx, halfy = dx/2.0, dy/2.0
    from shapely.geometry import box
    polys = [box(x-halfx, y-halfy, x+halfx, y+halfy) for x, y in zip(xs, ys)]
    gpoly = base_grid.copy(); gpoly.geometry = polys
    return gpoly

@st.cache_data
def load_state_boundary():
    path = Path('data/raw/state_geology.gpkg')
    if not path.exists():
        return None
    try:
        gdf = gpd.read_file(path)
        # Dissolve to single geometry
        gdf['__one'] = 1
        boundary = gdf.dissolve('__one').geometry.iloc[0]
        return gpd.GeoDataFrame({'geometry':[boundary]}, crs=gdf.crs)
    except Exception:
        return None

def clip_to_boundary(_gdf, _boundary_gdf):
    gdf = _gdf; boundary_gdf = _boundary_gdf
    if boundary_gdf is None:
        return gdf
    try:
        if gdf.crs != boundary_gdf.crs:
            boundary_gdf = boundary_gdf.to_crs(gdf.crs)
        # gpd.clip returns subset with partial geometries truncated
        clipped = gpd.clip(gdf, boundary_gdf)
        return clipped
    except Exception:
        return gdf

# --- Sidebar Controls ---
st.sidebar.title("Controls")

# Layer selection
display_mode = st.sidebar.radio("Display Mode", ["Model Outputs", "Raw Features"])

if display_mode == "Model Outputs":
    model_choices = {
        "Bayesian Mean": "mean_probs",
        "Bayesian Uncertainty": "std_probs",
        "Pre-computed RF": "rf_probs_critical",
        "On-the-fly RF": "on_the_fly_rf",
    }
    available_models = {name: key for name, key in model_choices.items() if key in outputs or key == "on_the_fly_rf"}
    layer_choice = st.sidebar.selectbox("Model Layer", list(available_models.keys()))
    active_layer_key = available_models[layer_choice]
else:
    # Allow viewing raw feature layers
    raw_feature_choices = {name: name for name in features if name != 'X_coords'}
    layer_choice = st.sidebar.selectbox("Raw Feature Layer", list(raw_feature_choices.keys()))
    active_layer_key = raw_feature_choices[layer_choice]


st.sidebar.header("Map Options")
show_targets = st.sidebar.checkbox("Show Target Locations", value=True)


# Feature selection for on-the-fly model
st.sidebar.header("On-the-fly Model Features")

# Determine if the toggles should be active. They are only active for the on-the-fly model.
is_on_the_fly_mode = (display_mode == "Model Outputs" and active_layer_key == "on_the_fly_rf")

if not is_on_the_fly_mode:
    st.sidebar.info("Feature selection is only available for the 'On-the-fly RF' model.")

feature_toggles = {}
# Exclude coords as it's always on
available_features = {k: v for k, v in features.items() if k != 'X_coords'}
for key in available_features:
    # The checkbox is disabled if not in on-the-fly mode.
    # The value is kept at True so that if the user switches back and forth,
    # the state of the toggles is not lost.
    feature_toggles[key] = st.sidebar.checkbox(
        f"Use {key.replace('X_', '')}", 
        value=True, 
        disabled=not is_on_the_fly_mode
    )


# --- Model Training & Feature Engineering ---
@st.cache_data
def assemble_and_train(toggles):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score

    # Always include coordinates
    parts = [features['X_coords']]
    names = feature_names['X_coords']
    
    for key, use_feature in toggles.items():
        if use_feature and key in features:
            feature_array = features[key]
            if feature_array.ndim == 1:
                feature_array = feature_array.reshape(-1, 1)
            parts.append(feature_array)
            names.extend(feature_names.get(key, [key]))

    X = np.hstack(parts)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y_labels)
    
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y_labels, probs)
    
    # Feature importances
    importances = pd.DataFrame({'feature': names, 'importance': model.feature_importances_})
    importances = importances.sort_values('importance', ascending=False).head(15)
    
    return probs, auc, importances

# --- Main Panel ---
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Map Visualization")
    
    # Determine values to plot
    if display_mode == "Model Outputs":
        if active_layer_key == "on_the_fly_rf":
            title = "On-the-fly RF Probability"
            with st.spinner("Training model with selected features..."):
                values, auc, importances = assemble_and_train(feature_toggles)
            st.metric("On-the-fly Model AUC", f"{auc:.3f}")
        else:
            title = layer_choice
            values = outputs[active_layer_key]
            importances = None # No feature importances for pre-computed models
    else: # Raw Features mode
        title = f"Raw Feature: {layer_choice}"
        raw_values = features[active_layer_key]
        if raw_values.ndim > 1 and raw_values.shape[1] > 1:
            # Get provided names; extend or replace if insufficient
            provided = feature_names.get(active_layer_key, [])
            base = active_layer_key.replace('X_', '')
            cols_full = [
                provided[i] if i < len(provided) else f"{base}_{i}"
                for i in range(raw_values.shape[1])
            ]
            col_index = st.sidebar.selectbox(
                "Feature Column", list(range(raw_values.shape[1])),
                format_func=lambda i: cols_full[i] if 0 <= i < len(cols_full) else str(i)
            )
            values = raw_values[:, col_index]
        else:
            values = raw_values if raw_values.ndim == 1 else raw_values[:,0]
        importances = None

    # Sidebar options for grid style
    clip_opt = st.sidebar.checkbox("Clip to State Boundary", value=True)
    square_opt = st.sidebar.checkbox("Render Square Grid", value=True)

    working_grid = grid
    if square_opt:
        working_grid = build_square_grid(working_grid)
    boundary_gdf = load_state_boundary() if clip_opt else None
    if clip_opt and boundary_gdf is not None:
        # Assign values before clipping so intersection keeps attribute
        temp = working_grid.assign(value=values)
        working_grid = clip_to_boundary(temp, boundary_gdf)
    else:
        working_grid = working_grid.assign(value=values)

    # Create Folium map
    center = [working_grid.geometry.centroid.y.mean(), working_grid.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron", control_scale=True)

    # Colormap setup
    cmap = 'viridis'
    if title and 'Uncertainty' in title:
        cmap = 'plasma'
    try:
        vmin, vmax = np.nanpercentile(values, [5, 95])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    except Exception:
        vmin, vmax = float(np.nanmin(values)), float(np.nanmax(values))
    mpl_cmap = plt.get_cmap(cmap)
    colors = [mpl_cmap(i) for i in np.linspace(0, 1, 256)]
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)

    def _style(feature):
        val = feature['properties'].get('value', None)
        if val is None or not isinstance(val, (int, float)) or not np.isfinite(val):
            return {'fillColor': 'transparent', 'color': 'grey', 'weight': 0.05, 'fillOpacity': 0.0}
        return {'fillColor': colormap(val), 'color': 'grey', 'weight': 0.05, 'fillOpacity': 0.8}

    folium.GeoJson(
        working_grid,
        style_function=_style,
        tooltip=folium.GeoJsonTooltip(fields=['value'], aliases=['Value:'], sticky=True)
    ).add_to(m)
    
    # Add targets
    if targets_gdf is not None and show_targets:
        for _, row in targets_gdf.iterrows():
            rf_p = row.get('rf_p')
            popup_text = f"Target Rank: {row.get('rank', 'N/A')}"
            if isinstance(rf_p, (int, float)):
                popup_text += f"<br>RF Prob: {rf_p:.3f}"
            else:
                popup_text += f"<br>RF Prob: N/A"

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=5,
                color='cyan',
                fill=True,
                fill_color='cyan',
                fill_opacity=0.8,
                popup=popup_text
            ).add_to(m)

    colormap.add_to(m)
    
    st_folium(m, width=700, height=500, returned_objects=[])

with col2:
    st.subheader("Analysis")
    if importances is not None:
        st.markdown("#### Feature Importances")
        st.bar_chart(importances.set_index('feature'))
    
    st.markdown("#### Value Distribution")
    fig, ax = plt.subplots()
    # Auto-range based on data; clamp if probabilities 0-1
    data = values[np.isfinite(values)]
    if len(data):
        dmin, dmax = np.min(data), np.max(data)
        hist_range = (0, 1) if (dmin >= 0 and dmax <= 1) else None
        ax.hist(data, bins=50, range=hist_range)
    ax.set_title("Distribution of Values")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

st.info("Tip: The on-the-fly model retrains automatically when you change feature selections.")

# --- Feature Plot Panels (replicate notebooks 02b–02e) ---
st.header("Feature Plots (Notebook Style)")
with st.expander("Show Static Feature Maps", expanded=False):
    feat_group = st.selectbox(
        "Feature Group",
        ["Geology", "Gravity", "Geochemistry", "Magnetics"],
        help="Select which engineered feature set to visualize in notebook style."
    )
    clip_feat = st.checkbox("Clip to State Boundary (static plots)", value=True)
    square_feat = st.checkbox("Use Square Grid (static plots)", value=True)
    run_feat = st.button("Generate Feature Plot")

    def _prep_grid():
        g = grid
        if square_feat:
            g = build_square_grid(g)
        b = load_state_boundary() if clip_feat else None
        if clip_feat and b is not None:
            g = clip_to_boundary(g, b)
        return g

    if run_feat:
        gwork = _prep_grid().copy()
        import matplotlib.pyplot as plt
        fig = None
        if feat_group == "Geology":
            if 'X_geo' not in features:
                st.warning("Geology features (X_geo.npy) not found.")
            else:
                Xg = features['X_geo']
                names_geo = feature_names.get('X_geo', [f'unit_{i}' for i in range(Xg.shape[1])])
                # Dominant unit index & label
                dom_idx = np.argmax(Xg, axis=1)
                dom_lab = np.array(names_geo)[dom_idx]
                gwork['dominant'] = pd.Categorical(dom_lab)
                gwork['dom_frac'] = Xg.max(axis=1)
                fig, axes = plt.subplots(1, 2, figsize=(13,6))
                gwork.plot(column='dominant', categorical=True, legend=True, cmap='tab20', linewidth=0, ax=axes[0])
                axes[0].set_title('Dominant Geology Unit')
                axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
                gwork.plot(column='dom_frac', cmap='viridis', legend=True, ax=axes[1])
                axes[1].set_title('Dominant Unit Coverage Fraction')
                axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
                plt.tight_layout()
        elif feat_group == "Gravity":
            if 'X_gravity' not in features:
                st.warning("Gravity feature (X_gravity.npy) not found.")
            else:
                gwork['grav'] = features['X_gravity'].reshape(-1)
                if 'X_gravity_grad' in features:
                    gwork['ggrad'] = features['X_gravity_grad'].reshape(-1)
                    fig, axes = plt.subplots(1, 2, figsize=(12,6))
                    gwork.plot(column='grav', cmap='viridis', legend=True, ax=axes[0])
                    axes[0].set_title('Gravity')
                    axes[0].set_xlabel('Longitude'); axes[0].set_ylabel('Latitude')
                    gwork.plot(column='ggrad', cmap='magma', legend=True, ax=axes[1])
                    axes[1].set_title('Gravity Gradient')
                    axes[1].set_xlabel('Longitude'); axes[1].set_ylabel('Latitude')
                    plt.tight_layout()
                else:
                    fig, axg = plt.subplots(figsize=(6.5,6.5))
                    gwork.plot(column='grav', cmap='viridis', legend=True, ax=axg)
                    axg.set_title('Gravity')
                    axg.set_xlabel('Longitude'); axg.set_ylabel('Latitude')
                    plt.tight_layout()
        elif feat_group == "Geochemistry":
            if 'X_geochem' not in features:
                st.warning("Geochemistry features (X_geochem.npy) not found.")
            else:
                Xgc = features['X_geochem']
                fn_gc = feature_names.get('X_geochem', [f'gc_{i}' for i in range(Xgc.shape[1])])
                # Pick up to 4 critical element percentile columns (CU, NI, CO, LI)
                wanted_prefix = ['CU_', 'NI_', 'CO_', 'LI_']
                chosen = []
                for pre in wanted_prefix:
                    for n in fn_gc:
                        if n.upper().startswith(pre):
                            chosen.append(n); break
                if not chosen:
                    # fallback to first up to 4 columns
                    chosen = fn_gc[:4]
                for i, n in enumerate(fn_gc):
                    if n in chosen:
                        gwork[n] = Xgc[:, i]
                ncols = len(chosen)
                fig, axes = plt.subplots(1, ncols, figsize=(6*ncols,6))
                if ncols == 1:
                    axes = [axes]
                for ax, col in zip(axes, chosen):
                    gwork.plot(column=col, cmap='inferno', legend=True, ax=ax)
                    ax.set_title(f'Geochem: {col}')
                    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
                plt.tight_layout()
        else: # Magnetics
            if 'X_mag' not in features:
                st.warning("Magnetics features (X_mag.npy) not found.")
            else:
                Xmag = features['X_mag']
                names_mag = feature_names.get('X_mag', [f'mag_{i}' for i in range(Xmag.shape[1])])
                for i, n in enumerate(names_mag):
                    gwork[n] = Xmag[:, i]
                # Use first three (val, hgm, tilt) if available
                cols = names_mag[:3]
                fig, axes = plt.subplots(1, len(cols), figsize=(5.5*len(cols),5.5))
                if len(cols) == 1:
                    axes = [axes]
                cmaps = ['viridis','magma','cividis']
                for ax, col, cm_ in zip(axes, cols, cmaps):
                    gwork.plot(column=col, cmap=cm_, legend=True, ax=ax)
                    ax.set_title(f'Magnetic {col}')
                    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
                plt.tight_layout()
        if fig is not None:
            st.pyplot(fig)
            if st.button("Save Feature Figure"):
                out_dir = Path('figures'); out_dir.mkdir(exist_ok=True)
                fname = f'feature_{feat_group.lower()}_panel.png'
                fig.savefig(out_dir / fname, dpi=180, bbox_inches='tight')
                st.success(f'Saved figures/{fname}')

# --- Model Comparison Panels (RF vs Bayesian) ---
st.header("Model Comparison (Static Maps)")
st.caption("Replicates notebook 03 / 03c style: Calibrated Random Forest vs Bayesian Logistic Regression (mean & uncertainty)")

with st.expander("Generate / View Comparison Figure", expanded=False):
    fast_mode = st.checkbox("Fast sampling (lower quality)", value=True, help="Uses fewer draws/chains for speed.")
    run_button = st.button("Run Model Comparison", type="primary")

    @st.cache_data(show_spinner=False)
    def run_model_comparison(fast: bool):
        import numpy as _np
        # Assemble full feature matrix (coords + all others)
        parts = [features['X_coords']]
        names = list(feature_names.get('X_coords', ['lon','lat']))
        for k, arr in features.items():
            if k == 'X_coords':
                continue
            if arr.ndim == 1:
                arr = arr.reshape(-1,1)
            if arr.shape[0] != parts[0].shape[0]:
                continue
            parts.append(arr)
            names.extend(feature_names.get(k, [k]))
        X = _np.hstack(parts)
        y = y_labels.astype(int)

        # Remove near-constant columns for Bayes stability
        std = X.std(axis=0, ddof=0)
        keep = std > 1e-8
        Xb = X[:, keep]
        Xb = (Xb - Xb.mean(axis=0)) / (Xb.std(axis=0) + 1e-9)

        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.metrics import roc_auc_score, average_precision_score

        rf = RandomForestClassifier(
            n_estimators=300 if fast else 600,
            max_features='sqrt',
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42
        )
        cal_rf = CalibratedClassifierCV(rf, method='isotonic', cv=3 if fast else 5)
        cal_rf.fit(X, y)
        rf_probs = cal_rf.predict_proba(X)[:,1]
        auc_rf = roc_auc_score(y, rf_probs)
        pr_rf = average_precision_score(y, rf_probs)

        # Bayesian logistic regression
        try:
            import pymc as pm
            import numpy as np
            from scipy.special import logit
            try:
                import aesara.tensor as at  # type: ignore
            except Exception:  # pragma: no cover
                at = None  # type: ignore
        except ImportError:
            return {
                'error': 'pymc not installed',
                'rf_probs': rf_probs,
                'auc_rf': auc_rf,
                'pr_rf': pr_rf
            }

        p0 = float(np.clip(y.mean(), 1e-4, 1-1e-4))
        mu_intercept = logit(p0)
        draws = 400 if fast else 1000
        tune = 600 if fast else 1500
        chains = 2 if fast else 4
        cores = 1
        with pm.Model() as model:
            beta = pm.Normal('beta', 0.0, 0.5, shape=Xb.shape[1])
            intercept = pm.Normal('intercept', mu_intercept, 2.0)
            if at is not None:
                logits = intercept + at.dot(Xb, beta)
            else:  # fallback (will still compile via numpy inside graph)
                logits = intercept + (Xb @ beta)
            pm.Bernoulli('y_obs', logit_p=logits, observed=y)
            idata = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                              init='jitter+adapt_diag', target_accept=0.94, progressbar=False,
                              random_seed=42, return_inferencedata=True)
        # Posterior is an xarray Dataset (chains, draws, parameter)
        beta_samp = idata.posterior['beta'].values   # type: ignore[attr-defined]
        int_samp = idata.posterior['intercept'].values  # type: ignore[attr-defined]
        logits = int_samp[..., None] + np.einsum('cdf,nf->cdn', beta_samp, Xb)
        post_p = 1.0 / (1.0 + np.exp(-logits))
        bayes_mean = post_p.mean(axis=(0,1))
        bayes_std = post_p.std(axis=(0,1))
        auc_bayes = roc_auc_score(y, bayes_mean)
        pr_bayes = average_precision_score(y, bayes_mean)

        return {
            'rf_probs': rf_probs,
            'auc_rf': auc_rf,
            'pr_rf': pr_rf,
            'bayes_mean': bayes_mean,
            'bayes_std': bayes_std,
            'auc_bayes': auc_bayes,
            'pr_bayes': pr_bayes,
            'n_samples': draws * chains
        }

    if run_button:
        with st.spinner("Training RF and Bayesian models..."):
            res = run_model_comparison(fast_mode)
        if 'error' in res:
            st.error("Bayesian model skipped: " + res['error'])
        # Plot panel
        if 'bayes_mean' in res:
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(18,6))
            base_for_plot = build_square_grid(grid)
            gtmp = base_for_plot.copy(); gtmp['val'] = res['rf_probs']
            gtmp.plot(column='val', ax=axs[0], cmap='viridis', legend=True, legend_kwds={'shrink':0.6})
            axs[0].set_title(f"RF (AUC={res['auc_rf']:.2f})")
            gtmp['val'] = res['bayes_mean']
            gtmp.plot(column='val', ax=axs[1], cmap='viridis', legend=True, legend_kwds={'shrink':0.6})
            axs[1].set_title(f"Bayes Mean (AUC={res['auc_bayes']:.2f})")
            gtmp['val'] = res['bayes_std']
            gtmp.plot(column='val', ax=axs[2], cmap='plasma', legend=True, legend_kwds={'shrink':0.6})
            axs[2].set_title("Bayes Uncertainty (Std)")
            for ax in axs:
                ax.set_xlabel('Lon'); ax.set_ylabel('Lat')
            plt.tight_layout()
            st.pyplot(fig)
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("RF ROC AUC", f"{res['auc_rf']:.3f}")
            col_a.metric("RF PR AUC", f"{res['pr_rf']:.3f}")
            col_b.metric("Bayes ROC AUC", f"{res['auc_bayes']:.3f}")
            col_b.metric("Bayes PR AUC", f"{res['pr_bayes']:.3f}")
            if fast_mode:
                col_c.info("Fast mode sampling")
            else:
                col_c.success(f"Samples: {res['n_samples']}")
            # Optional save
            if st.button("Save Figure & Arrays"):
                from pathlib import Path
                out_dir = Path('figures'); out_dir.mkdir(exist_ok=True)
                fig_path = out_dir / 'model_comparison.png'
                fig.savefig(fig_path, dpi=180, bbox_inches='tight')
                np.save('data/processed/rf_probs_comparison.npy', res['rf_probs'])
                np.save('data/processed/bayes_mean.npy', res['bayes_mean'])
                np.save('data/processed/bayes_std.npy', res['bayes_std'])
                st.success(f"Saved figure to {fig_path}")
        else:
            st.warning("Bayesian results unavailable; only RF probabilities produced.")




