import os
import numpy as np
import joblib
import geopandas as gpd
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
import folium
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
        # Use the first column if a feature has multiple
        raw_values = features[active_layer_key]
        values = raw_values[:, 0] if raw_values.ndim > 1 else raw_values
        importances = None

    # Create Folium map
    center = [grid.geometry.centroid.y.mean(), grid.geometry.centroid.x.mean()]
    m = folium.Map(location=center, zoom_start=7, tiles="CartoDB positron", control_scale=True)

    # Add data layer to map
    cmap = 'viridis'
    if title and 'Uncertainty' in title:
        cmap = 'plasma'
    
    # Normalize values for colormap
    vmin, vmax = np.nanpercentile(values, [5, 95])
    
    mpl_cmap = plt.get_cmap(cmap)
    colors = [mpl_cmap(i) for i in np.linspace(0, 1, 256)]
    colormap = cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax)


    folium.GeoJson(
        grid.assign(value=values),
        style_function=lambda feature: {
            'fillColor': colormap(feature['properties']['value']),
            'color': 'transparent',
            'weight': 0,
            'fillOpacity': 0.6
        },
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
    
    st.markdown("#### Probability Distribution")
    fig, ax = plt.subplots()
    ax.hist(values, bins=50, range=(0,1))
    ax.set_title("Distribution of Predictions")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

st.info("Tip: The on-the-fly model retrains automatically when you change feature selections.")



