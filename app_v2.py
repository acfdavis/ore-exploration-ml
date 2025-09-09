"""Streamlined app (v2).

Features:
    - Layer selector (Model vs Feature Panels) placed before mineral selection.
    - Mineral selector (Cu, Ni, Co, Li, Crit) loading precomputed predictions if present.
    - RF on-the-fly training (fast) with feature toggles (only when training).
    - Precomputed RF/Bayesian layers ignore feature toggles (all core feature blocks included at training time).
    - Display of RF probs, Bayesian mean & std (if arrays exist).
    - High quality feature panels (Geology, Gravity, Geochem, Magnetics) akin to notebooks.
    - Optional target overlay & boundary clipping (square grid enforced always).

Notes:
    Heavy Bayesian sampling is intentionally omitted live; encourage precomputing
    arrays (bayes_mean_<mineral>.npy / bayes_std_<mineral>.npy) in data/processed.
"""
from __future__ import annotations

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
from pathlib import Path
import pandas as pd

from src.app_data import load_common, load_mineral_labels, load_mineral_predictions, load_targets, mineral_code, load_rf_importances
from src.app_models import train_rf
from src.app_geo import build_square_grid, load_state_boundary, clip_to_boundary

st.set_page_config(page_title="Ore Exploration (Demo)", layout="wide")
st.title("Critical Mineral Prospectivity – Demo")
st.caption("Fast, precomputed visualization of critical mineral prospectivity.")

# --- Sidebar Layout ---
# (1) Layer type first
layer_mode = st.sidebar.radio(
    "Layer Type",
    ["Model", "Feature Panel"],
    help="Switch between probability layers and engineered feature panels."
)

# Placeholder container so Model Layer appears visually above mineral selector
model_layer_ct = st.sidebar.container()

# (2) Mineral selection (appears after model layer now)
minerals = ["Ni", "Co", "Li", "Cu", "Crit"]  # "Crit" for general critical minerals
mineral = st.sidebar.selectbox("Mineral", minerals, index=0)
min_code = mineral_code(mineral)

# Data loading (cached)
@st.cache_data(show_spinner=False)
def _load_all():
    return load_common()

grid, features, feature_names = _load_all()
labels = load_mineral_labels(min_code)
preds = load_mineral_predictions(min_code)
rf_importances_pre = load_rf_importances(min_code)
targets_gdf = load_targets(min_code)

feature_toggles = {}

if layer_mode == "Model":
    # Build model options based on current mineral's available predictions
    model_options = []
    if preds.get("rf") is not None:
        model_options.append("RF (Precomputed)")
    model_options.append("RF (On-the-fly)")
    if "bayes_mean" in preds:
        model_options.extend(["Bayesian Mean", "Bayesian Uncertainty"])
    with model_layer_ct:
        layer_choice = st.selectbox("Model Layer", model_options)
        if layer_choice == "RF (On-the-fly)":
            st.markdown("### Features (On-the-fly RF)")
            for k in sorted(features.keys()):
                if k == "X_coords":
                    continue
                nice = k.replace("X_", "")
                feature_toggles[k] = st.checkbox(nice, value=True)
        else:
            for k in features.keys():
                if k == "X_coords":
                    continue
                feature_toggles[k] = True
            st.caption("Precomputed model uses all feature groups.")
else:
    layer_choice = None

st.sidebar.markdown("### Map Display")
show_targets = st.sidebar.checkbox("Show Targets", value=True)
clip_boundary = st.sidebar.checkbox("Clip Boundary", value=True, help="Clip map to state boundary")

st.sidebar.markdown("### Data Sources")
st.sidebar.markdown("""
* **Geology (SGMC / State Geologic Map Compilation)** – USGS: https://pubs.usgs.gov/ds/1052/
* **Geochemistry (National Geochemical Database)** – USGS: https://mrdata.usgs.gov/geochemistry/
* **Gravity (U.S. Gravity / Isostatic Anomaly Grids)** – USGS/NCEI: https://www.ngdc.noaa.gov/mgg/gravity/
* **Magnetics (North American Magnetic Anomaly Grid)** – USGS: https://pubs.usgs.gov/of/2009/1258/
""")

# (old model layer block removed; replaced above)

# --- Prepare working grid (always square cells now) ---
working_grid = build_square_grid(grid.copy())
boundary = load_state_boundary() if clip_boundary else None


def _apply_values(gdf, values):
    gdf2 = gdf.copy()
    gdf2["value"] = values
    if boundary is not None and clip_boundary:
        gdf2 = clip_to_boundary(gdf2, boundary)
    return gdf2


def _color_map(values, cmap_name="viridis"):
    v = np.asarray(values)
    finite = v[np.isfinite(v)]
    if finite.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(finite, [2, 98])
        if np.isclose(vmin, vmax):
            vmin, vmax = float(finite.min()), float(finite.max())
    import matplotlib.pyplot as _plt
    cm_obj = _plt.get_cmap(cmap_name)
    colors = [cm_obj(i) for i in np.linspace(0, 1, 256)]
    return cm.LinearColormap(colors=colors, vmin=vmin, vmax=vmax), vmin, vmax


def _plot_static_map(gdf, column_name, title, cmap, categorical=False, targets_gdf=None, show_targets=False):
    """
    Generates and displays a single, static GeoPandas map for a given column.
    Handles both categorical and continuous data.
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    plot_kwargs = {
        "column": column_name,
        "cmap": cmap,
        "legend": True,
        "ax": ax,
    }
    if categorical:
        plot_kwargs["categorical"] = True
        plot_kwargs["linewidth"] = 0
        # Adjust legend for categorical data to prevent overlap
        plot_kwargs["legend_kwds"] = {'loc': 'upper left', 'bbox_to_anchor': (1, 1)}

    
    gdf.plot(**plot_kwargs)

    if show_targets and targets_gdf is not None:
        targets_gdf.plot(ax=ax, marker='*', color='white', edgecolor='black', markersize=65, label='Targets')

    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    st.pyplot(fig)


def _plot_feature_panel(panel: str, show_targets: bool, targets_gdf):
    g = working_grid.copy()
    if boundary is not None and clip_boundary:
        g = clip_to_boundary(g, boundary)
    
    st.markdown(f"### {panel} Features")

    if panel == "Geology" and "X_geo" in features:
        Xg = features["X_geo"]
        names_geo = feature_names.get("X_geo", [f"unit_{i}" for i in range(Xg.shape[1])])
        dom_idx = np.argmax(Xg, axis=1)
        g["dominant"] = np.array(names_geo)[dom_idx]
        g["dom_frac"] = Xg.max(axis=1)
        
        _plot_static_map(g, "dominant", "Dominant Geology Unit", "tab20", categorical=True, targets_gdf=targets_gdf, show_targets=show_targets)
        _plot_static_map(g, "dom_frac", "Dominant Unit Fraction", "viridis", targets_gdf=targets_gdf, show_targets=show_targets)

    elif panel == "Gravity" and "X_gravity" in features:
        g["gravity"] = features["X_gravity"].reshape(-1)
        _plot_static_map(g, "gravity", "Gravity", "viridis", targets_gdf=targets_gdf, show_targets=show_targets)
        
        if "X_gravity_grad" in features:
            g["gravity_grad"] = features["X_gravity_grad"].reshape(-1)
            _plot_static_map(g, "gravity_grad", "Gravity Gradient", "magma", targets_gdf=targets_gdf, show_targets=show_targets)

    elif panel == "Geochemistry" and "X_geochem" in features:
        Xgc = features["X_geochem"]
        names = feature_names.get("X_geochem", [f"gc_{i}" for i in range(Xgc.shape[1])])
        
        wanted = []
        for pre in ["CU_", "NI_", "CO_", "LI_"]:
            for n in names:
                if n.upper().startswith(pre):
                    wanted.append(n)
                    break
        if not wanted:
            wanted = names[:2]

        for n in wanted:
            if n in names:
                idx = names.index(n)
                g[n] = Xgc[:, idx]
                _plot_static_map(g, n, n, "inferno", targets_gdf=targets_gdf, show_targets=show_targets)

    elif panel == "Magnetics" and "X_mag" in features:
        Xmag = features["X_mag"]
        names = feature_names.get("X_mag", [f"mag_{i}" for i in range(Xmag.shape[1])])
        cmaps = ["viridis", "magma", "cividis"]

        for i, n in enumerate(names):
            g[n] = Xmag[:, i]
            _plot_static_map(g, n, n, cmaps[i % len(cmaps)], targets_gdf=targets_gdf, show_targets=show_targets)
    
    else:
        st.info("Panel not available for this selection or data is missing.")


col_map, col_side = st.columns([3, 1])

with col_map:
    st.subheader("Map")
    values = None  # ensure bound
    if layer_mode == "Model":
        # Determine values
        importances = None
        auc = None
        layer_label = layer_choice
        if layer_choice == "RF (Precomputed)" and "rf" in preds:
            values = preds["rf"]
        elif layer_choice == "RF (On-the-fly)":
            # train
            res = train_rf(features, feature_names, feature_toggles, labels)
            values = res["probs"]
            importances = res["importances"]
            auc = res["auc"]
        elif layer_choice == "Bayesian Mean" and "bayes_mean" in preds:
            values = preds["bayes_mean"]
        elif layer_choice == "Bayesian Uncertainty" and "bayes_std" in preds:
            values = preds["bayes_std"]
        else:
            st.warning("Selected layer unavailable.")
        if values is not None:
            # Ensure numpy array for downstream operations
            if not isinstance(values, np.ndarray):
                try:
                    values = np.asarray(values, dtype=float)
                except Exception:
                    values = np.array(values)
            gplot = _apply_values(working_grid, values)
            center = [gplot.geometry.centroid.y.mean(), gplot.geometry.centroid.x.mean()]
            fmap = folium.Map(location=center, zoom_start=6, tiles="CartoDB positron", control_scale=True)
            cmap_name = "plasma" if ("Uncertainty" in (layer_label or "")) else "viridis"
            colormap, vmin, vmax = _color_map(values, cmap_name=cmap_name)
            def _style(feat):
                val = feat["properties"].get("value")
                if val is None or not isinstance(val, (int, float)):
                    return {"fillColor": "transparent", "color": "#555", "weight": 0.05, "fillOpacity": 0.0}
                return {"fillColor": colormap(val), "color": "#888", "weight": 0.05, "fillOpacity": 0.8}
            folium.GeoJson(
                gplot,
                style_function=_style,
                tooltip=folium.GeoJsonTooltip(fields=["value"], aliases=["Value:"], sticky=True),
                name="Prospectivity"
            ).add_to(fmap)
            if targets_gdf is not None and show_targets:
                for _, r in targets_gdf.iterrows():
                    folium.CircleMarker([
                        r.geometry.y, r.geometry.x
                    ], radius=5, color="cyan", fill=True, fill_color="cyan", fill_opacity=0.85,
                    popup=str(r.get("rank", "target"))).add_to(fmap)
            colormap.add_to(fmap)
            st_folium(fmap, width=760, height=520, returned_objects=[])
        if importances is not None:
            try:
                st.markdown("#### RF Feature Importances")
                st.bar_chart(importances.set_index("feature"))  # type: ignore[attr-defined]
            except Exception:
                pass
        if isinstance(auc, (int, float)) and auc == auc:  # simple NaN check
            st.metric("RF AUC", f"{float(auc):.3f}")
    else:  # Feature Panel mode
        panel = st.selectbox("Feature Group", ["Geology", "Gravity", "Geochemistry", "Magnetics"], key="feat_panel")
        _plot_feature_panel(panel, show_targets, targets_gdf)

with col_side:
    st.subheader("Distribution")
    if layer_mode == "Model" and 'values' in locals() and values is not None:
        arr_vals = values if isinstance(values, np.ndarray) else np.asarray(values)
        finite = arr_vals[np.isfinite(arr_vals)] if arr_vals.size else arr_vals
        if finite.size:
            fig, ax = plt.subplots(figsize=(3.5,3.0))
            rng = (0,1) if finite.min() >= 0 and finite.max() <= 1 else None
            # Log-transform data (x-axis) rather than log-scaling counts
            eps = 1e-6
            # Shift if any non-positive values
            if (finite <= 0).any():
                shift = abs(finite.min()) + eps
            else:
                shift = 0.0
            finite_log = np.log10(finite + shift + eps)
            ax.hist(finite_log, bins=40)
            ax.set_xlabel("log10(Value + shift)")
            if shift > 0:
                ax.set_ylabel("Count (shifted to make positive before log)")
            else:
                ax.set_ylabel("Count")
            ax.set_title("Log10 Value Histogram")
            st.pyplot(fig)
    # Feature importances (only for On-the-fly RF since precomputed has no model object here)
    if layer_mode == "Model" and layer_choice == "RF (On-the-fly)" and 'importances' in locals() and importances is not None:
        if isinstance(importances, pd.DataFrame) and {'feature','importance'}.issubset(importances.columns):
            st.markdown("#### Top Features")
            imp_df = importances.head(12)
            fig_imp, ax_imp = plt.subplots(figsize=(3.5, 4.0))
            ax_imp.barh(list(imp_df['feature'])[::-1], list(imp_df['importance'])[::-1], color='steelblue')  # type: ignore[index]
            ax_imp.set_xlabel('Importance')
            ax_imp.set_ylabel('Feature')
            ax_imp.set_title('RF Importances')
            plt.tight_layout()
            st.pyplot(fig_imp)
    elif layer_mode == "Model" and layer_choice == "RF (Precomputed)" and rf_importances_pre is not None:
        try:
            if {'feature','importance'}.issubset(rf_importances_pre.columns):
                st.markdown("#### Top Features (Precomputed RF)")
                imp_df = rf_importances_pre.sort_values('importance', ascending=False).head(12)
                fig_imp2, ax_imp2 = plt.subplots(figsize=(3.5, 4.0))
                ax_imp2.barh(list(imp_df['feature'])[::-1], list(imp_df['importance'])[::-1], color='teal')  # type: ignore[index]
                ax_imp2.set_xlabel('Importance')
                ax_imp2.set_ylabel('Feature')
                ax_imp2.set_title('RF Importances')
                plt.tight_layout()
                st.pyplot(fig_imp2)
        except Exception:
            pass
    st.markdown("---")
    st.markdown("**Legend**")
    st.caption("RF (On-the-fly) retrains instantly. Bayesian layers require precomputed arrays.")

    if show_targets and targets_gdf is not None:
        try:
            st.markdown("### Top Targets")
            cols_show = [c for c in ["rank", "rf_prob", "bayes_mean", "bayes_std"] if c in targets_gdf.columns]
            st.dataframe(targets_gdf[cols_show].set_index("rank") if "rank" in cols_show else targets_gdf[cols_show].head(10))
        except Exception:
            pass

st.info("Tip: Place bayes_mean_<mineral>.npy & bayes_std_<mineral>.npy into data/processed to enable Bayesian layers.")
