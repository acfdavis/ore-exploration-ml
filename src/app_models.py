"""Model helpers for streamlined app.

Only Random Forest is trained live. Bayesian results are expected to be
precomputed; live PyMC sampling can be optionally added but is expensive
for a demo environment.
"""
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def assemble_matrix(features: Dict[str, np.ndarray], feature_names: Dict[str, list], toggles: Dict[str, bool]) -> Tuple[np.ndarray, List[str]]:
    parts = []
    names: List[str] = []
    # Always include coords first
    if "X_coords" in features:
        parts.append(features["X_coords"])
        names.extend(feature_names.get("X_coords", ["lon", "lat"]))
    for key, arr in features.items():
        if key == "X_coords":
            continue
        if not toggles.get(key, True):
            continue
        a = arr
        if a.ndim == 1:
            a = a.reshape(-1, 1)
        parts.append(a)
        names.extend(feature_names.get(key, [key]))
    X = np.hstack(parts) if parts else np.empty((0, 0))
    return X, names


def train_rf(features: Dict[str, np.ndarray], feature_names: Dict[str, list], toggles: Dict[str, bool], labels) -> Dict[str, object]:
    X, names = assemble_matrix(features, feature_names, toggles)
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X, labels)
    probs = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(labels, probs) if len(set(labels)) > 1 else float("nan")
    importances = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    importances = importances.sort_values("importance", ascending=False).head(20)
    return {"probs": probs, "auc": auc, "importances": importances}
