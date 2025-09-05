from __future__ import annotations
from typing import Tuple

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def train_baseline_rf(X: np.ndarray, y: np.ndarray, random_state: int = 42) -> Tuple[RandomForestClassifier, float]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return model, auc
