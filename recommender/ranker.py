from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Path to the saved LightGBM model (relative to the package root)
# This works whether running from notebooks/ or project root
def _get_model_path():
    # Get the directory containing this module
    module_dir = Path(__file__).parent
    # Go up one level to the project root and then to models
    project_root = module_dir.parent
    return project_root / "models" / "lgbm_ranker.pkl"

MODEL_PATH = _get_model_path()

# Cache the loaded model in a module-level variable
_lgbm_model = None

def _load_model():
    global _lgbm_model
    if _lgbm_model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"LightGBM model not found at {MODEL_PATH}. "
                "Train it first (see notebooks/06_train_lgbm_ranker.ipynb)."
            )
        _lgbm_model = joblib.load(MODEL_PATH)
    return _lgbm_model

def score_candidates(feature_df: pd.DataFrame) -> np.ndarray:
    """
    Given a DataFrame of candidate features, return predicted relevance scores.

    The feature_df should contain the same feature columns used during training,
    in the same order.
    """
    model = _load_model()
    proba = model.predict_proba(feature_df)[:, 1]
    return proba
