from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

# Path to the saved LightGBM model (relative to project root)
MODEL_PATH = Path("models/lgbm_ranker.pkl")

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
