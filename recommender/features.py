from __future__ import annotations

import numpy as np
import pandas as pd

from functools import lru_cache

from recommender.data import load_processed_ratings
from recommender.svd_model import (
    user_id_to_idx,
    item_id_to_idx,
    user_factors,
    item_factors,
    get_train_test,
)

# These are the same columns you used when training LightGBM
FEATURE_COLS = [
    "user_total_ratings",
    "user_avg_rating",
    "user_rating_std",
    "movie_popularity",
    "movie_avg_rating",
    "svd_score",
]


@lru_cache(maxsize=1)
def _get_train():
    """Cached access to the same train DF used in ranking dataset creation."""
    train, _ = get_train_test()
    return train


@lru_cache(maxsize=1)
def _compute_user_stats() -> pd.DataFrame:
    train = _get_train()
    user_stats = train.groupby("user_id")["rating"].agg(
        user_total_ratings="count",
        user_avg_rating="mean",
        user_rating_std="std",
    ).reset_index()
    user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0.0)
    return user_stats


@lru_cache(maxsize=1)
def _compute_item_stats() -> pd.DataFrame:
    train = _get_train()
    item_stats = train.groupby("movie_id")["rating"].agg(
        movie_popularity="count",
        movie_avg_rating="mean",
    ).reset_index()
    item_stats["movie_avg_rating"] = item_stats["movie_avg_rating"].fillna(
        item_stats["movie_avg_rating"].mean()
    )
    return item_stats


def _svd_score(user_id: int, movie_id: int) -> float:
    """
    Compute the MF (SVD) score for a given (user, movie) pair.
    Returns 0.0 if either ID is unknown.
    """
    try:
        u_idx = user_id_to_idx[user_id]
        i_idx = item_id_to_idx[movie_id]
    except KeyError:
        return 0.0
    return float(user_factors[u_idx] @ item_factors[i_idx])


def build_features_for_user_candidates(
    user_id: int,
    candidate_movie_ids: list[int],
) -> pd.DataFrame:
    """
    Build a feature DataFrame for (user_id, candidate_movie_ids).

    Returns:
        DataFrame with columns FEATURE_COLS in the same order as used for training.
    """
    train = _get_train()
    user_stats = _compute_user_stats()
    item_stats = _compute_item_stats()

    # Base DF with user_id & movie_id
    df = pd.DataFrame(
        {
            "user_id": [user_id] * len(candidate_movie_ids),
            "movie_id": candidate_movie_ids,
        }
    )

    # Merge user and item aggregates
    df = df.merge(user_stats, on="user_id", how="left")
    df = df.merge(item_stats, on="movie_id", how="left")

    # Fill potential NaNs
    df["user_total_ratings"] = df["user_total_ratings"].fillna(0.0)
    df["user_avg_rating"] = df["user_avg_rating"].fillna(0.0)
    df["user_rating_std"] = df["user_rating_std"].fillna(0.0)
    df["movie_popularity"] = df["movie_popularity"].fillna(0.0)
    df["movie_avg_rating"] = df["movie_avg_rating"].fillna(
        df["movie_avg_rating"].mean()
    )

    # Add SVD score
    df["svd_score"] = df.apply(
        lambda row: _svd_score(row["user_id"], row["movie_id"]),
        axis=1,
    )

    # Return only feature columns in the correct order
    feature_df = df[FEATURE_COLS].copy()
    feature_df.index = df["movie_id"]  # index by movie_id for convenience

    return feature_df
