from __future__ import annotations

from typing import List

from recommender.svd_model import recommend_svd
from recommender.features import build_features_for_user_candidates
from recommender.ranker import score_candidates


def recommend_two_stage(
    user_id: int,
    k: int = 10,
    n_candidates: int = 50,
) -> List[int]:
    """
    Two-stage recommendation:
      1) Candidate generation via SVD MF
      2) Re-ranking via LightGBM using rich features

    Returns:
        List of movie_ids (Top-k).
    """
    # 1) Get candidate movies via SVD-based MF
    candidate_movie_ids = recommend_svd(user_id, k=n_candidates)
    if not candidate_movie_ids:
        # In cold-start or failure, you might fall back to popularity later.
        return []

    # 2) Build features for (user, candidate_movie_ids)
    feature_df = build_features_for_user_candidates(user_id, candidate_movie_ids)

    # 3) Score candidates using LightGBM
    scores = score_candidates(feature_df)

    # 4) Sort by score and pick top-k
    feature_df = feature_df.copy()
    feature_df["score"] = scores

    # feature_df index is movie_id (set in features.py)
    ranked = feature_df.sort_values("score", ascending=False)
    top_movie_ids = ranked.index.to_list()[:k]

    return top_movie_ids
