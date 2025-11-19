import pandas as pd
from typing import Callable

def hit_at_k(test_df: pd.DataFrame, recommender_fn: Callable[[int, int], list[int]], k: int = 10) -> float:
    """
    Compute Hit@K over the given test_df for the provided recommender function.

    test_df is expected to have columns:
    - user_id
    - movie_id (the 'true' held-out item)
    """
    hits = 0
    total = 0

    for _, row in test_df.iterrows():
        user = row["user_id"]
        true_item = row["movie_id"]

        recs = recommender_fn(user, k)
        if not recs:
            continue

        total += 1
        if true_item in recs:
            hits += 1

    return hits / total if total > 0 else 0.0
