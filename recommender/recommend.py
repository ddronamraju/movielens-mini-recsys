from typing import Literal, List

from recommender.baselines import recommend_random, recommend_popularity
from recommender.svd_model import recommend_svd

ModelName = Literal["random", "popularity", "svd_mf"]

def recommend(user_id: int, k: int = 10, model: ModelName = "svd_mf") -> List[int]:
    """
    Unified interface for recommending movies.

    Parameters
    ----------
    user_id : int
        User ID from the ratings data.
    k : int
        Number of movies to recommend.
    model : {"random", "popularity", "svd_mf"}
        Which model to use for recommendation.

    Returns
    -------
    list[int]
        List of movie_ids.
    """
    if model == "random":
        return recommend_random(user_id, k)
    elif model == "popularity":
        return recommend_popularity(user_id, k)
    elif model == "svd_mf":
        return recommend_svd(user_id, k)
    # elif model == "two_stage":
        # return recommend_two_stage(user_id, k)
    else:
        raise ValueError(f"Unknown model: {model}")
