import numpy as np
import pandas as pd
from recommender.data import load_processed_ratings

# Load and prepare train/test split
df = load_processed_ratings()
df = df.sort_values("timestamp")

test = df.groupby("user_id").tail(1)
train = df.drop(test.index)

all_movies = df["movie_id"].unique()

# Popularity counts: global popularity baseline
popularity = train.groupby("movie_id").size().sort_values(ascending=False)
popular_movies = popularity.index.tolist()

def recommend_random(user_id: int, k: int = 10) -> list[int]:
    """
    Recommend k random movies from the entire catalog.
    No personalization, just random unseen items.
    """
    return np.random.choice(all_movies, size=k, replace=False).tolist()

def recommend_popularity(user_id: int, k: int = 10) -> list[int]:
    """
    Recommend k globally most popular movies that the user has not seen.
    """
    seen = set(train.loc[train.user_id == user_id, "movie_id"].values)
    recs = [m for m in popular_movies if m not in seen]
    return recs[:k]

def get_test_df() -> pd.DataFrame:
    """
    Expose the test DataFrame so evaluation utilities can reuse it.
    """
    return test.copy()
