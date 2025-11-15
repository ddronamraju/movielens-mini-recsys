import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/ratings_joined.parquet")
df = df.sort_values("timestamp")

test = df.groupby("user_id").tail(1)
train = df.drop(test.index)

all_movies = df["movie_id"].unique()
popularity = train.groupby("movie_id").size().sort_values(ascending=False)
popular_movies = popularity.index.tolist()

def recommend_random(user_id, k=10):
    return np.random.choice(all_movies, size=k, replace=False).tolist()

def recommend_popularity(user_id, k=10):
    watched = set(train[train.user_id == user_id].movie_id.values)
    recs = [m for m in popular_movies if m not in watched]
    return recs[:k]
