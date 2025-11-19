import pandas as pd
import os

def load_processed_ratings() -> pd.DataFrame:
    """
    Load the processed MovieLens ratings joined with movie metadata.

    Returns a DataFrame with at least:
    - user_id
    - movie_id
    - rating
    - timestamp
    - title (and possibly genre columns)
    """
     # directory of THIS file (recommender/data.py)
    base_dir = os.path.dirname(__file__)

    # go up one level to project root, then into data/processed/
    file_path = os.path.join(base_dir, "..", "data", "processed", "ratings_joined.parquet")
    file_path = os.path.abspath(file_path)
    return pd.read_parquet(file_path)
