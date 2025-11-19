import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from recommender.data import load_processed_ratings

# Load data
df = load_processed_ratings()
df = df.sort_values("timestamp")

# Same train/test split logic as baselines
test = df.groupby("user_id").tail(1)
train = df.drop(test.index)

# Map IDs to indices
unique_users = train["user_id"].unique()
unique_items = train["movie_id"].unique()

user_id_to_idx = {u: i for i, u in enumerate(unique_users)}
item_id_to_idx = {m: i for i, m in enumerate(unique_items)}
idx_to_item_id = {i: m for m, i in item_id_to_idx.items()}

n_users = len(unique_users)
n_items = len(unique_items)

# Build sparse user–item matrix
row_idx = train["user_id"].map(user_id_to_idx)
col_idx = train["movie_id"].map(item_id_to_idx)
vals = train["rating"].astype(float)

user_item = csr_matrix(
    (vals, (row_idx, col_idx)),
    shape=(n_users, n_items)
)

# Train SVD model (you can tweak n_components)
svd = TruncatedSVD(n_components=50, random_state=42)
user_factors = svd.fit_transform(user_item)   # shape: (n_users, k)
item_factors = svd.components_.T              # shape: (n_items, k)

def recommend_svd(user_id: int, k: int = 10) -> list[int]:
    """
    Recommend top-k movies for a user using SVD-based MF.
    """
    if user_id not in user_id_to_idx:
        # Cold-start fallback: return empty, or you could fall back to popularity
        return []

    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx]  # shape: (k,)

    # Score all items
    scores = item_factors @ u_vec  # (n_items,)

    # Mask already-seen items
    seen = set(train.loc[train.user_id == user_id, "movie_id"].map(item_id_to_idx))
    scores = scores.copy()
    for i in seen:
        scores[i] = -np.inf

    # Get top-k indices
    top_idx = np.argpartition(scores, -k)[-k:]
    top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

    return [idx_to_item_id[i] for i in top_idx]

def get_test_df() -> pd.DataFrame:
    """
    Expose test DataFrame (same split used in MF training).
    """
    return test.copy()

def get_train_test():
    """
    Return copies of the train and test DataFrames used for MF.
    """
    return train.copy(), test.copy()
