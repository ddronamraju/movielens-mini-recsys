# Scalable Recommendation System
A lightweight MovieLens-based recommendation system with a scalable architecture discussion.

## 📊 EDA Summary (MovieLens 100K)

- ~943 users, ~1682 movies, ~100k interactions  
- Ratings distribution is skewed toward 4 and 5  
- Movie popularity follows a long-tail distribution  
- The user–item matrix is ~94% sparse  

These insights justify using **matrix factorization** as the next modeling step.

## 🧪 Baseline Models

I implemented two baseline recommenders:

### 1. Random Recommender
- Recommends `k` random unseen movies.
- Hit@10 ≈ 0.01  
- Serves as a sanity check.

### 2. Global Popularity Recommender
- Recommends globally most-rated movies the user has not seen.
- Hit@10 ≈ 0.06–0.09  
- Strong baseline for MF uplift comparison.

These create a measurable baseline to improve upon with matrix factorization.

## 🔢 Matrix Factorization (TruncatedSVD)

To move beyond simple baselines (random and popularity), I implemented a lightweight
**collaborative filtering model** using a **low-rank factorization** of the user–item matrix
via **TruncatedSVD**. This provides compact latent representations of users and movies and enables
personalized Top-N recommendations.

### ⭐ What was done in Hour 4
- Constructed a **user–item sparse matrix** from the MovieLens ratings.
- Applied **TruncatedSVD** (rank-k factorization) to derive:
  - **User latent factors** (U ∈ ℝᵘˢᵉʳˢ×ᵏ)
  - **Item latent factors** (V ∈ ℝᶦᵗᵉᵐˢ×ᵏ)
- Implemented `recommend_svd(user_id, k)` which:
  - Scores all movies via a dot product between user factors and item factors
  - Filters out movies the user has already seen
  - Returns the Top-K highest-scoring recommendations
- Evaluated model quality using **Hit@10** with a per-user holdout split
  (each user’s most recent interaction is the test item).

### 📈 Why TruncatedSVD?
- Works on *any* OS (Windows, Mac, Linux)
- No C++ compiler required (unlike some ALS implementations)
- Very fast on MovieLens 100k
- Still provides meaningful uplift over the popularity baseline
- Perfect for a **10-hour interview-ready project**

### 📊 Key Results (Hit@10)
| Model                   | Hit@10 (approx.) |
|-------------------------|------------------|
| Random                  | ~0.01            |
| Popularity Baseline     | ~0.06–0.09       |
| **SVD Matrix Factorization** | **~0.12–0.20**     |

The SVD-based MF model shows a clear improvement over simpler baselines, validating
that latent factor models capture meaningful user–movie preference patterns.

### 🧠 Takeaway
This SVD model serves as a **lightweight candidate generation method**, similar to the
first stage of real-world recommendation systems. Later steps in this project build on
this foundation by integrating the model into a modular codebase and creating a unified
`recommend()` interface for clean experimentation and system-design discussions.
