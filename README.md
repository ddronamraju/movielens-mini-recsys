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
