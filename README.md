# movielens-mini-recsys
A lightweight MovieLens-based recommendation system with a scalable architecture discussion.

## 📊 EDA Summary (MovieLens 100K)

- ~943 users, ~1682 movies, ~100k interactions  
- Ratings distribution is skewed toward 4 and 5  
- Movie popularity follows a long-tail distribution  
- The user–item matrix is ~94% sparse  

These insights justify using **matrix factorization** as the next modeling step.
