# Mini Video Recommendation System (MovieLens)

## ⭐ Project Summary

This project implements a compact, end-to-end video recommendation system inspired by the retrieval-and-ranking architectures used in modern platforms such as YouTube. Using the MovieLens-100K dataset as a proxy for video interactions, the system is structured to resemble a realistic production pipeline while remaining fully runnable on a laptop.

The pipeline consists of:

- **Candidate Generation** using TruncatedSVD to quickly retrieve potentially relevant items.
- **Ranking** using a LightGBM model trained on user, item, and interaction features.
- **Evaluation** with metrics such as Hit@K, Recall@K, and NDCG@K.
- **A/B-style policy simulation**, demonstrating how different recommendation policies perform.
- **Feedback loop simulation** to illustrate how models can be refreshed as new data arrives.
- **Lightweight monitoring and logging** for traceability and inspection.

Although the project runs on a laptop with a small dataset, the structure, metrics, and components are designed to mirror patterns used in production systems described in:

- *Problem Statement and Metrics*  
- *Estimated Delivery Model*  
- *Video Recommendation System Design*

---

## 1. Problem Statement

Users interact with a large catalog of videos over time. The system should:

- Provide **personalized video recommendations** for each user.
- Adapt to evolving preferences and changing catalog.
- Optimize for **relevance** and **engagement**, while maintaining diversity and coverage.

In this project, MovieLens movies are treated as a proxy for videos. Each (user, movie, rating, timestamp) is interpreted as an implicit signal of interest. The system is responsible for ranking items that a user is likely to engage with next.

---

## 2. Dataset

**MovieLens 100K** is used as a stand‑in for a video interaction log:

- ~943 users
- ~1,682 movies
- 100,000 ratings with timestamps

The data is preprocessed into:

- A **ratings table** with user, movie, rating, timestamp.
- A **movies table** with metadata (title, release date, genres).

A per‑user temporal split is used:

- For each user, the **latest interaction** (by timestamp) is held out as test.
- All earlier interactions form the training set.

This split mimics a realistic “predict the next watched video” scenario.

---

## 3. System Overview

The project implements a simplified, two‑stage recommendation pipeline:

1. **Candidate Generation (Retrieval)**  
   - Uses **matrix factorization** (TruncatedSVD) on the user–item interaction matrix.
   - Produces a shortlist of candidate movies for each user.

2. **Ranking**  
   - Uses a **gradient boosting model (LightGBM)** trained on rich user, item, and interaction features.
   - Scores the candidate set and returns the final Top‑K recommendations.

In addition, the project includes:

- Offline evaluation (Hit@K, Recall@K, NDCG@K, coverage).
- A simple A/B‑style policy comparison.
- A simulated feedback loop and model refresh.
- Lightweight logging and monitoring hooks.

---
## 4. Architecture

The system is organized as a two-stage recommendation pipeline with a training and feedback loop:

```mermaid
flowchart LR
  U[User] --> REQ[Recommendation Request]

  subgraph RecSysEngine[Recommendation Engine]
    REQ --> CG[Candidate Generation (SVD)]
    CG --> RANK[Ranking (LightGBM)]
    RANK --> TOPK[Top-K Recommendations]
  end

  subgraph Logging[Logging & Feedback]
    TOPK --> LOG[Interaction Logs]
  end

  subgraph Training[Offline Training Pipeline]
    LOG --> FEAT[Feature Engineering]
    FEAT --> SVDM[Train SVD Model]
    FEAT --> LGBM[Train LightGBM Ranker]
    SVDM --> CG
    LGBM --> RANK
  end

  TOPK --> U

---

## 5. Modeling Approach

### 5.1 Candidate Generation – TruncatedSVD

The candidate generation stage uses **TruncatedSVD** to perform low‑rank factorization of the user–item interaction matrix.

Steps:

1. Build a sparse matrix `M` where rows are users, columns are movies, and values are ratings.
2. Apply TruncatedSVD with `k` latent dimensions to obtain:
   - User latent vectors \( U \in \mathbb{R}^{n_{\text{users}} \times k} \)
   - Item latent vectors \( V \in \mathbb{R}^{n_{\text{items}} \times k} \)
3. For a given user, compute a relevance score for each item via a dot product:
   \[
   s_{u,i} = U_u \cdot V_i
   \]
4. Filter out movies the user has already interacted with, then select the Top‑N as candidates.

This stage is used as:

- A **fast approximation of relevance**.
- A realistic stand‑in for embedding‑based retrieval in large systems.

### 5.2 Ranking – LightGBM

The ranking stage re‑orders candidate movies using a **LightGBM classifier** trained on per‑(user, item) feature vectors.

Example features:

- **User features**
  - Total number of ratings given.
  - Mean rating.
  - Rating variance or standard deviation.
- **Item features**
  - Popularity (number of ratings received).
  - Mean rating.
  - Genre indicators (one‑hot or multi‑hot).
- **Interaction features**
  - SVD score (latent relevance from candidate generator).
  - Optional: popularity ranks, recency indicators, etc.

The label is derived from ratings, e.g.:

- `label = 1` if rating ≥ 4
- `label = 0` otherwise

A binary LightGBM model is trained to estimate \( P(\text{engaged} \mid \text{user}, \text{item}, \text{features}) \). At recommendation time:

1. SVD provides a candidate set for a user.
2. Features are computed for each (user, candidate) pair.
3. LightGBM scores each candidate.
4. Candidates are sorted by score and the Top‑K are returned.

This mirrors common production patterns where candidate generation is relatively simple and fast, and most business logic and context is encoded in the ranking model.

---

## 6. Metrics

The project focuses on **offline metrics** that approximate relevance and ranking quality, and includes hooks and structure that correspond to **online metrics** used in production systems.

### 6.1 Offline Metrics (Implemented)

For each user in the test set (last interaction):

- **Hit@K**  
  Indicates whether the held‑out item appears in the Top‑K recommendations.

- **Recall@K**  
  Measures what fraction of relevant items are captured in the Top‑K list (here, usually a single held‑out item).

- **NDCG@K**  
  Discounted cumulative gain normalized by the ideal ranking, capturing both correctness and position of relevant items.

- **Coverage**  
  Fraction of items in the catalog that are ever recommended; gives a sense of how much of the catalog is being explored.

These metrics are computed for multiple policies:

- Random baseline
- Popularity baseline
- SVD‑only recommender
- Two‑stage SVD + LightGBM recommender

### 6.2 Online Metrics (Conceptual)

While no live system is deployed here, the design aligns with the metrics described in the supporting design documents, such as:

- Click‑through rate (CTR)
- Watch‑time or session duration (MovieLens rating can act as a very rough proxy)
- Long‑term engagement and return rate
- Diversity and novelty of recommendations

These concepts are documented to show how this codebase could be extended into a live system with real interaction data.

---

## 7. Evaluation and A/B Simulation

### 7.1 Offline Comparison of Policies

The evaluation module compares multiple recommendation strategies using the offline metrics above:

- Random
- Popularity
- SVD (matrix factorization)
- Two‑stage SVD + LightGBM

Typical behavior:

- Random provides a lower bound.
- Popularity captures global trends but ignores personalization.
- SVD improves personalization by modeling user–item affinities.
- Two‑stage SVD + LightGBM uses more signals and often performs best in ranking quality.

### 7.2 A/B‑Style Simulation

To mimic an A/B experiment:

1. Define two policies, e.g.:
   - Policy A: SVD‑only
   - Policy B: SVD + LightGBM
2. For each user in the test set, randomly assign them to A or B.
3. Serve recommendations from the assigned policy.
4. Check whether the held‑out item appears in the recommended list (a proxy for a “click”).
5. Aggregate metrics per policy (e.g., Hit@K as a stand‑in for CTR).

This doesn’t replace a real online A/B test but demonstrates how evaluation and experiment framing would work once integrated into a live product.

---

## 8. Feedback Loop and Model Refresh

The project includes a simulated feedback loop inspired by the delivery model in the supporting documents:

- Additional “days” of interactions can be generated or sampled to represent new user behavior.
- The LightGBM ranker can be retrained on:
  - Original data + new interactions.
  - Alternative sampling strategies (e.g., more recent data weighted more heavily).

This exercise illustrates:

- The importance of refreshing models as behavior and catalog shift.
- How a pipeline might be scheduled to retrain daily or weekly.
- How offline metrics can be tracked over successive training runs.

In a production setting, this retraining would be orchestrated via a workflow engine (e.g., Airflow, Kubeflow) with training jobs triggered by data availability and monitoring signals.

---

## 9. Monitoring and Logging

To mirror monitoring patterns in real systems, lightweight logging is added around the recommendation calls. The log entries can include:

- `user_id`
- candidate list from SVD
- final ranked list from LightGBM
- model scores
- timestamp

With these logs, one can:

- Inspect recommendation behavior for specific users.
- Analyze score distributions.
- Detect anomalies or drift in the model’s behavior over time.

In a larger system, this logging would feed:

- Data quality checks.
- Feature drift detectors.
- Dashboards tracking key metrics over time.
- Alerting systems when performance degrades.

---

## 10. Project Structure

A typical layout for this repository is:

```text
movielens-mini-recsys/
│
├── recommender/
│   ├── baselines.py        # Random and popularity recommenders
│   ├── data.py             # Data loading utilities
│   ├── svd_model.py        # Matrix factorization (TruncatedSVD) candidate generator
│   ├── features.py         # Feature construction for ranking
│   ├── ranker.py           # LightGBM ranking model
│   ├── eval.py             # Evaluation metrics and utilities
│   └── recommend.py        # Public recommend(user_id, model=..., k=...) interface
│
├── notebooks/
│   ├── 01_eda.ipynb                    # Dataset exploration and basic stats
│   ├── 02_baselines.ipynb              # Baseline models and Hit@K
│   ├── 03_svd.ipynb                    # TruncatedSVD candidate generation
│   ├── 04_demo.ipynb                   # Simple recommendation demos
│   ├── 05_build_ranking_dataset.ipynb  # Ranking dataset creation
│   ├── 06_train_lgbm_ranker.ipynb      # LightGBM training and validation
│   ├── 07_two_stage_demo.ipynb         # Two-stage pipeline demonstration
│   ├── 08_ab_simulation.ipynb          # A/B-style policy simulation
│   └── 09_feedback_loop.ipynb          # Simulated model refresh and analysis
│
├── data/
│   ├── raw/                            # Original MovieLens files (ignored by git)
│   └── processed/
│       ├── ratings_joined.parquet
│       └── ranking_dataset.parquet
│
├── models/                             # Saved models (ignored by git)
│   ├── svd_factors.pkl
│   └── lgbm_ranker.pkl
│
└── README.md
```

Exact file names may differ slightly from this sketch, but the intent is to keep the code modular, testable, and easy to extend.

---

## 11. How to Run

1. **Clone the repository**

```bash
git clone https://github.com/ddronamraju/movielens-mini-recsys.git
cd movielens-mini-recsys
```

2. **Create and activate a virtual environment** (example using `venv`):

```bash
python -m venv .venv
source .venv/bin/activate       # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download MovieLens 100K** into `data/raw/` (if not already present) and run the preprocessing / EDA notebook(s) as needed.

5. **Run notebooks in order** to reproduce the full pipeline, or import the `recommender` package in your own scripts, for example:

```python
from recommender.recommend import recommend

user_id = 42
recommendations = recommend(user_id=user_id, model="two_stage", k=10)
print(recommendations)
```

---

## 12. Possible Extensions

The current implementation is intentionally modest, but it is structured so that several extensions are straightforward:

- **Neural ranking models**  
  Replace LightGBM with a neural model that consumes user and item embeddings together with contextual features.

- **ANN‑based candidate generation**  
  Export item embeddings and use FAISS / ScaNN / Milvus to perform approximate nearest neighbor search at scale.

- **Richer context features**  
  Add time‑of‑day, device, session statistics, content freshness, or geography as features to the ranker.

- **Feature store integration**  
  Separate the computation and serving of features into a feature store to ensure online/offline consistency.

- **Real serving layer**  
  Wrap the `recommend()` function in a FastAPI (or similar) microservice and integrate logging into astream processor.

This project is meant to serve as a compact yet realistic reference implementation of a modern video recommendation pipeline, with enough structure to be extended toward production scenarios.