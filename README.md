# 🎬 Movie-Recommendation-System-Using-SVD  

**SVD-based Movie Recommendation System** for predicting user preferences and generating personalized movie suggestions.

---

## 🧩 Introduction  

This project implements a **Movie Recommendation System** using **Singular Value Decomposition (SVD)**, a powerful matrix-factorization technique widely used in collaborative filtering.  
The system predicts how a user would rate a movie based on their historical ratings and the behavior of similar users.  

Such models are commonly used by platforms like **Netflix, Amazon Prime, and Spotify** to recommend content based on user interests.

---

## 📊 Dataset Overview  

The dataset used in this project contains user IDs, movie IDs, and their corresponding ratings.  

**Example structure:**

| userId | movieId | rating |
|--------|----------|--------|
| 1 | 31 | 4.0 |
| 1 | 1029 | 5.0 |
| 2 | 1061 | 3.5 |
| ... | ... | ... |

**Source:** [MovieLens Dataset – Kaggle](https://www.kaggle.com/datasets/grouplens/movielens-100k-dataset) *(or replace with your dataset source)*  

**Features:**
- `userId` — Unique identifier for each user  
- `movieId` — Unique identifier for each movie  
- `rating` — User’s rating for the movie (e.g., 1–5)  

---

## 🧹 Data Preprocessing  

Steps performed before modeling:
1. **Load and clean** the dataset  
2. **Pivot the data** into a user–item matrix  
3. **Handle missing values** (fill with 0 or mean rating)  
4. **Normalize ratings** for consistent scale  

---

## 🧠 Model Training  

The project uses **Singular Value Decomposition (SVD)** for collaborative filtering.  

### Steps:
1. Construct the **user–item rating matrix**  
2. Apply **SVD decomposition** to capture latent features  
3. Reconstruct the matrix to predict missing ratings  
4. Recommend top-N movies with the highest predicted scores  

---

### Example Code Snippet  

```python
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import svds
import numpy as np

# Compute SVD
U, sigma, Vt = svds(ratings_matrix, k=50)
sigma = np.diag(sigma)

# Predict ratings
predicted_ratings = np.dot(np.dot(U, sigma), Vt)

# Example evaluation
rmse = np.sqrt(mean_squared_error(actual_ratings, predicted_ratings))
print("RMSE:", rmse)
```

##Model Evaluation

Metric Used:

RMSE (Root Mean Squared Error) — measures how accurately the model predicts unseen ratings.

Example output:

RMSE: 0.87


The lower the RMSE, the better the model performance.

---


##Example Recommendation Output

Top 5 Recommended Movies for User 25:
1. The Shawshank Redemption (1994)
2. The Dark Knight (2008)
3. Inception (2010)
4. Interstellar (2014)
5. The Matrix (1999)

---


##Possible Extensions

Integrate with Flask or Streamlit to build a web app

Add content-based filtering using genres and movie metadata

Combine SVD with neural networks for hybrid recommendation

Use real-time recommendations with APIs

---

##Prerequisites

Python 3.x

Libraries:

numpy

pandas

scikit-learn

scipy

matplotlib

---

##References

MovieLens Datasets – Kaggle

Scikit-learn Documentation

SVD in Recommender Systems – Medium Article

---

##About

Movie Recommendation System using SVD — a machine learning project to predict user ratings and provide intelligent movie recommendations based on collaborative filtering.
