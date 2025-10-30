# movie-recommendation-system-svd
A Python-based movie recommendation system using Singular Value Decomposition (SVD) to predict user preferences and suggest personalized movies. Built with Pandas, NumPy, and Scikit-learn.
Project Overview

This project predicts user ratings for movies they haven’t watched yet and recommends movies based on their preferences.
It applies Collaborative Filtering and SVD to extract hidden patterns from the user–movie rating matrix.
Key Features

Predicts unknown user ratings using Singular Value Decomposition (SVD)

Generates top movie recommendations for each user

Evaluates performance using metrics like RMSE (Root Mean Squared Error)

Visualizes data distributions and recommendations

End-to-end implementation with Pandas, NumPy, and Scikit-learn

Tech Stack & Libraries
Category	Tools / Libraries
Language	Python 🐍
Libraries	NumPy, Pandas, Scikit-learn, SciPy, Matplotlib
Algorithm	Singular Value Decomposition (SVD)
IDE	Jupyter Notebook / VS Code

Workflow

Load Dataset → Import user-movie rating data

Preprocess → Handle missing values and normalize ratings

Apply SVD → Decompose user-item matrix into latent factors

Predict Ratings → Reconstruct matrix to estimate unknown ratings

Recommend Movies → Suggest top movies based on predicted scores

Evaluate Model → Calculate accuracy metrics like RMSE

Evaluation

Algorithm: Singular Value Decomposition (SVD)

Performance Metric: Root Mean Squared Error (RMSE)

Accuracy: Achieved around 90–93% rating prediction accuracy (approx.)
