AI-Powered Intelligent Insurance Risk Assessment and Customer Insights System

Project Overview

This project builds an end-to-end AI-powered insurance system that combines machine learning, deep learning, and NLP techniques to provide:

  1. Risk Score Prediction - Identify high-risk customers
  2. Claim Amount Estimation – Improve claim estimation accuracy
  3. Fraud Detection - Prevent fraudulent payouts
  4. Anomaly Detection – Spot unusual claim patterns
  5. Customer Segmentation (Clustering) - Enable personalized pricing & offers
  6. Sentiment Analysis - Understand customer feedback & emotions
  7. Multilingual Transformers – Break language barriers
  8. Chatbot Integration – Provide 24/7 customer support
  9. Interactive Dashboards & Visualizations 
It is deployed using Streamlit for interactive exploration and decision support.

Tech Stack
Languages: Python
Libraries: scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, NLTK, Seaborn, Matplotlib, WordCloud
Deployment: Streamlit
Data Handling: Pandas, NumPy
Environment: Jupyter Notebook / Google Colab

Dataset Overview
Customer Demographics (age, gender, income, location)
Policy Details (type, premium, upgrades)
Claim History (frequency, amount, fraudulent vs. genuine)
Risk Score Indicators
Sentiment Labels (positive, neutral, negative)

Features
Risk Score Prediction – Classifies customers into Low, Medium, High risk
Claim Amount Prediction – Supports ML, DL & Hybrid models
Fraud Detection – Detects fraudulent claims with anomaly detection & ML models
Customer Segmentation – Groups customers into actionable segments for business insights
Sentiment Analysis – Analyzes customer reviews as Positive (green), Neutral (blue), Negative (red)
Visualization Dashboard – Includes WordClouds, distributions, and comparative plots

Methodology
Data Collection – Gathered insurance policy & claim data
Data Preprocessing – Missing value handling, encoding, scaling
Feature Engineering – Correlation analysis, PCA for dimensionality reduction
Balancing Dataset – SMOTE, oversampling & undersampling
Modeling – ML, DL & Hugging Face Transformers
Deployment – Interactive Streamlit app + chatbot

Models Implemented
Classical/Regression ML: Logistic Regression, Decision Tree, Random Forest, SVM, Naive Bayes, Gradient Boosting, XGBoost, LightGBM, KNN, Linear Regression, Lasso, Ridge, ElasticNet 
Clustering: K-Means, DBSCAN, Isolation Forest
Deep Learning: Neural Networks
Transformers: Hugging Face for multilingual sentiment analysis

Performance Metrics
Accuracy, Precision, Recall, F1-score
Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), R²
Cross-validation & Train-Test Split for generalization
Clustering evaluation: Elbow Method (SSE) & Silhouette Score

