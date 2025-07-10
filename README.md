# Credit Card Fraud Detection using Isolation Forest

## Overview
Credit card fraud detection is a crucial application of machine learning due to the massive volume of daily transactions and the significant financial loss caused by fraud. This project applies an unsupervised learning technique — Isolation Forest — to detect fraudulent transactions from anonymized credit card data.
Despite the high class imbalance (fraud cases make up just 0.17%), the model achieves a strong performance with ROC-AUC ≈ 0.96, demonstrating its ability to flag anomalies effectively without supervision.

![image](https://github.com/user-attachments/assets/8a698bb5-d288-4b9c-9626-1bdd3fc53db1)



## Dataset

- Source: Kaggle – Credit Card Fraud Detection
- Size: 284,807 transactions
- Features:
  - PCA-transformed features (V1 to V28)
  - Time and Amount (scaled)
  - Class: 0 = Legitimate, 1 = Fraud
    
This dataset contains anonymized credit card transactions made by European cardholders in 2013.

## Objectives

- Handle highly imbalanced classification using unsupervised anomaly detection.
- Use Isolation Forest to learn the pattern of normal transactions and flag potential outliers.
- Evaluate model performance using ROC-AUC, confusion matrix, and precision-recall metrics.

## Tools & Technologies

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Key Steps

### 1. Data Exploration & Preprocessing
- Visualized class imbalance and feature distributions
- Scaled Amount and Time using StandardScaler
- Verified dataset had no missing values

### 2. Model Training
- Used IsolationForest from scikit-learn with:
- contamination ≈ fraud ratio (0.0017)
- n_estimators = 100, max_samples = 'auto'
- Predicted anomalies and mapped them to fraud labels

## Results

| Metric |	Value |
| ------ | ------ |
| ROC-AUC	  |~0.96  |
|Fraud Ratio|	0.17% |
| Algorithm Used	| Isolation Forest (unsupervised)|
|Model Type	| Anomaly Detection|

## Visualizations:

- Class distribution plot
- Correlation heatmap
- ROC-AUC curve
- Confusion matrix

## Why Isolation Forest?

- Efficient on large datasets
- Handles high-dimensional data well
- No need for labeled training data (ideal for rare-event detection)
- Based on random partitioning of data — outliers require fewer splits to isolate

## Future Improvements

- Compare with other anomaly detectors (One-Class SVM, Autoencoders)
- Use feature engineering on Time (e.g., transaction hour)
- Build a simple Flask or Streamlit dashboard for real-time scoring
  
