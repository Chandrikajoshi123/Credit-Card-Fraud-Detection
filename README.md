# Credit Card Fraud Detection using Isolation Forest

## Overview
Credit card fraud detection is a crucial application of machine learning due to the massive volume of daily transactions and the significant financial loss caused by fraud. This project applies An end-to-end machine learning techniques to detect fraudulent credit card transactions using both **unsupervised anomaly detection** (Isolation Forest) and **supervised classifiers** (Random Forest & XGBoost). The project emphasizes real-world class imbalance handling, anomaly detection, model evaluation, and performance trade-offs.



![Cartoon thief stock vector_ Illustration of escape, bank - 31717335](https://github.com/user-attachments/assets/49bf28f9-abe8-4c6a-9535-8625335c88c3)



## Dataset

- Source: Kaggle – Credit Card Fraud Detection
- Size: 284,807 transactions
- Features:
  - PCA-transformed features (`V1` to `V28`)
  - `Time`, `Amount` (transaction context)
  - `Class`: target variable( 0 = Legitimate, 1 = Fraud)
   
 
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
  
