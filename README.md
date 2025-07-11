# Credit Card Fraud Detection 

## Overview
Credit card fraud detection is a crucial application of machine learning due to the massive volume of daily transactions and the significant financial loss caused by fraud. This project applies An end-to-end machine learning techniques to detect fraudulent credit card transactions using both **unsupervised anomaly detection** (Isolation Forest) and **supervised classifiers** (Random Forest & XGBoost). The project emphasizes real-world class imbalance handling, anomaly detection, model evaluation, and performance trade-offs.





## Dataset

- Source: Kaggle – Credit Card Fraud Detection
- Size: 284,807 transactions
- Features:
  - PCA-transformed features (`V1` to `V28`)
  - `Time`, `Amount` (transaction context)
  - `Class`: target variable( 0 = Legitimate, 1 = Fraud)
   
 
This dataset contains anonymized credit card transactions made by European cardholders in 2013.

**Note**: Fraud cases account for only **0.17%** of the data — creating a highly imbalanced classification challenge.


## Objectives

Build models that can accurately **identify fraudulent transactions** despite:
- Extreme **class imbalance**
- **Unclear fraud patterns** due to anonymized features
- **Very limited fraud examples** to learn from

## Tools & Technologies

- Python
- pandas, numpy
- scikit-learn
- matplotlib, seaborn

## Key Steps

###  1. Exploratory Data Analysis (EDA)
- Verified no missing values
- Detected severe class imbalance
- Scaled `Amount` and `Time` using `StandardScaler`

###  2. Models Implemented

#### **Isolation Forest (Unsupervised)**
- Detects anomalies without labels
- Tuned `contamination` to reflect true fraud rate
- Resulted in **low precision**, high false positives
- Best used for novelty detection — but struggled here due to subtle fraud patterns

####  **Random Forest (Supervised)**
- Balanced using SMOTE (Synthetic Minority Oversampling Technique)
- Achieved strong recall and accuracy
- Performance:
  - **Recall (Fraud)**: 0.78
  - **ROC-AUC**: 0.89
 
####  **XGBoost Classifier (Supervised)**
- Tuned with class weighting and SMOTE
- Outperformed all other models
- Performance:
  - **Precision (Fraud)**: 0.76
  - **Recall (Fraud)**: 0.80
  - **F1-Score (Fraud)**: 0.78
  - **ROC-AUC**: 0.97 ✅


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
  
