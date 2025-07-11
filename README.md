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
    
### 3. Hybrid Model
- Used **Isolation Forest anomaly scores as input features** to XGBoost
- Helped slightly improve recall
- Showcases a model-stacking approach to fraud detection


## Model Comparison

| Model              | Precision (Fraud) | Recall (Fraud) | F1-score | ROC-AUC |
|-------------------|-------------------|----------------|----------|---------|
| Isolation Forest   | 0.07–0.30         | 0.29–0.71      | Low      | 0.05–0.65 |
| Random Forest      | 0.89              | 0.78           | 0.83     | 0.89    |
| **XGBoost**        | **0.76**          | **0.80**       | **0.78** | **0.97** ✅ |

## Future Improvements

- Integrate time-series modeling (LSTM, RNN) to capture sequential fraud patterns.
- Deploy model using Streamlit or Flask as a demo dashboard.
- Experiment with autoencoders for deeper anomaly detection.
----
Made with ❤️ using:
<p align="center"> <img src="https://streamlit.io/images/brand/streamlit-logo-primary-colormark-darktext.svg" alt="Streamlit" height="40"/> </p>

----
## App Link: [🔗](https://fraudetecti.streamlit.app/)
