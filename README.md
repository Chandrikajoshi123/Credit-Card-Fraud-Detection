# Credit Card Fraud Detection using Isolation Forest
## Overview
Credit card fraud detection is a crucial application of machine learning due to the massive volume of daily transactions and the significant financial loss caused by fraud. This project applies an unsupervised learning technique — Isolation Forest — to detect fraudulent transactions from anonymized credit card data.
Despite the high class imbalance (fraud cases make up just 0.17%), the model achieves a strong performance with ROC-AUC ≈ 0.96, demonstrating its ability to flag anomalies effectively without supervision.

![image](https://github.com/user-attachments/assets/8a698bb5-d288-4b9c-9626-1bdd3fc53db1)


- 🔗 Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud
- 🧠 Task: Detect fraudulent transactions (unsupervised anomaly detection)
- ⚙ Tools: Python, Scikit-learn, IsolationForest


Dataset

- Source: Kaggle – Credit Card Fraud Detection
- Size: 284,807 transactions
- Features:
  - PCA-transformed features (V1 to V28)
  - Time and Amount (scaled)
  - Class: 0 = Legitimate, 1 = Fraud

## 📂 Structure:
fraud_detection/<br>
├── fraud_detect.py<br>
├── creditcard.csv<br>
