# import streamlit as st
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# # --- Page Config ---
# st.set_page_config(page_title="Fraud Detection", layout="wide")

# # --- CSS for Styling like your image ---
# st.markdown("""
# <style>
# body, .stApp {
#     background-color: #fdf6f0;
#     font-family: 'Segoe UI', sans-serif;
# }

# .sidebar .sidebar-content {
#     background-color: #111;
#     color: white;
# }

# h1, h2, h3 {
#     color: #1e1e1e;
# }

# .metric-box {
#     background-color: #f9f1ec;
#     padding: 1rem;
#     border-radius: 12px;
#     text-align: center;
#     box-shadow: 2px 2px 8px rgba(0,0,0,0.05);
#     margin-bottom: 1rem;
# }
# </style>
# """, unsafe_allow_html=True)

# # --- Sidebar ---
# st.sidebar.image("https://img.icons8.com/clouds/100/fraud.png", width=80)
# st.sidebar.title("💳 Fraud Detector")
# threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5, step=0.01)
# sample_size = st.sidebar.slider("Sample Size", 1000, 100000, 10000, step=1000)

# st.sidebar.markdown("---")
# st.sidebar.info("Upload credit card CSV with features: V1–V28, Time, Amount, Class")

#--- Main ---
# st.title("🧠 Real-Time Credit Card Fraud Detection")

# uploaded_file = st.file_uploader("📁 Upload Dataset", type=["csv"])

# if uploaded_file:
#     df = pd.read_csv(uploaded_file)

#     # Show data
#     st.subheader("📊 Dataset Preview")
#     st.dataframe(df.head(10))

#     # Downsample for speed
#     df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
#     X = df.drop("Class", axis=1)
#     y = df["Class"]

#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

#     # Model
#     model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
#     model.fit(X_train, y_train)

#     # Predict
#     probs = model.predict_proba(X_test)[:, 1]
#     preds = (probs > threshold).astype(int)

#     # --- Metrics ---
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.markdown('<div class="metric-box"><h3>Total Samples</h3><p>{}</p></div>'.format(len(X_test)), unsafe_allow_html=True)
#     with col2:
#         st.markdown('<div class="metric-box"><h3>Frauds Detected</h3><p>{}</p></div>'.format(sum(preds)), unsafe_allow_html=True)
#     with col3:
#         st.markdown('<div class="metric-box"><h3>ROC-AUC Score</h3><p>{:.3f}</p></div>'.format(roc_auc_score(y_test, probs)), unsafe_allow_html=True)

#     # --- Confusion Matrix ---
#     st.subheader("🧾 Confusion Matrix")
#     cm = confusion_matrix(y_test, preds)
#     fig, ax = plt.subplots()
#     sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
#     st.pyplot(fig)

#     # --- Classification Report ---
#     st.subheader("📋 Classification Report")
#     st.code(classification_report(y_test, preds), language='text')

#     # --- Download Results ---
#     result_df = X_test.copy()
#     result_df["Actual"] = y_test.values
#     result_df["Predicted"] = preds
#     result_df["Fraud Probability"] = probs
#     csv = result_df.to_csv(index=False).encode("utf-8")
#     st.download_button("📥 Download Prediction Results", csv, "fraud_predictions.csv", "text/csv")


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from datetime import datetime

# --- Page Config ---
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide", page_icon="💳")

# App title
st.title("🧠 Real-Time Credit Card Fraud Detection")
# --- Custom CSS Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .metric-box {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1.2rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        text-align: center;
        margin-bottom: 1rem;
    }
    .title-text {
        font-size: 2.5rem;
        font-weight: bold;
        color: #222;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar ---
st.sidebar.image("https://img.icons8.com/clouds/100/fraud.png", width=80)
st.sidebar.title("Information")
st.sidebar.markdown("### 👤 Profile")
st.sidebar.markdown("**Name:** Chandrika Joshi  \n**Social ID:** [Linkedln](https://www.linkedin.com/in/chandrika-j-0b1a98238?utm_source=share&utm_campaign=share_via&utm_content=profile&utm_medium=android_app) " "\n[GitHub](https://github.com/Chandrikajoshi123)" )

# Date picker
date_selected = st.sidebar.date_input("🗓️ Select Review Date", datetime.now())

# Controls
threshold = st.sidebar.slider("Fraud Threshold", 0.0, 1.0, 0.5, step=0.01)
sample_size = st.sidebar.slider("Sample Size", 1000, 100000, 10000, step=1000)

st.sidebar.markdown("---")
st.sidebar.info("Upload a file to start")

# --- Tabs ---
tab1, tab2 = st.tabs(["📁 Upload & Train", "📊 Results & Metrics"])

# --- Tab 1: Upload & Train ---
with tab1:
    st.markdown('<div class="title-text">📁 Upload Dataset & Train Model</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload a CSV file with `V1` to `V28`, `Time`, `Amount`, and `Class` columns", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success(f"Uploaded {uploaded_file.name} — {len(df)} rows")
        st.dataframe(df.head())

        if "Class" not in df.columns:
            st.error("❌ Dataset must include `Class` column.")
        else:
            # Downsample
            df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
            X = df.drop("Class", axis=1)
            y = df["Class"]

            # Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

            # Train model
            model = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
            model.fit(X_train, y_train)

            st.session_state["model"] = model
            st.session_state["X_test"] = X_test
            st.session_state["y_test"] = y_test

            st.success("✅ Model trained! Go to 'Results & Metrics' tab.")

# --- Tab 2: Results & Metrics ---
with tab2:
    st.markdown('<div class="title-text">📊 Model Performance Metrics</div>', unsafe_allow_html=True)

    if "model" in st.session_state:
        model = st.session_state["model"]
        X_test = st.session_state["X_test"]
        y_test = st.session_state["y_test"]

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > threshold).astype(int)

        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds, output_dict=True)
        roc_score = roc_auc_score(y_test, probs)

        # --- Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-box"><h4>Total Records</h4><p>{len(X_test)}</p></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-box"><h4>Frauds Detected</h4><p>{sum(preds)}</p></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-box"><h4>ROC-AUC Score</h4><p>{roc_score:.3f}</p></div>', unsafe_allow_html=True)

        # --- Confusion Matrix ---
        st.subheader("🧾 Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", ax=ax)
        st.pyplot(fig)

        # --- Classification Report ---
        st.subheader("📋 Classification Report")
        st.dataframe(pd.DataFrame(report).transpose())

        # --- Download Prediction Data ---
        results_df = X_test.copy()
        results_df["Actual"] = y_test.values
        results_df["Predicted"] = preds
        results_df["Fraud Probability"] = probs
        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button("📥 Download Prediction Results", csv, "fraud_predictions.csv", "text/csv")

    else:
        st.warning("Please train a model in the first tab.")
