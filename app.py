import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef,
                             confusion_matrix, classification_report)

TARGET = 'is_canceled'
MODEL_FILES = {
    'Logistic Regression': 'model/logistic_regression.joblib',
    'Decision Tree': 'model/decision_tree.joblib',
    'KNN': 'model/knn.joblib',
    'Naive Bayes': 'model/naive_bayes.joblib',
    'SVM': 'model/svm.joblib',
}

st.set_page_config(page_title="Hotel Booking Cancellation", layout="wide")
st.title("Hotel Booking Cancellation Prediction")
st.write("Upload the **test CSV** (features + `is_canceled` column), pick a model, and view the metrics.")


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def get_metrics(model, X, y):
    y_pred = model.predict(X)
    try:
        auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    except Exception:
        auc = float('nan')
    return {
        'Accuracy': accuracy_score(y, y_pred),
        'AUC': auc,
        'Precision': precision_score(y, y_pred, zero_division=0),
        'Recall': recall_score(y, y_pred, zero_division=0),
        'F1': f1_score(y, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y, y_pred),
    }, y_pred


uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None:
    st.info("Please upload test_data.csv to continue.")
    st.stop()

df = pd.read_csv(uploaded)
st.subheader("Data preview")
st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
st.dataframe(df.head())

if TARGET not in df.columns:
    st.error(f"The CSV must contain the target column '{TARGET}'.")
    st.stop()

y = df[TARGET].astype(int)

model_name = st.selectbox("Select a model", list(MODEL_FILES.keys()))
model = load_model(MODEL_FILES[model_name])

feature_cols = list(model.feature_names_in_)
missing = [c for c in feature_cols if c not in df.columns]
if missing:
    st.error(f"Missing required feature columns: {missing}")
    st.stop()
X = df[feature_cols]

metrics, y_pred = get_metrics(model, X, y)

st.subheader(f"Evaluation metrics: {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, metrics.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.subheader("Confusion matrix")
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Not Canceled', 'Canceled'],
                yticklabels=['Not Canceled', 'Canceled'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.subheader("Classification report")
    report = classification_report(y, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.round(4))

st.subheader("Comparison of all models on this data")
rows = []
for name, path in MODEL_FILES.items():
    m = load_model(path)
    mt, _ = get_metrics(m, df[list(m.feature_names_in_)], y)
    rows.append({'ML Model Name': name, **mt})
st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True)
