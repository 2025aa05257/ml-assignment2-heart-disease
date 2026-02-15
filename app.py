# ============================================
# Streamlit App - Heart Disease Classification
# Machine Learning Assignment 2
# ============================================

import streamlit as st
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier


# ============================================
# Streamlit Page Title
# ============================================

st.set_page_config(page_title="Heart Disease Prediction", layout="wide")

st.title("❤️ Heart Disease Classification Models")
st.markdown("### Machine Learning Assignment 2 - Streamlit Deployment")


# ============================================
# Upload Dataset
# ============================================

uploaded_file = st.file_uploader(
    "Upload Heart Disease Dataset (CSV format)",
    type=["csv"]
)

if uploaded_file is not None:

    # Load dataset
    df = pd.read_csv(uploaded_file)

    st.success("✅ Dataset Uploaded Successfully!")
    st.write("Preview of Dataset:")
    st.dataframe(df.head())

    # ============================================
    # Step 1: Convert Target Column
    # ============================================

    if "num" not in df.columns:
        st.error("❌ Target column 'num' not found in dataset!")
        st.stop()

    # Convert multi-class num → binary
    df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    # ============================================
    # Step 2: Handle Missing Values
    # ============================================

    categorical_cols = df.select_dtypes(include=["object"]).columns
    numeric_cols = df.select_dtypes(exclude=["object"]).columns

    # Impute numeric columns with mean
    num_imputer = SimpleImputer(strategy="mean")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Impute categorical columns with most frequent
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # ============================================
    # Step 3: One-Hot Encode Categorical Columns
    # ============================================

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and Target
    X = df.drop("num", axis=1)
    y = df["num"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ============================================
    # Model Selection Dropdown
    # ============================================

    st.subheader("📌 Select a Machine Learning Model")

    model_choice = st.selectbox(
        "Choose Classification Model:",
        [
            "Logistic Regression",
            "Decision Tree",
            "KNN",
            "Naive Bayes",
            "Random Forest",
            "XGBoost"
        ]
    )

    # Define Models
    if model_choice == "Logistic Regression":
        model = LogisticRegression(max_iter=5000)

    elif model_choice == "Decision Tree":
        model = DecisionTreeClassifier()

    elif model_choice == "KNN":
        model = KNeighborsClassifier()

    elif model_choice == "Naive Bayes":
        model = GaussianNB()

    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=200)

    else:
        model = XGBClassifier(eval_metric="logloss")

    # ============================================
    # Train Model
    # ============================================

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ============================================
    # Evaluation Metrics
    # ============================================

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    # ============================================
    # Display Metrics
    # ============================================

    st.subheader("📊 Evaluation Metrics")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", round(acc, 4))
    col1.metric("AUC Score", round(auc, 4))

    col2.metric("Precision", round(prec, 4))
    col2.metric("Recall", round(rec, 4))

    col3.metric("F1 Score", round(f1, 4))
    col3.metric("MCC Score", round(mcc, 4))

    # ============================================
    # Confusion Matrix
    # ============================================

    st.subheader("📌 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # ============================================
    # Classification Report
    # ============================================

    st.subheader("📄 Classification Report")

    report = classification_report(y_test, y_pred)
    st.text(report)

    st.success("🎉 Model Evaluation Completed Successfully!")
