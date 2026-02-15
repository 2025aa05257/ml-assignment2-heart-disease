# Machine Learning Assignment 2 - Heart Disease Classification

## Problem Statement
The objective of this project is to implement multiple machine learning classification
models to predict whether a person has heart disease based on clinical health attributes.
The project also demonstrates model evaluation, comparison, and deployment using
a Streamlit web application.

---

## Dataset Description
Dataset used: Heart Disease UCI Dataset  
Target Variable: `num`

- `0` → No Heart Disease  
- `1` → Presence of Heart Disease  

The dataset contains more than 500 instances and more than 12 clinical features,
making it suitable for classification tasks.

---

## Models Used and Evaluation Metrics

All six classification models were implemented on the same dataset, and the following
evaluation metrics were calculated:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|----------|--------|----------|------|
| Logistic Regression | 0.8478 | 0.9168 | 0.8785 | 0.8624 | 0.8704 | 0.6864 |
| Decision Tree | 0.7826 | 0.7895 | 0.8632 | 0.7523 | 0.8039 | 0.5693 |
| KNN | 0.8098 | 0.9093 | 0.8776 | 0.7890 | 0.8309 | 0.6195 |
| Naive Bayes | 0.8370 | 0.9096 | 0.8835 | 0.8349 | 0.8585 | 0.6680 |
| Random Forest (Ensemble) | 0.8913 | 0.9462 | 0.9159 | 0.8991 | 0.9074 | 0.7761 |
| XGBoost (Ensemble) | 0.8533 | 0.9385 | 0.9020 | 0.8440 | 0.8720 | 0.7026 |

---

## Observations on Model Performance

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | Achieved strong baseline performance with good balance between precision and recall. Suitable for linearly separable patterns. |
| Decision Tree | Easy to interpret but showed lower accuracy due to overfitting tendencies on this dataset. |
| KNN | Produced a high AUC score but slightly lower accuracy compared to ensemble models. Performance depends on feature scaling. |
| Naive Bayes | Performed consistently well with good precision and recall despite assuming feature independence. |
| Random Forest (Ensemble) | Delivered the best overall performance with highest accuracy, AUC, and MCC due to ensemble averaging and reduced overfitting. |
| XGBoost (Ensemble) | Achieved excellent results close to Random Forest, benefiting from boosting and advanced optimization techniques. |

---

## Streamlit Deployment
This project is deployed as an interactive Streamlit application using Streamlit
Community Cloud.

The application provides:

- Dataset upload option (CSV)
- Model selection dropdown
- Display of evaluation metrics
- Confusion matrix and classification report

Steps to deploy:

1. Push the project to GitHub
2. Go to https://streamlit.io/cloud
3. Click "New App"
4. Select repository and choose `app.py`
5. Deploy and share the live link

---

## Submission Requirements
The final submission PDF includes:

- GitHub repository link
- Streamlit live app link
- Screenshot of execution on BITS Virtual Lab
- README.md content
