| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|----------|--------|----------|------|
| Logistic Regression | 0.8478 | 0.9168 | 0.8785 | 0.8624 | 0.8704 | 0.6864 |
| Decision Tree | 0.7826 | 0.7895 | 0.8632 | 0.7523 | 0.8039 | 0.5693 |
| KNN | 0.8098 | 0.9093 | 0.8776 | 0.7890 | 0.8309 | 0.6195 |
| Naive Bayes | 0.8370 | 0.9096 | 0.8835 | 0.8349 | 0.8585 | 0.6680 |
| Random Forest (Ensemble) | 0.8913 | 0.9462 | 0.9159 | 0.8991 | 0.9074 | 0.7761 |
| XGBoost (Ensemble) | 0.8533 | 0.9385 | 0.9020 | 0.8440 | 0.8720 | 0.7026 |

## Model Performance Observations

| ML Model | Observation |
|---------|-------------|
| Logistic Regression | Achieved strong baseline performance with good balance between precision and recall. Suitable for linearly separable patterns. |
| Decision Tree | Simple and interpretable but showed lower accuracy due to overfitting tendencies on this dataset. |
| KNN | Produced good AUC score but slightly lower accuracy compared to ensemble models. Performance depends heavily on feature scaling. |
| Naive Bayes | Performed consistently well with high precision and recall, despite its assumption of feature independence. |
| Random Forest (Ensemble) | Delivered the best overall performance with highest accuracy, AUC, and MCC due to ensemble averaging and reduced overfitting. |
| XGBoost (Ensemble) | Achieved excellent results close to Random Forest, benefiting from boosting and advanced optimization. |
