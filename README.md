# cardiovascular-disease-prediction

## Overview

This project implements a machine learning pipeline to predict the risk of cardiovascular disease (CVD) based on patient data. The model analyzes clinical and demographic features to classify individuals as at-risk or not, helping with early detection and preventive care.

## Motivation

Cardiovascular diseases are the leading cause of death globally. Early detection is crucial for:

- Preventive interventions

- Personalized treatment planning

- Reducing hospitalizations and healthcare costs

- Machine learning enables fast, data-driven prediction from clinical records, complementing traditional diagnostic methods.

## Approach

1. Data Collection

- Public datasets such as the UCI Heart Disease dataset or similar clinical datasets

- Features include: age, gender, blood pressure, cholesterol, smoking status, BMI, and more

2. Data Preprocessing

- Handling missing values

- Normalization / standardization of numeric features

- Encoding categorical variables (e.g., sex, chest pain type)

- Feature Selection

- Identify the most predictive features using statistical analysis or feature importance from tree-based models

3. Model Training

Classification models tested:

- Logistic Regression

- Random Forest

- Gradient Boosting 

- Neural Networks 

4. Evaluation

- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Confusion matrix analysis

- Cross-validation

## Dataset
- **Source:** Heart Failure Clinical Records Dataset (Kaggle)
- **Records:** 299 patients
- **Target Variable:** DEATH_EVENT

## Dashboard Features
- KPI cards showing total patients, deaths, average age
- Demographic analysis by age, gender, diabetes, smoking
- Health metrics analysis (ejection fraction, serum creatinine)
- Interactive slicers for filtering

## Key Insights
- Patients aged 60+ show higher mortality rates
- Lower ejection fraction correlates with higher death events
- Serum creatinine is a strong indicator of risk

## 🖼 Dashboard Preview
![Overview](screenshots/overview.png)

## Technologies

- Python 3.x

- NumPy, pandas

- scikit-learn

- XGBoost 

- Matplotlib / Seaborn for visualization

## Limitations

- Prediction depends on dataset quality and diversity

- Model may not generalize to populations outside the dataset

- Missing or biased data can affect performance

## Future Work

- Incorporate larger, multi-center datasets for better generalization

- Explore deep learning models for feature extraction from raw clinical data

- Develop a web-based tool for real-time risk prediction

- Compare model performance across different demographic groups

