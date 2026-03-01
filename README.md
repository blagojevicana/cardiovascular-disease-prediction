# cardiovascular-disease-prediction

### 1. Introduction

This project implements a machine learning pipeline to predict the risk of cardiovascular disease (CVD) based on patient data. The model analyzes clinical and demographic features to classify individuals as at-risk or not, helping with early detection and preventive care.

### 2. Dashboard Preview
![Overview](screenshots/overview.png)

### 3. Motivation

Cardiovascular diseases are the leading cause of death globally. Early detection is crucial for:

- Preventive interventions

- Personalized treatment planning

- Reducing hospitalizations and healthcare costs

- Machine learning enables fast, data-driven prediction from clinical records, complementing traditional diagnostic methods.

## Approach

#### 1. Data Collection

- Public datasets such as the UCI Heart Disease dataset or similar clinical datasets

- Features include: age, sex, smoking, anaemia, creatinine phosphokinase, diabetes, ejection fraction, high blood pressure, platelets, serum creatinine, serum sodium

#### 2. Data Preprocessing

- Handling missing values

- Normalization / standardization of numeric features

- Encoding categorical variables

- Feature Selection

- Identify the most predictive features using statistical analysis or feature importance from tree-based models

![Dataset analysis](results/dataset_analysis.png)
![Histograms](results/histograms.png)
![Correlation](results/correlation.png)


#### 3. Model Training

Classification models tested:

- Random Forest

- Gradient Boosting 

- Support Vector Machine

![Models](results/models.png)


#### 4. Evaluation

- Metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC

- Confusion matrix analysis

- Cross-validation

| Method | F1-score | Accuracy | TP rate | TN rate |
|----------|----------|----------|----------|----------|
| Random Forest  | 0.5267 | 0.7390  | 0.5729  | 0.8127 |
| Gradient Boost | 0.5403 | 0.7480 | 0.4809 | 0.8730 |
| Support Vector Machine | 0.5079 | 0.7340 | 0.4463 | 0.8716 |

After an analysis, it has been concluded that the same results could be achieved with just two features instead of 12. Reducing dimensionality is a big plus when it comes to complex machine learning algorithms.

| Method | F1-score | Accuracy | TP rate | TN rate |
|----------|----------|----------|----------|----------|
| Logistic regression (on ejection fraction and serum creatinine)  | 0.5267 | 0.7390  | 0.5729  | 0.8127 |
| Logistic regression (on all 12 features) | 0.5403 | 0.7480 | 0.4809 | 0.8730 |

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

## Limitations

- Prediction depends on dataset quality and diversity

- Model may not generalize to populations outside the dataset

- Missing or biased data can affect performance

## Future Work

- Incorporate larger, multi-center datasets for better generalization

- Explore deep learning models for feature extraction from raw clinical data

- Develop a web-based tool for real-time risk prediction

- Compare model performance across different demographic groups

