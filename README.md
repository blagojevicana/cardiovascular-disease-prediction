# cardiovascular-disease-prediction

### 1. Introduction

This project implements a machine learning pipeline to predict the risk of cardiovascular disease (CVD) based on patient data. The model analyzes clinical and demographic features to classify individuals as at-risk or not, helping with early detection and preventive care.

### 2. Dashboard 
![Overview](screenshots/overview.png)

### 3. EDA (Exploratory Data Analysis)

This dataset has 12 features and 1 outcome, with 299 patients.

|Variable Name	|Role|	Type|	Demographic|	Description|	Units|	Missing Values
|---|---|---|---|---|---|---|
age	|Feature	|Integer|	Age	|age of the patient	|years	|no
anaemia	|Feature|	Binary	|	|decrease of red blood cells or hemoglobin|	|	no
creatinine_phosphokinase	|Feature|	Integer	| |	level of the CPK enzyme in the blood	|mcg/L	|no
diabetes|	Feature|	Binary	| |	if the patient has diabetes|	|	no
ejection_fraction	|Feature	|Integer|	|	percentage of blood leaving the heart at each contraction	|%	|no
high_blood_pressure|	Feature	|Binary|	|	if the patient has hypertension|	|	no
platelets	|Feature	|Continuous|	|	platelets in the blood	|kiloplatelets/mL	|no
serum_creatinine	|Feature	|Continuous|	|	level of serum creatinine in the blood|	mg/dL|	no
serum_sodium|	Feature	|Integer|	|	level of serum sodium in the blood	|mEq/L|	no
sex	|Feature	|Binary	|Sex	|woman or man	| |	no

First we need to check if all of our values are in appropriate intervals. We can do this by checking the information about the dataset.

| |	age	|anaemia|	creatinine_phosphokinase|	diabetes|	ejection_fraction|	high_blood_pressure|	platelets|	serum_creatinine|	serum_sodium|	sex|	smoking|	time|	DEATH_EVENT|
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
count	|299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0|	299.0
mean	|60.83389297658862|	0.431438127090301	|581.8394648829432	|0.4180602006688963|	38.08361204013378	|0.3511705685618729	|263358.02926421404	|1.3938795986622072	|136.62541806020067	|0.6488294314381271|	0.3210702341137124|	130.2608695652174	|0.3210702341137124
std	|11.89480907404447|	0.4961072681330793	|970.2878807124362	|0.49406706510360904|	11.834840741039171|	0.47813637906274475	|97804.2368685983	|1.0345100640898541	|4.412477283909235	|0.47813637906274475|	0.46767042805677167	|77.61420795029339|	0.46767042805677167
min	|40.0|	0.0|	23.0|	0.0|	14.0|	0.0|	25100.0|	0.5|	113.0	|0.0|	0.0|	4.0|	0.0
25%|	51.0	|0.0|	116.5	|0.0	|30.0	|0.0|	212500.0	|0.9|	134.0	|0.0	|0.0|	73.0	|0.0
50%	|60.0|	0.0|	250.0	|0.0|	38.0	|0.0|	262000.0|	1.1|	137.0|	1.0|	0.0|	115.0|	0.0
75%|	70.0	|1.0|	582.0	|1.0|	45.0	|1.0|	303500.0	|1.4	|140.0	|1.0	|1.0|	203.0	|1.0
max	|95.0|	1.0|	7861.0	|1.0|	80.0	|1.0|	850000.0	|9.4|	148.0	|1.0|	1.0|	285.0|	1.0

Next, we should check if the classes are balanced or not.


..................................................................................................

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

