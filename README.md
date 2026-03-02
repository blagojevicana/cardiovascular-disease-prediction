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

<img src="results/class_distribution.png" width="500" alt="class_distribution.png"/>

We can see that the classes are not balanced, which happens often in real life datasets, and we should keep this in mind.

We can also take a look at means and standard deviations of distributions of continious features.

![Dataset analysis](results/dataset_analysis.png)
![Dataset analysis](results/data_analysis2.png)
![Histograms](results/histograms.png)

It is useful to also look at correlation between features themselves, and also with the outcome.

![Correlation](results/correlation.png)

Based on the correlation heatmap matrix, we can notice which features have the most correlation with the outcome.
Feature |Correlation with outcome
|---|---|
time| -0.54 
serum_creatinine| 0.37 
ejection_fraction| -0.29 
age |0.22 
serum_sodium| -0.21 
high_blood_pressure| 0.079 
anaemia| 0.066 
platelets |-0.046 
creatinine_phosphokinase| 0.024 
smoking |-0.013 
sex| -0.0043 
diabetes|-0.0019 

It is interesting to see which features are correlated whith each other.

Features | Correlation
|---|---|
sex & smoking |0.45 
serum_creatinine & serum_sodium|-0.3 

We can also calculate Information Gain (IG) for each feature.

Feature |IG coefficient 
---|---
time| 0.235042 
serum_creatinine |0.097463 
ejection_fraction| 0.072724 
serum_sodium| 0.069616 
age |0.050366 
creatinine_phosphokinase| 0.041297 
sex |0.002352 
anaemia |0.000000 
diabetes| 0.000000 
high_blood_pressure| 0.000000 
platelets| 0.000000 
smoking| 0.000000 

### 4. Simple models

Let us first start with the most simple models. One of the most simple models is just counting appearances of each class, and using that class as an outcome for every new patient. This would give us:

F1-score | Accuracy | TP rate | TN rate
---|---|---|---
0.0000|0.6789|0.0000|1.0000

which is a pretty bad result, but it gives as a benchmark for other models.

Next, we can try a simple tree.

<img src="results/tree.png" width="700" alt="tree.png"/>

Metrics of this kind of classifier are:

F1-score | Accuracy | TP rate | TN rate
---|---|---|---
0.5238|0.6567|0.4783|0.7038

which are much better than before, but still not good enough. Next, we will try more complicated models and use these ones as a benchmark.

### 5. Other models

First, let us try **Random Forrest**. Random Forrest is an ansamble of tree, and decision is being made as a majority vote. If we want this model to have good generalization, trees should be uncorrelated. We achive this by using bootstrapping. The idea is that every tree chooses its own subset from the original dataset. Every tree will be trained on different set, which means trees will be uncorrelated and won't make the same mistakes.

When using Random Forrest models, there are a few hyperparameteres to consider:
1. Ansamble size - increasing the number of trees increases the overall accuracy of model, but if we increase it too much, accuracy stays the same, but time to execute program increases.
2. Tree size - increasing the size of a tree increases complexity of the model, which means the model can solve more complex problems, but if we increase it too much, then the model would overfit.
3. Number of predictors - increasing the number of predictors increases overall accuracy, but could lead to overfitting.

A useful information we can get from Random Forests is feature importance. This gives us insight into how many times a predictor has been chosen as a best predictor. Feature importance can be calculated in two ways:
1. Calculating how much impurity has decreased after choosing this predictor. The more impurity decreases, the more useful the predictor is.

Feature | Feature importance 
---|---
serum_creatinine |0.1991 
ejection_fraction| 0.1731 
age |0.1437 
platelets |0.1296 
creatinine_phosphokinase |0.1291 
serum_sodium |0.1166 
anaemia |0.0226 
high_blood_pressure |0.0222 
diabetes| 0.0220 
sex |0.0217 
smoking |0.0203 

2. Calculating accuracy of the model on validation set, then observing if it increases when the predictor is used.

Feature | Feature importance 
---|---
serum_creatinine| 0.0479 
ejection_fraction| 0.0410 
age |0.0157 
serum_sodium |0.0088 
sex |0.0020 
creatinine_phosphokinase| 0.0010 
anaemia |-0.0006 
diabetes |-0.0031 
smoking |-0.0033 
high_blood_pressure |-0.0039 
platelets |-0.0059 

The results after training are:

Model |F1-score|Accuracy|TP rate | TN rate
---|---|---|---|---
Random Forest|0.5304|0.7390|0.4803|0.8631

Next, we will try **Gradient Boosting**. Gradient Boosting is an ansamble algorithm that combines weak models into one strong model. This algorithm iteratively adds new trees that fix the mistakes of trees before them.

When using Gradient Boosting, there are a few hyperparameters to consider:
1. Ansamble size - the bigger the ansamble, the better the accuracy, but we have to pay attention to overfitting.
2. Tree size - the bigger the tree, the more complex the model is, but we have to pay attention to overfitting.
3. Learning rate - increasing learning rate leads to overfitting, but small learning rate slows down the algorithm.

The results after training are:

Model |F1-score|Accuracy|TP rate | TN rate
---|---|---|---|---
Gradient Boosting|0.5315|0.7283|0.4973|0.8383|

Next, we will try **SVM (Support Vector Machine)**. SVM tries to find the best boundary known as hyperplane that separates different classes in the data. The main goal of SVM is to maximize the margin between the two classes. The larger the margin the better the model performs on new and unseen data. There is one important parameter we need to find before training the model, and that is C. C is a regularization term balancing margin maximization and misclassification penalties. A higher C value forces stricter penalty for misclassifications. We can find C by looking at hinge loss on validation set.

<img src="results/find_C.png" width="500" alt="find_C.png"/>

The results after training are:

Model |F1-score|Accuracy|TP rate | TN rate
---|---|---|---|---
SVM|0.2143|0.6333|0.1765|0.8140|

Next, we will try **Naive Bayes**, which is an algorithm based on Bayes' theorem. . It is "naive" because it assumes all features are independent, meaning each predictor contributes equally and independently to the probability of a class.

The results after training are:

Model |F1-score|Accuracy|TP rate | TN rate
---|---|---|---|---
Naive Bayes|0.3359|0.7107|0.2443|0.8989

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

