import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

y=data[:,-1]

SC=data[:,7]
EF=data[:,4]
FU=data[:,11]

plt.figure()
plt.scatter(SC[y == 0], EF[y == 0], color='blue', marker='o', alpha=0.8)
plt.scatter(SC[y == 1], EF[y == 1], color='red', marker='*', alpha=0.8)
plt.xlabel('serum creatinine')
plt.ylabel('ejection fraction')


X = np.column_stack((SC, EF))
#%% Random forest

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_train)

sc_min, sc_max = SC.min() - 0.2, SC.max() + 0.2
ef_min, ef_max = EF.min() - 2, EF.max() + 2
xx, yy = np.meshgrid(np.linspace(sc_min, sc_max, 200),
                     np.linspace(ef_min, ef_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = rf_clf.predict(grid_points).reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)

plt.scatter(SC[y == 0], EF[y == 0], color='blue', marker='o')
plt.scatter(SC[y == 1], EF[y == 1], color='red', marker='*')

plt.xlabel('Serum Creatinine (mg/dL)')
plt.ylabel('Ejection Fraction (%)')
plt.title('Random Forest Decision Regions')
plt.legend(['Negative','Positive'])
plt.xlim(sc_min, sc_max)
plt.ylim(ef_min, ef_max)

#%% Random forest * 100
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    rf_clf = RandomForestClassifier(random_state=i)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    accuracies.append(accuracy)
    f1_scores.append(f1)
    tp_rates.append(tp_rate)
    tn_rates.append(tn_rate)


mean_accuracy = np.mean(accuracies)
mean_f1 = np.mean(f1_scores)
mean_tp_rate = np.mean(tp_rates)
mean_tn_rate = np.mean(tn_rates)

print("===== Random Forest =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

#%% Gradient Boosting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

gb_clf = GradientBoostingClassifier(random_state=42)
gb_clf.fit(X_train, y_train)

sc_min, sc_max = SC.min() - 0.2, SC.max() + 0.2
ef_min, ef_max = EF.min() - 2, EF.max() + 2
xx, yy = np.meshgrid(np.linspace(sc_min, sc_max, 200),
                     np.linspace(ef_min, ef_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]

Z = gb_clf.predict(grid_points).reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)

plt.scatter(SC[y == 0], EF[y == 0], color='blue', marker='o')
plt.scatter(SC[y == 1], EF[y == 1], color='red', marker='*')

plt.xlabel('Serum Creatinine (mg/dL)')
plt.ylabel('Ejection Fraction (%)')
plt.title('Gradient Boosting')
plt.legend(['Negative','Positive'])
plt.xlim(sc_min, sc_max)
plt.ylim(ef_min, ef_max)

#%% Gradient Boosting * 100
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    gb_clf = GradientBoostingClassifier(random_state=i)
    gb_clf.fit(X_train, y_train)
    y_pred = gb_clf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    accuracies.append(accuracy)
    f1_scores.append(f1)
    tp_rates.append(tp_rate)
    tn_rates.append(tn_rate)


mean_accuracy = np.mean(accuracies)
mean_f1 = np.mean(f1_scores)
mean_tp_rate = np.mean(tp_rates)
mean_tn_rate = np.mean(tn_rates)

print("===== Gradient Boosting =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

#%% Radial SVM
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_clf = SVC(kernel='rbf',gamma=1,class_weight='balanced')
svm_clf.fit(X_train_scaled, y_train)

sc_min, sc_max = SC.min() - 0.2, SC.max() + 0.2
ef_min, ef_max = EF.min() - 2, EF.max() + 2
xx, yy = np.meshgrid(np.linspace(sc_min, sc_max, 200),
                     np.linspace(ef_min, ef_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = scaler.transform(grid_points)
Z = svm_clf.predict(grid_points_scaled).reshape(xx.shape)

plt.figure()

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)

plt.scatter(SC[y == 0], EF[y == 0], color='blue', marker='o')
plt.scatter(SC[y == 1], EF[y == 1], color='red', marker='*')

plt.xlabel('Serum Creatinine (mg/dL)')
plt.ylabel('Ejection Fraction (%)')
plt.title('Radial SVM')
plt.legend(['Negative','Positive'])
plt.xlim(sc_min, sc_max)
plt.ylim(ef_min, ef_max)

#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['rbf']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='f1')
grid_search.fit(X_train_scaled, y_train)

print("Best parameters:", grid_search.best_params_)
best_model = grid_search.best_estimator_

#%% Radial SVM * 100
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    svm_clf = SVC(kernel='rbf')
    svm_clf.fit(X_train_scaled, y_train)
    
    y_pred = svm_clf.predict(X_test_scaled)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tp_rate = tp / (tp + fn)
    tn_rate = tn / (tn + fp)

    accuracies.append(accuracy)
    f1_scores.append(f1)
    tp_rates.append(tp_rate)
    tn_rates.append(tn_rate)


mean_accuracy = np.mean(accuracies)
mean_f1 = np.mean(f1_scores)
mean_tp_rate = np.mean(tp_rates)
mean_tn_rate = np.mean(tn_rates)

print("===== Radial SVM =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

