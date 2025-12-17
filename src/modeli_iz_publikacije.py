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
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:11] # exluding time
y=data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Random Forest
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []
perm_importances = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    rf_clf = RandomForestClassifier(random_state=i)
    rf_clf.fit(X_train, y_train)
    y_pred = rf_clf.predict(X_test)

    feature_importances.append(rf_clf.feature_importances_)
    perm = permutation_importance(rf_clf, X_test, y_test, n_repeats=10, random_state=i)
    perm_importances.append(perm.importances_mean)
    
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
mean_importances = np.mean(feature_importances, axis=0)
mean_perm_importances = np.mean(perm_importances, axis=0)

print("===== Random Forest =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

print("\nMean Feature Importances:")
for name, imp in zip(data_pd.columns[:12], mean_importances):
    print(f"{name}: {imp:.4f}")

print("\nMean Permutation (Accuracy Reduction) Importances:")
for name, imp in zip(data_pd.columns[:12], mean_perm_importances):
    print(f"{name}: {imp:.4f}")

#%% Gradient Boosting
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []

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

#%% Linear Regression
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    y_pred_continuous = lin_reg.predict(X_test)
    y_pred = (y_pred_continuous >= 0.5).astype(int)

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

print("===== Linear Regression =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

#%% Naive Bayes
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    nb_clf = GaussianNB()
    nb_clf.fit(X_train, y_train)
    y_pred = nb_clf.predict(X_test)

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

print("===== Naive Bayes =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

#%% Linear SVM
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    svm_clf = SVC(kernel='linear', random_state=i)
    svm_clf.fit(X_train, y_train)
    y_pred = svm_clf.predict(X_test)


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

print("===== Linear SVM =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")

#%% Radial SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_clf = SVC(kernel='rbf',gamma=0.4, random_state=42)
svm_clf.fit(X_train_scaled, y_train)
y_pred = svm_clf.predict(X_test_scaled)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tp_rate = tp / (tp + fn)
tn_rate = tn / (tn + fp)

print("===== Radial SVM =====")
print(f"Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"TP rate (Recall): {tp_rate:.4f}")
print(f"TN rate (Specificity): {tn_rate:.4f}")
