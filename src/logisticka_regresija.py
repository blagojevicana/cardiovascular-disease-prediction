import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

y=data[:,-1]

SC=data[:,7]
EF=data[:,4]
FU=data[:,11]

X = np.column_stack((SC, EF, FU))

#%% Logisticka regresija * 100
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)

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

print("===== Logistic Regression =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")


#%% Logisticka regresija * 100, sva obelezja
X=data[:,0:12]

accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []

for i in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    log_reg = LogisticRegression()
    log_reg.fit(X_train_scaled, y_train)
    y_pred = log_reg.predict(X_test_scaled)

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

print("===== Logistic Regression =====")
print(f"Mean Accuracy: {mean_accuracy:.4f}")
print(f"Mean F1-Score: {mean_f1:.4f}")
print(f"Mean TP rate (Recall): {mean_tp_rate:.4f}")
print(f"Mean TN rate (Specificity): {mean_tn_rate:.4f}")





