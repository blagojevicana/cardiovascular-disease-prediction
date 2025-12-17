import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.metrics import hinge_loss

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:11] # exluding time
y=data[:,-1]

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,shuffle=True,random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

C_values = np.linspace(0.01, 0.3, 50)
hinge_losses = []

for C in C_values:
    svm_clf = LinearSVC(C=C, loss='hinge', max_iter=100000, random_state=42)
    svm_clf.fit(X_train_scaled, y_train)
    
    decision_values = svm_clf.decision_function(X_valid_scaled)
    loss = hinge_loss(y_valid, decision_values)
    hinge_losses.append(loss)

hinge_losses = np.array(hinge_losses) 
plt.figure()
plt.plot(C_values,hinge_losses)
plt.xlabel('C')
plt.ylabel('hinge loss')

C_opt=C_values[np.argmin(hinge_losses)]
#%%
accuracies = []
f1_scores = []
tp_rates = []
tn_rates = []
feature_importances = []

for i in range(1):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    
    svm_clf = SVC(kernel='linear', C=C_opt, random_state=i)
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


