import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:12]
y=data[:,-1]

y_true = data_pd['DEATH_EVENT']
majority_class = y_true.value_counts().idxmax()
print(f"Majority class is: {majority_class}")

y_pred = np.full_like(y_true, fill_value=majority_class)

tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
tp_rate = tp / (tp + fn)  
tn_rate = tn / (tn + fp)  


print(f"Confusion Matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"TP rate {tp_rate:.4f}")
print(f"TN rate: {tn_rate:.4f}")

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_pred = dummy.predict(X_test)

print(classification_report(y_test, y_pred))







