import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:12]
y=data[:,-1]

feature = 'time'
threshold = 200

y_true = data_pd['DEATH_EVENT'].astype(int)

y_pred = (data_pd[feature] > threshold).astype(int)

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
scores = data_pd[feature]  

fpr, tpr, thresholds = roc_curve(y_true, scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') 
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Recall)')
plt.title(f'ROC Curve for Feature: {feature}')
plt.legend(loc="lower right")


