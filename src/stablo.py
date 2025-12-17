import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:12]
y=data[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_clf = DecisionTreeClassifier(random_state=42)  
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
tp_rate = tp / (tp + fn)  
tn_rate = tn / (tn + fp)  


print(f"Confusion Matrix:\nTN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"TP rate {tp_rate:.4f}")
print(f"TN rate: {tn_rate:.4f}")

plt.figure(figsize=(12, 6))
plot_tree(tree_clf, feature_names=data_pd.columns[:12], class_names=['0', '1'], filled=True, rounded=True)
plt.title("Decision Tree Classifier")


