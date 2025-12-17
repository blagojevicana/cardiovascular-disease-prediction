import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif

plt.close('all')

data_pd=pd.read_csv('C:/Users/Korisnik/Desktop/masinsko projekat/heart_failure_clinical_records_dataset.csv')
data=pd.DataFrame.to_numpy(data_pd)
data=data[1:]

X=data[:,0:12]
y=data[:,-1]


print(data_pd.info())
stats=data_pd.describe();
data_pd.hist()
#data_pd.dropna(axis=0,inplace=True) 

#%%
unique, counts = np.unique(y, return_counts=True)

for cls, count in zip(unique, counts):
    print(f"Class {cls}: {count} samples ({count / len(y) * 100:.2f}%)")

plt.bar(unique.astype(str), counts, color=['skyblue', 'salmon'])
plt.xlabel('Class')
plt.ylabel('Number of samples')
plt.title('Class Distribution')

#%%
import seaborn as sns
sns.set_theme(style="ticks", palette="pastel")

selected_features = [
    'age',
    'creatinine_phosphokinase',
    'ejection_fraction',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'time'
]

data_pd.columns = list(data_pd.columns[:-1]) + ['Class']
X = data_pd[selected_features]
y = data_pd['Class']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
data_scaled = pd.concat([X_scaled, y], axis=1)
data_long = pd.melt(data_scaled, id_vars='Class', var_name='Feature', value_name='Value')

plt.figure()
sns.boxplot(x='Feature', y='Value', hue='Class', data=data_long, palette=["m", "g"])
sns.despine(offset=10, trim=True)
plt.xticks(rotation=45)

#%%
if data_pd.columns[-1] != 'Class':
    data_pd.columns = list(data_pd.columns[:-1]) + ['Class']

data_pos = data_pd[data_pd['Class'] == 1]
data_neg = data_pd[data_pd['Class'] == 0]

features = data_pd.columns[:-1]

n_features = len(features)
n_cols = 4  
n_rows = (n_features + n_cols - 1) // n_cols  

plt.figure(figsize=(20, 4 * n_rows))

for i, feature in enumerate(features, 1):
    plt.subplot(n_rows, n_cols, i)
    plt.hist(data_neg[feature], bins=20, alpha=0.5, label='Negative (0)', color='skyblue', density=True)
    plt.hist(data_pos[feature], bins=20, alpha=0.5, label='Positive (1)', color='salmon', density=True)
    plt.title(feature)
    plt.xlabel(feature)
    plt.ylabel('Density')
    if i == 1:
        plt.legend()

plt.tight_layout()

#%%
corr=data_pd.corr(method='spearman');
plt.figure()
sns.heatmap(corr,annot=True)
#%%
#sns.set_theme(style="ticks", palette="pastel")

selected_features = [
    'age',
    'creatinine_phosphokinase',
    'ejection_fraction',
    'platelets',
    'serum_creatinine',
    'serum_sodium',
    'time'
]

data_pd.columns = list(data_pd.columns[:-1]) + ['Class']
X = data_pd[selected_features]
y = data_pd['Class']
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
data_scaled = pd.concat([X_scaled, y], axis=1)
data_long = pd.melt(data_scaled, id_vars='Class', var_name='Feature', value_name='Value')

plt.figure()
darker_palette=["#8B008B", "#006400"]
sns.violinplot(x='Feature', y='Value', hue='Class', data=data_long, palette=darker_palette,split=True, inner="quart", fill=False)
sns.despine(offset=10, trim=True)
plt.xticks(rotation=45)

#%%
X = data_pd.iloc[:, :-1]
y = data_pd.iloc[:, -1]

info_gain = mutual_info_classif(X, y, discrete_features='auto')
info_gain_series = pd.Series(info_gain, index=X.columns)

print("Information Gain of each feature:")
print(info_gain_series.sort_values(ascending=False))



