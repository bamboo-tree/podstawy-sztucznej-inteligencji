import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC as svm
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


data = pd.read_csv('voice_extracted_features.csv', sep=',')
arr = data.values

# zamiana tekstowej cechy jakościowej na logiczną 1/0
mask = arr[:,-1] == 'male'
arr[mask, -1] = 1 # male
arr[~mask, -1] = 0 # female
arr.astype(float)

x = data.values[:,:-1]
y = data.values[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)

pca_transform = PCA()
pca_transform.fit(x_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()

plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0,1.1,0.1))

PC_num = (cumulated_variances >= 0.99).sum()
x_pcaed = PCA(2).fit_transform(x_train)

fig, ax = plt.subplots(1,1)
females = y_train == 0
ax.scatter(x_pcaed[females,0], x_pcaed[females,1], label='female')
ax.scatter(x_pcaed[~females,0], x_pcaed[~females,1], label='male')
ax.legend()

# pipeline
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
pipe = Pipeline([('scaler', StandardScaler()), ('transformer', PCA(2)), ('svm', svm())])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)

print(accuracy_score(y_pred, y_test))
print(pipe.score(x_test, y_test))

