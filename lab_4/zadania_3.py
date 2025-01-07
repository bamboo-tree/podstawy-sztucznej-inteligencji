import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.datasets import load_digits
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import numpy as np

# pobranie danych
data = pd.read_csv('voice_extracted_features.csv', sep=',')
title = list(data.columns)
arr = data.values

# zamiana tekstowej cechy jakościowej na logiczną 1/0
mask = arr[:,-1] == 'male'
arr[mask, -1] = 1 # male
arr[~mask, -1] = 0 # female
arr.astype(float)

# podział na zbiory
x = arr[:,:-1]
y = arr[:,-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)
# przygotowanie danych do trenowania
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
pca_transform = PCA()
pca_transform.fit(x_train)
variances = pca_transform.explained_variance_ratio_
cumulated_variances = variances.cumsum()

# tworzenie wykresu
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))

pc_num = (cumulated_variances<0.95).sum()
x_pcaed = PCA(2).fit_transform(x_train)

fig, ax = plt.subplots(1,1)
males = y_train == 1
ax.scatter(x_pcaed[males,0], x_pcaed[males,1], label="MALE")
ax.scatter(x_pcaed[~males,0], x_pcaed[~males,1], label="FEMALE")
ax.legend()

# X, _ = load_digits(return_X_y=True)
# transformer = FastICA(n_components=7, random_state=0)
# X_transformed = transformer.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6, stratify=y)
#przygotowanie danych ze skalowaniem i PCA
pca_transform = PCA(9)
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

x_train_pcaed = pca_transform.fit_transform(x_train)
x_test_pcaed = pca_transform.transform(x_test)

x_train_pcaed_scaled = scaler.fit_transform(x_train_pcaed)
x_test_pcaed_scaled = scaler.transform(x_test_pcaed)

model = knn(5, weights = 'distance')
model.fit(x_train_pcaed_scaled, y_train) # problem ???
y_predict = model.predict(x_test_pcaed_scaled)
print(accuracy_score(y_test, y_predict))


# to samo co wyżej tylko mniej pisania
from sklearn.pipeline import Pipeline
pipe = Pipeline([['transformer', PCA(11)], ['scaler', StandardScaler()], ['classifier', knn(weights='distance')]])
pipe.fit(x_train, y_train)
y_pred = pipe.predict(x_test)
print(accuracy_score(y_test, y_pred))