import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error


data = pd.read_excel('practice_lab_2.xlsx')
title = data.columns
arr = data.values

x = arr[:,:-1]
y = arr[:,-1]


# zadanie 2.1.
fig, ax = plt.subplots(1, 1, figsize=(10,10))
corr_matrix = data.corr()
ax.imshow(corr_matrix)

fig, ax = plt.subplots(3, 4, figsize=(10,10))
for i in range(3):
  for j in range(4):
    ax[i,j].scatter(x[:,i*3+j],y)
    ax[i,j].set_xlabel(title[i*3+j])
    ax[i,j].set_ylabel(title[-1])
fig.tight_layout()


# zadanie 2.2.
n = 5
for i in range(n):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
  model = LinearRegression()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  mape = mean_absolute_percentage_error(y_test, y_pred)
  print(f'MAPE dla próby {i+1}: {mape}')


# zadanie 2.3.
n = 5
for i in range(n):
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
  outliers = np.abs((y_train - y_train.mean())/y_train.std()) > 3
  x_train = x_train[~outliers]
  y_train = y_train[~outliers]

  model = LinearRegression()
  model.fit(x_train, y_train)
  y_pred = model.predict(x_test)
  mape = mean_absolute_percentage_error(y_test, y_pred)
  print(f'MAPE dla próby {i+1}: {mape}')