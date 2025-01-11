import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

data = pd.read_excel('practice_lab_1.xlsx')

title = data.columns
arr = data.values

# zadanie 1.2.
arr_1 = arr[::2, :]
arr_2 = arr[1::2, :]
result = arr_1 - arr_2

mean = arr.mean()
std = arr.std()
result = (arr-mean)/std

mean = arr.mean(axis=0)
std = arr.std(axis=0)
result = (arr-mean)/(std+np.spacing(std))

mean = arr.mean(axis=0)
mask = arr > mean
result = mask.sum(axis=0)

mask = arr == 0
mask = mask.argmax(axis=0) == 1
result = title[mask]

mask = arr[::2,:].sum(axis=0) > arr[1::2,:].sum(axis=0)
result = title[mask]

# zadanie 1.3.
x = np.arange(-5, 5, 0.01)

y = np.tanh(x)

fig, ax = plt.subplots(1,1)
ax.plot(x,y)
ax.set_title('Wykres funkcji tanh')
ax.set_xlabel('x')
ax.set_ylabel('y')
fig.tight_layout() 

# zadanie 1.4.
cor_matrix = data.corr()
fig, ax = plt.subplots(1,1)
ax.imshow(cor_matrix, cmap='jet')
fig.tight_layout()


fig, ax = plt.subplots(cor_matrix.shape[0], cor_matrix.shape[0], figsize=(10,10))
for i in range(cor_matrix.shape[0]):
    for j in range(cor_matrix.shape[1]):
        ax[i,j].scatter(arr[:,i], arr[:,j])
        ax[i,j].set_xlabel(title[i])
        ax[i,j].set_ylabel(title[j])
fig.tight_layout()
