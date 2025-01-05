# biblioteki
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

data = pd.read_excel('practice_lab_2.xlsx')
title = list(data)[:-1]
arr = data.values

def zadanie_1():
    fig, ax = plt.subplots(4, 4, figsize=(10, 10))
    matrix = data.corr()
    for i in range(len(title)):
        ax[i//4, i%4].scatter(arr[:,i], arr[:,-1])
        ax[i//4, i%4].set_xlabel(title[i])
        ax[i//4, i%4].set_ylabel("Cena")
    fig.tight_layout()

    plt.imshow(matrix, 'jet')
    
def zadanie_2():
    n = 5
    x, y = arr[:,:-1], arr[:, -1]
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        lin_reg = LinearRegression().fit(x_train, y_train)
        y_pred = lin_reg.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"MAPE = {mape}")

def zadanie_3():
    print("NO OUTLIERS")
    n = 5
    x, y = arr[:,:-1], arr[:, -1]
    for i in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        
        outliers = np.abs((y_train - y_train.mean())/y_train.std())>3
        x_train_no_outliers = x_train[~outliers,:]
        y_train_no_outliers = y_train[~outliers]
        
        lin_reg = LinearRegression().fit(x_train_no_outliers, y_train_no_outliers)
        y_pred = lin_reg.predict(x_test)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        print(f"MAPE = {mape}")
    
# ZADANIE 2.1.
zadanie_1()

# ZADANIE 2.2.
zadanie_2()

# ZADANIE 2.3.
zadanie_3()


