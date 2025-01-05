# zawsze się przyda
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import pandas as pd

# przygotowanie danych
x = np.arange(-3, 3, 0.1).reshape((-1, 1))
y = np.tanh(x) + np.random.randn(*x.shape)*0.2 #dodanie szumu do wartości

# trenowanie
ypred = LinearRegression().fit(x,y).predict(x)

# wykres 
#plt.scatter(x,y) # punktowy danych 'poprawnych'
#plt.xlabel('x')
#plt.ylabel('y')
#plt.plot(x, ypred) # liniowy z wartościami regresji liniowej



data = pd.read_excel('practice_lab_2.xlsx')
data_values = data.values
# x -> dane z każdej kolumny bez ostaniej
# y -> dane z ostatniej kolumny (poprawny wynik)
x, y = data_values[:, :-1], data_values[:,-1]
# podział danych na treningowe i testowe w stosunku 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15, shuffle=True)

lin_reg = LinearRegression()
lin_reg.fit(x_train, y_train) # trenowanie na danych treningowych
y_pred = lin_reg.predict(x_test) # wartosci szacowane na podstawie probki testowej
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
# wykres - porównanie wartości szacowanej z faktycznymi danymi
plt.scatter(y_test, y_pred)
plt.plot([min_val, max_val], [min_val, max_val])
plt.xlabel('x')
plt.ylabel('y')

mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f"MSE = {mse}\nMAE = {mae}\nMAPE = {mape}")


fig, ax = plt.subplots(1,1)
title = list(data)[:-1]
X = np.arange(len(title))
weight = lin_reg.coef_
ax.bar(X, weight)
ax.set_xticks(X)
ax.set_xticklabels(title, rotation=90)