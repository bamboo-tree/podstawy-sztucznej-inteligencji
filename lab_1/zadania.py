# ZADANIE 1.2.

import pandas as pd
import numpy as np

# pobranie danych z pliku csv
data = pd.read_excel('practice_lab_1.xlsx')
nazwy_kolumn = list(data.columns)
wartosci = data.values

# 1)
arr_1 = wartosci[::2, :]
arr_2 = wartosci[1::2, :]
wynik = arr_1 - arr_2

# 2)
srednia = wartosci.mean()
odchylenie = wartosci.std()
wynik = (wartosci - srednia) / odchylenie

# 3)
srednia = wartosci.mean(axis=0)
odchylenie = wartosci.std(axis=0)
wynik = (wartosci - srednia) / (odchylenie + np.spacing(odchylenie))

# 4)
wspolczynnik = wartosci.mean(axis=0) / (wartosci.std(axis=0) + np.spacing(wartosci.std(axis=0)))

# 5)
max_wspolczynnik = wspolczynnik.argmax(axis=0)

# 6)
maska = wartosci > wartosci.mean(axis=0)
elementy_wieksze_od_sredniej = maska.sum(axis=0)

# 7)
max_index_kolumny = wartosci.max(axis=0) == wartosci.max()
kolumny_max = np.array(nazwy_kolumn)[max_index_kolumny]

# 8)
maska = wartosci == 0
max_ilosc_zer = maska.sum(axis=0) == maska.sum(axis=0).max()
kolumna_max_zer = np.array(nazwy_kolumn)[max_ilosc_zer]

# 9)
maska = wartosci[::2,:].sum(axis=0) > wartosci[1::2,:].sum(axis=0)
kolumny = np.array(nazwy_kolumn)[maska]


# ZADANIE 1.3.

from matplotlib import pyplot as plt

x = np.arange(-5, 5, 0.01)

# 1) 
y = np.tanh(x)
# 2)
y = (np.e**(x) - np.e**(-x)) / (np.e**(x) + np.e**(-x))
# 3)
y = 1 / (1 + np.e**(-x))
# 4)
y = np.where(x <= 0, 0, x)
# 5)
y = np.where(x <= 1, (np.e**x) - 1, x)

plt.plot(x, y)
plt.xlabel('X')
plt.ylabel('Y')


# ZADANIE 1.4.
korelacja = data.corr()

liczba_kolumn = len(nazwy_kolumn)
fig, ax = plt.subplots(liczba_kolumn, liczba_kolumn, figsize=(liczba_kolumn*4, liczba_kolumn*4))

for i in range(liczba_kolumn):
    for j in range(liczba_kolumn):
        ax[i,j].scatter(wartosci[:,i],wartosci[:,j])
        ax[i,j].set_xlabel(nazwy_kolumn[i])
        ax[i,j].set_ylabel(nazwy_kolumn[j])
fig.tight_layout()





