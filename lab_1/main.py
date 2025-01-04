# operacje na tablicach
import numpy as np

arr = np.array([[2,3,5,1],
                [5,1,2,8],
                [5,1,6,-1]])

# wiersz 0 kolumna 3 ---> 1
pojedynczy_element = arr[0,3]
# wiersz 3 od konca, kolumna 1 od konca ---> 1
pojedynczy_element_od_konca = arr[-3,-1]

# dla wszystkich wierszy, element z indeksu 0 ---> cała kolumna 0
kolumna = arr[:,0]
# dla wiersza 2, wszystkie elementy ---> cały wiersz 2
wiersz = arr[2,:]
# obszar dla wierszy 0-1 oraz kolumn 1-3
obszar = arr[0:2, 1:4]

obszar_pomijajac_koncowe_indeksy = arr[:2, 1:]
parzyste_kolumny = arr[:, ::2]
nieparzyste_kolumny = arr[:, 1::2]
odwrocone_kolumny = arr[:, ::-1]

# maskowanie tablic
indeksy = [1,1,-1]
wybrane_kolumny = arr[:,indeksy]
maska = arr>2
arr_maskowane = arr[maska]


# dane z pliku
import pandas as pd

data_csv = pd.read_csv('practice_lab_1.csv', sep=';')
# wydobycie danych z pliku
nazwy_kolumn = list(data_csv.columns)
wartosc_kolumn = data_csv.values


# wykresy
from matplotlib import pyplot as plt
# różne przebiegi na jednym wykresie
x = np.arange(0, 10, 0.1)
y = np.sin(x**2 - 5*x + 3)
plt.scatter(x,y)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')

# różne przebiegi oddzielnie
fig, ax = plt.subplots(1, 3, figsize=(15,5)) # 1 wiersz, 3 kolumny, rozmiar 10x5
ax[0].plot(x,y)
ax[0].set_xlabel('x')
ax[0].set_ylabel('y')
ax[1].scatter(x,y)
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')
ax[2].scatter(x,y)
ax[2].plot(x,y)
ax[2].set_xlabel('x')
ax[2].set_ylabel('y')
fig.tight_layout()
