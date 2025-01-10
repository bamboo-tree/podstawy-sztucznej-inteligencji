import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model

"""
np.set_printoptions(precision=2)
x = np.arange(0,1,0.01)
y = x.copy()
X, Y = np.meshgrid(x,y)
wx = 0.1
wy = 0.3
s = wx*X + wy*Y
out = s>0.15

fig, ax = plt.subplots(1,1)
ax.imshow(out)

ticks = np.around(np.arange(-0.2, 1.1, 0.2), 3)
ax.set_xticklabels(ticks)
ax.set_yticklabels(ticks)
plt.gca().invert_yaxis()
"""

# pobranie danych
data = load_iris()
y = data.target
x = data.data

# konwersja 1-z-n dla wyników
y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]


model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=('accuracy'))
model.summary()

plot_model(model, to_file='siec_neuronowa.png')

