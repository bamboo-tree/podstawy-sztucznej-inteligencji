import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model

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

data = load_iris()
y = data.target
x = data.data

y = pd.Categorical(y)
y = pd.get_dummies(y).values
class_num = y.shape[1]

model = Sequential()
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(class_num, activation='softmax'))
learning_rate = 0.0001
model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model.fit(x_train, y_train, epochs=500, batch_size=32, validation_data=(x_test, y_test), verbose=2)



plot_model(model, to_file='model.png')

historia = model.history.history
floss_train = historia['loss']
floss_test = historia['val_loss']
acc_train = historia['accuracy']
acc_test = historia['val_accuracy']
fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 500)
ax[0].plot(epochs, floss_train, label = 'floss_train')
ax[0].plot(epochs, floss_test, label = 'floss_test')
ax[0].set_title('Funkcje strat')
ax[0].legend()
ax[1].set_title('Dokladnosci')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()