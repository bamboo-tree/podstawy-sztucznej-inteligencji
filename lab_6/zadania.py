import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from keras.regularizers import l1, l2
from keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, LayerNormalization, Input
from keras.models import Sequential
from keras.optimizers import Adam, RMSprop, SGD
from sklearn.model_selection import train_test_split

# załadowanie danych
data = load_iris()
x = data.data
y = data.target

# usuniecie odchylen w danych
y = pd.Categorical(y)
y = pd.get_dummies(y).values

# podział danych
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# argumenty do głębokiej sieci neuronowej
neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001
reg_rate = [0, 0.0001, 0.001, 0.01, 0.1]
acc = []
block = [Dense]
args = [(neuron_num, 'relu'),(),(),(do_rate,),(noise,)]

for e in reg_rate:
    model = Sequential()
    model.add(Input(shape=(x_train.shape[1],)))
    model.add(Dense(neuron_num, activation='selu', kernel_regularizer=l2(e)))
    
    repeat_num = 2

    for i in range(repeat_num):
      for layer, arg in zip(block, args):
        model.add(layer(*arg))
    
    model.add(Dense(y_train.shape[1], activation='softmax'))
    model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])

    history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), verbose=0)
    acc.append(max(history.history['val_accuracy']))

    print(f'dla regularyzacji {e} maksymalna dokładność = {acc[-1]}')

model.summary()

plt.figure(figsize=(8,8))
plt.scatter(reg_rate, acc)
plt.xlabel('Regularyzacja')
plt.ylabel('Dokładność')

neuron_num = 64
do_rate = [0, 0.2, 0.3, 0.5]
noise = 0.1
learning_rate = 0.001
reg_rate = 0.001
acc = []

for rate in do_rate:
  block = [Dense, Dropout]
  args = [(neuron_num, 'relu'),(rate,)]

  model = Sequential()
  model.add(Input(shape=(x_train.shape[1],)))
  model.add(Dense(neuron_num, activation='relu'))
  
  repeat_num = 2
  for i in range(repeat_num):
    for layer, arg in zip(block, args):
      model.add(layer(*arg))
  
  model.add(Dense(y_train.shape[1], activation='softmax'))
  model.compile(optimizer=Adam(learning_rate), loss='categorical_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
  model.add(Dense(y_train.shape[1], activation='sigmoid'))
  model.compile(optimizer=Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy', 'Recall', 'Precision'])
  
  history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test), verbose=1)
  acc.append(max(history.history['val_accuracy']))

  print(f'dla dropoutu {rate} maksymalna dokładność = {acc[-1]}')

  model.summary()


