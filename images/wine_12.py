student_index = 11112

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.utils import plot_model


# import danych
data = load_wine()
x = data.data
y = data.target

# usuniecie odpowiedniej kolumny ze zbioru danych; (axis=1) --> kolumna
x = np.delete(x, student_index%10, axis=1)

# podział danych
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# skalowanie danych
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

# model svm
model_reg = LogisticRegression()
model_reg.fit(x_train, y_train)
y_pred_reg = model_reg.predict(x_test)
print(f'accuracy_score: {accuracy_score(y_test, y_pred_reg)}')
print(f'confusion_matrix:\n{confusion_matrix(y_test, y_pred_reg)}')

# model sieci neuronowej
model_nn = Sequential()
model_nn.add(Input(shape=(x_train.shape[1:])))
model_nn.add(Dense(64, activation='relu'))
model_nn.add(Dropout(0.2))
model_nn.add(Dense(12, activation='relu'))
model_nn.add(Dense(3, activation='softmax'))

model_nn.compile(optimizer=Adam(0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model_nn.summary()

# uczenie
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
model_nn.fit(x_train, to_categorical(y_train), epochs=50, batch_size=32, validation_data=(x_test, to_categorical(y_test)))

# metryki
model_history = model_nn.history.history
loss_train = model_history['loss']
loss_test = model_history['val_loss']
acc_train = model_history['accuracy']
acc_test = model_history['val_accuracy']

# wykres uczenia
fig,ax = plt.subplots(1,2, figsize=(20,10))
epochs = np.arange(0, 50)

ax[0].plot(epochs, loss_train, label = 'loss_train')
ax[0].plot(epochs, loss_test, label = 'loss_test')
ax[0].set_title('LOSS FUNCTION')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')
ax[0].legend()
ax[1].set_title('ACCURACY')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy')
ax[1].plot(epochs, acc_train, label = 'acc_train')
ax[1].plot(epochs, acc_test, label = 'acc_test')
ax[1].legend()

fig.tight_layout()
plt.show()