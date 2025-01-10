from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Input, Dropout
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.optimizers import Adam
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

train, test = mnist.load_data()

X_train, y_train = train[0], train[1]
X_test, y_test = test[0], test[1]
print('Wymiar X_train ',X_train.shape )

# Dodanie nowego wymiaru na końcu tablicy = liczba warst
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)
print('Wymiar X_train ',X_train.shape)

num_fig = 2024
fig, ax = plt.subplots(1,2, figsize = (20,20))
ax[0].imshow(X_train[num_fig], cmap = plt.get_cmap('gray'))
ax[0].set_title('Zbior treningowy cyfra: '+str(y_train[num_fig]))
ax[1].imshow(X_test[num_fig], cmap = plt.get_cmap('gray'))
ax[1].set_title('Zbior testowy cyfra: '+str(y_test[num_fig]))
plt.show()

class_cnt = np.unique(y_train).shape[0]
neuron_cnt = 32
learning_rate = 0.0001
act_func = 'relu'

filter_cnt = 32
kernel_size = (3,3)
conv_rule = 'same'
pooling_size = (2,2)

print(X_train.shape[1:])

# Definiowanie modelu
model = Sequential()
# Warstwa wejsciowa
model.add(Input(shape = X_train.shape[1:]))
# Warstawa 1 - konwolucyjna
model.add(Conv2D(filters=filter_cnt, kernel_size = kernel_size, padding = conv_rule, activation = act_func))
# Warstawa 2 - pooling
model.add(MaxPooling2D(pooling_size))
# Warstawa 3 - wypłaszczanie
model.add(Flatten())
# Warstwa wyjsciowa - klasyfikacja
model.add(Dense(class_cnt, activation='softmax'))

model.summary()

# Kompilowanie modelu
model.compile(optimizer=Adam(learning_rate), loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

# Trenowanie modelu
model.fit(x = X_train, y = y_train, epochs = class_cnt, validation_data=(X_test, y_test))

# Ocena modelu
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')

# Predykcja
y_pred = model.predict(X_test)
print(np.argmax(y_pred, axis=1))

# Macierz pomyłek
cm = confusion_matrix(y_test, np.argmax(y_pred, axis=-1))
print(cm)