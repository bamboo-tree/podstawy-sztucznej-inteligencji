from keras.layers import Dense, BatchNormalization, Dropout, GaussianNoise, LayerNormalization
from kearas.models import Sequential
from keras.optimizers import Adam

neuron_num = 64
do_rate = 0.5
noise = 0.1
learning_rate = 0.001

block = [
    Dense,
    LayerNormalization(),
    BatchNormalization,
    Dropout,
    GaussianNoise
]
args = [
    (neuron_num, 'selu'), (), (), (do_rate), (noise,)
]

model = Sequential()
model.add(Dense(neuron_num, activation='relu', input_shape=(x.shape[1],)))

for i in range(2):
  for layer, arg in zip(block, args):
    model.add(layer(*arg))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(learing_rate), loss='binary_crossentropy', metrics=('accuracy', 'Recall', 'Precision'))
