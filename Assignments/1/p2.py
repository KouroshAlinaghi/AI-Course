import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

num_classes = 10

input_shape = (28, 28, 1)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

batch_size=64
epochs=10

def build_model(optimizer):
    model=Sequential()
    model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])

    return model

optimizers = ['SGD', 'Adagrad', 'RMSprop']
for opt in optimizers:
    model = build_model(opt)
    hist=model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test,y_test))

import matplotlib.pyplot as plt 
import numpy as np

green = '#57cc99'
blue = '#22577a'
red = '#e56b6f'

fig, ax = plt.subplots()
x_values = np.arange(0, 10)

SGD_acc = [0.7516, 0.8848, 0.9043, 0.9184, 0.9262, 0.9315, 0.9366, 0.9396, 0.9425, 0.9457]
AdaGrad_acc = [0.5760, 0.7952, 0.8340, 0.8511, 0.8630, 0.8732, 0.8791, 0.8837, 0.8879, 0.8915]
RMSProp_acc = [0.9238, 0.9712, 0.9786, 0.9825, 0.9841, 0.9863, 0.9878, 0.9884, 0.9890, 0.9897]

plt.plot(x_values, SGD_acc, label='SGD', color=green)
plt.plot(x_values, AdaGrad_acc, label='AdaGrad', color=red)
plt.plot(x_values, RMSProp_acc, label='RMSProp', color=blue)

ax.set_xticks(x_values)
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_title("Accuracy over time")
ax.legend()
plt.grid(True)
plt.show()
