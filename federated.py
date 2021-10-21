import numpy as np
import pandas as pd

import tensorflow as tf
from keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import backend as K

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label


def load_data():
    """Load data"""

    print("[X] Loading data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # data cleaning
    image_size = x_train.shape[1] # 28
    x_train = np.reshape(x_train, [-1, image_size])
    x_test = np.reshape(x_test, [-1, image_size])

    x_train /= 255
    x_test /= 255
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    return list(zip(x_train, y_train))


def shard_data(data, num_clients=10):
    """Shard data"""

    print("[X] Sharding data...")
    client_names = [f"Device_{i}" for i in range(num_clients)]
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    return {client_names[i]: shards[i] for i in range(len(client_names))}


data = load_data()
clients = shard_data(data)

for (client, data) in clients.items():
    pass

class FederatedMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

fed_model = FederatedMLP()
global_model = fed_model.build(784, 10)
global_weights = global_model.get_weights()

for round in rounds:
    # retrieve weights of global models
    global_weights = global_model.get_weights()

    # simulate client
    for client in client_names:
        fed_model = FederatedMLP()
        local_model = fed_model.build(784, 10)
        local_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        local_model.set_weights(global_weights)

        local_model.fit(clients_batched[client], epochs=1, verbose=0)

        K.clear_session()

    average_weights = aggregation()