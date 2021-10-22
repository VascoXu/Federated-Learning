import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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

    # Data cleaning
    image_size = x_train.shape[1] # 28

    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)

    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)    

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return list(zip(x_train, y_train))


def shard_data(data, num_clients=10):
    """Shard data"""

    print("[X] Sharding data...")
    client_names = [f"device{i}" for i in range(num_clients)]
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]
    
    return {client_names[i]: shards[i] for i in range(len(client_names))}


print("[X] Simulating clients...")
data = load_data()
clients = shard_data(data)

for (client, data) in clients.items():
    print(f"Client: {client}")

class FederatedMLP:
    @staticmethod
    def build(shape, classes=10):
        num_classes = 10
        input_shape = (28, 28, 1)
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(num_classes, activation="softmax"),
            ]
        )
        return model



fed_model = FederatedMLP()
global_model = fed_model.build(784, 10)
global_weights = global_model.get_weights()


def weights_scaling_factor(clients):
    client_names = list(clients.keys())
    bs = list(clients[client_name])[0][0].shape[0]
    global_count = sum([tf.data.experimental.cardinality(clients[client_name]).numpy() for client_name in client_names])*bs
    local_count = tf.data.experimental.cardinality(clients[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[1])
    return final_weight


def sum_scaled_weights(scaled_weight_lsit):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    
    return avg_grad


rounds = 1
for myround in range(rounds):
    # Get weights of global models
    global_weights = global_model.get_weights()

    # Get client names
    client_names = list(clients.keys())

    # Simulate multiple clients
    for client in client_names:
        fed_model = FederatedMLP()
        local_model = fed_model.build(784, 10)
        local_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )

        # Set weights of local model to that of global model
        local_model.set_weights(global_weights)

        # Scaled weights
        scaled_local_weights = list()

        # Simulate training the model locally
        data, label = zip(*clients[client])
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_train = np.expand_dims(x_train, -1)

        x_test = x_test.astype('float32') / 255.0
        x_test = np.expand_dims(x_test, -1)    

        num_classes = 10
        y_train = to_categorical(y_train, num_classes)
        y_test = to_categorical(y_test, num_classes)

        local_model.fit(x_train, y_train, batch_size=32, epochs=1, verbose=0)

        # Scale the model
        scaling_factor = weights_scaling_factor(clients, client)
        scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor) 
        scaled_local_weights.append(scaled_weights)

        K.clear_session()

    average_weights = sum_scaled_weights(scaled_local_weights)

global_model.set_weights(average_weights)

# Test model accuracy
global_acc, global_loss = test_model(x_test, y_test, global_model)