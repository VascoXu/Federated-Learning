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


from models import FederatedMLP


def load_data():
    """Load data"""

    print("[X] Loading data...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.0
    x_train = np.expand_dims(x_train, -1)

    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)    

    num_classes = 10
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return (list(zip(x_train, y_train)), list(zip(x_test, y_test)))


def shard_data(data, num_clients=10):
    """Shard data"""

    print("[X] Sharding data...")
    client_names = [f"device{i}" for i in range(num_clients)]
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    return {client_names[i]: shards[i] for i in range(num_clients)}


def weights_scaling_factor(clients, client_name):
    client_names = list(clients.keys())
    bs = list(clients[client_name])[0][0].shape[0]
    global_count = sum(len(clients[client_name]) for client_name in client_names)*bs
    local_count = len(clients[client_name])*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    avg_grad = list()
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)
    
    return avg_grad


def test_model(x_test, y_test, rounds, model):

    optimizer = SGD(
        lr=0.01,
        decay=0.01 / rounds,
        momentum=0.9
    )
    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    dataset = tf.data.Dataset.from_tensor_slices((list(x_test), list(y_test))).shuffle(len(y_test)).batch(32)
    score = model.evaluate(dataset, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


### SZ added
def shard_data2(data, num_clients=10):
    """Shard data + add noise to shards
    num_clients: number of mobile devices
    """

    print("[X] Sharding data...")
    client_names = [f"device{i}" for i in range(num_clients)]
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]


    return {client_names[i]: shards[i] if i % 2 == 0 else add_noise(shards[i]) for i in range(num_clients)}

### SZ added
def add_noise(shard):
  ## shard[i][j]: 1st index: ith data (out of 6000); 2nd index: 0 means data from x_train vs 1 means data from y_train

  Guassian_filter = tf.keras.layers.GaussianNoise(0.2)
  shard_guas_noise = [(Guassian_filter(shard[k][0][:, :],training=True), shard[k][1]) for k in range(len(shard))]
  return shard_guas_noise


def main():
    """Main function"""

    # 
    tf.debugging.set_log_device_placement(True)

    # Define global modal
    fed_model = FederatedMLP()
    global_model = fed_model.build((28, 28, 1), 10)
    global_weights = global_model.get_weights()

    # Shard data
    train_data, test_data = load_data()
    shards = shard_data(train_data)

    comm_rounds = 100
    for comm_round in range(comm_rounds):
        # Get weights of global models
        global_weights = global_model.get_weights()

        # Get client names
        client_names = list(shards.keys())

        # Define optimizer
        optimizer = SGD(
            lr=0.01,
            decay=0.01/comm_rounds,
            momentum=0.9
        )

        # Simulate multiple clients
        for client in client_names:
            fed_model = FederatedMLP()
            local_model = fed_model.build((28, 28, 1), 10)
            local_model.compile(
                loss='categorical_crossentropy',
                optimizer=optimizer,
                metrics=['accuracy']
            )

            print(f"[X] Training {client}")

            # Set weights of local model to that of global model
            local_model.set_weights(global_weights)

            # Scaled weights
            scaled_local_weights = list()

            # Simulate training the model locally
            data, label = zip(*shards[client])
            dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label))).shuffle(len(label)).batch(32)
            local_model.fit(dataset, batch_size=32, epochs=1, verbose=0)

            # Scale the model
            scaling_factor = weights_scaling_factor(shards, client)
            scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor) 
            scaled_local_weights.append(scaled_weights)

            K.clear_session()

        average_weights = sum_scaled_weights(scaled_local_weights)

    global_model.set_weights(average_weights)

    # Test model accuracy
    x_test, y_test = zip(*test_data)
    score = test_model(x_test, y_test, comm_rounds, global_model)


main()