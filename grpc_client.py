from __future__ import print_function

from io import BytesIO

import grpc
import federated_pb2
import federated_pb2_grpc

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


class FederatedMLP:
    @staticmethod
    def build(shape, classes=10):
        model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=shape),
                tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(classes, activation="softmax"),
            ]
        )
        return model


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
    # global_count = sum([tf.data.experimental.cardinality(clients[client_name]).numpy() for client_name in client_names])*bs
    # local_count = tf.data.experimental.cardinality(clients[client_name]).numpy()*bs
    return local_count/global_count


def scale_model_weights(weight, scalar):
    weight_final = list()
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


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """Serialize NumPy ndarray to bytes."""
    bytes_io = BytesIO()
    np.save(bytes_io, ndarray, allow_pickle=False)
    return bytes_io.getvalue()


def make_message(message):
    return federated_pb2.Empty(
        message=message
    )


def generate_messages():
    messages = [
        make_message("One"),
        make_message("Two"),
        make_message("Three"),
        make_message("Four"),
        make_message("Five")
    ]
    for msg in messages:
        print(f"Sending server message: {msg.message}")
        yield msg


def send_message(stub):
    responses = stub.GetServerResponse(generate_messages())
    for response in responses:
        print(f"Message received from server: {response.message}")

class Client:
    def __init__(self):
        pass

    def start_client(self, server_address):
        # Define global modal
        fed_model = FederatedMLP()
        global_model = fed_model.build((28, 28, 1), 10)
        global_weights = global_model.get_weights()

        # Shard data
        train_data, test_data = load_data()
        shards = shard_data(train_data)

        comm_rounds = 1
        for comm_round in range(comm_rounds):
            # Get weights of global models
            global_weights = global_model.get_weights()

            # Define optimizer
            optimizer = SGD(
                lr=0.01,
                decay=0.01/comm_rounds,
                momentum=0.9
            )

            # Simulate multiple clients
            client = "device0"
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

            # Convert weights to nparray
            scaled_weights = [ndarray_to_bytes(ndarray) for ndarray in scaled_weights]

            K.clear_session()
       
            with grpc.insecure_channel(server_address) as channel:
                stub = federated_pb2_grpc.FederatedStub(channel)
                model = federated_pb2.Model()
                model.weights = scaled_weights
                response = stub.Join(model)

