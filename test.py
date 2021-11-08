import numpy as np
import pandas as pd
import random

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
from tensorflow.keras import layers


def MNIST_IID(data, num_clients):
    """Shard data"""

    # Shuffle data

    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    return {i: shards[i] for i in range(num_clients)}


def MNIST_NONIID():
    pass


class Client(tf.keras.Model):
    def __init__(self, x_train, y_train, malicious):
        """Constructor for Client class"""
        super(Client, self).__init__()

        # Num samples
        self.num_samples = x_train.shape[0]

        # Inputs and Labels
        self.train_inputs, self.train_labels = x_train, y_train

        # Configurations
        self.num_classes = 10
        self.num_epochs = 20
        self.hidden_size = 200
        self.batch_size = self.num_samples // 30
        self.malicious = malicious

        # Initializer (for random weights of Keras layers)
        # init = tf.keras.initializers.RandomNormal(stddev=0.1)
        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.hidden_size, activation='relu',
        #                           kernel_initializer=init, bias_initializer=init),
        #     tf.keras.layers.Dense(self.num_classes, activation='softmax',
        #                           kernel_initializer=init, bias_initializer=init)
        # ])
        # self.model.build((None, 784))


        # Layers
        input_shape = (28, 28, 1)
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

        # Loss
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()


    def call(self, inputs):
        return self.model(inputs)


    def loss(self, labels, logits):
        l = self.loss_function(labels, logits)
        return tf.reduce_mean(l)


    def train(self, weights):
        """Train a model"""

        # Set weights to that of global model
        self.model.set_weights(weights)

        # Shuffle data
        shuffled = tf.random.shuffle(range(self.num_samples))
        inputs = tf.gather(self.train_inputs, shuffled)
        labels = tf.gather(self.train_labels, shuffled)
        labels = to_categorical(labels, self.num_classes)

        # Define optimizer and compile
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)
        self.model.compile(loss="categorical_crossentropy", optimizer=self.optimizer, metrics=["accuracy"])
        self.model.fit(inputs, labels, epochs=self.num_epochs, verbose=0)

        # # Iterate through epochs
        # for _ in range(self.num_epochs):
        #     # Shuffle data
        #     shuffled = tf.random.shuffle(range(self.num_samples))
        #     inputs = tf.gather(self.train_inputs, shuffled)
        #     labels = tf.gather(self.train_labels, shuffled)

        #     # Iterate over batches of the dataset
        #     for i in range(self.num_samples//self.batch_size):
        #         start, end = i*self.batch_size, (i+1)*self.batch_size

        #         with tf.GradientTape() as tape:
        #             # Run the forward pass of the layer.
        #             logits = self.model(inputs[start:end], training=True)  # Logits for this minibatch

        #             # Compute the loss value for this minibatch.
        #             loss_value = self.loss(labels[start:end], logits)

        #         grads = tape.gradient(loss_value, self.trainable_variables)
        #         self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # return self.model.get_weights()


    def test(self):
        logits = self.call(self.test_inputs)
        preds = np.argmax(logits, axis=1)
        num_correct = np.sum(self.test_labels == preds)
        return num_correct
    

class Server:
    def __init__(self, clients, x_test, y_test):
        """Constructor for the Server class"""

        self.clients = clients
        self.num_clients = len(clients)
        self.total_samples = sum(c.num_samples for c in self.clients)

        self.C_fixed = False

        self.num_rounds = 100
        self.num_classes = self.clients[0].num_classes
        self.hidden_size = self.clients[0].hidden_size

        self.num_samples = x_test.shape[0]
        shuffled = tf.random.shuffle(range(self.num_samples))
        self.test_inputs = tf.gather(x_test, shuffled)
        self.test_labels = tf.gather(y_test, shuffled)

        # Initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.1)

        # Layers (Needs to be built before call)
        input_shape = (28, 28, 1)
        self.model = tf.keras.Sequential(
            [
                tf.keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

        # self.model = tf.keras.Sequential([
        #     tf.keras.layers.Dense(self.hidden_size, activation='relu',
        #                         kernel_initializer=init, bias_initializer=init),
        #     tf.keras.layers.Dense(self.hidden_size, activation='relu',
        #                         kernel_initializer=init, bias_initializer=init),                                                                    
        #     tf.keras.layers.Dense(self.num_classes, activation='softmax',
        #                         kernel_initializer=init, bias_initializer=init)
        # ])
        # self.model.build((None, 784))


    def update(self):
        for t in range(self.num_rounds):
            # Create random sample of clients
            if self.C_fixed: 
                p = 1
            else:
                p = min(1, 1.1 * random.random())
            # m = max(1, int(self.num_clients * p))
            m = max(1, int(self.num_clients * 0.1))
            sample = np.random.choice(self.clients, m, replace=False)

            print(f"ROUND: {t+1} / {self.num_rounds} NUM CLIENTS: {m}")
            
            # Update clients weights
            for client in sample:
                client.train(self.model.get_weights())

            # Scale weights
            scaled_weights = []
            for client in self.clients:
                pk = client.num_samples / self.total_samples
                cw = client.get_weights()
                scaled_weights.append([pk*w for w in cw])
                
            # Sum weights
            sum_weights = []
            for wt in zip(*scaled_weights):
                layer_sum = tf.math.reduce_sum(wt, axis=0)
                sum_weights.append(layer_sum)
            
            # Updated server weights with FedAvg
            self.model.set_weights(sum_weights)

            # Testing
            acc = self.test()
            print(f"ROUND {t+1} / {self.num_rounds} UPDATE ACCURACY: {acc*100:.2f}")
            
        self.model.save_weights('./weights')
    
        return acc


    def call(self, inputs):
        return self.model(inputs)

    
    def test(self):
        """Test accuracy of global model"""
        logits = self.call(self.test_inputs)
        preds = np.argmax(logits, axis=1)
        num_correct = np.sum(self.test_labels == preds)
        return num_correct / self.num_samples

        # correct, samples = 0, 0
        # for c in self.clients:
        #     samples += c.test_inputs.shape[0]
        #     correct += c.test()
        # return correct / samples


def build_dict(y_train):
    """Build dictionary of clients"""
    train_dict, test_dict = {}, {}
    
    for label in range(10):
      train_dict[label] = np.argwhere(label==y_train).reshape(-1)
    #   test_dict[label] = np.argwhere(label==y_test).reshape(-1)

    return train_dict


def create_clients(x_train, y_train, num_clients=100, num_malicious=0):
    """"""

    # Client Data Size
    client_size = x_train.shape[0] // num_clients # MNIST: 600 (for 100 clients)
    shard_size = client_size // 2 # MNIST: 300 (for 100 clients)

    # clients = []
    # for i in range(num_clients):
    #     train_ind = [np.random.choice(x_train.reshape(-1), client_size)]
    #     train_ind = [i for l in train_ind for i in l]
    #     train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
    #     train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)

    #     test_ind = [np.random.choice(x_test.reshape(-1)
    #     , client_size)]
    #     test_ind = [i for l in train_ind for i in l]
    #     test_inputs = tf.cast(tf.gather(x_test, train_ind), tf.float32)
    #     test_labels = tf.cast(tf.gather(y_test, train_ind), tf.uint8)

    #     clients.append(Client(train_inputs, train_labels, test_inputs, test_labels))

    # Build dictionaries (shards)
    train_dict = build_dict(y_train)

    # clients = []
    # for i in range(num_clients):
    #     labels = np.random.choice(10, 2)
    #     if (i % 20 == 0): print(f"Client {i} contains shards: {labels}")

    #     train_ind = [np.random.choice(train_dict[l], shard_size) for l in labels]
    #     train_ind = [i for l in train_ind for i in l]
    #     train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
    #     train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)

    #     test_ind = [np.random.choice(test_dict[l], shard_size) for l in labels]
    #     test_ind = [i for l in test_ind for i in l]
    #     test_inputs = tf.cast(tf.gather(x_test, test_ind), tf.float32)
    #     test_labels = tf.cast(tf.gather(y_test, test_ind), tf.uint8)

    #     clients.append(Client(train_inputs, train_labels, False))

    clients = []
    # shuffled = tf.random.shuffle(range(x_train.shape[0]))
    # train_inputs = tf.gather(x_train, shuffled)
    # train_labels = tf.gather(y_train, shuffled)


    # input_shards = [train_inputs[i:i + client_size] for i in range(0, client_size*num_clients, client_size)]
    # label_shards = [train_labels[i:i + client_size] for i in range(0, client_size*num_clients, client_size)]
    for i in range(num_clients):
        train_ind = np.random.randint(x_train.shape[0], size=client_size)
        train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
        train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)

        clients.append(Client(train_inputs, train_labels, malicious=False))
    
    return clients


def main():    
    # Debug
    tf.debugging.set_log_device_placement(False)

    # Load and normalize data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    # x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    # x_test = x_test.reshape(10000, 784).astype('float32') / 255.0 
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    # Create list of clients
    clients = create_clients(x_train, y_train, num_malicious=1)

    server = Server(clients, x_test, y_test)
    server.update()

main()