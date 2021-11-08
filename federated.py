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


from models import FederatedMLP


class Client(tf.keras.Model):
    def __init__(self, x_train, y_train, x_test, y_test):
        """Constructor for Client class"""

        super(Client, self).__init__()
        self.num_samples = x_train.shape[0]

        # Inputs and Labels
        self.train_inputs, self.train_labels = x_train, y_train
        self.test_inputs, self.test_labels = x_test, y_test

        self.num_classes = 10
        self.num_epochs = 10
        self.hidden_size = 200
        self.batch_size = self.num_samples // 30

        # Initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.1)

        # Layers (Need to be built prior to call)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu',
                                  kernel_initializer=init, bias_initializer=init),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer=init, bias_initializer=init)
        ])
        self.model.build((None, 784))

        # Optimizer
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

        # Loss
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()


    def call(self, inputs):
        return self.model(inputs)


    def loss(self, logits, labels):
        l = self.loss_function(labels, logits)
        return tf.reduce_mean(l)


    def train(self, weights):
        self.model.set_weights(weights)
        for _ in range(self.num_epochs):
            shuffled = tf.random.shuffle(range(self.num_samples))
            inputs = tf.gather(self.train_inputs, shuffled)
            labels = tf.gather(self.train_labels, shuffled)
            for i in range(self.num_samples//self.batch_size):
                start, end = i*self.batch_size, (i+1)*self.batch_size

                with tf.GradientTape() as tape:
                    l = self.loss(self.call(inputs[start:end]), labels[start:end])
                
                g = tape.gradient(l, self.trainable_variables)
                self.optimizer.apply_gradients(zip(g, self.trainable_variables))

        return self.model.get_weights()


    def test(self):
        logits = self.call(self.test_inputs)
        preds = np.argmax(logits, axis=1)
        num_correct = np.sum(self.test_labels == preds)
        return num_correct
    

class Server:

    def __init__(self, clients):
        """Constructor for the Server class"""

        self.clients = clients
        self.num_clients = len(clients)
        self.total_samples = sum(c.num_samples for c in self.clients)

        self.C_fixed = False

        self.num_rounds = 25
        self.num_classes = self.clients[0].num_classes
        self.hidden_size = self.clients[0].hidden_size

        # Initializer
        init = tf.keras.initializers.RandomNormal(stddev=0.1)

        # Layers (Needs to be built before call)
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_size, activation='relu',
                                  kernel_initializer=init, bias_initializer=init),
            tf.keras.layers.Dense(self.num_classes, activation='softmax',
                                  kernel_initializer=init, bias_initializer=init)
        ])
        self.model.build((None, 784))


    def update(self):
        for t in range(self.num_rounds):
            # Create Random Sample of Clients
            if self.C_fixed: 
                p = 1
            else:
                p = min(1, 1.1 * random.random())

            m = max(1, int(self.num_clients * p))
            sample = np.random.choice(self.clients, m)
            print('ROUND %d / %d NUM SAMPLES : %d' % (t+1, self.num_rounds, m))
            
            # Update Client Weights (Train) in Sample
            for client in sample:
                weights = client.train(self.model.get_weights())
                client.model.set_weights(weights)

            # Update Server Weights
            server_weights = self.model.get_weights()
            for client in self.clients:
                pk = client.num_samples / self.total_samples

                for sw, w in zip(server_weights, client.model.get_weights()):
                    sw = sw + w * pk

            self.model.set_weights(server_weights)

            # Testing
            acc = self.test()
            print('ROUND %d / %d UPDATE ACCURACY : %.2f %%' % (t+1, self.num_rounds, acc * 100))
            
        return acc


    def call(self, inputs):
        return self.model(inputs)

    
    def test(self):
        correct, samples = 0, 0
        for c in self.clients:
            samples += c.test_inputs.shape[0]
            correct += c.test()
        return correct / samples


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
    client_names = [i for i in range(num_clients)]
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


def test_model(x_test, y_test, model):

    # Define optimizer
    optimizer = SGD(lr=0.01)

    model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    dataset = tf.data.Dataset.from_tensor_slices((list(x_test), list(y_test))).shuffle(len(y_test)).batch(32)
    score = model.evaluate(dataset, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    return score


def shard_data_wih_noise(data, num_clients=10):
    """Shard data + add noise to shards
    num_clients: number of mobile devices
    """

    print("[X] Sharding data...")
    client_names = [i for i in range(num_clients)]
    size = len(data)//num_clients
    shards = [data[i:i + size] for i in range(0, size*num_clients, size)]

    return {client_names[i]: shards[i] if i % 2 == 0 else add_noise(shards[i]) for i in range(num_clients)}


def add_noise(shard):
  ## shard[i][j]: 1st index: ith data (out of 6000); 2nd index: 0 means data from x_train vs 1 means data from y_train

  Guassian_filter = tf.keras.layers.GaussianNoise(0.2)
  shard_guas_noise = [(Guassian_filter(shard[k][0][:, :],training=True), shard[k][1]) for k in range(len(shard))]
  return shard_guas_noise


def build_dict(y_train, y_test):
    train_dict, test_dict = {}, {}
    
    for label in range(10):
      train_dict[label] = np.argwhere(label==y_train).reshape(-1)
      test_dict[label] = np.argwhere(label==y_test).reshape(-1)

    return train_dict, test_dict

def create_clients(x_train, y_train, x_test, y_test, num_clients, malicious):

    """
      Builds a list of Non-IID clients for the demo
        - Generates dictionaries of train and test dataset indices according to labels
        - Based on dictionaries, pick two random labels with replacement
        - Pick 100 random samples from the picked labels for train and test set
    """

    # Client Data Size
    client_size = x_train.shape[0] // num_clients
    shard_size = client_size // 2

    # Build Dictionaries
    train_dict, test_dict = build_dict(y_train, y_test)

    clients = []
    for i in range(num_clients):
        if (i % 20 == 0): print("Generating %d-th Clients ..." % (i+20))
        labels = np.random.choice(10, 2)

        #create unique label list and shuffle
        unique_labels = np.unique(np.array(y_train))
        random.shuffle(unique_labels)

        #create sub label lists based on x
        sub_lab_list = [unique_labels[i:i + 1] for i in range(0, len(unique_labels), 1)]
            
        for item in sub_lab_list:
            class_data = [(image, label) for (image, label) in zip(x_train, y_train) if label in item]
        
            #decouple tuple list into seperate image and label lists
            images, labels = zip(*class_data)
            
            # train_ind = [np.random.choice(train_dict[l], shard_size) for l in labels]
            # train_ind = [i for l in train_ind for i in l]
            # train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
            # train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)
            clients.append(Client(np.asarray(images), np.asarray(labels), np.asarray(x_test), np.asarray(y_test)))

        

        # train_ind = [np.random.choice(train_dict[l], shard_size) for l in labels]
        # train_ind = [i for l in train_ind for i in l]
        # train_inputs = tf.cast(tf.gather(x_train, train_ind), tf.float32)
        # train_labels = tf.cast(tf.gather(y_train, train_ind), tf.uint8)

        # test_ind = [np.random.choice(test_dict[l], shard_size) for l in labels]
        # test_ind = [i for l in test_ind for i in l]
        # test_inputs = tf.cast(tf.gather(x_test, test_ind), tf.float32)
        # test_labels = tf.cast(tf.gather(y_test, test_ind), tf.uint8)

    
    return clients


def main():
    """Main function"""

    tf.debugging.set_log_device_placement(False)

    # Create List of Clients
    # Load and Normalize Data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype('float32') / 255.0
    x_test = x_test.reshape(10000, 784).astype('float32') / 255.0 

    # Create Dictionaries for Non-IID Data Generation

    # Create List of Clients
    clients = create_clients(x_train, y_train, x_test, y_test, 10)

    server = Server(clients)
    server.update()

    # clients = create_clients(x_train, y_train, x_test, y_test, 100)

    # # Define global modal
    # fed_model = FederatedMLP()
    # global_model = fed_model.build((28, 28, 1), 10)
    # global_weights = global_model.get_weights()

    # # Shard data
    # train_data, test_data = load_data()
    # shards = shard_data(train_data)

    # num_rounds = 25
    # for t in range(num_rounds):
    #     # Get weights of global models
    #     global_weights = global_model.get_weights()

    #     # Get client names
    #     client_names = list(shards.keys())

    #     # Define optimizer
    #     optimizer = SGD(
    #         lr=0.01,
    #         decay=0.01/comm_rounds,
    #         momentum=0.9
    #     )

    #     # Simulate multiple clients
    #     for client in client_names:
    #         fed_model = FederatedMLP()
    #         local_model = fed_model.build((28, 28, 1), 10)
    #         local_model.compile(
    #             loss='categorical_crossentropy',
    #             optimizer=optimizer,
    #             metrics=['accuracy']
    #         )

    #         print(f"[X] Training {client}")

    #         # Set weights of local model to that of global model
    #         local_model.set_weights(global_weights)

    #         # Scaled weights
    #         scaled_local_weights = list()
            
    #         """
    #         for client in clients:
    #             pk = client.num_samples / self.total_samples

    #             for sw, w in zip(server_weights, client.mode.get_weights()):
    #                 sw = sw + w * pk
            
    #         self.model.set_weights(server_weights)
    #         """

    #         # Simulate training the model locally
    #         data, label = zip(*shards[client])
    #         dataset = tf.data.Dataset.from_tensor_slices((list(data), list(label))).shuffle(len(label)).batch(32)
    #         local_model.fit(dataset, batch_size=32, epochs=1, verbose=0)

    #         # Scale the model
    #         scaling_factor = weights_scaling_factor(shards, client)
    #         scaled_weights = scale_model_weights(local_model.get_weights(), scaling_factor) 
    #         scaled_local_weights.append(scaled_weights)

    #         K.clear_session()

    #     average_weights = sum_scaled_weights(scaled_local_weights)

    # global_model.set_weights(average_weights)

    # # Test model accuracy
    # x_test, y_test = zip(*test_data)
    # score = test_model(x_test, y_test, comm_rounds, global_model)


main()