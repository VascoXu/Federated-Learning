import tensorflow as tf
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