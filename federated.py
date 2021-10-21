import tensorflow as tf
import cv2


def load_data():
    """Load data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    data = list()
    labels = list()

