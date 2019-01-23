"""
This module provide a way to apply transfer learning
for image classification, based on imagenet weights. Given a new set 
of classes, we want to retrain our model depending on the following cases:
retrain full CNN model, retrain some layers of CNN,
only train the classifier (Dense neural network).
"""
import tensorflow as tf
import base
from tf.keras.layers import Dense, GlobalAveragePooling2D, Flatten
from tf.keras.models import Model

losses=["binary_crossentropy","categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_logarithmic_error",
        "cosine_proximity"]

optimizers={
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
    "gradient_descent": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

def assemble(model, classifier):
    """Takes a ModelConstructor instance and a Classifier instance
    We return a Model instance
    """
    pass