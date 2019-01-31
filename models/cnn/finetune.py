"""
This module provide a way to apply transfer learning
for image classification, based on imagenet weights. Given a new set 
of classes, we want to retrain our model depending on the following cases:
retrain full CNN model, retrain some layers of CNN,
only train the classifier (Dense neural network).
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from . import base
from .base import ModelConstructor, Classifier
from utils.training import monitor

losses=["binary_crossentropy","categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_logarithmic_error",
        "cosine_proximity"]


keras_optimizers={
    "adadelta": tf.keras.optimizers.Adadelta,
    "adagrad": tf.keras.optimizers.Adagrad,
    "adam": tf.keras.optimizers.Adam,
    "adamax": tf.keras.optimizers.Adamax,
    "nadam": tf.keras.optimizers.Nadam, 
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop
    }

native_optimizers = {
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "adam": tf.train.AdamOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

regularizers = {

}



def get_loss(label_type, num_classes):
    """
    Given label_type as sparse or onehot and
    number of categories we want to train on,
    return the nounb of the appropriate loss
    """
    if label_type == "sparse":
        return "sparse_categorical_crossentropy"
    if num_classes == 2:
        return "binary_crossentropy" 
    return "categorical_crossentropy"

def get_metrics(label_type, num_classes):
    metrics = []
    if num_classes==2:
            metrics.append(tf.keras.metrics.binary_accuracy)
    else:
        if label_type=="sparse":
            metrics.append(tf.keras.metrics.sparse_categorical_accuracy)
            metrics.append(tf.keras.metrics.sparse_top_k_categorical_accuracy)
        else:
            metrics.append(tf.keras.metrics.categorical_accuracy)
            metrics.append(tf.keras.metrics.top_k_categorical_accuracy)
    return metrics

def assemble(model, classifier, 
            optimizer_noun="adam", learning_rate=1e-4,
            label_type="onehot",
            distribute=False):
    """Takes a ModelConstructor instance and a Classifier instance
    We return a Model instance
    Args:
        model: ModelConstructor instance
        classifier: Classifier instance
        optimizer_noun: (optional) optimizer noun to use. Must be in
        "optimizers" dictionnary
        learning_rate: (optional). Float number representing the exponential learning 
        rate decay
        label_type: (optional) sparse or onehot 
    Return:
        A Keras Model with defined loss, optimizer and metrics.
    """
    if distribute:
        assert optimizer_noun in native_optimizers.keys()
        optimizer = native_optimizers.get(optimizer_noun)(learning_rate)
    else:
        assert optimizer_noun in keras_optimizers.keys()
        optimizer = keras_optimizers.get(optimizer_noun)(learning_rate, decay=0.00001)
    # Using the ModelConstructor instance, we build our CNN architecture
    features = model.construct()
    # Using the features previously extracted, we also build our classifier 
    logits = classifier.construct(features)
    assembly = Model(inputs=model.input_placeholder, outputs=logits)
    # Using "get_loss" func, we retrieve the loss type (loss argument accepts a noun)
    # Using "optimizers" dict, we use retrieve our optimizer, and pass the learning rate
    # as it is required
    loss = get_loss(label_type, classifier.num_classes)
    metrics = get_metrics(label_type, classifier.num_classes)

    assembly.compile(optimizer, loss, metrics)
    return assembly


def trainable_layers():
    """Given the defined architecture within this class, we choose
        the non-trainable layers of the CNN model. By default, all
        CNN layers are trainable.
        num_trainable (Integer) represents the number of layers we want to train,
        begining from to queue of the model and going back to the first layer of the model
    """
    pass

def add_regularizer(name, weight_decay):
    """
    Given a name of a regularizer, 
    """
    pass