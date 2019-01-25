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


losses=["binary_crossentropy","categorical_crossentropy",
        "sparse_categorical_crossentropy",
        "mean_squared_logarithmic_error",
        "cosine_proximity"]

optimizers={
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "adam": tf.train.AdamOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
    "gradient_descent": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
    }


def get_loss(label_type, num_classes):
    """
    Given label_tye as sparse or onehot and
    number of categories we want to train on,
    return the nounb of the appropriate loss
    """
    if label_type == "sparse":
        return "sparse_categorical_crossentropy"
    if num_classes == 2:
        return "binary_crossentropy" 
    return "categorical_crossentropy"
def assemble(model, classifier, 
            optimizer_noun="adam", learning_rate=1e-3,
            label_type="onehot"):
    """Takes a ModelConstructor instance and a Classifier instance
    We return a Model instance
    Args:
        model: ModelConstructor instance
        classifier: Classifier instance
        optimizer_noun: (optional) optimizer noun to use. Must be in
        "optimizers" dictionnary
        learning_rate: (optional). float number representing the learning 
        rate
        label_type: (optional) sparse or onehot 
    Return:
        A Keras Model with defined loss, optimizer and metrics.
    """
    assert optimizer_noun in optimizers.keys()
    # Using the ModelConstructor instance, we build our CNN architecture
    features = model.construct()
    # Using the features previously extracted, we also build our classifier 
    logits = classifier.construct(features)
    assembly = Model(inputs=model.input_placeholder, outputs=logits)
    # Using "get_loss" func, we retrieve the loss type (loss argument accepts a noun)
    # Using "optimizers" dict, we use retrieve our optimizer, and pass the learning rate
    # as it is required
    assembly_args = {
                    "loss": get_loss(label_type, classifier.num_classes),
                    "optimizer": optimizers.get(optimizer_noun)(learning_rate),
                    "metrics": ["accuracy"]
                    }
    assembly.compile(assembly_args.get("optimizer"),
                    assembly_args.get("loss"),
                    assembly_args.get("metrics"))
    return assembly

def assemble_gpus(model, classifier, 
            optimizer_noun="adam", learning_rate=1e-3,
            label_type="onehot"):
    """Takes a ModelConstructor instance and a Classifier instance
    We return a Model instance
    Args:
        model: ModelConstructor instance
        classifier: Classifier instance
        optimizer_noun: (optional) optimizer noun to use. Must be in
        "optimizers" dictionnary
        learning_rate: (optional). float number representing the learning 
        rate
        label_type: (optional) sparse or onehot 
    Return:
        A Keras Model with defined loss, optimizer and metrics.
    """
    assert optimizer_noun in optimizers.keys()
    # Using the ModelConstructor instance, we build our CNN architecture
    with tf.device('/gpu:0'):
        features = model.construct()
    # Using the features previously extracted, we also build our classifier 
    with tf.device('/gpu:1'):
        logits = classifier.construct(features)
    assembly = Model(inputs=model.input_placeholder, outputs=logits)
    # Using "get_loss" func, we retrieve the loss type (loss argument accepts a noun)
    # Using "optimizers" dict, we use retrieve our optimizer, and pass the learning rate
    # as it is required
    assembly_args = {
                    "loss": get_loss(label_type, classifier.num_classes),
                    "optimizer": optimizers.get(optimizer_noun)(learning_rate),
                    "metrics": ["accuracy"]
                    }
    assembly.compile(assembly_args.get("optimizer"),
                    assembly_args.get("loss"),
                    assembly_args.get("metrics"))
    return assembly

def trainable_layers():
    """Given the defined architecture within this class, we choose
        the non-trainable layers of the CNN model. By default, all
        CNN layers are trainable.
        num_trainable (Integer) represents the number of layers we want to train,
        begining from to queue of the model and going back to the first layer of the model
    """
    pass