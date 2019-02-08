"""
This module provide a way to apply transfer learning
for image classification, based on imagenet weights. Given a new set 
of classes, we want to retrain our model depending on the following cases:
retrain full CNN model, retrain some layers of CNN,
only train the classifier (Dense neural network).
"""
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten
from . import base
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

def get_keras_loss(label_type, num_classes):
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

def get_keras_metrics(label_type, num_classes):
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

def assemble(model_name, input_type, 
            num_classes, learning_rate,
            classification_layers=None, 
            classification_type="multiclass",
            activation_func = tf.nn.relu,
            optimizer_noun="adam", 
            label_type="onehot",
            distribute=False):
    """
    Takes a ModelConstructor instance and a Classifier instance
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
        assert type(learning_rate) is dict
        initial_lr = learning_rate.get("initial")
        #TODO:decay_lr = get_decaylr(learning_rate, global_step)
        optimizer = native_optimizers.get(optimizer_noun)(initial_lr)
    else:
        assert optimizer_noun in keras_optimizers.keys()
        initial_lr = learning_rate.get("initial")
        optimizer = keras_optimizers.get(optimizer_noun)(initial_lr, decay=1e-10)
    # Using the ModelConstructor instance, we build our CNN architecture
    model = base.ModelConstructor(model_name, input_type)
    #take the expected image_size by the model 
    image_size = model.input_shape
    features = model.construct()
    # Using the features previously extracted, we also build our classifier 
    classifier = base.Classifier(num_classes)
    logits = classifier.construct(features)
    
    assembly = tf.keras.Model(inputs = model.input_placeholder, outputs = logits)
    merge_summaries = monitor.get_summary(assembly)
    # Using "get_loss" func, we retrieve the loss type (loss argument accepts a noun)
    # Using "optimizers" dict, we use retrieve our optimizer, and pass the learning rate
    # as it is required
    loss = get_keras_loss(label_type, classifier.num_classes)
    metrics = get_keras_metrics(label_type, classifier.num_classes)
    assembly.compile(optimizer, loss, metrics)
    return assembly, image_size, merge_summaries

def get_loss():
    """
    Return the loss to pass into the model_fn
    function (Depending on the label_type and
    classification task)
    """
    pass

def get_optimizer():
    """
    Return the optimizer function needed during
    the training. This utility function is needed
    when we create a model-fn for an estimator
    """
    pass

def get_modelfn():
    """
    This utility function provides a dynamic way
    to construct our model (cnn + classifier) depending
    on the desired convolutional neural network model,
    classifier specs,
    """
    pass

def trainable_layers():
    """
    Given the defined architecture within this class, we choose
    the non-trainable layers of the CNN model. By default, all
    CNN layers are trainable.
    num_trainable (Integer) represents the number of layers we 
    want to train, begining from to queue of the model and going 
    back to the first layer of the model
    """
    pass

def add_regularizer(name, weight_decay):
    """
    Given a name of a regularizer, 
    """
    pass

def get_decaylr(initial_lr,decay_steps,decay_factor, global_step):
    """
    Utility function to get a decayed learning rate

    Args:
        initial_lr: the initial value of learning rate
        decay_factor: factor by which the learning rate decreases over steps
        decay_steps: number of steps it has to wait before starting decreasing
        the learning rate

    Return:
        Learning rate with exponential decay
    """
    lr = tf.train.exponential_decay(learning_rate=initial_lr,
                                    global_step=global_step,
                                    decay_steps=decay_steps,
                                    decay_rate = decay_factor)
    tf.summary.scalar('learning_rate', lr)
    return lr