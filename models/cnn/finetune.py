"""
This module provide a way to apply transfer learning
for image classification, based on imagenet weights. Given a new set 
of classes, we want to retrain our model depending on the following cases:
retrain full CNN model, retrain some layers of CNN,
only train the classifier (Dense neural network).
"""

import base
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

losses={

}

optimizers={
    
}

def construct(name, num_classe, classification_layers=None):
    """
    Args:
        name: the name of the cnn model that we want 
        num_classe: number of classes that you want to train the model on
        classification
        classification_layers:(optional) intermediate layers coming before
        the Dense(num_classe) layer
    Return:
        Instance of Model class
    """
    pass

def optimize(model, loss_name, optimizer_name, learning_rate, momentum):
    """
    Args:
        model: Instance of Model class
        loss_name: loss to be used, it is included in the following list::
        []
        optimizer_name: the name of the optimizer that we want to use
        learning_rate: Desired learning rate (Hyperparameter)
        momentum: For regularization (Hyperparameter)
    Return:
        Compiled Keras model that we want to pass to Estimator class
        (from Estimator API)
    """
    assert loss_name in losses.keys()
    assert optimizer_name in optimizers.keys()
    
    pass