import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten
from . import famous_cnn
import os

"""Models have the same arguments: 
#include_top, weights, input_tensor,
#input_shape, pooling, classes
#Mobilenet models have two additionnal arguments:
alpha, depth_multiplier"""

native_optimizers = {
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "adam": tf.train.AdamOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

def get_input_shape(name, image_type):
    """
    Based on the model's name and the image type
    (grayscale, RGB, RGBA), the function returns
    a tuple of 3 dims representing the input shape
    """
    #Dictionnary that contain the default
    #height and width for each model name
    height_width = {
    "densenet121": (224,224),
    "densenet169": (224,224),
    "densenet201": (224,224),
    "densenet264": (224,224),
    "inceptionv3": (299,299),
    "inception_resnet_v2": (299,299),
    "mobilenet": (224,224),
    "mobilenetv2": (224,224),
    "nasnet_mobile": (224,224),
    "nasnet_large": (331,331),
    "resnet": (224,224),
    "vgg16": (224,224),
    "vgg19": (224,224),
    "xception": (299,299)
    }
    channels = {
        "gray": (1,),
        "rgb": (3,),
        "rgba": (4,)
    }
    assert name in height_width.keys()
    shape = height_width.get(name) + channels.get(image_type)
    return shape

class Classifier(object):
    """
    Multiclass classification makes the assumption that each sample is assigned to one and only one label
    Multilabel classification assigns to each sample a set of target labels
    """
    def __init__(self, num_classes,
                classification_type,
                classification_layers, 
                activation_func):
        """
        Args:
            features: A Layer instance representing the last layer of a CNN model
            classification_type: "multiclass" or "multilabel". String
            classification_layers: A list. Each element is an integer representing
                                    the number of neurons in each layer.
            num_classes: A integer. Number of classes (categories) 
                            that you want to train your model into.
        """
        #Classification_type refers to the definition above
        self.classification_type = classification_type
        assert type(classification_layers) is list
        self.classification_layers = classification_layers
        #It also accepts features coming from the last layer of the CNN
        self.num_classes = num_classes
        # Check if module has this func
        if hasattr(tf.nn, activation_func):
            #define the activation function 
            self.activation = getattr(tf.nn, activation_func) 
    
    def construct(self, features):
        """
        We construct a Neural Network with the number of layers equivalent to
        len classification_layers list.
        Args:
            features: features layer 
        """
        #Create intermediate variable representing the intermediate layers
        #of the neural networks:
        inter = Flatten()(features)
        if self.classification_layers:
            for size in self.classification_layers:
                inter = Dense(size, activation=self.activation)(inter)
        if self.classification_type=="multiclass":
            logits = Dense(self.num_classes, activation=tf.nn.softmax)(inter)
        else:
            logits = Dense(self.num_classes, activation=tf.nn.sigmoid)(inter)
        return logits

#TODO: Instead of Downloading keras weights (.h5 format)
# we download checkpoint from slim repository:
# ex:inceptionv1 (http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz)
class AssembleModel():

    def __init__(self, model_name, input_type,
    num_classes, classification_type="multiclass",
    classification_layers=[], activation_func="relu"):
        super(AssembleModel, self).__init__()
        # Get the CNN model base on the given name
        self.cnn_model = famous_cnn.architectures.get(model_name)
        # Define the image type : 'Grayscale' or 'RGB'
        self.input_type = input_type
        # Define the classifier as an instance of Classifier:
        self.classifier = Classifier(num_classes,
                                    classification_type,
                                    classification_layers,
                                    activation_func)

    def call(self, inputs):
        features = self.cnn_model(inputs)
        logits = self.classifier.construct(features)
        return logits

    def get_hyperparams(self):
        """
        Using stdio inputs, we ask the user to define
        a value for each hyperparameter of the model, depending on the
        CNN model that is used (epsilon, batch_norm, alpha for mobilenet &
        mobilenetv2)
        # Return : 
            A dict containing the value of each hyperparameter
        """
        pass
    
    def get_loss_function(self):
        """
        Depending on the number of classes and the classification type,
        we return the loss function that we will pass into
        model_fn
        """
        pass




def  model_fn(features, labels, mode, params):
    """
    Model_fn function that we will pass to the 
    estimator.
    # Arguments :
        - features : batch of images as input
        - labels : batch of labels (true prediction)
        - mode : train, eval or predict. (ref to Estimator docs)
        - params : from the configuration file, we take the following params as
        dicts :
            * image_type : input type ('rgb' or 'gray')
            * model.name : model name to pass to the AssembleModel
            * num_classes : number of classes 
            * classification : includes the classification type and the layers to construct
            * optimizer_noun : the optimizer function we want to use during training
            * learning_rate : initial, decay_factor and before_decay to define the
            a decayed learning rate
            
    # Return : 
        model_fn function
    """
    model = AssembleModel(params["model_name"], params["image_type"],
                        params["num_classes"], params["classification_type"],
                        params["activation_func"])
    logits = model.call(features["image"])