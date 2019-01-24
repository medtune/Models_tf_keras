import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Flatten

import os

"""Models have the same arguments: 
#include_top, weights, input_tensor,
#input_shape, pooling, classes
#Mobilenet models have two additionnal arguments:
alpha, depth_multiplier"""


def get_model(name):
    famous_cnn = {
    "densenet121": keras.applications.DenseNet121,
    "densenet169": keras.applications.DenseNet169,
    "densenet201": keras.applications.DenseNet201,
    "inceptionv3": keras.applications.InceptionV3,
    "inception_resnet_v2": keras.applications.InceptionResNetV2,
    "mobilenet": keras.applications.MobileNet,
    "mobilenetv2": keras.applications.MobileNetV2,
    "nasnet_mobile": keras.applications.NASNetMobile,
    "nasnet_large": keras.applications.NASNetLarge,
    "resnet": keras.applications.ResNet50,
    "vgg16": keras.applications.VGG16,
    "vgg19": keras.applications.VGG19,
    "xception": keras.applications.Xception
    }
    """Extract the desired model from 'models'
    dictionnary."""
    assert name in famous_cnn.keys()
    return famous_cnn.get(name)

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
    return height_width.get(name)+channels.get(image_type)

def check_args(name):
    pass

def print_layers_names(name):
    """
    Utility function to quickly visualize
    each layer name
    """
    model = get_model(name)
    for i, layer in enumerate(model.layers):
        print(i, layer.name)

class ModelConstructor(object):
    """
    This class provides methods to construct the CNN model and define
    the trainable layers (By default, all layers are trainable)
    """
    def __init__(self, name, 
                input_shape=None, image_type="rgb"):

        #Name referring to the CNN model
        self.name = name 
        #Image type as "gray", "rgb", "rgba"
        self.image_type = image_type
        #Check if we got an input shape different from dafault shape
        # (ex: mnist input shape)
        if input_shape:
            self.input_shape = input_shape
        else:
            self.input_shape = get_input_shape(name, image_type)
        #If a specific input shape is given (ex: mnist input shapes)
        self.input_placeholder = keras.Input(self.input_shape) #Input tensor
        #Convolutional neural network structure
        self.architecture = get_model(self.name)
    
    def construct(self):
        """
        Return:
            Instance of Layer representing last layer of the CNN model
        """
        #Layer representing final features extracted from CNN model
        #The following line builds the entire model, and takes input
        #data as argument
        features = self.architecture(input_shape=self.input_shape,
                                    input_tensor=self.input_placeholder,
                                    include_top=False).output
        return features
    
class Classifier(object):
    """
    Multiclass classification makes the assumption that each sample is assigned to one and only one label
    Multilabel classification assigns to each sample a set of target labels
    """
    def __init__(self, num_classes,
                classification_layers=None, 
                classification_type="multiclass",
                activation_func=tf.nn.relu):
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
        self.classification_layers = classification_layers
        #It also accepts features coming from the last layer of the CNN
        self.num_classes = num_classes
        self.activation = activation_func #define the activation function 
    
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
            x = Dense(self.num_classes, activation=tf.nn.softmax)(inter)
        else:
            x = Dense(self.num_classes, activation=tf.nn.sigmoid)(inter)
        return x
