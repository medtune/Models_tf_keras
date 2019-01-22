import tensorflow.keras as keras
import os

"""Models have the same arguments: 
#include_top, weights, input_tensor,
#input_shape, pooling, classes
#Mobilenet models have two additionnal arguments:
alpha, depth_multiplier"""

models = {
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
    "vgg19": keras.applications.VGG19
}

def get_model(name):
    """Extract the desired model from 'models'
    dictionnary."""
    assert name in models.keys()
    return models.get(name)

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