

import tensorflow.keras as keras
from . import densenet
from . import mobilenet
from . import mobilenetv2
from . import inception_v3
from . import inception_resnet
from . import inception_resnet_v2
from . import resnet
from . import vgg16
from . import vgg19

# This dictionnary contains the different Convolutional
# Neural Nertworks that we can invoke in order to perform
# image recognition 

architectures  = {
    "densenet121": densenet.densenet121,
    "densenet169": densenet.densenet169,
    "densenet201": densenet.densenet201,
    "densenet264": densenet.densenet264,
    "mobilenet": mobilenet.mobilenet,
    "mobilenetv2": mobilenetv2.mobilenetv2,
    "vgg16": vgg16.vgg16,
    "vgg19": vgg19.vgg19
    }

