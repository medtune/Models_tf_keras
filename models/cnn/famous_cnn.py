

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

def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling.
    # Arguments
        input_size: An integer or tuple/list of 2 integers.
        kernel_size: An integer or tuple/list of 2 integers.
    # Returns
        A tuple.
    """
    img_dim = 2 if keras.backend.image_data_format() == 'channels_first' else 1
    input_size = keras.backend.int_shape(inputs)[img_dim:(img_dim + 2)]

    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)

    if input_size[0] is None:
        adjust = (1, 1)
    else:
        adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)

    correct = (kernel_size[0] // 2, kernel_size[1] // 2)

    return ((correct[0] - adjust[0], correct[0]),
            (correct[1] - adjust[1], correct[1]))