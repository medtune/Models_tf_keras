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

default_names = ['densenet_121', 'densenet_169', 'densenet_201',
                'densenet_264', 'mobilenet_v1', 'mobilenet_v2', 
                'vgg_16', 'vgg_19']

# Dict to retrieve naming that we pass to warm-start settings
naming_mapping = {
    'densenet_121': 'densenet121',
    'densenet_169': 'densenet169',
    'densenet_201': 'densenet201',
    'densenet_264': 'densenet264',
    'mobilenet_v1': 'MobilenetV1',
    'mobilenet_v2': 'MobilenetV2',
    'vgg_16': 'vgg_16',
    'vgg_19': 'vgg_19'
}

# Dict to retrieve CNN architecture
architectures  = {
    'densenet_121': densenet.densenet_121,
    'densenet_169': densenet.densenet_169,
    'densenet_201': densenet.densenet_201,
    'densenet_264': densenet.densenet_264,
    'mobilenet_v1': mobilenet.mobilenet_v1,
    'mobilenet_v2': mobilenetv2.mobilenet_v2,
    'vgg_16': vgg16.vgg_16,
    'vgg_19': vgg19.vgg_19
}

# URL of checkpoints of famous CNN models trained on imagenet
# (.ref https://github.com/tensorflow/models/tree/master/research/slim)

checkpoints = {
    'densenet_121': 'https://drive.google.com/uc?authuser=0&id=0B_fUSpodN0t0eW1sVk1aeWREaDA&export=download',
    'densenet_169': 'https://drive.google.com/uc?authuser=0&id=0B_fUSpodN0t0TDB5Ti1PeTZMM2c&export=download',
    'mobilenet_v1_1.0': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz',
    'mobilenet_v1_0.5': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz',
    'mobilenet_v1_0.25': 'http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz',
    'mobilenet_v2_1.0': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz',
    'mobilenet_v2_1.4': 'https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz',
    'vgg_16': 'http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz',
    'vgg_19': 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'
}

# Slim models contains their own variable_scope. It differs from tf.layers & tf.keras
# especially for Conv2d/ DepthwiseConv2D. We have to write the mapping from
# new variables scopes to old variables scopes
var_name_to_prev_var_name = {
    'densenet_121': densenet.slim_to_keras_namescope(densenet.DENSENET121_BLOCKS),

    'densenet_169': densenet.slim_to_keras_namescope(densenet.DENSENET169_BLOCKS),

    'densenet_201': densenet.slim_to_keras_namescope(densenet.DENSENET201_BLOCKS),

    'densenet_264': densenet.slim_to_keras_namescope(densenet.DENSENET264_BLOCKS),

    'mobilenet_v1': mobilenet.slim_to_keras_namescope(),

    'mobilenet_v2':mobilenetv2.slim_to_keras_namescope(),

    'vgg_16': {},

    'vgg_19': {},
}