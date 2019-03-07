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

# URL of checkpoints of famous CNN models trained on imagenet
# (.ref https://github.com/tensorflow/models/tree/master/research/slim)

checkpoints = {
    "densenet121": "https://drive.google.com/open?id=0B_fUSpodN0t0eW1sVk1aeWREaDA",
    "densenet169": "https://drive.google.com/open?id=0B_fUSpodN0t0TDB5Ti1PeTZMM2c",
    "mobilenet_1.0_224": "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz",
    "mobilenet_0.50_160": "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz",
    "mobilenet_0.25_128": "http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz",
    "mobilenetv2_1.0_224": "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz",
    "mobilenetv2_1.4_224": "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz",
    "vgg16": "http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz",
    "vgg19": "http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz"
}