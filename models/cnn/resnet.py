"""ResNet, ResNetV2, and ResNeXt models for Keras.
# Reference papers
- [Deep Residual Learning for Image Recognition]: ResNet
    (https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
- [Identity Mappings in Deep Residual Networks] : ResnetV2
    (https://arxiv.org/abs/1603.05027) (ECCV 2016)
- [Aggregated Residual Transformations for Deep Neural Networks] : ResNeXt
    (https://arxiv.org/abs/1611.05431) (CVPR 2017)
# Reference implementations
- [TensorNets]
    (https://github.com/taehoonlee/tensornets/blob/master/tensornets/resnets.py)
- [Caffe ResNet]
    (https://github.com/KaimingHe/deep-residual-networks/tree/master/prototxt)
- [Torch ResNetV2]
    (https://github.com/facebook/fb.resnet.torch/blob/master/models/preresnet.lua)
- [Torch ResNeXt]
    (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
-  [keras_applications]
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet_common.py
"""
import tensorflow.keras as keras

def _identity_block(input_tensor, kernel_size, id, name_prefix):
    """
    Arguments:
        - input tensor : tensor from the previous layer
        - kernel_size 
        - id : id of the current block, useful for wreiting layer
        scope
        - name_prefix : name_prefix to add to layer scope
    """
    pass




def _conv_block():
    pass

def resNet(inputs,
            pooling=None,
            activation="relu",
            momentum=0.99,
            epsilon=0.001):

    axis = 3 if keras.backend.image_data_format() == 'channels_last' else 1
