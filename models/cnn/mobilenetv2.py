"""
# Reference
This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
    (https://arxiv.org/abs/1801.04381)
# Implementations:
    (https://github.com/JonathanCMitchell/mobilenet_v2_keras)
    (https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py)
"""

import tensorflow.keras as keras
from famous_cnn import correct_pad


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

def _inverted_res_block(inputs,
                        expansion,
                        stride,
                        alpha,
                        filters,
                        block_id,
                        momentum=0.999,
                        epsilon=1e-3):
    
    in_channels = keras.backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = 'block_{}_'.format(block_id)

    if block_id:
        # Expand
        x = keras.layers.Conv2D(expansion * in_channels,
                          kernel_size=1,
                          padding='same',
                          use_bias=False,
                          activation=None,
                          name=prefix + 'expand')(x)
        x = keras.layers.BatchNormalization(epsilon=epsilon,
                                      momentum=momentum,
                                      name=prefix + 'expand_BN')(x)
        x = keras.layers.ReLU(6., name=prefix + 'expand_relu')(x)
    else:
        prefix = 'expanded_conv_'

    if stride == 2:
        x = keras.layers.ZeroPadding2D(padding=correct_pad(keras.backend, x, 3),
                                 name=prefix + 'pad')(x)
    x = keras.layers.DepthwiseConv2D(kernel_size=3,
                               strides=stride,
                               activation=None,
                               use_bias=False,
                               padding='same' if stride == 1 else 'valid',
                               name=prefix + 'depthwise')(x)
    x = keras.layers.BatchNormalization(epsilon=epsilon,
                                  momentum=momentum,
                                  name=prefix + 'depthwise_BN')(x)

    x = keras.layers.ReLU(6., name=prefix + 'depthwise_relu')(x)
    x = keras.layers.Conv2D(pointwise_filters,
                      kernel_size=1,
                      padding='same',
                      use_bias=False,
                      activation=None,
                      name=prefix + 'project')(x)
    x = keras.layers.BatchNormalization(epsilon=epsilon, 
                                        momentum=momentum, name=prefix + 'project_BN')(x)
    if in_channels == pointwise_filters and stride == 1:
        return keras.layers.Add(name=prefix + 'add')([inputs, x])
    return x

def mobilenetv2()