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
                        name_prefix,
                        momentum=0.999,
                        epsilon=1e-3):
    
    in_channels = keras.backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    prefix = name_prefix + 'block_{}_'.format(block_id)

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
        x = keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
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

def mobilenetv2(inputs,
                alpha=1.0,
                depthwise_multiplier=1,
                pooling=None,
                momentum=0.99,
                epsilon=0.001):

    """
    Args:
        alpha: alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
        is called the resolution multiplier in the MobileNet paper.
    
    Returns:
        output features from MobilenetV2 
    """
    
    naming = 'mobilenetv2'
    axis  = keras.backend.image_data_format() #Channels axis
    if alpha not in [0.35, 0.50, 0.75, 1.0, 1.3, 1.4]:
        raise ValueError('alpha can be one of `0.35`, `0.50`, `0.75`, '
                            '`1.0`, `1.3` or `1.4` only.')
    if depthwise_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if axis ==  'channels_first':
        keras.backend.set_image_data_format('channels_last')
    
    first_block_filter = _make_divisible(32*alpha, 8)
    x = keras.layers.ZeroPadding2D(padding=correct_pad(inputs, kernel_size=3),
                                  name=naming+'conv1_pad')(inputs)
    x1 = keras.layers.Conv2D(first_block_filter,
                      kernel_size=3,
                      strides=(2, 2),
                      padding='valid',
                      use_bias=False,
                      name=naming+'conv1')(x)
    x2 = keras.layers.BatchNormalization(
        epsilon=epsilon, momentum=momentum, name=naming+'conv1_bn')(x1)                  
    x3 = keras.layers.ReLU(6., name=naming+'conv1_relu')(x2)
    x4 = _inverted_res_block(x3, filters=16, alpha=alpha, stride=1,
                            expansion=1, block_id=0, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x5 = _inverted_res_block(x4, filters=24, alpha=alpha, stride=2,
                            expansion=6, block_id=1, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x6 = _inverted_res_block(x5, filters=24, alpha=alpha, stride=1,
                            expansion=6, block_id=2, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x7 = _inverted_res_block(x6, filters=32, alpha=alpha, stride=2,
                            expansion=6, block_id=3, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x8 = _inverted_res_block(x7, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=4, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x9 = _inverted_res_block(x8, filters=32, alpha=alpha, stride=1,
                            expansion=6, block_id=5, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x10 = _inverted_res_block(x9, filters=64, alpha=alpha, stride=2,
                            expansion=6, block_id=6, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x11 = _inverted_res_block(x10, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=7, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x12 = _inverted_res_block(x11, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=8, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x13 = _inverted_res_block(x12, filters=64, alpha=alpha, stride=1,
                            expansion=6, block_id=9, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x14 = _inverted_res_block(x13, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=10, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x15 = _inverted_res_block(x14, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=11, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x16 = _inverted_res_block(x15, filters=96, alpha=alpha, stride=1,
                            expansion=6, block_id=12, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x17 = _inverted_res_block(x16, filters=160, alpha=alpha, stride=2,
                            expansion=6, block_id=13, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x18 = _inverted_res_block(x17, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=14, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    x19 = _inverted_res_block(x18, filters=160, alpha=alpha, stride=1,
                            expansion=6, block_id=15, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)

    x20  = _inverted_res_block(x19, filters=320, alpha=alpha, stride=1,
                            expansion=6, block_id=16, name_prefix=naming,
                            momentum=momentum, epsilon=epsilon)
    # no alpha applied to last conv as stated in the paper:
    # if the width multiplier is greater than 1 we
    # increase the number of output channels
    if alpha > 1.0:
        last_block_filters = _make_divisible(1280 * alpha, 8)
    else:
        last_block_filters = 1280
    #TODO: Check for layers name (from checkpoints) to correct setup
    x21 = keras.layers.Conv2D(last_block_filters,
                      kernel_size=1,
                      use_bias=False,
                      name=naming+'conv_1')(x20)
    x22 = keras.layers.BatchNormalization(epsilon=epsilon,
                                  momentum=momentum,
                                  name=naming+'conv_1_bn')(x21)
    x_final = keras.layers.ReLU(6., name=naming+'out_relu')(x22)
    if pooling == 'avg':
        x_final = keras.layers.GlobalAveragePooling2D()(x_final)
    else:
        x_final = keras.layers.GlobalMaxPool2D()(x_final)
    
    return x_final