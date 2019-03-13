"""
# Reference
This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
    (https://arxiv.org/abs/1801.04381)
# Implementations:
    (https://github.com/JonathanCMitchell/mobilenet_v2_keras)
    (https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet_v2.py)
"""
import tensorflow as tf
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
                        momentum=0.999,
                        epsilon=1e-3):
    
    in_channels = keras.backend.int_shape(inputs)[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = _make_divisible(pointwise_conv_filters, 8)
    x = inputs
    
    if block_id:
        prefix = 'expanded_conv_{}'.format(block_id)
    else:
        prefix = 'expanded_conv'

    with tf.name_scope(prefix):
        with tf.name_scope('expand'):
            x = keras.layers.Conv2D(expansion * in_channels,
                                    kernel_size=1,
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='Conv2D')(x)
            x = keras.layers.BatchNormalization(epsilon=epsilon,
                                                momentum=momentum,
                                                name='BatchNorm')(x)
            x = keras.layers.ReLU(6., name='Relu6')(x)


        if stride == 2:
            x = keras.layers.ZeroPadding2D(padding=correct_pad(x, 3),
                                    name='pad')(x)
        with tf.name_scope('depthwise'):
            x = keras.layers.DepthwiseConv2D(kernel_size=3,
                                            strides=stride,
                                            activation=None,
                                            use_bias=False,
                                            padding='same' if stride == 1 else 'valid',
                                            name='Conv2D')(x)
            x = keras.layers.BatchNormalization(epsilon=epsilon,
                                                momentum=momentum,
                                                name='BatchNorm')(x)

            x = keras.layers.ReLU(6., name='Relu6')(x)
        with tf.name_scope('project'):
            x = keras.layers.Conv2D(pointwise_filters,
                                    kernel_size=1,
                                    padding='same',
                                    use_bias=False,
                                    activation=None,
                                    name='Conv2D')(x)
            x = keras.layers.BatchNormalization(epsilon=epsilon, 
                                                momentum=momentum, name='BatchNorm')(x)
        if in_channels == pointwise_filters and stride == 1:
            return keras.layers.Add(name='add')([inputs, x])
    return x

def mobilenet_v2(inputs,
                alpha,
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
    
    naming = 'MobilenetV2'
    axis  = keras.backend.image_data_format() #Channels axis
    if depthwise_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if axis ==  'channels_first':
        keras.backend.set_image_data_format('channels_last')
    first_block_filter = _make_divisible(32*alpha, 8)
    
    with tf.name_scope(naming):

        with tf.name_scope('Conv'):
            x = keras.layers.ZeroPadding2D(padding=correct_pad(inputs, kernel_size=3),
                                        name='pad')(inputs)
            x = keras.layers.Conv2D(first_block_filter,
                                    kernel_size=3,
                                    strides=(2, 2),
                                    padding='valid',
                                    use_bias=False,
                                    name='Conv2D')(x)
            x = keras.layers.BatchNormalization(
                epsilon=epsilon, momentum=momentum, name='BatchNorm')(x)                  
            x = keras.layers.ReLU(6., name='Relu6')(x)

        x = _inverted_res_block(x, filters=16, alpha=alpha, stride=1,
                                expansion=1, block_id=None,
                                momentum=momentum, epsilon=epsilon)

        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=2,
                                expansion=6, block_id=1,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=24, alpha=alpha, stride=1,
                                expansion=6, block_id=2,
                                momentum=momentum, epsilon=epsilon)

        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=2,
                                expansion=6, block_id=3,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=4,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=32, alpha=alpha, stride=1,
                                expansion=6, block_id=5,
                                momentum=momentum, epsilon=epsilon)

        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=2,
                                expansion=6, block_id=6,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=7,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=8,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=64, alpha=alpha, stride=1,
                                expansion=6, block_id=9,
                                momentum=momentum, epsilon=epsilon)

        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=10,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=11,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=96, alpha=alpha, stride=1,
                                expansion=6, block_id=12,
                                momentum=momentum, epsilon=epsilon)

        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=2,
                                expansion=6, block_id=13,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                expansion=6, block_id=14,
                                momentum=momentum, epsilon=epsilon)
        x = _inverted_res_block(x, filters=160, alpha=alpha, stride=1,
                                expansion=6, block_id=15,
                                momentum=momentum, epsilon=epsilon)

        x  = _inverted_res_block(x, filters=320, alpha=alpha, stride=1,
                                expansion=6, block_id=16,
                                momentum=momentum, epsilon=epsilon)
        # no alpha applied to last conv as stated in the paper:
        # if the width multiplier is greater than 1 we
        # increase the number of output channels
        if alpha > 1.0:
            last_block_filters = _make_divisible(1280 * alpha, 8)
        else:
            last_block_filters = 1280
        #TODO: Check for layers name (from checkpoints) to correct setup
        with tf.name_scope('Conv_1'):
            x = keras.layers.Conv2D(last_block_filters,
                                    kernel_size=1,
                                    use_bias=False)(x)
            x = keras.layers.BatchNormalization(epsilon=epsilon,
                                                momentum=momentum,
                                                name='BatchNorm')(x)
            x_final = keras.layers.ReLU(6., name='Relu6')(x)
        if pooling == 'avg':
            x_final = keras.layers.GlobalAveragePooling2D()(x_final)
        else:
            x_final = keras.layers.GlobalMaxPool2D()(x_final)
        
    return x_final