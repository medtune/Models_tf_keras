"""
The weights for all 16 models are obtained and translated
from TensorFlow checkpoints found at :
    (https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.md)
# Reference
- [MobileNets: Efficient Convolutional Neural Networks for
   Mobile Vision Applications](https://arxiv.org/pdf/1704.04861.pdf))
# Keras Implementation:
    (https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py)
 The following table describes the size and accuracy of the 100% MobileNet
on size 224 x 224:
----------------------------------------------------------------------------
Width Multiplier (alpha) | ImageNet Acc |  Multiply-Adds (M) |  Params (M)
----------------------------------------------------------------------------
|   1.0 MobileNet-224    |    70.6 %     |        529        |     4.2     |
|   0.75 MobileNet-224   |    68.4 %     |        325        |     2.6     |
|   0.50 MobileNet-224   |    63.7 %     |        149        |     1.3     |
|   0.25 MobileNet-224   |    50.6 %     |        41         |     0.5     |
----------------------------------------------------------------------------
The following table describes the performance of
the 100 % MobileNet on various input sizes:
------------------------------------------------------------------------
      Resolution      | ImageNet Acc | Multiply-Adds (M) | Params (M)
------------------------------------------------------------------------
|  1.0 MobileNet-224  |    70.6 %    |        529        |     4.2     |
|  1.0 MobileNet-192  |    69.1 %    |        529        |     4.2     |
|  1.0 MobileNet-160  |    67.2 %    |        529        |     4.2     |
|  1.0 MobileNet-128  |    64.4 %    |        529        |     4.2     |
------------------------------------------------------------------------
"""
import tensorflow.keras as keras

def _conv_block(inputs,
                filters,
                alpha,
                kernel,
                strides,
                momentum,
                epsilon,
                name_prefix):
    """
    Adds a convolutional layer to the architecture
    
    Args:
        inputs: input tensor of shape (batch, height, width, channels)
        if channels_last or (batch, channels, height, with) if channels first
        filters : Number of output filters (How many filters are involved in
        the convolution)
        alpha: alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        kernel: The size of the convolution that is used
        strides: Strides of the convolution along height and width
    """
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    # tuple of 2 tuples of 2 ints: interpreted as ((top_pad, bottom_pad), (left_pad, right_pad))
    x = keras.layers.ZeroPadding2D(padding=((0, 1), (0, 1)),
                                name=name_prefix+'conv1_pad')(inputs)
    x = keras.layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name=name_prefix+'conv1')(x)
    #Batch Normalization of the output of the conv 2D
    x = keras.layers.BatchNormalization(axis=channel_axis,
                                        momentum=momentum,
                                        epsilon=epsilon,
                                        name=name_prefix+'conv1_bn')(x)
    x = keras.layers.ReLU(6., name=name_prefix+'conv1_relu')(x)
    return x


def _depthwise_conv(inputs,
                    filters,
                    alpha,
                    depthwise_multiplier,
                    kernel,
                    strides,
                    block_id,
                    momentum,
                    epsilon,
                    name_prefix): 
    """
    Args:
        filters: pointwise convolutional filters 
        alpha: controls the width of the network.
            - If `alpha` < 1.0, proportionally decreases the number
                of filters in each layer.
            - If `alpha` > 1.0, proportionally increases the number
                of filters in each layer.
            - If `alpha` = 1, default number of filters from the paper
                 are used at each layer.
        depth_multiplier: depth multiplier for depthwise convolution. This
        is called the resolution multiplier in the MobileNet paper.
    """
    channel_axis = 1 if keras.backend.image_data_format() == 'channels_first' else -1
    # Pointwise convolution:we determine the number of filter with alpha
    pointwise_conv_filters = int(filters * alpha)
    if strides == (1, 1):
        x = inputs
    else:
        x = keras.layers.ZeroPadding2D(((0, 1), (0, 1)),
                                 name=name_prefix+'conv_pad_%d' % block_id)(inputs)
    x = keras.layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depthwise_multiplier,
                               strides=strides,
                               use_bias=False,
                               name=name_prefix+'conv_dw_%d' % block_id)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis,
                                        momentum=momentum,
                                        epsilon=epsilon,
                                        name=name_prefix+'conv_dw_%d_bn' % block_id)(x)
    x = keras.layers.ReLU(6., name=name_prefix+'conv_dw_%d_relu' % block_id)(x)

    x = keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name=name_prefix+'conv_pw_%d' % block_id)(x)
    x = keras.layers.BatchNormalization(axis=channel_axis,
                                        momentum=momentum,
                                        epsilon=epsilon,
                                        name=name_prefix+'conv_pw_%d_bn' % block_id)(x)
    x = keras.layers.RELU(6., name=name_prefix+'conv_pw_%d_relu')(x)
    return x

def mobilenet_v1(inputs,
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
        output features from Mobilenet model
    """
    naming = 'MobilenetV1_'
    axis  = keras.backend.image_data_format()
    if depthwise_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    if alpha not in [0.25, 0.50, 0.75, 1.0]:
        raise ValueError('alpha can be one of'
                        '`0.25`, `0.50`, `0.75` or `1.0` only.')
    if axis ==  'channels_first':
        keras.backend.set_image_data_format('channels_last')
    x = _conv_block(inputs, 32, alpha, kernel=(3,3), strides=(2,2),
                    momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 64, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=1,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 128, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(2,2), block_id=2,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 128, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=3,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 256, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(2,2), block_id=4,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 256, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=5,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(2,2), block_id=6,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=7,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=8,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=9,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=10,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=11,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 1024, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=12,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)

    x = _depthwise_conv(x, 1024, alpha, depthwise_multiplier,
                        kernel=(3,3), strides=(1,1), block_id=13,
                        momentum=momentum, epsilon=epsilon, name_prefix=naming)
    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D(name=naming+'avg_pool')(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPool2D(name=naming+'max_pool')(x)
    return x