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
import tensorflow as tf
import tensorflow.keras as keras

def _conv_block(inputs,
                filters,
                alpha,
                kernel,
                strides,
                momentum,
                epsilon):
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
                                   name='conv1_pad')(inputs)
    with tf.variable_scope('Conv2d_0', 'Conv2d_0'):
        x = keras.layers.Conv2D(filters, kernel,
                                padding='valid',
                                use_bias = False,
                                strides=strides)(x)
        #Batch Normalization of the output of the conv 2D
        x = keras.layers.BatchNormalization(axis=channel_axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name='BatchNorm')(x)
        x = keras.layers.ReLU(6., name='ReLU')(x)
    return x

def _depthwise_conv(inputs,
                    filters,
                    alpha,
                    depthwise_multiplier,
                    kernel,
                    strides,
                    block_id,
                    momentum,
                    epsilon): 
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
                                 name='Conv2d_%d_pad' % block_id)(inputs)

    with tf.variable_scope('Conv2d_%d_depthwise' % block_id):
        x = keras.layers.DepthwiseConv2D((3, 3),
                                        padding='same' if strides == (1, 1) else 'valid',
                                        depth_multiplier=depthwise_multiplier,
                                        strides=strides,
                                        use_bias = False)(x)
        x = keras.layers.BatchNormalization(axis=channel_axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name='BatchNorm')(x)
        x = keras.layers.ReLU(6., name='Relu6')(x)

    with tf.variable_scope('Conv2d_%d_pointwise' % block_id):
        x = keras.layers.Conv2D(pointwise_conv_filters, (1, 1),
                                padding='same',
                                use_bias = False,
                                strides=(1, 1))(x)
        x = keras.layers.BatchNormalization(axis=channel_axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name='BatchNorm')(x)
        x = keras.layers.ReLU(6., name='Relu6')(x)
    return x

def mobilenet_v1(inputs,
                 alpha,
                 pooling,
                 momentum,
                 epsilon,
                 num_classes,
                 activation_func,
                 classification_layers,
                 classification_type,
                 depthwise_multiplier=1):
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
    naming = 'MobilenetV1'
    axis  = keras.backend.image_data_format()
    if depthwise_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    print('\n Here is alpha:'+ str(alpha))
    if alpha not in [0.25, 0.50, 1.0]:
        raise ValueError('alpha can be one of'
                        '`0.25`, `0.50` or `1.0` only.')
    if axis ==  'channels_first':
        keras.backend.set_image_data_format('channels_last')
    with tf.variable_scope(naming, 'MobilenetV1'):
        x = _conv_block(inputs, 32, alpha, kernel=(3,3), strides=(2,2),
                        momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 64, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=1,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 128, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(2,2), block_id=2,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 128, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=3,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 256, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(2,2), block_id=4,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 256, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=5,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(2,2), block_id=6,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=7,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=8,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=9,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=10,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 512, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=11,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 1024, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=12,
                            momentum=momentum, epsilon=epsilon)

        x = _depthwise_conv(x, 1024, alpha, depthwise_multiplier,
                            kernel=(3,3), strides=(1,1), block_id=13,
                            momentum=momentum, epsilon=epsilon)
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPool2D(name='max_pool')(x)
        # Create intermediate variable representing the intermediate layers
        # of the neural networks:
        if hasattr(tf.nn, activation_func):
            #define the activation function 
            activation = getattr(tf.nn, activation_func)
        with tf.variable_scope("Logits"):
            inter = keras.layers.Flatten()(x)
            if classification_layers:
                for size in classification_layers:
                    inter = keras.layers.Dense(size, activation=activation)(inter)
            if classification_type=="multilabel":
                logits = keras.layers.Dense(num_classes, activation=tf.nn.sigmoid)(inter)
            else:
                logits = keras.layers.Dense(num_classes, activation=tf.nn.softmax)(inter)
    return logits


def slim_to_keras_namescope():
    """
    Utility function that produces a mapping btw
    old names scopes of MobilenetV1 variables
    """
    nameMapping = {}
    nameMapping['MobilenetV1/Conv2d_0/conv2d/kernel'] = 'MobilenetV1/Conv2d_0/weights'
    for i in range(1,14):
        newNameDepthwise = 'MobilenetV1/Conv2d_%d_depthwise/depthwise_conv2d/depthwise_kernel' %i
        oldNameDepthwise = 'MobilenetV1/Conv2d_%d_depthwise/depthwise_weights' %i
        newNamePointwise = 'MobilenetV1/Conv2d_%d_pointwise/conv2d/kernel' %i
        oldNamePointwise = 'MobilenetV1/Conv2d_%d_pointwise/weights' %i
        nameMapping[oldNameDepthwise] = newNameDepthwise
        nameMapping[oldNamePointwise] = newNamePointwise
    return nameMapping