"""
DenseNet models for Keras.
# Reference paper
- [Densely Connected Convolutional Networks]
    (https://arxiv.org/abs/1608.06993) (CVPR 2017 Best Paper Award)
# Reference implementation
- [TensorNets]
    (https://github.com/taehoonlee/tensornets/blob/master/tensornets/densenets.py)
-keras_applications:
    (https://github.com/keras-team/keras-applications/blob/master/keras_applications/densenet.py)
"""
import tensorflow as tf
import tensorflow.keras as keras

def _transition_block(x, reduction, name, activation="relu",
                    momentum=0.99, epsilon=0.001):
    """
    Transition block defined in Densenet architecture

    Args:
        x: input tensor form the precedent layer
        reduction : float number, compression rate (ref. paper)
        activation: string, name of the activation function we want to use 
        (ref Keras Docs)
        name: string, scope/name of current layer
    Return:
        Output tensor of the block
    """
    axis = 3 if keras.backend.image_data_format()=='channels_last' else 1
    with tf.name_scope(name):
    # Batch normalization layer 
        x = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name = "bn")(x)
        # Activation layer
        x = keras.layers.Activation(activation, name = activation)(x)
        #number of filters, filter size = 1, stride = 1
        x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[axis] * reduction), 1,
                                use_bias=False,
                                name='conv')(x)
        # Average pooling
        x = keras.layers.AveragePooling2D(2, strides=2, name="pool")(x)
    return x

def conv_block(x, growth_rate, name, activation="relu",
                momentum=0.99, epsilon=0.001):
    """Convolutional block defined in Densenet architecture,
       It is its base layers

    Args:
        x: input tensor form the precedent layer
        growth_rate : float number, growth rate or number of filters (ref. paper)
        activation: string, name of the activation function we want to use 
        (ref Keras Docs)
        name: string, scope/name of current layer
    Return:
        Output tensor of the block
        """
    with tf.name_scope(name):    
        axis = 3 if keras.backend.image_data_format()=='channels_last' else 1
        x_c = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name="bn")(x)
        x_c = keras.layers.Activation(activation, name=activation)(x_c)
        # number of filters=4*growth_rate, filter size = 1
        x_c = keras.layers.Conv2D(4*growth_rate,1,
                                use_bias=False,
                                name="conv_1")(x_c)
        x_c = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name="bn")(x_c)
        x_c = keras.layers.Activation(activation, name=activation)(x_c)
        # number of filters=growth_rate, filter size = 3
        x_c = keras.layers.Conv2D(growth_rate, 3, padding="same",
                                use_bias=False, name="conv_2")(x_c)
        output = keras.layers.Concatenate(axis=axis, name="concat")([x, x_c])
    return output

def _dense_block(x, blocks, name):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    with tf.name_scope(name):
        for i in range(blocks):
            x = conv_block(x, 32, name='block' + str(i + 1))
    return x


def densenet(blocks,
            inputs,
            pooling,
            activation="relu",
            momentum=0.99,
            epsilon=0.001):
    """
    Args:
        blocks: list of integer, each element represents
        the number of convolutional blocks
        inputs: image input (1 or batch)
        classes: number of classes we want to train on
        pooling: 'avg' for average, 'max for max pooling
        activation: string for the activation name
        momentum: value for batch norm
        epsilon: value foir batch norm
    
    """
    assert len(blocks)==4
    axis = 3 if keras.backend.image_data_format()=='channels_last' else 1
    if blocks == [6, 12, 24, 16]:
        naming = 'Densenet121'
    elif blocks == [6, 12, 32, 32]:
        naming = 'Densenet169'
    elif blocks == [6, 12, 48, 32]:
        naming = 'Densenet201'
    elif blocks == [6, 12, 64, 48]:
        naming= 'Densenet264'
    else:
        naming = 'Densenet'
    
    with tf.name_scope(naming):
        x = keras.layers.ZeroPadding2D(((3,3), (3,3)))(inputs)
        x = keras.layers.Conv2D(64, 7, 
                                strides=2, 
                                use_bias=False, 
                                name="conv1")(x)
        x = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name="conv1_bn")(x)
        x = keras.layers.Activation(activation, name="conv1_"+activation)(x)
        x = keras.layers.ZeroPadding2D(((1,1),(1,1)))(x)
        x = keras.layers.MaxPool2D(3, strides=2, name='pool1')(x)
        x = _dense_block(x, blocks[0], 'conv2')
        x = _transition_block(x, 0.5, 'pool2')
        x = _dense_block(x, blocks[1], 'conv3')
        x = _transition_block(x, 0.5, 'pool3')
        x = _dense_block(x, blocks[2], 'conv4')
        x = _transition_block(x, 0.5, 'pool4')
        x = _dense_block(x, blocks[3], 'conv5')
        x = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name='bn')(x)
        x = keras.layers.Activation(activation, name=activation)(x)
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPool2D(name='max_pool')(x)
        return x

def densenet_121(inputs,
            pooling='avg',
            activation="relu",
            momentum=0.99,
            epsilon=0.001):
    return densenet([6,12,24,16],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_169(inputs,
            pooling='avg',
            activation="relu",
            momentum=0.99,
            epsilon=0.001):
    return densenet([6, 12, 32, 32],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_201(inputs,
                pooling='avg',
                activation="relu",
                momentum=0.99,
                epsilon=0.001):
    return densenet([6, 12, 48, 32],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_264(inputs,
                pooling='avg',
                activation="relu",
                momentum=0.99,
                epsilon=0.001):
    return densenet([6, 12, 64, 48],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

setattr(densenet_121, '__doc__', densenet.__doc__)
setattr(densenet_169, '__doc__', densenet.__doc__)
setattr(densenet_201, '__doc__', densenet.__doc__)
setattr(densenet_264, '__doc__', densenet.__doc__)