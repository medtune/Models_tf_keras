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

def _transition_block(x, reduction, name, activation,
                    momentum, epsilon):
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
    with tf.variable_scope(name):
        with tf.variable_scope('blk'):
            # Batch normalization layer 
            x = keras.layers.BatchNormalization(axis=axis,
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                name = "BatchNorm")(x)
            # Activation layer
            x = keras.layers.Activation(activation, name = activation)(x)
            #number of filters, filter size = 1, stride = 1
            x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[axis] * reduction), 1,
                                    use_bias=False,
                                    name='Conv')(x)
            # Average pooling
            x = keras.layers.AveragePooling2D(2, strides=2, name="pool")(x)
    return x

def conv_block(x, growth_rate, name, activation,
                momentum, epsilon):
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
    with tf.variable_scope(name):    
        axis = 3 if keras.backend.image_data_format()=='channels_last' else 1
        with tf.variable_scope('x1'):
            x_c = keras.layers.BatchNormalization(axis=axis,
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                name='BatchNorm')(x)
            x_c = keras.layers.Activation(activation, name=activation)(x_c)
            # number of filters=4*growth_rate, filter size = 1
            x_c = keras.layers.Conv2D(4*growth_rate,1,
                                    use_bias=False,
                                    name="Conv")(x_c)
        with tf.variable_scope('x2'):
            x_c = keras.layers.BatchNormalization(axis=axis,
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                name='BatchNorm')(x_c)
            x_c = keras.layers.Activation(activation, name=activation)(x_c)
            # number of filters=growth_rate, filter size = 3
            x_c = keras.layers.Conv2D(growth_rate, 3, padding="same",
                                    use_bias=False, name="Conv")(x_c)
        output = keras.layers.Concatenate(axis=axis, name="concat")([x, x_c])
    return output

def _dense_block(x, blocks, name, activation, momentum, epsilon):
    """A dense block.
    # Arguments
        x: input tensor.
        blocks: integer, the number of building blocks.
        name: string, block label.
    # Returns
        output tensor for the block.
    """
    with tf.variable_scope(name):
        for i in range(blocks):
            x = conv_block(x, 
                           32,
                           'conv_block' + str(i + 1), 
                           activation,
                           momentum, epsilon)
    return x


def densenet(blocks,
            inputs,
            pooling,
            activation,
            momentum,
            epsilon):
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
        naming = 'densenet121'
    elif blocks == [6, 12, 32, 32]:
        naming = 'densenet169'
    elif blocks == [6, 12, 48, 32]:
        naming = 'densenet201'
    elif blocks == [6, 12, 64, 48]:
        naming= 'densenet264'
    else:
        naming = 'densenet'
    
    with tf.variable_scope(naming):
        x = keras.layers.ZeroPadding2D(((3,3), (3,3)))(inputs)
        x = keras.layers.Conv2D(64, 7, 
                                strides=2, 
                                use_bias=True, 
                                name="conv1")(x)
        x = keras.layers.BatchNormalization(axis=axis,
                                            momentum=momentum,
                                            epsilon=epsilon,
                                            name="BatchNorm")(x)
        x = keras.layers.Activation(activation, name="conv1_"+activation)(x)
        x = keras.layers.ZeroPadding2D(((1,1),(1,1)))(x)
        x = keras.layers.MaxPool2D(3, strides=2, name='pool1')(x)
        x = _dense_block(x, blocks[0], 'dense_block1',activation, momentum, epsilon)
        x = _transition_block(x, 0.5, 'transition_block1', activation, momentum, epsilon)
        x = _dense_block(x, blocks[1], 'dense_block2',activation, momentum, epsilon)
        x = _transition_block(x, 0.5, 'transition_block2',activation, momentum, epsilon)
        x = _dense_block(x, blocks[2], 'dense_block3',activation, momentum, epsilon)
        x = _transition_block(x, 0.5, 'transition_block3',activation, momentum, epsilon)
        x = _dense_block(x, blocks[3], 'dense_block4',activation, momentum, epsilon)
        with tf.variable_scope('final_block'):
            x = keras.layers.BatchNormalization(axis=axis,
                                                momentum=momentum,
                                                epsilon=epsilon,
                                                name='BatchNorm')(x)
            x = keras.layers.Activation(activation, name=activation)(x)
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPool2D(name='max_pool')(x)
        return x

def densenet_121(inputs,
            pooling,
            activation,
            momentum,
            epsilon):
    return densenet([6,12,24,16],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_169(inputs,
            pooling,
            activation,
            momentum,
            epsilon):
    return densenet([6, 12, 32, 32],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_201(inputs,
                pooling,
                activation,
                momentum,
                epsilon):
    return densenet([6, 12, 48, 32],
                    inputs,
                    pooling=pooling,
                    activation=activation,
                    momentum=momentum,
                    epsilon=epsilon)

def densenet_264(inputs,
                pooling,
                activation,
                momentum,
                epsilon):
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


def slim_to_keras_namescope(blocks):
    nameMapping = {}
    if blocks == [6, 12, 24, 16]:
        naming = 'densenet121'
    elif blocks == [6, 12, 32, 32]:
        naming = 'densenet169'
    elif blocks == [6, 12, 48, 32]:
        naming = 'densenet201'
    elif blocks == [6, 12, 64, 48]:
        naming= 'densenet264'
    else:
        naming = 'densenet'
    nameMapping['%s/conv1/Conv2D/kernel'%naming] = '%s/conv1/weights'%naming
    for i, value in enumerate(blocks):
        for j in range(1, value):
            newNameXone= '%s/dense_block%d/conv_block%d/x1/Conv2D/kernel' %(naming, i+1, j)
            oldNameXone = '%s/dense_block_%d/conv_block%d/x1/Conv/weights' %(naming, i+1, j)
            newNameXtwo = '%s/dense_block%d/conv_block%d/x2/Conv2D/kernel' %(naming, i+1, j)
            oldNameXtwo = '%s/dense_block%d/conv_block%d/x2/Conv/weights' %(naming, i+1, j)
            nameMapping[newNameXone] = oldNameXone
            nameMapping[newNameXtwo] = oldNameXtwo
        if i <= 2:
            newNameTransition = '%s/transition_block%d/blk/Conv2D/kernel' %(naming, i)
            oldNameTrasnition = '%s/transition_block%d/blk/Conv/weights' %(naming, i)
            nameMapping[newNameTransition] = oldNameTrasnition
    return nameMapping