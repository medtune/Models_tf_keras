"""
VGG16 model
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""
import tensorflow.keras as keras
import tensorflow as tf



def vgg_16(inputs,
           pooling='max',
           activation="relu"):
    """
    Arguments:
        - inputs: image input tensor 
        - pooling : avg for Average Pooling, 'max' for Max Pooling
        - activation: name of the activation function to use, default to
        'relu'
        - momentum: value to use for batch normalization
        - epsilon: value to use in the denominatort of batch normalization
    
    Returns:
        - Tensor representing features 
    """
    naming = 'vgg_16'
    with tf.name_scope(naming):
        #Block 1:
        with tf.name_scope('conv1'):
            x = keras.layers.Conv2D(64, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv1_1')(inputs)
            x = keras.layers.Conv2D(64, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv1_2')(x)
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool1')(x)

        #Block 2:
        with tf.name_scope('conv2'):
            x = keras.layers.Conv2D(128, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name ='conv2_1')(x)
            x = keras.layers.Conv2D(128, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name ='conv2_2')(x)
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool2')(x)

        #Block 3:
        with tf.name_scope('conv3'):
            x = keras.layers.Conv2D(256, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv3_1')(x)
            x = keras.layers.Conv2D(256, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv3_2')(x)
            x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation,
                            padding='same',
                            name='conv3_3')(x)
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool3')(x)

        #Block 4:
        with tf.name_scope('conv4'):
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv4_1')(x)
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv4_2')(x)
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation,
                            padding='same',
                            name='conv4_3')(x)
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool4')(x)

        #Block 5:
        with tf.name_scope('conv5'):
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = 'conv5_1')(x)
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation,
                                    padding = 'same',
                                    name = naming+'conv5_2')(x)
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation,
                            padding='same',
                            name='conv5_3')(x)
        x = keras.layers.MaxPool2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool5')(x)

        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D(name=naming+'avg_pool')(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D(name=naming+'max_pool')(x)
    return x