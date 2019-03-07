"""
VGG19 
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""
import tensorflow.keras as keras

def vgg19(inputs,
        pooling=None,
        activation="relu",
        momentum=0.99,
        epsilon=0.001):
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
    axis = 3 if keras.backend.image_data_format()=='channels_last' else 1
    naming = 'Vgg19_'

    #Block 1:
    x = keras.layers.Conv2D(64, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block1_conv1')(inputs)
    
    x = keras.layers.Conv2D(64, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block1_conv2')(x)
    
    x = keras.layers.MaxPooling2D((2, 2), 
                                strides=(2, 2),
                                name=naming+'block1_pool')(x)

    #Block 2:
    x = keras.layers.Conv2D(128, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block2_conv1')(x)
    
    x = keras.layers.Conv2D(128, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block2_conv2')(x)
    
    x = keras.layers.MaxPooling2D((2, 2), 
                                strides=(2, 2),
                                name=naming+'block2_pool')(x)

    #Block 3:
    x = keras.layers.Conv2D(256, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block3_conv1')(x)
    
    x = keras.layers.Conv2D(256, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block3_conv2')(x)
    
    x = keras.layers.Conv2D(256, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block3_conv3')(x)
    
    x = keras.layers.Conv2D(256, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block3_conv4')(x)
    
    x = keras.layers.MaxPooling2D((2, 2), 
                                strides=(2, 2),
                                name=naming+'block3_pool')(x)

    #Block 4:
    x = keras.layers.Conv2D(512, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block4_conv1')(x)
    
    x = keras.layers.Conv2D(512, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block4_conv2')(x)
    
    x = keras.layers.Conv2D(512, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block4_conv3')(x)
    
    x = keras.layers.Conv2D(512, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block4_conv4')(x)
    
    x = keras.layers.MaxPooling2D((2, 2), 
                                strides=(2, 2),
                                name=naming+'block4_pool')(x)

    #Block 5:
    x = keras.layers.Conv2D(512, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block5_conv1')(x)
    
    x = keras.layers.Conv2D(512, (3,3),
                            activation=activation,
                            padding = 'same',
                            name = naming+'block5_conv2')(x)
    
    x = keras.layers.Conv2D(512, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block5_conv3')(x)
    
    x = keras.layers.Conv2D(512, (3, 3),
                      activation=activation,
                      padding='same',
                      name='block5_conv4')(x)
    
    x = keras.layers.MaxPooling2D((2, 2), 
                                strides=(2, 2),
                                name=naming+'block5_pool')(x)

    if pooling == 'avg':
        x = keras.layers.GlobalAveragePooling2D(name=naming+'avg_pool')(x)
    elif pooling == 'max':
        x = keras.layers.GlobalMaxPool2D(name=naming+'max_pool')(x)
    return x