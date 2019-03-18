"""
VGG19 
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""
import tensorflow as tf
import tensorflow.keras as keras

def vgg_19(inputs,
          pooling,
          num_classes,
          activation_func,
          classification_layers,
          classification_type):
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
    naming = 'vgg_19'
    with tf.name_scope(naming):
        #Block 1:
        with tf.name_scope('conv1'):
            x = keras.layers.Conv2D(64, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv1_1')(inputs)
            
            x = keras.layers.Conv2D(64, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv1_2')(x)
        
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool1')(x)

        #Block 2:
        with tf.name_scope('conv2'):
            x = keras.layers.Conv2D(128, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv2_1')(x)
            
            x = keras.layers.Conv2D(128, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv2_2')(x)
        
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool2')(x)

        #Block 3:
        with tf.name_scope('conv3'):
            x = keras.layers.Conv2D(256, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv3_1')(x)
            
            x = keras.layers.Conv2D(256, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv3_2')(x)
            
            x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv3_3')(x)
        
            x = keras.layers.Conv2D(256, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv3_4')(x)
            
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool3')(x)

        #Block 4:
        with tf.name_scope('conv4'):
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv4_1')(x)
            
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv4_2')(x)
            
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv4_3')(x)
            
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv4_4')(x)
        
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool4')(x)

        #Block 5:
        with tf.name_scope('conv5'):
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv5_1')(x)
            
            x = keras.layers.Conv2D(512, (3,3),
                                    activation=activation_func,
                                    padding = 'same',
                                    name = 'conv5_2')(x)
            
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv5_3')(x)
            
            x = keras.layers.Conv2D(512, (3, 3),
                            activation=activation_func,
                            padding='same',
                            name='conv5_4')(x)
        
        x = keras.layers.MaxPooling2D((2, 2), 
                                    strides=(2, 2),
                                    name='pool5')(x)
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
    convolutions = [2,2,4,4,4]
    nameMapping = {}
    for i, value in enumerate(convolutions):
        for j in range(1, value+1):
            newVariablescope = 'vgg_19/conv%d/conv%d_%d/Conv2D/kernel'%(i+1, i+1, j)
            oldVariablescope =  'vgg_19/conv%d/conv%d_%d/Conv/weights'%(i+1, i+1, j)
            nameMapping[oldVariablescope] = newVariablescope
    return nameMapping