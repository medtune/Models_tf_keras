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

def mobilenet(inputs,
            classes=1000,
            pooling=None,
            activation="relu",
            momentum=0.99,
            epsilon=0.001,
            include_top=False):

    axis  = keras.backend.image_data_format()
    if axis ==  'channels_first':
        keras.backend.set_image_data_format('channels_last')
    