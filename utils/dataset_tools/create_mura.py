import tensorflow as tf
import os
import utils.images.visu_tensorflow as visu_tensorflow

def load_mura(filenames_pattern, num_channels, image_extension='jpg'):
    """
    Args:
    filenames_list : A string representing path pattern of each image
    
    Return:
    tf.data.Dataset containing raw image data, decoded and flattened
    """
    def parse_fn(filename):
        label = visu_tensorflow.extract_label(filename, label_pos=-2)
        image = visu_tensorflow.extract_image(filename, num_channels)
        return (image, label)
    dataset = tf.data.Dataset.list_files(filenames_pattern, shuffle=False)
    dataset = tf.data.Dataset.map(lambda x: parse_fn(x),
                                num_parallel_calls=os.cpu_count())
    return dataset