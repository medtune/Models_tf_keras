import tensorflow as tf
import os

def get_tfrecord(phase_name, file_pattern, image_size,
                num_classes, batch_size=32, num_epochs=-1,
                shuffle_buffer_size=1024, is_training=False):
    """Creates dataset based on phased_name(train or evaluation), datatset_dir."""
    def _parse_fn(example, is_training=is_training):
        # Create the keys_to_features dictionary for the decoder    
        feature = {
            'image/encoded':tf.FixedLenFeature((), tf.string),
            'image/class/id':tf.FixedLenFeature((), tf.int64),
            }
        parsed_example = tf.parse_single_example(example, feature)
        label = parsed_example["image/class/id"]
        label = tf.one_hot(label, num_classes)
        image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=image_size[2])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size[:2])
        image = _augment(image, is_training)
        return (image, label)
    # On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'eval']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or eval as the phase_name' % (phase_name))
    # TODO: Remove counting num_samples. num_samples have to be fixed before
    # Compte le nombre total d'examples dans tous les fichiers
    # file_pattern will have the following format:
    # alpha/beta/datasetname_*.tfrecord: * represents  the phase name:
    file_pattern_for_counting = file_pattern.replace("*", phase_name)  
    files = tf.data.Dataset.list_files(file_pattern_for_counting)
    dataset = files.interleave(tf.data.TFRecordDataset, 1)
    dataset = dataset.map(_parse_fn)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(-1)    
    else:
        dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    return dataset

def get_flat(phase_name, file_pattern, image_size,
            num_classes, batch_size=32, num_epochs=-1,
            shuffle_buffer_size=1024, is_training=False):
    """Creates dataset based on phased_name(train or evaluation), """
    def _parse_fn(filename):
        #Create the keys_to_features dictionary for the decoder    
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=image_size[2])
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size[:2])
        image = _augment(image, is_training)
        return image
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'eval']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or eval as the phase_name' % (phase_name))
    #TODO: Remove counting num_samples. num_samples have to be fixed before
    #Compte le nombre total d'examples dans tous les fichiers
    file_pattern_for_counting = file_pattern+ '_' + phase_name
    dataset = tf.data.Dataset.list_files(file_pattern_for_counting)
    dataset = dataset.map(_parse_fn)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)    
    dataset = dataset.batch(batch_size)
    return dataset


def _augment(image, is_training=False):
    """
    Helper function for Data augmentation, depending on
    the case: train or validation set of data
    """
    return image

