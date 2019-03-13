import tensorflow as tf
import os
from . import preprocessing

def get_tfrecord(phase_name, file_pattern, image_size, image_channels,
                num_classes, batch_size = 32, num_epochs = 1,
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
        label = tf.cast(tf.one_hot(label, num_classes), tf.float32)
        image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=image_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        image = _augment(image, is_training)
        return image, label
    # On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'valid']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or valid as the phase_name' % (phase_name))
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
            image_channels, num_classes, 
            batch_size=32, num_epochs=-1,
            shuffle_buffer_size=1024, is_training=False):
    """Creates dataset based on phased_name(train or evaluation), """
    def _parse_fn(filename):
        #Create the keys_to_features dictionary for the decoder    
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=image_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        image = _augment(image, is_training)
        return image
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'valid']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or valid as the phase_name' % (phase_name))
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

def get_Mura(phase_name, file_pattern, image_size, image_channels,
            names_to_labels, num_classes, batch_size=32, num_epochs=-1,
            shuffle_buffer_size=1024, is_training=False):
    """Creates dataset based on phased_name(train or evaluation) for
    MURA dataset
    """
    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(names_to_labels))
    def _parse_fn(filename):
        #Create the keys_to_features dictionary for the decoder    
        filename_split = tf.string_split([filename], delimiter=os.sep).values
        #NOTE:The Following line is a way of extracting label for MURA
        #(ex: \data\MURA-v1.1\train\XR_ELBOW\patient00011\study1_negative\image.png)
        label = tf.string_split([filename_split[-2]], delimiter = "_").values[-1]
        label = tf.cast(tf.one_hot(table.lookup(label), num_classes), tf.float32)
        image = tf.image.decode_png(tf.read_file(filename), channels=image_channels)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        image = _augment(image, is_training)
        return image, label
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'valid']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or valid as the phase_name' % (phase_name))
    #Using file_pattern, we replace the phase_name:
    file_pattern_for_counting = file_pattern.replace("phase_name", phase_name)
    #Use list file utiliy function, resulting in a tf.data.Dataset of filenames
    dataset = tf.data.Dataset.list_files(file_pattern_for_counting)
    #Introduce the parse_fn function in order to obtain the image and it's label for MURA dataset
    dataset = dataset.map(_parse_fn, num_parallel_calls=os.cpu_count())
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)    
    dataset = dataset.batch(batch_size)
    return dataset

def get_GED(phase_name, file_pattern, image_size, image_channels,
            num_classes, batch_size=32, num_epochs=-1,
            shuffle_buffer_size=1024, is_training=False):
    """Creates dataset based on phased_name (train or evaluation) for
    rvl-cdip dataset
    """
    images_dir = os.path.dirname(file_pattern).replace("labels", "images")
    def _parse_fn(line):
        #Create the keys_to_features dictionary for the decoder    
        split = tf.string_split([line], delimiter=" ").values
        filename, label = tf.strings.join([images_dir,split[0]], separator=os.sep), tf.strings.to_number(split[1], tf.int32)
        image = tf.read_file(filename)
        image = tf.image.decode_jpeg(image, channels=image_channels)
        #NOTE:The Following line is an efficient way of extracting label
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        label = tf.cast(tf.one_hot(label, num_classes), tf.float32)
        return (image, label)
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'val', 'test']:
        raise ValueError('The phase_name %s is not recognized.\
                          Please input either train or eval as the phase_name' % (phase_name))
    #Using file_pattern, we replace the phase_name:
    file_pattern_for_counting = file_pattern.replace("phase_name", phase_name)
    assert os.path.exists(file_pattern_for_counting)
    #Use list file utiliy function, resulting in a tf.data.Dataset of filenames
    dataset = tf.data.TextLineDataset([file_pattern_for_counting])
    #Introduce the parse_fn function in order to obtain the image and it's label for MURA dataset
    dataset = dataset.map(_parse_fn, num_parallel_calls=os.cpu_count())
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


def get_input_fn(mode, datasetSpecs):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    file_pattern = os.path.join(datasetSpecs.get("dataset_dir"), 
                                datasetSpecs.get("file_pattern")) 
    with tf.name_scope("dataset"):
        phase_name = "train" if train_mode else "val"
        # We first split file_pattern given the os seperator
        # Then split the last element of the resulting list
        # using dot separator
        file_type = file_pattern.split(os.sep)[-1].split(".")[-1]
        if file_type=="tfrecord":
            def input_fn():
                dataset = get_tfrecord(phase_name,
                                        file_pattern=file_pattern,
                                        image_size=datasetSpecs.get("image_size"),
                                        image_channels=datasetSpecs.get("image_channels"),
                                        num_classes=datasetSpecs.get("num_classes"),
                                        batch_size=datasetSpecs.get("batch_size"),
                                        num_epochs=datasetSpecs.get("num_epochs"),
                                        shuffle_buffer_size=datasetSpecs.get("shuffle_buffer_size"),
                                        is_training=train_mode)
                return dataset
        else:
            def input_fn():
                dataset = get_GED(phase_name,
                                    file_pattern=file_pattern,
                                    image_size=datasetSpecs.get("image_size"),
                                    image_channels=datasetSpecs.get("image_channels"),
                                    num_classes=datasetSpecs.get("num_classes"),
                                    batch_size=datasetSpecs.get("batch_size"),
                                    num_epochs=datasetSpecs.get("num_epochs"),
                                    shuffle_buffer_size=datasetSpecs.get("shuffle_buffer_size"),
                                    is_training=train_mode)
                return dataset
    return input_fn