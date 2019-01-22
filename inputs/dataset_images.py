import tensorflow as tf
import os

def get_dataset(dataset_dir, phase_name, file_pattern, shuffle_buffer_size,
                batch_size,labels_to_name, num_epochs=-1, is_training=False
                ):
    """Creates dataset based on phased_name(train or evaluation), datatset_dir. """
    def _parse_fn(example, is_training=is_training):
        #Create the keys_to_features dictionary for the decoder    
        feature = {
            'image/encoded':tf.FixedLenFeature((), tf.string),
            'image/class/id':tf.FixedLenFeature((), tf.int64),
        }
        parsed_example = tf.parse_single_example(example, feature)
        parsed_example['image/encoded'] = tf.image.decode_image(parsed_example['image/encoded'], channels=3)
        parsed_example['image/encoded'] = _augment(parsed_example['image/encoded'], is_training)
        return parsed_example
    #On vérifie si phase_name est 'train' ou 'validation'
    if phase_name not in ['train', 'eval']:
        raise ValueError('The phase_name %s is not recognized. Please input either train or eval as the phase_name' % (phase_name))
    #TODO: Remove counting num_samples. num_samples have to be fixed before
    #Compte le nombre total d'examples dans tous les fichiers
    file_pattern_for_counting = phase_name + '_' + file_pattern 
    files = tf.data.Dataset.list_files(file_pattern_for_counting)
    dataset = files.interleave(tf.data.TFRecordDataset)
    dataset = dataset.map(_parse_fn)
    if is_training:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)    
    return dataset

def _augment(image, is_training=False):
    """
    Helper function for Data augmentation, depending on
    the case: train or validation set of data
    """
    return image