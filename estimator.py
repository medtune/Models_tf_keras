import tensorflow as tf
import os
import sys
from yaml import load

##New imports##
from Models_tf_keras.inputs.images import dataset_images
from Models_tf_keras.models.cnn import finetune, base


def input_fn(mode, file_pattern, image_size,
            names_to_labels, batch_size,
            num_epochs, shuffle_buffer_size):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("dataset"):
        phase_name = "train" if train_mode else "eval"
        file_type = file_pattern.split(os.sep)[-1].split(".")[-1]
        if file_type=="tfrecord":
            dataset = dataset_images.get_tfrecord(phase_name,
                                            file_pattern=file_pattern,
                                            image_size=image_size,
                                            names_to_labels=names_to_labels,
                                            batch_size=batch_size,
                                            num_epochs=num_epochs,
                                            shuffle_buffer_size=shuffle_buffer_size,
                                            is_training=train_mode)
        else:
            dataset = dataset_images.get_flat(phase_name,
                                            file_pattern=file_pattern,
                                            image_size=image_size,
                                            names_to_labels=names_to_labels,
                                            batch_size=batch_size,
                                            num_epochs=num_epochs,
                                            shuffle_buffer_size=shuffle_buffer_size,
                                            is_training=train_mode)
    return dataset 

def main():
    #Open and read the yaml file:
    cwd = os.getcwd()
    with open(os.path.join(cwd, "yaml","config.yaml")) as stream:
        config = load(stream)
    #==================================#
    #=======Dataset Informations=======#
    #==================================#
    dataset_dir = config.dataset.dataset_dir # Absolute direction to the dataset 
    #file_pattern of each example of data
    file_pattern = os.path.join(dataset_dir, config.dataset.file_pattern) 
    # Number of examples in the dataset
    num_samples = config.dataset.num_samples 
    # Number of categories we want to train the model on:
    num_classes = config.dataset.num_classes
    # Dictionnary mapping each label's name to an integer value
    names_to_labels = config.dataset.names_to_labels 
    #labels_to_names = data["labels_to_names"]
    #==================================#

    #==================================#
    #=======Model Informations======#
    model_name = config.model.name
    #==================================#

    #==================================#
    #=======Training Informations======#
    #==================================#
    #Number of epochs for training the model
    num_epochs = config.train.num_epochs
    #State your batch size
    batch_size = config.train.batch_size
    #Optimizer noun that we want to use
    optimizer_noun = config.train.optimizer_noun
    #Learning rate information and configuration
    initial_lr = config.train.learning_rate.initial
    #Decay factor
    decay_factor = config.train.learning_rate.decay_factor
    before_decay = config.train.learning_rate.before_decay

    #Calculus of batches/epoch, number of steps after decay learning rate
    num_batches_per_epoch = int(num_samples / batch_size)
    #num_batches = num_steps for one epcoh
    decay_steps = int(before_decay * num_batches_per_epoch)
    train_dir = os.path.join(cwd, "train_"+ model_name)
    #==================================#
    #Create log_dir:argscope_config
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    #Set the verbosity to INFO level
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Define max steps:
    max_step = num_epochs*num_batches_per_epoch
    strategy = tf.contrib.distribute.MirroredStrategy()
    # Define ModelConstructor instance base on the model_name:
    # input_shape and image_type are optional:
    model = base.ModelConstructor(model_name)
    image_size = model.input_shape
    # Define ModelConstructor instance base on the model_name:
    classifier = base.Classifier(num_classes)
    # Assemble both classifier and CNN model:
    # We get a keras Model instance, and it's argument that we'll
    # pass with assembly.compile(**assembly_args)
    assembly, assembly_args = finetune.assemble(model, classifier)
    assembly.compile(**assembly_args)
    # Define configuration:
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=num_batches_per_epoch,keep_checkpoint_max=num_epochs,
                                        train_distribute=strategy, eval_distribute=strategy)
    #Turn the Keras model to an estimator, so we can use Estimator API
    estimator = tf.keras.estimator.model_to_estimator(assembly, config=run_config)
    
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.TRAIN,
                                                file_pattern,
                                                image_size,
                                                names_to_labels,
                                                batch_size,
                                                num_epochs,
                                                shuffle_buffer_size=1024), 
                                                max_steps=max_step)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.EVAL,
                                                    file_pattern,
                                                    image_size,
                                                    names_to_labels,
                                                    batch_size,
                                                    num_epochs,
                                                    shuffle_buffer_size=256))       
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
if __name__ == '__main__':
    main()