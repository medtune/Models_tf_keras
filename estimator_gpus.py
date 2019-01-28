import tensorflow as tf
import os
import sys
from yaml import load

##New imports##
from inputs.images import dataset_images
from models.cnn import finetune, base

def input_fn(mode, file_pattern, image_size,
            names_to_labels, num_classes, batch_size,
            num_epochs, shuffle_buffer_size):
    train_mode = mode==tf.estimator.ModeKeys.TRAIN
    with tf.name_scope("dataset"):
        phase_name = "train" if train_mode else "valid"
        if os.sep in file_pattern:
            # We first split file_pattern given the os seperator
            # Then split the last element of the resulting list
            # using dot separator
            file_type = file_pattern.split(os.sep)[-1].split(".")[-1]
        else:
            file_type = file_pattern.split(".")[-1]
        if file_type=="tfrecord":
            dataset = dataset_images.get_tfrecord(phase_name,
                                            file_pattern=file_pattern,
                                            image_size=image_size,
                                            names_to_labels=names_to_labels,
                                            num_classes=num_classes,
                                            batch_size=batch_size,
                                            num_epochs=num_epochs,
                                            shuffle_buffer_size=shuffle_buffer_size,
                                            is_training=train_mode)
        else:
            dataset = dataset_images.get_Mura(phase_name,
                                            file_pattern=file_pattern,
                                            image_size=image_size,
                                            names_to_labels=names_to_labels,
                                            num_classes=num_classes,
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
    dataset_spec = config.get("dataset")
    dataset_dir = dataset_spec.get("dataset_dir") # Absolute direction to the dataset 
    #file_pattern of each example of data
    file_pattern = os.path.join(dataset_dir, dataset_spec.get("file_pattern")) 
    # Number of examples in the dataset
    num_samples = dataset_spec.get("num_samples") 
    # Number of categories we want to train the model on:
    num_classes = dataset_spec.get("num_classes")
    # Dictionnary mapping each label's name to an integer value
    names_to_labels = dataset_spec.get("names_to_labels") 
    #labels_to_names = data["labels_to_names"]
    #==================================#

    #==================================#
    #=======Model Informations======#
    model_name = config.get("model").get("name")
    #==================================#

    #==================================#
    #=======Training Informations======#
    #==================================#
    #Number of epochs for training the model
    train_spec = config.get("train")
    num_epochs = train_spec.get("num_epochs")
    #State your batch size
    batch_size = train_spec.get("batch_size")
    #Optimizer noun that we want to use
    optimizer_noun = train_spec.get("optimizer_noun")
    #Learning rate information and configuration
    learning_rate = train_spec.get("learning_rate")
    initial_lr = learning_rate.get("initial")
    #Decay factor
    decay_factor = learning_rate.get("decay_factor")
    before_decay = learning_rate.get("before_decay")
    del config
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
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Define max steps:
    max_step = num_epochs*num_batches_per_epoch
    # Define ModelConstructor instance base on the model_name:
    # input_shape and image_type are optional:
    model = base.ModelConstructor(model_name)
    #take the expected image_size by the model 
    image_size = model.input_shape
    # Define ModelConstructor instance base on the model_name:
    
    classifier = base.Classifier(num_classes)
    # Assemble both classifier and CNN model:
    # We get a keras Model instance, and it's argument that we'll
    # pass with assembly.compile(**assembly_args) : 
    assembly = finetune.assemble_gpus(model, classifier,
                                optimizer_noun=optimizer_noun)
    # Define configuration:
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=num_batches_per_epoch,
                                        keep_checkpoint_max=num_epochs,
                                        model_dir=train_dir)
    #Turn the Keras model to an estimator, so we can use Estimator API
    estimator = tf.keras.estimator.model_to_estimator(assembly, config=run_config)
    
    #Define trainspec estimator, including max number of step for training 
    train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.TRAIN,
                                                file_pattern,
                                                image_size,
                                                names_to_labels,
                                                num_classes,
                                                batch_size,
                                                num_epochs,
                                                shuffle_buffer_size=1024), 
                                                max_steps=max_step)
    #Define evalspec estimator
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(tf.estimator.ModeKeys.EVAL,
                                                    file_pattern,
                                                    image_size,
                                                    names_to_labels,
                                                    num_classes,
                                                    batch_size,
                                                    num_epochs,
                                                    shuffle_buffer_size=256))       
    #Run the training and evaluation (1 eval/epoch)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


if __name__ == '__main__':
    main()