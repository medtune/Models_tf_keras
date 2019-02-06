import tensorflow as tf
import os
import sys
from yaml import load

##New imports##
from inputs.images import dataset_images
import models.cnn.finetune as finetune
import models.cnn.base as base
from utils.training import monitor

#Open and read the yaml file:
cwd = os.getcwd()
with open(os.path.join(cwd, "yaml","config_2.yaml")) as stream:
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
# list containing each label's name :
names_to_labels = dataset_spec.get("names_to_labels")
image_type = dataset_spec.get("image_type")
#labels_to_names = data["labels_to_names"]
#==================================#

#==================================#
#=======Model Informations======#
model_name = config.get("model").get("name")
#==================================#

#==================================#
#=======Training Informations======#
#==================================#
train_spec = config.get("train")
distribute = train_spec.get("distribute")
xla = train_spec.get("xla")
#Get the standard image size for the desired model
image_size = base.get_input_shape(model_name, image_type)
#Number of epochs for training the model
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
#Create log_dir:
if not os.path.exists(train_dir):
    os.mkdir(train_dir)

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

def model_fn(features, labels, mode):
    #Visualize images on tensorboard
    tf.summary.image("final_image_hist", features)
    #Visualize pixel's distribution on tensorboard
    tf.summary.histogram("final_image_hist", features)
    # Define ModelConstructor instance base on the model_name:
    # input_shape and image_type are optional:
    # Define ModelConstructor instance base on the model_name:
    logits = finetune.assemble_modelfn(base.ModelConstructor(model_name, image_type=image_type),
                                       base.Classifier(num_classes),
                                       features)
    #Find the max of the predicted class and change its data type
    labels = labels
    if mode != tf.estimator.ModeKeys.PREDICT:
        #Defining losses and regulization ops:
        with tf.name_scope("loss_op"):
            loss = tf.losses.softmax_cross_entropy(labels, logits)
            total_loss = tf.losses.get_total_loss() #obtain the regularization losses as well
        metrics = {
        'Accuracy': tf.metrics.accuracy(labels, logits, name="acc_op"),
        'Precision': tf.metrics.precision(labels, logits, name="precision_op"),
        'Recall': tf.metrics.recall(labels, logits, name="recall_op"),
        }
        #For Evaluation Mode
        if mode == tf.estimator.ModeKeys.EVAL:
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, eval_metric_ops=metrics)
        else:
            #During training, we want to evaluate update_op of each metric:
            for name, value in metrics.items():
                tf.summary.scalar(name, value[1]) 
            #Create the global step for monitoring the learning_rate and training:
            global_step = tf.train.get_or_create_global_step()
            with tf.name_scope("learning_rate"):    
                #Create learning rate:
                lr = monitor.get_decaylr(initial_lr,
                                        decay_factor,
                                        decay_steps,
                                        global_step)
            #Define Optimizer with decay learning rate:
            with tf.name_scope("optimizer"):
                optimizer = tf.train.AdamOptimizer(learning_rate = lr)      
                train_op = optimizer.minimize(total_loss, global_step=global_step)
            return tf.estimator.EstimatorSpec(mode, loss=total_loss, train_op=train_op)
    #For Predict/Inference Mode:
    else:
        predictions = {
            'probabilities': tf.nn.softmax(logits, name="Softmax")
            }      
        export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
        }
        
        return tf.estimator.EstimatorSpec(mode,predictions=predictions,
                                            export_outputs=export_outputs)

def main():
    #==================================#
    #Create log_dir:argscope_config
    if not os.path.exists(train_dir):
        os.mkdir(train_dir)
    #===================================================================== Training ===========================================================================#
    #Adding the graph:
    #Set the verbosity to INFO level
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Define max steps:
    max_step = num_epochs * num_batches_per_epoch
    #Define strategy training variable
    strategy= None
    #Define variable for xla computations
    jit_level = 0
    if distribute:
        strategy = tf.contrib.distribute.MirroredStrategy()
    if xla:
        jit_level = tf.OptimizerOptions.ON_1
    # Define tf.ConfigProto() as config to pass to estimator config:
    config = tf.ConfigProto()
    #Define optimizers options based on jit_level:
    config.graph_options.optimizer_options.global_jit_level = jit_level
    # Define configuration:
    run_config = tf.estimator.RunConfig(save_checkpoints_steps=num_batches_per_epoch,
                                        keep_checkpoint_max=num_epochs,
                                        model_dir=train_dir,
                                        train_distribute=strategy,
                                        eval_distribute=strategy,
                                        session_config=config)
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
    estimator = tf.estimator.Estimator(model_fn, config=run_config)
    #Run the training and evaluation (1 eval/epoch)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == '__main__':
    main()