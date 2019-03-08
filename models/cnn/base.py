import tensorflow as tf
import tensorflow.keras as keras
import os
from tensorflow.keras.layers import Dense, Flatten

import utils.training.monitor as monitor
from . import famous_cnn

"""
Mobilenet models have two additionnal arguments:
alpha, depth_multiplier
"""
native_optimizers = {
    "adadelta": tf.train.AdadeltaOptimizer,
    "adagrad": tf.train.AdagradOptimizer,
    "adam": tf.train.AdamOptimizer,
    "ftrl": tf.train.FtrlOptimizer,
    "sgd": tf.train.GradientDescentOptimizer,
    "momentum": tf.train.MomentumOptimizer,
    "rmsprop": tf.train.RMSPropOptimizer
}

def get_loss_function(classification_type):
        """
        Depending on the label type (sparse or one_hot) and the classification type,
        we return the loss function that we will pass into
        model_fn
        """
        # labels is a batch, so the first dimension is the batch size
        if classification_type == "multilabel":
            return tf.losses.sigmoid_cross_entropy
        return tf.losses.softmax_cross_entropy

def get_aggregation_function(classification_type):
    """
    Depending on the label type (sparse or one_hot) and the classification type,
    we return the aggregation function of the logits
    """
    if classification_type == "multilabel":
        return tf.nn.sigmoid
    return tf.nn.softmax

class AssembleComputerVisionModel():

    def __init__(self, params):
        """
        params : from the configuration file, we take the following params as
            dict :
                * model.name : model name to pass to the AssembleModel
                * image_type : input type ('rgb' or 'gray')
                * num_classes : number of classes 
                * classification_type : multiclass or multilabel
                * classification_layers : a list representing the hidden layers we want to
                implement
                * optimizer_noun : the optimizer function we want to use during training
                * learning_rate : initial, decay_factor and before_decay to define the
                a decayed learning rate
                * activation_func : name of activation function we want to use
                * num_samples : number of training examples. Used to define a decay learning rate
                * batch_size : integer representing number of examples per batch 
        """
        # Get the CNN model base on the given name
        self.modelName = params["name"]
        self.checkpointName = self.modelName
        self.cnn_model = famous_cnn.architectures.get(self.modelName)
        # Define the image type : 'Grayscale' or 'RGB'
        self.input_type = params["image_type"]
        # Number of classes
        self.numClasses = params["num_classes"]
        # classification type
        self.classificationType = params["classification_type"]
        self.classificationLayers = params["classification_layers"]
        self.activationFunc = params["activation_func"]
        # Dict learning rate containing: initial lr, decay factor, epochs
        # before decay:
        self.learningRate = params["learning_rate"]
        # String representing the noun of the optimizer we want to use 
        # (ref. list of nouns)
        self.optimizerNoun = params["optimizer_noun"]
        # Int. we use it to calculate the decay step 
        self.num_batches_per_epoch = int(params["num_samples"] / params["batch_size"])
        self.decay_steps = int(self.learningRate["before_decay"] * self.num_batches_per_epoch)
        del params
    
    def get_hyperparams(self):
        """
        Using stdio inputs, we ask the user to define
        a value for each hyperparameter of the model, depending on the
        CNN model that is used (epsilon, batch_norm, alpha for mobilenet &
        mobilenetv2)
        # Return : 
            A dict containing the value of each hyperparameter
        """
        pass
    
    def get_modelName(self):
        return self.modelName

    def get_inputType(self):
        return self.input_type

    def get_numClasses(self):
        return self.numClasses

    def  model_fn(self, features, labels, mode):
        """
        Model_fn function that we will pass to the 
        estimator.
        # Arguments :
            - features : batch of images as input
            - labels : batch of labels (true prediction)
            - mode : train, eval or predict. (ref to Estimator docs)

        # Return : 
            model_fn function
        """
        # Calculate CNN features (last layer output) :
        cnn_features = self.cnn_model(features)
        # Calculate the classification results : 
        logits = self.construct(cnn_features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            _ , top_5 =  tf.nn.top_k(logits, k=5)
            predictions = {
                'top_1': tf.argmax(logits, -1),
                'top_5': top_5,
                    'probabilities': tf.nn.softmax(logits),
                'logits': logits,
            }
            export_outputs = {
        'prediction': tf.estimator.export.PredictOutput(predictions)
        }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions, 
                                                export_outputs=export_outputs)
        else :
            # Define the classification loss : 
            classification_loss = get_loss_function(self.classificationType)\
                                                   (labels, logits)
            # Add the regularization loss : 
            total_loss = tf.losses.get_total_loss()
            # Metrics 
            metrics = {
            'Accuracy': tf.metrics.accuracy(labels, logits, name="acc_op"),
            'Precision': tf.metrics.precision(labels, logits, name="precision_op"),
            'Recall': tf.metrics.recall(labels, logits, name="recall_op"),
            #'Acc_Class': tf.metrics.mean_per_class_accuracy(labels, predicted_classes,len(labels_to_names), name="per_class_acc")
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                evaluationHook = tf.train.SummarySaverHook(save_steps=100,
                                summary_op = tf.summary.image("validation_images", features))
                #TODO: Add evaluation hooks
                return tf.estimator.EstimatorSpec(mode, loss=total_loss,
                                                eval_metric_ops=metrics,
                                                evaluation_hooks=[evaluationHook])

            else :
                for name, value in metrics.items():
                    tf.summary.scalar(name, value[1])
                #Create the global step for monitoring the learning_rate and training:
                global_step = tf.train.get_or_create_global_step()
                with tf.name_scope("learning_rate"):    
                    lr = tf.train.exponential_decay(learning_rate=self.learningRate["initial"],
                                            global_step=global_step,
                                            decay_steps=self.decay_steps,
                                            decay_rate = self.learningRate["decay_factor"],
                                            staircase=True)
                    tf.summary.scalar('learning_rate', lr)
                #Define Optimizer with decay learning rate:
                with tf.name_scope("optimizer"):
                    optimizer = native_optimizers.get(self.optimizerNoun)(lr)
                    train_op = optimizer.minimize(total_loss)
                trainHook = tf.train.SummarySaverHook(save_steps=self.num_batches_per_epoch,
                                        summary_op=self.getSummariesComputerVision())
                return tf.estimator.EstimatorSpec(mode, 
                                                  loss=total_loss, 
                                                  train_op=train_op,
                                                  training_hooks=[trainHook])
    
    def initModel(self, jobPath):
        """
        Given a folder path, we check the existence of 
        the checkpoint from a previous training session.
        If it doesn't exists, we download the "imagenet"
        checkpoint model using modelNaming.
        """
        if jobPath is None:
            raise ValueError("Path must not be None. Please provide\
                            the folder path  from which you wish to restore/finetune\
                            your model.\
                            Example:\
                            |_ jobPath\
                                |_imagenet_weights\
                                |_train\
                                |_eval")
        modelPath = tf.train.latest_checkpoint(os.path.join(jobPath,"train"))
        if modelPath:
            warmStartSetting = tf.estimator.WarmStartSettings(modelPath, vars_to_warm_start=[".*"])
        else:
            downloadDir = os.path.join(jobPath,"imagenet_weights")
            modelPath = tf.train.latest_checkpoint(downloadDir)
            if not modelPath:
                # Extract url from checkpoints dict using the attribute checkpointName 
                url = famous_cnn.checkpoints.get(self.checkpointName)
                monitor.download_imagenet_checkpoints(self.modelName, url, downloadDir)
            # We create train and eval dir inside the job folder : 
            trainDir = os.path.join(jobPath,"train")
            if not os.path.exists(trainDir):
                os.makedirs(trainDir)
            evalDir = os.path.join(jobPath, "eval")
            if not os.path.exists(evalDir):
                os.makedirs(evalDir)
            warmStartSetting = tf.estimator.WarmStartSettings(modelPath, vars_to_warm_start=[self.modelName])
        return warmStartSetting


    def construct(self, features):
        """
        We construct a Neural Network with the number of layers equivalent to
        len classification_layers list.
        Args:
            features: features layer 
        """
        # Create intermediate variable representing the intermediate layers
        # of the neural networks:
        if hasattr(tf.nn, self):
            #define the activation function 
            activation = getattr(tf.nn, self.activationFunc) 
        with tf.name_scope("Logits"):
            inter = Flatten()(features)
            if self.classificationLayers:
                for size in self.classificationLayers:
                    inter = Dense(size, activation=activation)(inter)
            if self.classificationType=="multiclass":
                logits = Dense(self.numClasses, activation=tf.nn.softmax)(inter)
            else:
                logits = Dense(self.numClasses, activation=tf.nn.sigmoid)(inter)
        return logits

    def getSummariesComputerVision(self):
        """
        We compute and return the serialized of trainable
        variables. In the case of computer vision, it is kernel
        filters, neural network weights and biases, `Summary` protocol  
        buffer resulting from merging all summaries present in the
        graph
        """
        graph = tf.get_default_graph()
        trainableVariables = graph.get_collection("TRAINABLE_VARIABLES ")
        if trainableVariables:
            for variable in trainableVariables:
                tf.summary.histogram(variable.name, variable)
        merge_summaries = tf.summary.merge_all()
        return merge_summaries