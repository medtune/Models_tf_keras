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
_native_optimizers = {
    "adadelta": ('Adadelta',tf.train.AdadeltaOptimizer),
    "adagrad": ('Adagrad',tf.train.AdagradOptimizer),
    "adam": ('Adam',tf.train.AdamOptimizer),
    "ftrl": ('Ftrl',tf.train.FtrlOptimizer),
    "sgd": ('GradientDescent',tf.train.GradientDescentOptimizer),
    "momentum": ('Momentum',tf.train.MomentumOptimizer),
    "rmsprop": ('RMSProp',tf.train.RMSPropOptimizer)
}

_non_batchnorm_models = ["vgg_16", "vgg_19"]
_batchnorm_models = ["densenet_121","densenet_169","densenet_201", "densenet_264"]
_mobilenet_model = ["mobilenet_v1", "mobilenet_v2"]

def _get_loss_function(classification_type):
        """
        Depending on the label type (sparse or one_hot) and the classification type,
        we return the loss function that we will pass into
        model_fn
        """
        # labels is a batch, so the first dimension is the batch size
        if classification_type == "multilabel":
            return tf.losses.sigmoid_cross_entropy
        return tf.losses.softmax_cross_entropy

def _get_aggregation_function(classification_type):
    """
    Depending on the label type (sparse or one_hot) and the classification type,
    we return the aggregation function of the logits
    """
    if classification_type == "multilabel":
        return tf.nn.sigmoid
    return tf.nn.softmax

def _get_alpha(modelName):
    """
    This a hyperparameter related to the mobilenetV1 and mobilenetV2
    models. Please refer to models documentation for more details
    """
    alpha = 0.
    if modelName not in ["mobilenet_v1", "mobilenet_v2"]:
        pass
    elif modelName == "mobilenet_v1":
        allowedValues = [0.25, 0.50, 1.0]
        while alpha not in allowedValues:
            demand = "Choose between the following values of alpha:\
                    [0.25, 0.50, 1.0]\n"
            alpha = float(get_input(demand))
    else:
        allowedValues = [1.0, 1.4]
        while alpha not in allowedValues:
            demand = "Please choose between the following values of alpha:\
                    [1.0, 1.4]\n"
            alpha = float(get_input(demand))
    return alpha

def _get_epsilon():
    """
    We ask the user to input a value for epsilon.
    Epsilon is a hyperparameter that is used in the
    batch normalization denominator
    """
    epsilon = 0.01
    while epsilon > 0.001:
            demand = "Please choose a value for epsilon that is below 0.001 (or 1e-3)"
            epsilon = float(get_input(demand))
    return epsilon

def _get_momentum():
    """
    We ask the user to input a value for momentum.
    Momentum is a hyperparameter that is used in the
    batch normalization calculus
    """
    momentum = 0.
    while momentum < 0.899 or momentum >= 1:
            demand = "Please choose a value for momentum that is between 0.99\
                      and 0.9999... [0.99; 1[\n"
            momentum = float(get_input(demand))
    return momentum

def _get_pooling():
    """
    We ask the user to input a value for momentum.
    Momentum is a hyperparameter that is used in the
    batch normalization calculus'
    """
    pooling = ''
    while pooling not in ['avg', 'max']:
            demand = "Please choose a value for pooling argument that is `avg`\
                      or `max`\n"
            pooling = str(get_input(demand))
    return pooling

def _get_depthwise():
    """
    We ask the user to input a value for depthwise.
    Depthwise is an integer hyperparameter that is used in the
    mobilenet-like model. Please refer to famous_cnn submodule
    or to mobilenets paper
    # Default: 1
    """
    depth = ''
    while depth not in ['avg', 'max']:
            demand = "Please choose a value for pooling argument that is `avg`\
                      or `max`\n"
            pooling = str(get_input(demand))
    return pooling

def get_input(demand):
    try:
        answer = raw_input(demand)
    except NameError:
        answer = input(demand)
    return answer

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
        _optimizer= _native_optimizers.get(params["optimizer_noun"])
        self.optimizerNoun = _optimizer[0]
        self.optimizerObject = _optimizer[1]
        # Int. we use it to calculate the decay step 
        self.num_batches_per_epoch = int(params["num_samples"] / params["batch_size"])
        self.decay_steps = int(self.learningRate["before_decay"] * self.num_batches_per_epoch)
        self.hyperParameters = {}
        del params
    
    def initModel(self, jobPath):
        """
        Given a job folder path, we check the existence of 
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
        # TODO: Check how to configure training dir in Estimator
        trainDir = os.path.join(jobPath,"train")
        if not os.path.exists(trainDir):
            os.makedirs(trainDir)
        # TODO: Check how to configure eval dir in Estimator
        evalDir = os.path.join(jobPath, "eval")
        if not os.path.exists(evalDir):
            os.makedirs(evalDir)
        modelPath = tf.train.latest_checkpoint(jobPath)
        if modelPath:
            warmStartSetting = tf.estimator.WarmStartSettings(modelPath, vars_to_warm_start=[".*"])
        else:
            # We call 'get_hyperParameters' method in order to define the model
            # And it's HP (learning_rate, Batch_norm & epsilon(divisor))
            self.get_hyperParameters()
            downloadDir = os.path.join(jobPath,"imagenet_weights")
            print("Imagenet weights Download direction :" + downloadDir +"\n")
            modelPath = os.path.exists(os.path.join(downloadDir,self.checkpointName+'.ckpt*'))
            if not modelPath:
                # Extract url from checkpoints dict using the attribute checkpointName 
                url = famous_cnn.checkpoints.get(self.checkpointName)
                monitor.download_imagenet_checkpoints(self.checkpointName, url, downloadDir)
            # We retrieve naming according to the model name : 
            variablesPattern = famous_cnn.naming_mapping.get(self.modelName) + '[^/%s]'%(self.optimizerNoun)
            # We define warm_start settings for loading variables from checkpoint
            warmStartSetting = tf.estimator.WarmStartSettings(downloadDir, vars_to_warm_start=[variablesPattern])
        return warmStartSetting
    
    def get_modelName(self):
        return self.modelName

    def get_inputType(self):
        return self.input_type

    def get_numClasses(self):
        return self.numClasses

    def get_hyperParameters(self):
        """
        Using stdio inputs, we ask the user to define
        a value for each hyperparameter of the model, depending on the
        CNN model that is used (epsilon, batch_norm, alpha for mobilenet_v1 &
        mobilenet_v2)
        (Inspired from https://github.com/tensorflow/tensorflow/blob/master/configure.py)
        # Return : 
            A dict containing the value of each hyperparameter
        """
        if self.modelName in _mobilenet_model:
            self.hyperParameters["alpha"] = _get_alpha(self.modelName)
            self.hyperParameters["depthwise_multiplier"] = 1
            self.hyperParameters["momentum"] = _get_momentum()
            self.hyperParameters["epsilon"] = _get_epsilon()
            self.checkpointName = self.modelName + '_' + str(self.hyperParameters["alpha"])
        else:
            if self.modelName in _batchnorm_models:
                self.hyperParameters["momentum"] = _get_momentum()
                self.hyperParameters["epsilon"] = _get_epsilon()
            self.hyperParameters["activation"] = self.activationFunc
            
        self.hyperParameters["pooling"] = _get_pooling()
        

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
        cnn_features = self.cnn_model(features, **self.hyperParameters)
        # Calculate the classification results : 
        logits = self.classify(cnn_features)
        if mode == tf.estimator.ModeKeys.PREDICT:
            print("\nPredict Mode\n")
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
            classification_loss = _get_loss_function(self.classificationType)\
                                                   (labels, logits)
            # Add the regularization loss : 
            # total_loss = tf.losses.get_total_loss()

            # Metrics 
            metrics = {
            'Accuracy': tf.metrics.accuracy(labels, logits, name="acc_op"),
            'Precision': tf.metrics.precision(labels, logits, name="precision_op"),
            'Recall': tf.metrics.recall(labels, logits, name="recall_op"),
            #'Acc_Class': tf.metrics.mean_per_class_accuracy(labels, predicted_classes,len(labels_to_names), name="per_class_acc")
            }
            if mode == tf.estimator.ModeKeys.EVAL:
                print("\n Eval Mode\n")
                evaluationHook = tf.train.SummarySaverHook(save_steps=100,
                                summary_op = tf.summary.image("validation_images", features))
                return tf.estimator.EstimatorSpec(mode, loss=classification_loss,
                                                eval_metric_ops=metrics,
                                                evaluation_hooks=[evaluationHook])
            print("\n Train Mode\n")
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
                optimizer = self.optimizerObject(lr)
                train_op = optimizer.minimize(classification_loss, global_step=global_step)
            trainHook = tf.train.SummarySaverHook(save_steps=100,
                                    summary_op=self.getSummariesComputerVision(features, labels))
            imageHook = tf.train.SummarySaverHook(save_steps=100,
                                                summary_op=tf.summary.image("training_images", features))                                    
            return tf.estimator.EstimatorSpec(mode, 
                                              loss=classification_loss, 
                                              train_op=train_op,
                                              training_hooks=[trainHook, imageHook])



    def classify(self, features):
        """
        We construct a Neural Network with the number of layers equivalent to
        len classification_layers list.
        Args:
            features: features layer 
        """
        # Create intermediate variable representing the intermediate layers
        # of the neural networks:
        if hasattr(tf.nn, self.activationFunc):
            #define the activation function 
            activation = getattr(tf.nn, self.activationFunc) 
        with tf.name_scope("Logits"):
            inter = Flatten()(features)
            if self.classificationLayers:
                for size in self.classificationLayers:
                    inter = Dense(size, activation=activation)(inter)
            if self.classificationType=="multilabel":
                logits = Dense(self.numClasses, activation=tf.nn.sigmoid)(inter)
            else:
                logits = Dense(self.numClasses, activation=tf.nn.softmax)(inter)
        return logits

    def getSummariesComputerVision(self, features, labels):
        """
        We compute and return the serialized of trainable
        variables. In the case of computer vision, it is kernel
        filters, neural network weights and biases, `Summary` protocol  
        buffer resulting from merging all summaries present in the
        graph
        """
        tf.summary.histogram("features_hist", features)
        tf.summary.histogram("labels", labels)
        trainableVariables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        if trainableVariables:
            for variable in trainableVariables:
                tf.summary.histogram(variable.name, variable)
        merge_summaries = tf.summary.merge_all()
        return merge_summaries