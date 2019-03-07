import tensorflow as tf
import os

"""
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

def getSummariesComputerVision():
    """
    We compute and return the serialized `Summary` protocol  
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

def initModel(jobPath, modelNaming):
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
        variableToRestore = tf.get_collection("GLOBAL_VARIABLES")
    else:
        modelPath = tf.train.latest_checkpoint(os.path.join(jobPath,"imagenet_weights"))

