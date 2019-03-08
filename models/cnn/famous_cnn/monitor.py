import tensorflow as tf
import os
import urllib
"""
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

def getSummariesComputerVision():
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

def download_imagenet_checkpoints(url, downloadDir):
    """
    Given the name of the model, we first extract the
    Imagenet URL weights from checkpoints dicts (.ref: famous_cnn)
    
    # Arguments:
        - url : Correspond to the url to tar/zip file
        - downloadDir : Correspond to jobPath/imagenet_weights
    """
