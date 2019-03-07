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

