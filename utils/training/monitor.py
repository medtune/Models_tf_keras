import tensorflow as tf

"""
This module contains the utility functions that help us define
custom hyperparameters like decay learning rate
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

def getSummariesComputerVision():
    """
    Given a Graph instance (the model graph),
    we compute and return a tf.summary instance
    """
    graph = tf.get_default_graph()
    trainableVariables = graph.get_collection("trainable")
    for variable in trainableVariables:
        tf.summary.histogram(variable.name, variable)
    merge_summaries = tf.summary.merge_all()
    return merge_summaries


