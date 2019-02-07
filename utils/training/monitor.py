import tensorflow as tf

"""
This module contains the utility functions that help us define
custom hyperparameters like decay learning rate
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

_graph = tf.get_default_graph()
_summaries = _graph.get_collection("variables")

def get_summary(model):
    """
    Given a Model instance (a combination of a CNN model & a classifier),
    we compute and return a tf.summary instance
    """
    tf.summary.image("image", model.layers[0].output)
    for i, layer in enumerate(model.layers[1:]):
        tf.summary.histogram(layer.name, layer.output)
    merge_summaries = tf.summary.merge_all() 
    return merge_summaries

class TrainStats(tf.train.SessionRunHook):
    """Logs training summaries into Tensorboard """

    def __init__(self):
        """

        """
        self.graph = tf.get_default_graph()
        self.summaries = self.graph.get_collection("summaries")
    
    def begin(self):
        """
        :param session:
            Tensorflow session
        :param coord:
            unused
        """
        pass


class TrainLogs(tf.train.LoggingTensorHook):
    """
    Create and stream logs to IO output
    """
    def __init__(self):
        pass

    def begin(self):
        pass