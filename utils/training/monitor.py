import tensorflow as tf

"""
This module contains the utility functions that help us define
custom hyperparameters like decay learning rate
It also defines training hooks for monitoring hardware performance and
time processing
It also provides utily functions to call tf.summary.xxx depending on the
model
"""

def get_decaylr(initial_lr, decay_factor, decay_steps, global_step):
    """
    Utility function to get a decayed
    learning rate
    Args:
        initial_lr: the initial value of learning rate
        decay_factor: factor by which the learning rate decreases over steps
        decay_steps: number of steps it has to wait before starting decreasing
        the learning rate
    Return:
        Learning rate with exponential decay
    """
    
    lr = tf.train.exponential_decay(learning_rate=initial_lr,
                            global_step=global_step,
                            decay_steps=decay_steps,
                            decay_rate = decay_factor,
                            staircase=True)
    tf.summary.scalar('learning_rate', lr)
    return lr

def get_summary(model):
    """
    Given a Model instance (a combination of a CNN model & a classifier),
    we compute and return a tf.summary instance
    """
    tf.summary.image("image", model.layers[0].output)
    for i, layer in enumerate(model.layers[1:]):
        output = layer.output
        tf.summary.histogram(layer.name, output)