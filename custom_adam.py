import tensorflow as tf


class CustomAdam(tf.train.AdamOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-4,
                 use_locking=True, name="CustomAdam"):
        super(CustomAdam, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)

    def minimize(self, grads_and_vars, global_step=None, name=None):
        return self.apply_gradients(grads_and_vars, global_step, name)