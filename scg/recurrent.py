from core import *
from deterministic import dispatch_function, glorot_normal


class RNN(NodePrototype):
    def __init__(self, input_size, hidden_size, fun='tanh', init=glorot_normal):
        NodePrototype.__init__(self)
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.Wxh = init(input_size, hidden_size, fun)
        self.Whh = init(hidden_size, hidden_size, fun)
        self.b = tf.Variable(tf.zeros((1, hidden_size)))
        self.fun = fun

        if self.fun == 'prelu':
            self.p = tf.Variable(tf.random_uniform((hidden_size,), minval=-0.02, maxval=0.02))

    def flow(self, input=None, state=None):
        assert input is not None
        assert state is not None

        h = tf.matmul(input, self.Wxh) + tf.matmul(state, self.Whh) + self.b
        args = {}
        if self.fun == 'prelu':
            args['p'] = self.p

        return dispatch_function(h, fun=self.fun, **args)

