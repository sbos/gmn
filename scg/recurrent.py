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

    @property
    def variables(self):
        return [self.Wxh, self.Whh, self.b] + [self.p] if self.fun == 'prelu' else []


class GRU(NodePrototype):
    def __init__(self, input_size, hidden_size, fun='tanh', init=glorot_normal):
        NodePrototype.__init__(self)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fun = fun

        self.w_gates = init(input_size + hidden_size, 2 * hidden_size, 'sigmoid')
        self.w_candidate = init(input_size + hidden_size, hidden_size, fun)

        self.b_gates = tf.Variable(tf.zeros((hidden_size + hidden_size,)))
        self.b_candidate = tf.Variable(tf.zeros((hidden_size,)))

        self.args = {}
        if self.fun == 'prelu':
            self.args['p'] = tf.Variable(tf.random_uniform((hidden_size,), minval=-0.02, maxval=0.02))

    def flow(self, input=None, state=None):
        assert input is not None
        assert state is not None

        total_input = tf.concat(1, [input, state])
        r, u = tf.split(1, 2, tf.sigmoid(tf.matmul(total_input, self.w_gates) + self.b_gates))

        gated_input = tf.concat(1, [input, r * state])
        c = dispatch_function(tf.matmul(gated_input, self.w_candidate) + self.b_candidate,
                              self.fun, **self.args)

        return u * state + (1 - u) * c
