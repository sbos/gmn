from core import *
from deterministic import dispatch_function, glorot_normal, norm_init


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

        _init = norm_init(init)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fun = fun

        self.w_gates = init(input_size + hidden_size, 2 * hidden_size, 'sigmoid')
        self.w_candidate = _init(input_size + hidden_size, hidden_size, fun)

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


class Attention(NodePrototype):
    def __init__(self, strength=1.):
        NodePrototype.__init__(self)
        self.strength = strength

    def flow(self, mem=None, key=None, strength=None):
        assert mem is not None
        assert key is not None
        if strength is None:
            strength = self.strength

        strength = tf.clip_by_value(strength, 1e-10, 10.)

        # shape = batch x num_cells
        key_norm = tf.sqrt(tf.reduce_sum(tf.square(key), [1]))
        # shape = batch x num_cells
        mem_norm = tf.sqrt(tf.reduce_sum(tf.square(mem), [2]))
        key_norm = tf.clip_by_value(key_norm, 1e-45, 1e45)
        mem_norm = tf.clip_by_value(mem_norm, 1e-45, 1e45)
        sim = tf.batch_matmul(tf.expand_dims(key, 1), mem, adj_y=True)
        sim = tf.squeeze(sim, [1])
        sim = tf.transpose(sim)
        sim /= tf.transpose(mem_norm)
        sim /= tf.transpose(key_norm)
        sim = tf.transpose(sim)
        max_sim = tf.reduce_max(sim, 1)
        sim = tf.transpose(tf.transpose(sim) - tf.transpose(max_sim))
        sim = tf.exp(sim * strength)
        sim_sum = tf.reduce_sum(sim, [1])
        return tf.transpose(tf.transpose(sim) / sim_sum)


class AttentiveReader(NodePrototype):
    def __init__(self):
        NodePrototype.__init__(self)

    def flow(self, attention=None, mem=None):
        assert attention is not None
        assert mem is not None

        # with tf.control_dependencies([tf.Print(attention, [attention[0, :]])]):
        return tf.squeeze(tf.batch_matmul(tf.expand_dims(attention, 1), mem))
