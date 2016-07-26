from core import *
from deterministic import dispatch_function, glorot_normal, Affine


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


class NTM(NodePrototype):
    def __init__(self, input_size, cell_size, num_cells, controller_size=512,
                 num_reads=2, num_writes=1, use_decay=0.99):
        NodePrototype.__init__(self)
        self.cell_size = cell_size
        self.num_cells = num_cells
        self.controller_size = controller_size

        self.controller = GRU(input_size, controller_size)
        self.use_decay = use_decay

        class Reader:
            def __init__(self):
                self.read_key = Affine(input_size + controller_size, cell_size,
                                       fun='tanh')()

            def read(self, controller_state, input, mem, strength=1.):
                key = self.read_key.flow(input=tf.concat(0, controller_state, input))
                return NTM.content_focus(key, mem, strength)

        class Writer:
            def __init__(self):
                self.write_key = Affine(input_size + controller_size, cell_size, fun='tanh')()
                self.write_gate = Affine(input_size + controller_size, 1, fun='sigmoid')()

            def write(self, controller_state, input, mem, w_r, w_lu):
                input = tf.concat(0, controller_state, input)
                key = tf.expand_dims(self.write_key.flow(input=input), 1)
                gate = self.write_gate.flow(input=input)

                weights = tf.expand_dims(gate * w_r + (1 - gate) * w_lu, 1)
                mem_updated = mem + tf.transpose(tf.batch_matmul(key, weights, adj_x=True))

                return weights, mem_updated

        self.readers = map(lambda _: Reader(), xrange(num_reads))
        self.writers = map(lambda _: Writer(), xrange(num_writes))


    def initial_memory(self):
        return tf.zeros(tf.pack([self.num_cells, self.cell_size]))

    @staticmethod
    def content_focus(key, mem, strength=1.):
        key_norm = tf.sqrt(tf.reduce_sum(tf.square(key), [1]))
        # shape = batch x num_cells
        mem_norm = tf.sqrt(tf.reduce_sum(tf.square(mem), [2]))
        # shape = batch x num_cells
        sim = tf.squeeze(tf.batch_matmul(tf.expand_dims(key, 1), mem, adj_y=True))
        sim /= mem_norm
        sim = tf.transpose(tf.transpose(sim) / key_norm)
        sim = tf.exp(sim * strength)
        sim_sum = tf.reduce_sum(sim, [1])
        return tf.transpose(tf.transpose(sim) / sim_sum)

    @staticmethod
    def retrieve(weights, mem):
        return tf.batch_matmul(tf.expand_dims(weights, 1), mem, adj_y=True)

    def flow(self, state=None, mem=None, input=None, use_weights=None):
        assert state is not None
        assert mem is not None
        assert input is not None

        w_r = 0.
        reads = []
        for reader in self.readers:
            loc = reader.read(state, input, mem)
            w_r += loc / len(self.readers)
            reads.append(NTM.retrieve(loc, mem))
        reads = tf.concat(0, reads)

        min_weight = tf.squeeze(-tf.nn.top_k(-use_weights, len(self.readers))[0][:, 0])
        w_lu = tf.cast(tf.less_equal(use_weights, min_weight), tf.float32)

        w_w = 0

        new_state = dict()
        new_state['use_weights'] = self.use_decay * use_weights + w_r + w_w
