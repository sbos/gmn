import scg
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class ResNet:
    @staticmethod
    def res_block(input_shape, kernel_size, num_filters, init=scg.he_normal, lastfun=True):
        # conv1 = scg.Convolution2d(input_shape, kernel_size, num_filters, padding='SAME',
        #                           fun='prelu', init=init)
        conv2 = scg.Convolution2d(input_shape, kernel_size, num_filters, padding='SAME', init=init)
        f = scg.Nonlinearity(fun='prelu' if lastfun else None, input_shape=input_shape)

        def _apply(x):
            y = x
            # y = conv1(input=x)
            return f(input=scg.add(x, conv2(input=y)))

        return _apply

    @staticmethod
    def section(input_shape, scale_kernel, scale_filters, stride,
                block_kernel, num_blocks=1, shortcut=True, downscale=True, lastfun=True):
        init = scg.norm_init(scg.he_normal)

        conv = scg.Convolution2d(input_shape, scale_kernel, scale_filters,
                                 stride, padding='VALID', fun='prelu', init=scg.he_normal,
                                 transpose=False if downscale else True)

        if shortcut:
            scale = scg.Convolution2d(input_shape, [1, 1], scale_filters,
                                      padding='VALID', init=init,
                                      transpose=False if downscale else True)

            scaled_shape = input_shape[:2] + [scale_filters]

            if downscale:
                pool = scg.Pooling(scaled_shape, scale_kernel, [stride, stride])
            if not downscale:
                pool = scg.ResizeImage(scaled_shape, float(conv.shape[0]) / float(scaled_shape[0]))

        blocks = [ResNet.res_block(conv.shape, block_kernel, scale_filters, init=init) for l in xrange(num_blocks - 1)]
        blocks.append(ResNet.res_block(conv.shape, block_kernel, scale_filters, init=init, lastfun=lastfun))

        def _apply(x):
            h = conv(input=x)
            for layer in xrange(num_blocks):
                h = blocks[layer](h)
            if shortcut:
                h = scg.add(pool(input=scale(input=x)), h)
            return h

        return _apply


class Memory:
    @staticmethod
    def build(entries):
        mem = []
        for entry in entries:
            def transform(input=None):
                return tf.expand_dims(input, 1)

            entry = scg.apply(transform, input=entry)
            mem.append(entry)
        return scg.concat(mem, 1)


class SetRepresentation:
    def __init__(self, proto_dim, matching_dim, hidden_dim, num_dummies=1):
        self.proto_dim = proto_dim
        self.matching_dim = matching_dim
        self.hidden_dim = hidden_dim

        self.cell = scg.GRU(proto_dim + hidden_dim, hidden_dim, fun='prelu', init=scg.he_normal)

        self.dummy_match = []
        self.dummy_proto = []
        for i in xrange(num_dummies):
            self.dummy_proto.append(scg.Constant(tf.Variable(tf.random_uniform([proto_dim],
                                                                               minval=-1. / proto_dim,
                                                                               maxval=1. / proto_dim),
                                                             trainable=True))())
            self.dummy_match.append(scg.Constant(tf.Variable(tf.random_uniform([matching_dim],
                                                                               minval=-1. / proto_dim,
                                                                               maxval=1. / proto_dim),
                                                             trainable=True))())

        self.init_state = scg.Constant(tf.Variable(tf.random_uniform([hidden_dim],
                                                                     minval=-1. / proto_dim, maxval=1. / proto_dim),
                                                   trainable=True))()
        self.match = scg.Affine(self.proto_dim, self.matching_dim, fun='prelu', init=scg.he_normal)

    def recognize(self, obs, timestep, query, num_steps, dummy=True, strength=lambda state: 1.):
        # assert num_steps > 0
        state = scg.batch_repeat(self.init_state, obs[0])

        data = obs[:timestep]
        if dummy:
            data += [scg.batch_repeat(dummy, state) for dummy in self.dummy_proto]
        proto_mem = Memory.build(data)

        data = [self.match(input=obs[t]) for t in xrange(timestep)]
        if dummy:
            data += [scg.batch_repeat(dummy, state) for dummy in self.dummy_match]
        match_mem = Memory.build(data)

        if num_steps == 0:
            def avg(input=None):
                return tf.reduce_mean(input, 1)

            r = scg.apply(avg, input=proto_mem)
            state = self.cell(input=scg.concat([r, state]), state=state)
            return r, state

        r = None
        for step in xrange(num_steps):
            q = query(state)
            a = scg.Attention()(mem=match_mem, key=q, strength=strength(state))
            r = scg.AttentiveReader()(attention=a, mem=proto_mem)
            state = self.cell(input=scg.concat([r, state]), state=state)

        return r, state


def put_new_data(data, batch, max_classes, classes=None, conditional=False):
    import numpy as np

    if classes is None:
        classes = np.random.choice(data.shape[0], [batch.shape[0], max_classes])
    else:
        classes = np.repeat(classes[None, :], batch.shape[0], 0)

    for j in xrange(batch.shape[0]):
        # classes_idx = [424, 424, 323, 323, 424, 424, 323, 323, 323, 323]
        # objects_idx = [4, 11, 2, 6, 18, 19, 0, 3, 10, 13]
        # classes_idx = [2, 7, 2, 7, 2, 7, 2, 7, 7, 2]
        # objects_idx = [0, 1, 101, 102, 203, 204, 305, 306, 307, 308]
        if not conditional:
            classes_idx = np.random.choice(classes[j], batch.shape[1])
        else:
            classes_idx = np.concatenate([classes[j], np.random.choice(classes[j], batch.shape[1] - max_classes)])
        objects_idx = np.random.choice(data.shape[1], batch.shape[1])
        # print classes_idx, objects_idx
        batch[j] = data[classes_idx, objects_idx]
    return classes


def load_data(path):
    raw_data = np.load(path)
    data = []
    min_size = min([raw_data[f].shape[0] for f in raw_data.files])
    max_value = max([raw_data[f].max() for f in raw_data.files])
    for cl in raw_data.files:
        class_data = raw_data[cl][:min_size]
        class_data = class_data.reshape(min_size, np.prod(class_data.shape[1:]))
        np.true_divide(class_data, max_value, out=class_data, casting='unsafe')
        # reverse_data = class_data.copy()
        # reverse_data[class_data > 0.] = 0.
        # reverse_data[class_data <= 0.95] = 1.
        # data.append(reverse_data[None, :, :])
        data.append(class_data[None, :, :])
    return np.concatenate(data, axis=0)


def lower_bound(w, start_from=0):
    return tf.reduce_mean(tf.reduce_sum(w[start_from:, :], 0))


def predictive_lb(w):
    return tf.reduce_mean(w, 1)


def predictive_ll(w):
    w = tf.transpose(w)
    max_w = tf.reduce_max(w, 0)
    adjusted_w = w - max_w
    ll = tf.log(tf.reduce_mean(tf.exp(adjusted_w), 0)) + max_w
    return ll


def likelihood_classification(w, n_classes, n_samples):
    # w has shape ()
    w = tf.reshape(w, [n_classes, n_samples])
    ll = predictive_ll(w)
    return ll
    # return tf.arg_max(ll, 0)


def draw_episode(episode):
    episode_length = episode.shape[0]
    img = []
    for t in xrange(episode_length):
        img.append(episode[t].reshape(28, 28))
    img = np.hstack(img)
    plt.imshow(img)
    plt.show()
    plt.close()
