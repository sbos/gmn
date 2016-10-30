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
            if downscale:
                pool = scg.Pooling(scale.shape, scale_kernel, [stride, stride])
            if not downscale:
                pool = scg.ResizeImage(scale.shape, float(conv.shape[0]) / float(scale.shape[0]))

        blocks = [ResNet.res_block(conv.shape, block_kernel, scale_filters, init=init) for l in xrange(num_blocks-1)]
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
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.cell = scg.GRU(input_dim + hidden_dim, hidden_dim, fun='prelu', init=scg.he_normal)

        self.dummy_obs = scg.Constant(tf.Variable(tf.random_uniform([input_dim],
                                                                    minval=-1. / input_dim,
                                                                    maxval=1. / input_dim), trainable=True))()
        # self.dummy_obs = scg.Constant(tf.Variable(tf.ones([input_dim]), trainable=True))()
        self.init_state = scg.Constant(tf.Variable(tf.random_uniform([hidden_dim],
                                                                     minval=-1. / input_dim, maxval=1. / input_dim),
                                                   trainable=True))()

    def recognize(self, obs, timestep, query, num_steps, dummy=True, strength=lambda state: 1.):
        # assert num_steps > 0
        state = scg.batch_repeat(self.init_state, obs[0])

        data = obs[:timestep]
        if dummy:
            data += [scg.batch_repeat(self.dummy_obs, state)]
        mem = Memory.build(data)

        if num_steps == 0:
            def avg(input=None):
                return tf.reduce_mean(input, 1)
            r = scg.apply(avg, input=mem)
            state = self.cell(input=scg.concat([r, state]), state=state)
            return r, state

        r = None
        for step in xrange(num_steps):
            q = query(state)
            a = scg.Attention()(mem=mem, key=q, strength=strength(state))
            r = scg.AttentiveReader()(attention=a, mem=mem)
            state = self.cell(input=scg.concat([r, state]), state=state)

        return r, state


def put_new_data(data, batch, max_classes, classes=None):
    import numpy as np

    if classes is None:
        classes = np.random.choice(data.shape[0], [batch.shape[0], max_classes])
    else:
        classes = np.repeat(classes[None, :], batch.shape[0], 0)

    for j in xrange(batch.shape[0]):
        classes_idx = np.random.choice(classes[j], batch.shape[1])
        batch[j] = data[classes_idx, np.random.choice(data.shape[1], batch.shape[1])]

    np.true_divide(batch, 255., out=batch, casting='unsafe')
    return classes


def load_data(path):
    raw_data = np.load(path)
    data = []
    min_size = min([raw_data[f].shape[0] for f in raw_data.files])
    for cl in raw_data.files:
        data.append(raw_data[cl][None, :min_size, :])
    return np.concatenate(data, axis=0)


def lower_bound(w):
    return tf.reduce_sum(tf.reduce_mean(w, 1))


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
