import scg
import tensorflow as tf


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

        if num_steps == 0:
            return scg.batch_repeat(scg.Concat(tf.zeros([self.input_dim]))(), obs[0]), state

        data = obs[:timestep]
        if dummy:
            data += [scg.batch_repeat(self.dummy_obs, state)]
        mem = Memory.build(data)

        r = None
        for step in xrange(num_steps):
            q = query(state)
            a = scg.Attention()(mem=mem, key=q, strength=strength(state))
            r = scg.AttentiveReader()(attention=a, mem=mem)
            state = self.cell(input=scg.concat([r, state]), state=state)

        return r, state
