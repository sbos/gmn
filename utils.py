import scg


class ResNet:
    @staticmethod
    def res_block(input_shape, kernel_size, num_filters, init=scg.he_normal, lastfun=True):
        conv1 = scg.Convolution2d(input_shape, kernel_size, num_filters, padding='SAME',
                                  fun='prelu', init=init)
        conv2 = scg.Convolution2d(input_shape, kernel_size, num_filters, padding='SAME', init=init)
        f = scg.Nonlinearity(fun='prelu' if lastfun else None, input_shape=input_shape)

        def _apply(x):
            return f(input=scg.add(x, conv2(input=conv1(input=x))))
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
