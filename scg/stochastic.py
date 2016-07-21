from core import *


def mvn_diag_density(x, mu, sigma):
    with tf.variable_scope('mvn_density'):
        return -0.5 * tf.reduce_sum(tf.square((x - mu) / sigma) +
                                    tf.log(2 * np.pi) + 2 * tf.log(sigma), 1)


def bernoulli_logit_density(x, f):
    with tf.variable_scope('bernoulli_density'):
        logp = -tf.nn.softplus(-f)
        logip = -tf.nn.softplus(f)

        return tf.reduce_sum(x * logp + (1. - x) * logip, 1)


class Normal(StochasticPrototype):
    def __init__(self, size=None):
        StochasticPrototype.__init__(self)

        self.size = size

    def noise(self, batch=1):
        return tf.random_normal(tf.pack([batch, self.size]))

    def params(self, mu=None, pre_sigma=None):
        if mu is None:
            assert self.size is not None, 'size can not be recovered'
            mu = tf.zeros((self.size,))
        if pre_sigma is None:
            sigma = tf.ones((self.size,))
        else:
            sigma = tf.nn.softplus(tf.clip_by_value(pre_sigma, -10, 10))

        return mu, sigma

    def transform(self, eps, mu=None, pre_sigma=None):
        mu, sigma = self.params(mu, pre_sigma)
        return eps * sigma + mu

    def likelihood(self, value, mu=None, pre_sigma=None):
        mu, sigma = self.params(mu, pre_sigma)
        return mvn_diag_density(value, mu, sigma)


class Bernoulli(StochasticPrototype):
    def __init__(self):
        StochasticPrototype.__init__(self)

    def flow(self, logit=None, batch=1):
        assert logit is not None

        shape = tf.shape(logit)
        eps = tf.random_uniform(shape)

        return tf.cast(tf.less_equal(eps, tf.sigmoid(logit)), logit.dtype)

    def likelihood(self, value, logit=None):
        assert logit is not None

        return bernoulli_logit_density(value, logit)
