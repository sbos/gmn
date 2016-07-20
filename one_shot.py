import tensorflow as tf
import numpy as np
import scg
import sys
from threading import Thread
from multiprocessing import Pool, Process
import matplotlib.pyplot as plt


data_dim = 28*28
episode_length = 10


class GenerativeModel:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        # raw_data = np.load('data/train_small.npz')
        # train_data = []
        # for cl in raw_data.files:
        #     train_data.append(raw_data[cl][None, :, :])
        # train_data = np.concatenate(train_data, axis=0)

        self.prior = scg.Normal(50)
        self.h1 = scg.Affine(50, 200, fun='prelu', init=scg.he_normal)
        self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
        # self.bias = tf.Variable(train_data.mean(axis=0).mean(axis=0).astype(np.float32))
        self.logit = scg.Affine(200, data_dim, init=scg.he_normal)

    def generate(self, observed_name, hidden_name, **params_input):
        z = self.prior(name=hidden_name)
        logit = self.logit(input=self.h2(input=self.h1(input=z, **params_input)), name=observed_name + '_logit')
        return scg.Bernoulli()(logit=logit, name=observed_name)


class RecognitionModel:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        self.h1 = scg.Affine(data_dim, 200, fun='prelu', init=scg.he_normal)
        self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
        self.mu = scg.Affine(200, 50, init=scg.he_normal)
        self.sigma = scg.Affine(200, 50, init=scg.he_normal)

    def recognize(self, input_data, observed_name, hidden_name, **theta):
        obs = scg.Constant(value=input_data)(name=observed_name)
        h1 = self.h1(input=obs)
        h2 = self.h2(input=h1, **theta)
        mu = self.mu(input=h2)
        sigma = self.sigma(input=h2)
        z = scg.Normal(50)(mu=mu, pre_sigma=sigma, name=hidden_name)
        return z


class VAE:
    @staticmethod
    def hidden_name(step):
        return 'z_' + str(step)

    @staticmethod
    def observed_name(step):
        return 'x_' + str(step)

    def __init__(self, input_data, hidden_dim, gen, rec):
        self.gen = gen(hidden_dim)
        self.rec = rec(hidden_dim)

        self.z = []
        self.x = []

        for i in xrange(episode_length):
            current_data = input_data[:, i, :]

            self.z.append(self.rec.recognize(current_data, VAE.observed_name(i),
                                             VAE.hidden_name(i)))
            self.x.append(self.gen.generate(VAE.observed_name(i), VAE.hidden_name(i)))

    def sample(self, cache=None):
        for i in xrange(episode_length):
            cache = self.z[i].backtrace(cache)
            cache = self.x[i].backtrace(cache)
        return cache

    def lower_bound(self, cache):
        gen_ll = {}
        rec_ll = {}

        vlb = 0.

        for i in xrange(episode_length):
            scg.likelihood(self.x[i], cache, gen_ll)
            scg.likelihood(self.z[i], cache, rec_ll)
            vlb += gen_ll[VAE.observed_name(i)] + gen_ll[VAE.hidden_name(i)] - rec_ll[VAE.hidden_name(i)]

        return tf.reduce_mean(vlb) / episode_length

data_queue = tf.FIFOQueue(1000, tf.float32, shapes=[episode_length, data_dim])

new_data = tf.placeholder(tf.float32, [None, episode_length, data_dim])
enqueue_op = data_queue.enqueue_many(new_data)
batch_size = 20
input_data = data_queue.dequeue_many(batch_size)
binarized = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(input_data)), input_data), tf.float32)

vae = VAE(binarized, 50, GenerativeModel, RecognitionModel)
train_samples = vae.sample(None)
vlb = vae.lower_bound(train_samples)

ema = tf.train.ExponentialMovingAverage(0.99)
avg_op = ema.apply([vlb])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.placeholder(tf.float32)
opt_op = tf.train.AdamOptimizer(epsilon=1e-4, learning_rate=learning_rate, use_locking=False).minimize(-vlb, global_step=global_step)

gen_cache = vae.x[0].backtrace(batch=5)

with tf.control_dependencies([opt_op]):
    train_op = tf.group(avg_op)

avg_vlb = ema.average(vlb)

with tf.Session() as sess:
    def put_new_data(data, batch):
        import numpy as np

        for j in xrange(batch.shape[0]):
            batch[j] = data[np.random.choice(data.shape[0], 10),
                            np.random.choice(data.shape[1], 10)]

        np.true_divide(batch, 255., out=batch, casting='unsafe')
        sess.run(enqueue_op, feed_dict={new_data: batch})

    def data_loop(coordinator=None):
        raw_data = np.load('data/train_small.npz')
        train_data = []
        for cl in raw_data.files:
            train_data.append(raw_data[cl][None, :, :])
        train_data = np.concatenate(train_data, axis=0)
        batch = np.zeros((1, episode_length, data_dim))
        # test_data = np.load('data/test_small.npz')

        while coordinator is None or not coordinator.should_stop():
            put_new_data(train_data, batch)

    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())

    # for i in xrange(20):
    #     p = Process(data_loop)
    #     p.start()

    data_threads = [Thread(target=data_loop, args=[coord]) for i in xrange(30)]

    for t in data_threads:
        t.start()
    # coord.join(data_threads)

    # train_data = np.load('data/train_small.npz')

    log_file = open('log.txt', 'w')
    for epochs, lr in zip([900, 900, 900], [1e-3, 3e-4, 1e-4]):
        for epoch in xrange(epochs):
            for batch in xrange(24345 / batch_size / episode_length):
                lb, i, _ = sess.run([avg_vlb, global_step, train_op],
                                    feed_dict={learning_rate: lr})

                sys.stdout.write('\repoch {0}, batch {1}, lower bound: {2}'.format(epoch, i, lb))
                log_file.write('\repoch {0}, batch {1}, lower bound: {2}'.format(epoch, i, lb))
                log_file.flush()

                if np.random.rand() < 0.003:
                    sample = sess.run(gen_cache['x_0_logit'])
                    plt.imshow(sample.reshape(28 * 5, 28))
                    plt.savefig('samples.png')
            print '' \

    log_file.close()
    coord.request_stop()