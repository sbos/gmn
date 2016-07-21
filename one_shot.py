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

        self.prior = scg.Normal(50)
        self.h1 = scg.Affine(50, 200, fun='prelu', init=scg.he_normal)
        self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
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

    def recognize(self, obs, hidden_name, **theta):
        h1 = self.h1(input=obs)
        h2 = self.h2(input=h1, **theta)
        mu = self.mu(input=h2)
        sigma = self.sigma(input=h2)
        z = scg.Normal(50)(mu=mu, pre_sigma=sigma, name=hidden_name)
        return z


class ParamRecognition:
    def __init__(self, state_dim, param_dim):
        self.param_dim = param_dim
        self.rnn = scg.RNN(data_dim, state_dim, fun='prelu', init=scg.he_normal)
        self.b_mu = scg.Affine(state_dim, param_dim, init=scg.he_normal)
        self.b_sigma = scg.Affine(state_dim, param_dim, init=scg.he_normal)

    def update(self, state, obs):
        state = self.rnn(input=obs, state=state)
        return state

    def get_params(self, state, param_name):
        return scg.Normal(self.param_dim)(name=param_name, mu=self.b_mu(input=state), pre_sigma=self.b_sigma(input=state))


class VAE:
    @staticmethod
    def hidden_name(step, j):
        return 'z_' + str(step) + '_' + str(j)

    @staticmethod
    def observed_name(step):
        return 'x_' + str(step)

    @staticmethod
    def params_name(step):
        return 'theta_' + str(step)

    def __init__(self, input_data, hidden_dim, gen, rec, par=None):
        with tf.variable_scope('generation') as vs:
            self.gen = gen(hidden_dim)
            self.gen_vars = [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('recognition') as vs:
            self.rec = rec(hidden_dim)
            self.init_state = None
            self.par = par
            if par is not None:
                state_dim = 512
                param_dim = 200
                self.par = par(state_dim, param_dim)
                self.init_state = scg.Constant(tf.Variable(np.zeros((1, state_dim), np.float32)))()

            self.rec_vars = [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        self.z = []
        self.x = []
        self.params_gen = []
        self.params_rec = []

        state = self.init_state
        for timestep in xrange(episode_length):
            current_data = input_data[:, timestep, :]
            obs = scg.Constant(value=current_data)(name=VAE.observed_name(timestep))

            params_input = {}
            if par is not None:
                params_input['b'] = self.par.get_params(state, VAE.params_name(timestep))
                self.params_gen.append(scg.Normal(hidden_dim)(name=VAE.params_name(timestep)))
                self.params_rec.append(params_input['b'])

            self.z.append([])
            self.x.append([])

            for j in xrange(timestep+1):
                self.z[timestep].append(self.rec.recognize(obs, VAE.hidden_name(timestep, j),
                                                           **params_input))
                self.x[timestep].append(self.gen.generate(VAE.observed_name(timestep),
                                                          VAE.hidden_name(timestep, j),
                                                          **params_input))

            if self.par is not None:
                state = self.par.update(state, obs)

    def sample(self, cache=None):
        for i in xrange(episode_length):
            for j in xrange(i+1):
                cache = self.z[i][j].backtrace(cache)
                cache = self.x[i][j].backtrace(cache)
        return cache

    def lower_bound(self, cache):
        gen_ll = {}
        rec_ll = {}

        vlb_gen = 0.
        vlb_rec = 0.

        n_gen = 0
        n_rec = 0

        for i in xrange(episode_length):
            for j in xrange(i+1):
                scg.likelihood(self.x[i][j], cache, gen_ll)
                scg.likelihood(self.z[i][j], cache, rec_ll)

            L_i = gen_ll[VAE.params_name(i)] - rec_ll[VAE.params_name(i)]
            for j in xrange(i+1):
                local_ll = gen_ll[VAE.observed_name(j)] + gen_ll[VAE.hidden_name(i, j)] \
                           - rec_ll[VAE.hidden_name(i, j)]
                if j < i:
                    L_i += local_ll
                    n_gen += 1

                vlb_rec += local_ll
                n_rec += 1

            vlb_gen += L_i

        vlb_gen = tf.reduce_mean(vlb_gen) / n_gen
        vlb_rec = tf.reduce_mean(vlb_rec) / n_rec

        return vlb_gen, vlb_rec

data_queue = tf.FIFOQueue(1000, tf.float32, shapes=[episode_length, data_dim])

new_data = tf.placeholder(tf.float32, [None, episode_length, data_dim])
enqueue_op = data_queue.enqueue_many(new_data)
batch_size = 20
input_data = data_queue.dequeue_many(batch_size)
binarized = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(input_data)), input_data), tf.float32)

vae = VAE(binarized, 50, GenerativeModel, RecognitionModel, ParamRecognition)
train_samples = vae.sample(None)
vlb_gen, vlb_rec = vae.lower_bound(train_samples)

ema = tf.train.ExponentialMovingAverage(0.99)
avg_op = ema.apply([vlb_gen])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.placeholder(tf.float32)

vars_losses = zip([vae.gen_vars, vae.rec_vars], [-vlb_gen, -vlb_rec])
opt_opts = [tf.train.AdamOptimizer(beta2=0.99, epsilon=1e-4, learning_rate=learning_rate,
                                   use_locking=False).minimize(loss, global_step=global_step,
                                                               var_list=vars)
            for vars, loss in vars_losses]

gen_cache = vae.x[0][0].backtrace(batch=5)

with tf.control_dependencies(opt_opts):
    train_op = tf.group(avg_op)

avg_vlb = ema.average(vlb_gen)

with tf.Session() as sess:
    def put_new_data(data, batch):
        import numpy as np

        for j in xrange(batch.shape[0]):
            random_classes = np.random.choice(data.shape[0], 2)
            batch[j] = data[np.random.choice(random_classes, batch.shape[1]),
                            np.random.choice(data.shape[1], batch.shape[1])]

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
    for epochs, lr in zip([500, 500, 500], [1e-3, 3e-4, 1e-4]):
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