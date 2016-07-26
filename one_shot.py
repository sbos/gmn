import tensorflow as tf
import numpy as np
import scg
import sys
from threading import Thread
import matplotlib.pyplot as plt


data_dim = 28*28
episode_length = 10


class GenerativeModel:
    def __init__(self, hidden_dim, state_dim):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.prior = scg.Normal(self.hidden_dim)

        self.h1 = scg.Affine(hidden_dim + state_dim, 100, fun='prelu', init=scg.he_normal)
        # self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
        self.logit = scg.Affine(100, data_dim, init=scg.he_normal)

    def generate(self, state, observed_name, hidden_name):
        z = self.prior(name=hidden_name)
        # h = self.h1(input=state)
        h = self.h1(input=scg.Concat()(a=z, b=state))
        logit = self.logit(input=h, name=observed_name + '_logit')
        return scg.Bernoulli()(logit=logit, name=observed_name)


class RecognitionModel:
    def __init__(self, hidden_dim, state_dim):
        self.hidden_dim = hidden_dim
        self.state_dim = state_dim

        self.h = scg.Affine(state_dim + data_dim, 100, fun='prelu', init=scg.he_normal)
        self.mu = scg.Affine(100, 50, init=scg.he_normal)
        self.sigma = scg.Affine(100, 50, init=scg.he_normal)

    def recognize(self, obs, state, hidden_name):
        h = self.h(input=scg.Concat()(a=obs, b=state))
        mu = self.mu(input=h)
        sigma = self.sigma(input=h)
        z = scg.Normal(50)(mu=mu, pre_sigma=sigma, name=hidden_name)
        return z


class ParamRecognition:
    def __init__(self, state_dim):
        self.cell = scg.GRU(data_dim, state_dim, fun='prelu', init=scg.he_normal)

    def update(self, state, obs):
        state = self.cell(input=obs, state=state)
        return state


class VAE:
    @staticmethod
    def hidden_name(step, j):
        return 'z_' + str(step) + '_' + str(j)

    @staticmethod
    def observed_name(step, j):
        return 'x_' + str(step) + '_' + str(j)

    @staticmethod
    def params_name(step):
        return 'theta_' + str(step)

    def __init__(self, input_data, hidden_dim, gen, rec, par=None):
        state_dim = 256

        with tf.variable_scope('both') as vs:
            self.init_state = scg.Constant(tf.Variable(tf.zeros((2 * state_dim,))))()
            self.both_vars = [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('generation') as vs:
            self.gen = gen(hidden_dim, state_dim)

            self.params_gen = []
            for i in xrange(episode_length+1):
                mu = scg.Slice(0, state_dim)(input=self.init_state)
                pre_sigma = scg.Slice(state_dim, state_dim)(input=self.init_state)
                self.params_gen.append(scg.Normal(state_dim)(mu=mu, pre_sigma=pre_sigma,
                                                             name=VAE.params_name(i)))

            self.gen_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('recognition') as vs:
            self.rec = rec(hidden_dim, state_dim)
            self.par = par

            if par is not None:
                self.par = par(2 * state_dim)

            self.rec_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        self.z = []
        self.x = []

        self.params_rec = []

        self.obs = [None] * (episode_length+1)
        for t in xrange(episode_length+1):
            self.obs[t] = [None] * episode_length
            for j in xrange(episode_length):
                current_data = input_data[:, j, :]
                self.obs[t][j] = scg.Constant(value=current_data)(name=VAE.observed_name(t, j))

        state = scg.BatchRepeat()(batch=scg.StealBatch()(input=self.obs[0][0]),
                                  input=self.init_state)
        for timestep in xrange(episode_length+1):
            mu = scg.Slice(0, state_dim)(input=state)
            pre_sigma = scg.Slice(state_dim, state_dim)(input=state)
            state_sample = scg.Normal(state_dim)(mu=mu, pre_sigma=pre_sigma,
                                                 name=VAE.params_name(timestep))
            self.params_rec.append(state_sample)

            self.z.append([])
            self.x.append([])

            for j in xrange(min(timestep+1, episode_length)):
                self.z[timestep].append(self.rec.recognize(self.obs[timestep][j],
                                                           state_sample,
                                                           VAE.hidden_name(timestep, j)))
                self.x[timestep].append(self.gen.generate(state_sample,
                                                          VAE.observed_name(timestep, j),
                                                          VAE.hidden_name(timestep, j)))

            if self.par is not None and timestep < episode_length:
                state = self.par.update(state, self.obs[timestep][timestep])

    def sample(self, cache=None):
        for i in xrange(episode_length):
            for j in xrange(i+1):
                cache = self.z[i][j].backtrace(cache)
                cache = self.x[i][j].backtrace(cache)
        return cache

    def importance_weights(self, cache):
        gen_ll = {}
        rec_ll = {}

        weights = [None] * (episode_length+1)
        param_weights = [None] * (episode_length+1)

        for i in xrange(episode_length+1):
            for j in xrange(0, min(i+1, episode_length)):
                scg.likelihood(self.z[i][j], cache, rec_ll)
                scg.likelihood(self.x[i][j], cache, gen_ll)

            if i > 0:
                param_weights[i] = gen_ll[VAE.params_name(i)] - rec_ll[VAE.params_name(i)]
            else:
                param_weights[i] = tf.zeros(tf.shape(gen_ll[VAE.observed_name(i, 0)]))

            weights[i] = [None] * episode_length

            for j in xrange(min(i+1, episode_length)):
                local_ll = gen_ll[VAE.observed_name(i, j)] + gen_ll[VAE.hidden_name(i, j)] \
                           - rec_ll[VAE.hidden_name(i, j)]
                weights[i][j] = local_ll
            for j in xrange(i+1, episode_length):
                weights[i][j] = tf.zeros(tf.shape(weights[i][0]))

        param_weights = tf.pack(param_weights)
        weights = tf.pack(weights)

        return param_weights, weights

    def lower_bound(self, param_weights, weights):
        vlb_gen = 0.
        vlb_rec = 0.

        for i in xrange(episode_length):
            vlb_gen += tf.reduce_mean(param_weights[i+1, :])
            vlb_rec += tf.reduce_mean(param_weights[i, :])

            vlb_gen += tf.reduce_sum(tf.reduce_mean(weights[i+1, :i+1, :], [1]))
            vlb_rec += tf.reduce_sum(tf.reduce_mean(weights[i, :i+1, :], [1]))

        return vlb_gen, vlb_rec

    def predictive_lb(self, weights):
        ll = [None] * episode_length

        for i in xrange(len(ll)):
            ll[i] = tf.reduce_mean(weights[i, i, :])

        return tf.pack(ll)

    def predictive_likelihood(self, weights):
        ll = [None] * episode_length

        for i in xrange(len(ll)):
            w_i = weights[i, i, :]
            max_w = tf.reduce_max(w_i)
            ll[i] = tf.log(tf.reduce_sum(tf.exp(w_i - max_w))) - tf.log(tf.cast(tf.shape(w_i)[0], tf.float32)) \
                    + max_w

        return tf.pack(ll)


class CustomAdam(tf.train.AdamOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-4,
                 use_locking=True, name="CustomAdam"):
        super(CustomAdam, self).__init__(learning_rate, beta1, beta2, epsilon, use_locking, name)

    def minimize(self, grads_and_vars, global_step=None, name=None):
        return self.apply_gradients(grads_and_vars, global_step, name)

data_queue = tf.FIFOQueue(1000, tf.float32, shapes=[episode_length, data_dim])

new_data = tf.placeholder(tf.float32, [None, episode_length, data_dim])
enqueue_op = data_queue.enqueue_many(new_data)
batch_size = 20
input_data = data_queue.dequeue_many(batch_size)
binarized = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(input_data)), input_data), tf.float32)

vae = VAE(binarized, 50, GenerativeModel, RecognitionModel, ParamRecognition)
train_samples = vae.sample(None)
weights = vae.importance_weights(train_samples)

test_samples = vae.sample(None)
test_weights = vae.importance_weights(test_samples)

train_pred_ll = vae.predictive_lb(weights[1])
test_pred_ll = vae.predictive_likelihood(test_weights[1])

vlb_gen, vlb_rec = vae.lower_bound(*weights)

ema = tf.train.ExponentialMovingAverage(0.99)
avg_op = ema.apply([vlb_gen, train_pred_ll])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.placeholder(tf.float32)

vars_losses = zip([vae.gen_vars, vae.rec_vars], [-vlb_gen, -vlb_rec])


def grads_and_vars(vars_and_losses):
    grads = dict()
    for vars, loss in vars_and_losses:
        for var, grad in zip(vars, tf.gradients(loss, vars)):
            if var not in grads:
                grads[var] = grad
            else:
                grads[var] += grad
    return [(grad, var) for var, grad in grads.iteritems()]

opt_opt = CustomAdam(beta2=0.99, epsilon=1e-4, learning_rate=learning_rate,
                     use_locking=False).minimize(grads_and_vars(vars_losses), global_step)

gen_cache = vae.x[0][0].backtrace(batch=5)

with tf.control_dependencies([opt_opt]):
    train_op = tf.group(avg_op)

avg_vlb = ema.average(vlb_gen)
avg_pred_lb = ema.average(train_pred_ll)


def put_new_data(data, batch):
    import numpy as np

    for j in xrange(batch.shape[0]):
        random_classes = np.random.choice(data.shape[0], 1)
        classes = np.random.choice(random_classes, batch.shape[1])
        batch[j] = data[classes, np.random.choice(data.shape[1], batch.shape[1])]

    np.true_divide(batch, 255., out=batch, casting='unsafe')


def load_data(path):
    raw_data = np.load(path)
    data = []
    for cl in raw_data.files:
        data.append(raw_data[cl][None, :, :])
    return np.concatenate(data, axis=0)

with tf.Session() as sess:
    def data_loop(coordinator=None):
        train_data = load_data('data/train_small_aug10.npz')
        batch = np.zeros((1, episode_length, data_dim))
        # test_data = np.load('data/test_small.npz')

        while coordinator is None or not coordinator.should_stop():
            put_new_data(train_data, batch)
            sess.run(enqueue_op, feed_dict={new_data: batch})

    coord = tf.train.Coordinator()
    sess.run(tf.initialize_all_variables())

    # for i in xrange(20):
    #     p = Process(data_loop)
    #     p.start()

    data_threads = [Thread(target=data_loop, args=[coord]) for i in xrange(1)]

    for t in data_threads:
        t.start()


    def test():
        print
        test_data = load_data('data/test_small.npz')
        avg_pred_ll = np.zeros(episode_length)
        batch = np.zeros((batch_size, episode_length, data_dim), dtype=np.float32)
        for j in xrange(1000):
            put_new_data(test_data, batch)
            pred_ll = sess.run(train_pred_ll, feed_dict={input_data: batch})
            avg_pred_ll += (pred_ll - avg_pred_ll) / (j+1)
            sys.stdout.write('\rtesting %d' % j)
            for t in xrange(episode_length):
                sys.stdout.write(' %.2f' % avg_pred_ll[t])
        print

    log_file = open('log.txt', 'w')
    for epochs, lr in zip([500, 500, 500], [1e-3, 3e-4, 1e-4]):
        for epoch in xrange(epochs):
            for batch in xrange(24345 / batch_size / episode_length):
                lb, pred_lb, i, _ = sess.run([avg_vlb, avg_pred_lb, global_step, train_op],
                                    feed_dict={learning_rate: lr})

                sys.stdout.write('\repoch {0}, batch {1} '.format(epoch, i))
                for t in xrange(episode_length):
                    sys.stdout.write(' %.2f' % pred_lb[t])
                log_file.write('\repoch {0}, batch {1}, lower bound: {2}'.format(epoch, i, lb))
                log_file.flush()

                if np.random.rand() < 0.003:
                    sample = sess.run(train_samples['x_2_0_logit'])
                    plt.imshow(sample.reshape(28 * 20, 28))
                    plt.savefig('samples.png')

                if epoch % 50 == 0 and batch == 0:
                    test()
            print '' \

    log_file.close()
    coord.request_stop()