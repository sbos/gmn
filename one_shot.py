import tensorflow as tf
import numpy as np
import scg
import sys
from threading import Thread
import matplotlib.pyplot as plt
import time
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--hidden_dim', type=int, default=50)
args = parser.parse_args()

data_dim = 28*28
episode_length = args.episode


class GenerativeModel:
    def __init__(self, hidden_dim):
        self.hidden_dim = hidden_dim

        self.prior = scg.Normal(self.hidden_dim)

        self.h1 = scg.Affine(hidden_dim, 200, fun='prelu', init=scg.he_normal)
        self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
        self.logit = scg.Affine(200, data_dim, init=scg.he_normal)

    def generate(self, observed_name, hidden_name):
        z = self.prior(name=hidden_name)
        h = self.h1(input=z)
        h = self.h2(input=h)
        logit = self.logit(input=h, name=observed_name + '_logit')
        return scg.Bernoulli()(logit=logit, name=observed_name)


class RecognitionModel:
    def __init__(self, hidden_dim, param_dim):
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim

        self.h1 = scg.Affine(data_dim + param_dim, 200, fun='prelu', init=scg.he_normal)
        self.h2 = scg.Affine(200, 200, fun='prelu', init=scg.he_normal)
        self.mu = scg.Affine(200, hidden_dim, init=scg.he_normal)
        self.sigma = scg.Affine(200, hidden_dim, init=scg.he_normal)

    def recognize(self, obs, param, hidden_name):
        h = self.h1(input=scg.concat([obs, param]))
        h = self.h2(input=h)
        mu = self.mu(input=h)
        sigma = self.sigma(input=h)
        z = scg.Normal(self.hidden_dim)(mu=mu, pre_sigma=sigma, name=hidden_name)
        return z


class ParamRecognition:
    def __init__(self, state_dim, feature_dim=100, param_dim=100):
        self.feature_dim = feature_dim
        self.param_dim = param_dim

        self.cell = scg.GRU(data_dim, state_dim, fun='prelu', init=scg.he_normal)
        # self.img_features = scg.Affine(data_dim, feature_dim, fun='prelu', init=scg.he_normal)
        # self.source_encoder = scg.Affine(feature_dim + state_dim, param_dim,
        #                                  fun='prelu', init=scg.he_normal)
        # self.query_encoder = scg.Affine(feature_dim + state_dim, param_dim,
        #                                 fun='prelu', init=scg.he_normal)

        self.source_encoder = scg.Affine(data_dim + state_dim, param_dim,
                                         fun='prelu', init=scg.he_normal)
        self.query_encoder = scg.Affine(data_dim + state_dim, param_dim,
                                        fun='prelu', init=scg.he_normal)

    def update(self, state, obs):
        state = self.cell(input=obs, state=state)
        return state

    def encode_source(self, state, obs):
        # features = self.img_features(input=obs)
        features = obs
        return features, self.source_encoder(input=scg.concat([features, state]))

    def encode_query(self, state, obs):
        # features = self.img_features(input=obs)
        features = obs
        return self.query_encoder(input=scg.concat([features, state]))

    def build_memory(self, state, observations, time_step):
        if time_step == 0:
            return None

        sources = [None] * time_step
        features = [None] * time_step
        for i in xrange(time_step):
            features[i], sources[i] = self.encode_source(state, observations[i])

            def transform(input=None):
                return tf.expand_dims(input, 1)

            features[i] = scg.apply(transform, input=features[i])
            sources[i] = scg.apply(transform, input=sources[i])
        features, sources = scg.concat(features, 1), scg.concat(sources, 1)
        return sources

    def query(self, sources, state, obs):
        if sources is None:
            return scg.BatchRepeat()(input=scg.Constant(tf.zeros([self.param_dim]))(),
                                     batch=scg.StealBatch()(input=state))

        query = self.encode_query(state, obs)
        attention = scg.Attention()(mem=sources, key=query)
        return scg.AttentiveReader()(attention=attention, mem=sources)


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
        state_dim = 100
        param_dim = 100
        feature_dim = 100

        with tf.variable_scope('both') as vs:
            self.init_state = scg.Constant(tf.Variable(tf.zeros((state_dim,))))()
            self.both_vars = [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('generation') as vs:
            self.gen = gen(hidden_dim)

            self.gen_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('recognition') as vs:
            self.rec = rec(hidden_dim, param_dim)
            self.par = par

            if par is not None:
                self.par = par(state_dim, feature_dim, param_dim)

            self.rec_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        self.z = []
        self.x = []

        # allocating observations
        self.obs = [None] * (episode_length+1)
        for t in xrange(episode_length+1):
            self.obs[t] = [None] * episode_length
            for j in xrange(episode_length):
                current_data = input_data[:, j, :]
                self.obs[t][j] = scg.Constant(value=current_data)(name=VAE.observed_name(t, j))

        state = scg.BatchRepeat()(batch=scg.StealBatch()(input=self.obs[0][0]),
                                  input=self.init_state)
        for timestep in xrange(episode_length+1):
            self.z.append([])
            self.x.append([])

            sources = self.par.build_memory(state, self.obs[timestep], timestep)

            for j in xrange(min(timestep+1, episode_length)):
                param = self.par.query(sources, state, self.obs[timestep][j])

                self.z[timestep].append(self.rec.recognize(self.obs[timestep][j],
                                                           state,
                                                           VAE.hidden_name(timestep, j)))
                self.x[timestep].append(self.gen.generate(VAE.observed_name(timestep, j),
                                                          VAE.hidden_name(timestep, j)))

            if self.par is not None and timestep < episode_length:
                state = self.par.update(state, self.obs[timestep+1][timestep])

    def sample(self, cache=None):
        for i in xrange(episode_length):
            for j in xrange(i+1):
                cache = self.z[i][j].backtrace(cache)
                cache = self.x[i][j].backtrace(cache)
        return cache

    def importance_weights(self, cache):
        gen_ll = {}
        rec_ll = {}

        # weights[t][i] -- likelihood ratio for the i-th object after t objects has been seen
        weights = [0.] * (episode_length+1)

        for i in xrange(episode_length+1):
            for j in xrange(0, min(i+1, episode_length)):
                scg.likelihood(self.z[i][j], cache, rec_ll)
                scg.likelihood(self.x[i][j], cache, gen_ll)

            weights[i] = [None] * episode_length

            for j in xrange(min(i+1, episode_length)):
                local_ll = gen_ll[VAE.observed_name(i, j)] + gen_ll[VAE.hidden_name(i, j)] \
                           - rec_ll[VAE.hidden_name(i, j)]
                weights[i][j] = local_ll
            for j in xrange(i+1, episode_length):
                weights[i][j] = tf.zeros(tf.shape(weights[i][0]))

        weights = tf.pack(weights)

        return weights

    def lower_bound(self, weights, randomize=True):
        vlb_gen = [0.] * episode_length
        vlb_rec = [0.] * episode_length

        for i in xrange(episode_length):
            vlb_gen[i] += tf.reduce_sum(tf.reduce_mean(weights[i+1, :i+1, :], [1]))
            vlb_rec[i] += tf.reduce_sum(tf.reduce_mean(weights[i, :i+1, :], [1]))

        if randomize:
            vlb_gen, vlb_rec = tf.pack(vlb_gen), tf.pack(vlb_rec)

            logits = tf.expand_dims(tf.log([1. / episode_length] * episode_length), 0)
            random_idx = tf.cast(tf.multinomial(logits, 1)[0, 0], tf.int32)
            vlb_gen = tf.squeeze(tf.slice(vlb_gen, [random_idx], [1]))
            vlb_rec = tf.squeeze(tf.slice(vlb_rec, [random_idx], [1]))
        else:
            vlb_gen, vlb_rec = sum(vlb_gen), sum(vlb_rec)

        return vlb_gen, vlb_rec

    def predictive_lb(self, weights):
        ll = [0. for i in xrange(episode_length)]

        for i in xrange(len(ll)):
            ll[i] += tf.reduce_mean(weights[i, i, :])

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

vae = VAE(binarized, args.hidden_dim, GenerativeModel, RecognitionModel, ParamRecognition)
train_samples = vae.sample(None)
weights = vae.importance_weights(train_samples)

test_samples = vae.sample(None)
test_weights = vae.importance_weights(test_samples)

train_pred_ll = vae.predictive_lb(weights)

vlb_gen, vlb_rec = vae.lower_bound(weights, randomize=False)

ema = tf.train.ExponentialMovingAverage(0.99)
avg_op = ema.apply([vlb_gen, train_pred_ll])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.placeholder(tf.float32)

epoch_passed = tf.Variable(0)
increment_passed = epoch_passed.assign_add(1)

#vars_losses = zip([vae.gen_vars, vae.rec_vars], [-vlb_gen, -vlb_rec])
vars_losses = zip([vae.gen_vars + vae.rec_vars], [-vlb_gen])


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

reconstructions = [None] * episode_length
for i in xrange(episode_length):
    reconstructions[i] = tf.sigmoid(train_samples[VAE.observed_name(i, i) + '_logit'][0, :])
reconstructions = tf.pack(reconstructions)
original_input = input_data[0, :, :]


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

saver = tf.train.Saver()
with tf.Session() as sess:
    def data_loop(coordinator=None):
        train_data = load_data('data/train_small_aug10.npz')
        batch = np.zeros((1, episode_length, data_dim))
        # test_data = np.load('data/test_small.npz')

        while coordinator is None or not coordinator.should_stop():
            put_new_data(train_data, batch)
            sess.run(enqueue_op, feed_dict={new_data: batch})

    coord = tf.train.Coordinator()
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        saver.restore(sess, args.checkpoint)
    else:
        sess.run(tf.initialize_all_variables())

    data_threads = [Thread(target=data_loop, args=[coord]) for i in xrange(1)]

    for t in data_threads:
        t.start()


    def test():
        test_data = load_data('data/test_small_aug4.npz')
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

    num_epochs = 0
    done_epochs = epoch_passed.eval(sess)
    log_file = open('log.txt', 'w')
    for epochs, lr in zip([150, 150, 150], [1e-3, 3e-4, 1e-4]):
        for epoch in xrange(epochs):
            if num_epochs < done_epochs:
                num_epochs += 1
                continue

            epoch_started = time.time()
            for batch in xrange(24345 / batch_size / episode_length):

                # lb, i, _ = sess.run([avg_vlb, global_step, train_op], feed_dict={learning_rate: lr})
                lb, pred_lb, i, _ = sess.run([avg_vlb, avg_pred_lb, global_step, train_op],
                                    feed_dict={learning_rate: lr})

                sys.stdout.write('\repoch {0}, batch {1} '.format(epoch, i))
                for t in xrange(episode_length):
                    sys.stdout.write(' %.2f' % pred_lb[t])
                log_file.write('\repoch {0}, batch {1}, lower bound: {2}'.format(epoch, i, lb))
                log_file.flush()

                if np.random.rand() < 0.01:
                    sample, original = sess.run([reconstructions, original_input])
                    plt.matshow(np.hstack([sample.reshape(28 * episode_length, 28),
                                           original.reshape(28 * episode_length, 28)]),
                                cmap=plt.get_cmap('Greys'))
                    plt.savefig('samples.png')
                    plt.close()

            print
            print 'time for epoch: %f' % (time.time() - epoch_started)

            sess.run(increment_passed)
            if epoch % 10 == 0 and args.checkpoint is not None:
                saver.save(sess, args.checkpoint)

            if epoch % 20 == 0 and epoch > 0:
                test()

    log_file.close()
    coord.request_stop()