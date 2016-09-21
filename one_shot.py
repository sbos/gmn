import argparse
import logging
import os
import sys
import time
from threading import Thread
from utils import ResNet

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import scg

parser = argparse.ArgumentParser()
parser.add_argument('--episode', type=int, default=10)
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--hidden-dim', type=int, default=50)
parser.add_argument('--test', type=int, default=None)
parser.add_argument('--max-classes', type=int, default=2)
parser.add_argument('--test-episodes', type=int, default=1000)
parser.add_argument('--reconstructions', action='store_const', const=True)
parser.add_argument('--generate', type=int, default=None)
parser.add_argument('--test-dataset', type=str, default='data/test_small_aug4.npz')
parser.add_argument('--train-dataset', type=str, default='data/train_small_aug10.npz')
parser.add_argument('--batch', type=int, default=20)
args = parser.parse_args()

data_dim = 28*28
episode_length = args.episode


class GenerativeModel:
    def __init__(self, hidden_dim, state_dim, param_dim):
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim

        self.hp = scg.Affine(state_dim, 200, 'prelu', scg.he_normal)
        self.mu = scg.Affine(200, self.hidden_dim, None, scg.he_normal)
        self.pre_sigma = scg.Affine(200, self.hidden_dim, None, scg.he_normal)
        self.prior = scg.Normal(self.hidden_dim)

        self.h0 = scg.Affine(hidden_dim + param_dim, 3*3*32, fun=None, init=scg.he_normal)
        self.h1 = ResNet.section([3, 3, 32], [2, 2], 32, 2, [3, 3], downscale=False)
        self.h2 = ResNet.section([6, 6, 32], [3, 3], 32, 2, [3, 3], downscale=False)
        self.h3 = ResNet.section([13, 13, 32], [4, 4], 16, 2, [3, 3], downscale=False)
        self.conv = scg.Convolution2d([28, 28, 16], [1, 1], 1, padding='VALID')

        self.strength = scg.Affine(state_dim, 1, init=scg.he_normal)

    def generate_prior(self, state, hidden_name):
        # hp = self.hp(input=state)
        # z = self.prior(name=hidden_name, mu=self.mu(input=hp),
        #                pre_sigma=self.pre_sigma(input=hp))

        z = self.prior(name=hidden_name)
        return z

    def generate(self, z, param, observed_name):
        with tf.variable_scope(observed_name + '_h0'):
            h = self.h0(input=scg.concat([z, param]))
        with tf.variable_scope(observed_name + '_h1'):
            h = self.h1(h)
        with tf.variable_scope(observed_name + '_h2'):
            h = self.h2(h)
        with tf.variable_scope(observed_name + '_h3'):
            h = self.h3(h)

        h = self.conv(input=h, name=observed_name + '_logit')
        return scg.Bernoulli()(logit=h, name=observed_name)


class RecognitionModel:
    h1 = staticmethod(ResNet.section([28, 28, 1], [4, 4], 16, 2, [3, 3]))
    h2 = staticmethod(ResNet.section([13, 13, 16], [3, 3], 32, 2, [3, 3]))
    h3 = staticmethod(ResNet.section([6, 6, 32], [2, 2], 32, 2, [3, 3]))

    def __init__(self, hidden_dim, param_dim, state_dim):
        self.hidden_dim = hidden_dim
        self.param_dim = param_dim

        self.init = scg.norm_init(scg.he_normal)

        self.mu = scg.Affine(RecognitionModel.features_dim + param_dim, hidden_dim)
        self.sigma = scg.Affine(RecognitionModel.features_dim + param_dim, hidden_dim)

        self.strength = scg.Affine(state_dim, 1, init=scg.he_normal)

    @staticmethod
    def get_features(obs):
        h = RecognitionModel.h1(obs)
        h = RecognitionModel.h2(h)
        h = RecognitionModel.h3(h)

        return h

    features_dim = 3 * 3 * 32

    def recognize(self, h, param, hidden_name):
        # h = RecognitionModel.get_features(obs)
        h = scg.concat([h, param])
        mu = self.mu(input=h, name=hidden_name + '_mu')
        sigma = self.sigma(input=h, name=hidden_name + '_sigma')
        z = scg.Normal(self.hidden_dim)(mu=mu, pre_sigma=sigma, name=hidden_name)
        return z


class ParamRecognition:
    h1 = staticmethod(ResNet.section([28, 28, 1], [4, 4], 8, 2, [3, 3]))
    h2 = staticmethod(ResNet.section([13, 13, 8], [3, 3], 16, 2, [3, 3]))
    h3 = staticmethod(ResNet.section([6, 6, 16], [2, 2], 16, 2, [3, 3]))

    feature_dim = 3 * 3 * 16

    def __init__(self, state_dim, hidden_dim, mem_dim=100, param_dim=100):
        init = scg.he_normal  # scg.norm_init(scg.he_normal)

        self.param_dim = param_dim
        self.source_encoder = scg.Affine(ParamRecognition.feature_dim + state_dim, mem_dim,
                                         fun='prelu', init=init)

        self.cell = scg.GRU(ParamRecognition.feature_dim, state_dim, fun='prelu', init=scg.norm_init(scg.he_normal))

        self.query_encoder = scg.Affine(hidden_dim + state_dim, mem_dim,
                                         fun='prelu', init=init)
        self.param_encoder = scg.Affine(ParamRecognition.feature_dim + state_dim, self.param_dim,
                                        fun='prelu', init=init)
        self.dummy_mem = scg.Constant(tf.Variable(tf.random_uniform([1, mem_dim],
                                                                    -1. / mem_dim,
                                                                    1. / mem_dim)))()
        self.dummy_param = scg.Constant(tf.Variable(tf.zeros([1, self.param_dim])))()

    def update(self, state, obs):
        state = self.cell(input=obs, state=state)
        return state

    @staticmethod
    def get_features(obs):
        h = ParamRecognition.h1(obs)
        h = ParamRecognition.h2(h)
        h = ParamRecognition.h3(h)

        return h

    def encode_source(self, state, features):
        # features = ParamRecognition.get_features(obs)
        return self.param_encoder(input=scg.concat([features, state])), \
            self.source_encoder(input=scg.concat([features, state]))
        # return self.param_encoder(input=features), \
        #     self.source_encoder(input=features)

    def encode_query(self, state, z):
        return self.query_encoder(input=scg.concat([z, state]))
        # return self.query_encoder(input=z)

    # returns parameters and features
    # latter are used to compute kernel weights
    def build_memory(self, state, observations, time_step, dummy=True):
        mem = []
        params = []
        if dummy:
            mem.append(scg.batch_repeat(self.dummy_mem, state))
            params.append(scg.batch_repeat(self.dummy_param, state))

        for t in xrange(time_step):
            param, cell = self.encode_source(state, observations[t])

            def transform(input=None):
                return tf.expand_dims(input, 1)

            param = scg.apply(transform, input=param)
            cell = scg.apply(transform, input=cell)

            params.append(param)
            mem.append(cell)

        params, mem = scg.concat(params, 1), scg.concat(mem, 1)
        return params, mem

    def query(self, resources, query, strength):
        features, sources = resources

        attention = scg.Attention()(mem=sources, key=query,
                                    strength=strength)

        return scg.AttentiveReader()(attention=attention, mem=features)


def lower_bound(w):
    vlb_gen = 0.

    for i in xrange(episode_length):
        vlb_gen += tf.reduce_mean(w[i, i, :])

    return vlb_gen


def predictive_lb(w):
    ll = [0.] * episode_length

    for i in xrange(len(ll)):
        ll[i] += tf.reduce_mean(w[i, i, :])

    return tf.pack(ll)


def predictive_ll(w):
    ll = [0.] * episode_length

    for i in xrange(len(ll)):
        max_w = tf.reduce_max(w[i, i, :])
        adjusted_w = w[i, i, :] - max_w
        ll[i] += tf.log(tf.reduce_mean(tf.exp(adjusted_w))) + max_w

    return tf.pack(ll)


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
        state_dim = 200

        with tf.variable_scope('both') as vs:
            self.init_state = scg.Constant(tf.Variable(tf.zeros((state_dim,)), trainable=True))()
            self.both_vars = [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('recognition') as vs:
            param_dim = 200
            self.rec = rec(hidden_dim, param_dim, state_dim)
            self.par = par(state_dim, hidden_dim, param_dim=param_dim)

            self.rec_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        with tf.variable_scope('generation') as vs:
            self.gen = gen(hidden_dim, state_dim, self.par.param_dim)

            self.gen_vars = self.both_vars + [var for var in tf.all_variables() if var.name.startswith(vs.name)]

        self.z = []
        self.x = []

        # allocating observations

        self.obs = [None] * (episode_length+1)
        for t in xrange(episode_length+1):
            self.obs[t] = [None] * episode_length
            for j in xrange(episode_length):
                current_data = input_data[:, j, :]
                self.obs[t][j] = scg.Constant(value=current_data, shape=[28*28])(name=VAE.observed_name(t, j))

        # pre-computing features
        self.features = []
        self.param_features = []
        for j in xrange(episode_length):
            self.features.append(RecognitionModel.get_features(self.obs[j][j]))
            self.param_features.append(ParamRecognition.get_features(self.obs[j][j]))

        state = scg.BatchRepeat()(batch=scg.StealBatch()(input=self.obs[0][0]),
                                  input=self.init_state)
        self.states = []
        self.mem = []
        self.clear_mem = []
        for timestep in xrange(episode_length+1):
            self.z.append([])
            self.x.append([])
            self.states.append(state)

            resources = self.par.build_memory(state, self.param_features, timestep)
            self.mem.append(resources)
            self.clear_mem.append(self.par.build_memory(state, self.param_features, timestep, False))

            for j in xrange(min(timestep+1, episode_length)):
                param = self.par.query(resources,
                                       # self.par.encode_source(state, self.obs[timestep][j])[1],
                                       self.par.encode_source(state, self.param_features[j])[1],
                                       self.rec.strength(input=state))

                self.z[timestep].append(self.rec.recognize(self.features[j],
                                                           param,
                                                           VAE.hidden_name(timestep, j)))

                self.x[timestep].append(self.generate(timestep, j))

            if self.par is not None and timestep < episode_length:
                state = self.par.update(state, self.param_features[timestep])

    def generate(self, timestep, j, dummy=True):
        state = self.states[timestep]
        mem = self.mem
        if not dummy:
            mem = self.clear_mem
        resources = mem[timestep]

        z_prior = self.gen.generate_prior(state, VAE.hidden_name(timestep, j))
        param = self.par.query(resources,
                               self.par.encode_query(state, z_prior),
                               self.gen.strength(input=state, name='gen_strength_%d_%d' % (timestep, j)))
        return self.gen.generate(z_prior, param, VAE.observed_name(timestep, j))

    def sample(self, cache=None):
        for i in xrange(episode_length+1):
            for j in xrange(min(episode_length, i+1)):
                cache = self.z[i][j].backtrace(cache)
                cache = self.x[i][j].backtrace(cache)
        return cache

    def importance_weights(self, cache):
        gen_ll = {}
        rec_ll = {}

        # w[t][i] -- likelihood ratio for the i-th object after t objects has been seen
        w = [0.] * (episode_length+1)

        for i in xrange(episode_length+1):
            for j in xrange(0, min(i+1, episode_length)):
                scg.likelihood(self.z[i][j], cache, rec_ll)
                scg.likelihood(self.x[i][j], cache, gen_ll)

            w[i] = [None] * episode_length

            for j in xrange(min(i+1, episode_length)):
                local_ll = gen_ll[VAE.observed_name(i, j)] + gen_ll[VAE.hidden_name(i, j)] \
                           - rec_ll[VAE.hidden_name(i, j)]
                w[i][j] = local_ll
            for j in xrange(i+1, episode_length):
                w[i][j] = tf.zeros(tf.shape(w[i][0]))

        w = tf.pack(w)

        return w


data_queue = tf.FIFOQueue(1000, tf.float32, shapes=[episode_length, data_dim])

new_data = tf.placeholder(tf.float32, [None, episode_length, data_dim])
enqueue_op = data_queue.enqueue_many(new_data)
batch_size = args.batch if args.test is None else args.test
input_data = data_queue.dequeue_many(batch_size)
binarized = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(input_data)), input_data), tf.float32)

vae = VAE(binarized, args.hidden_dim, GenerativeModel, RecognitionModel, ParamRecognition)
train_samples = vae.sample(None)
weights = vae.importance_weights(train_samples)

test_samples = vae.sample(None)
test_weights = vae.importance_weights(test_samples)

train_pred_lb = predictive_lb(weights)
train_pred_ll = predictive_ll(weights)

vlb_gen = lower_bound(weights)

ema = tf.train.ExponentialMovingAverage(0.99)
avg_op = ema.apply([vlb_gen, train_pred_lb])

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.placeholder(tf.float32)

epoch_passed = tf.Variable(0)
increment_passed = epoch_passed.assign_add(1)


def grads_and_vars(vars_and_losses):
    grads = dict()
    for vars, loss in vars_and_losses:
        for var, grad in zip(vars, tf.gradients(loss, vars)):
            if var not in grads:
                grads[var] = grad
            else:
                grads[var] += grad
    return [(grad, var) for var, grad in grads.iteritems()]

opt_opt = tf.train.AdamOptimizer(beta2=0.99, epsilon=1e-8, learning_rate=learning_rate,
                                 use_locking=False).minimize(-vlb_gen, global_step)

with tf.control_dependencies([opt_opt]):
    train_op = tf.group(avg_op)

avg_vlb = ema.average(vlb_gen)
avg_pred_lb = ema.average(train_pred_lb)

reconstructions = [None] * episode_length
for i in xrange(episode_length):
    reconstructions[i] = tf.sigmoid(train_samples[VAE.observed_name(i, i) + '_logit'][0, :])
reconstructions = tf.pack(reconstructions)
original_input = input_data[0, :, :]


def put_new_data(data, batch):
    import numpy as np

    for j in xrange(batch.shape[0]):
        random_classes = np.random.choice(data.shape[0], args.max_classes)
        classes = np.random.choice(random_classes, batch.shape[1])
        batch[j] = data[classes, np.random.choice(data.shape[1], batch.shape[1])]

    np.true_divide(batch, 255., out=batch, casting='unsafe')


def load_data(path):
    raw_data = np.load(path)
    data = []
    min_size = min([raw_data[f].shape[0] for f in raw_data.files])
    for cl in raw_data.files:
        data.append(raw_data[cl][None, :min_size, :])
    return np.concatenate(data, axis=0)

saver = tf.train.Saver()
with tf.Session() as sess:
    log = logging.getLogger()
    log.setLevel(10)
    log.addHandler(logging.StreamHandler())
    if args.checkpoint is not None:
        log.addHandler(logging.FileHandler(args.checkpoint + '.log'))

    def data_loop(coordinator=None):
        train_data = load_data(args.train_dataset) if not args.reconstructions else load_data(args.test_dataset)
        batch = np.zeros((1, episode_length, data_dim))
        # test_data = np.load('data/test_small.npz')

        while coordinator is None or not coordinator.should_stop():
            put_new_data(train_data, batch)
            sess.run(enqueue_op, feed_dict={new_data: batch})

    coord = tf.train.Coordinator()
    if args.checkpoint is not None and os.path.exists(args.checkpoint):
        print 'checkpoint found, restoring'
        saver.restore(sess, args.checkpoint)
    else:
        print 'starting from scratch'
        sess.run(tf.initialize_all_variables())

    data_threads = [Thread(target=data_loop, args=[coord]) for i in xrange(1)]

    for t in data_threads:
        t.start()


    def test(full=False):
        test_data = load_data(args.test_dataset)
        avg_predictive_ll = np.zeros(episode_length)
        batch_data = np.zeros((batch_size, episode_length, data_dim), dtype=np.float32)

        target = train_pred_lb if not full else train_pred_ll

        for j in xrange(args.test_episodes):
            if full:
                put_new_data(test_data, batch_data[:1, :, :])
                for t in xrange(1, batch_data.shape[0]):
                    batch_data[t] = batch_data[0]
            else:
                put_new_data(test_data, batch_data[:, :, :])

            pred_ll = sess.run(target, feed_dict={input_data: batch_data})
            avg_predictive_ll += (pred_ll - avg_predictive_ll) / (j+1)

            msg = '\rtesting %d' % j
            for t in xrange(episode_length):
                msg += ' %.2f' % avg_predictive_ll[t]
            sys.stdout.write(msg)
            if j == args.test_episodes-1:
                print
                log.info(msg)

    num_epochs = 0
    done_epochs = epoch_passed.eval(sess)

    if args.test is not None:
        test(full=True)
        sys.exit()
    elif args.reconstructions:
        while True:
            sample, original = sess.run([reconstructions, original_input])
            plt.matshow(np.hstack([sample.reshape(28 * episode_length, 28),
                                   original.reshape(28 * episode_length, 28)]),
                        cmap=plt.get_cmap('Greys'))
            plt.show()
            plt.close()
        sys.exit()
    elif args.generate is not None:
        assert args.generate <= episode_length

        time_step = args.generate
        num_object = args.generate+1

        # train_samples['gen_strength_%d_%d' % (time_step, num_object)] =
        # np.ones([batch_size, 1], dtype=np.float32) * 100.

        obs = vae.generate(time_step, num_object, False)
        gen_samples = dict()
        obs.backtrace(gen_samples, batch=episode_length)

        data = load_data(args.test_dataset)
        input_batch = np.zeros([batch_size, episode_length, data_dim])

        logits = tf.sigmoid(gen_samples[VAE.observed_name(time_step, num_object) + '_logit'])
        # strength = train_samples['gen_strength_%d_%d' % (time_step, num_object)]

        while True:
            put_new_data(data, input_batch)
            for j in xrange(1, input_batch.shape[0]):
                input_batch[j] = input_batch[0]

            f, axs = plt.subplots(1, 11, sharey=True, squeeze=True)
            axs[0].matshow(input_batch[0].reshape(input_batch.shape[1] * 28, 28), cmap=plt.get_cmap('gray'))
            axs[0].set_yticklabels(())
            axs[0].set_xticklabels(())
            plt.subplots_adjust(wspace=0.001)
            axs[0].axis('off')

            for ax in axs[1:]:
                img = sess.run(logits, feed_dict={input_data: input_batch})
                ax.matshow(img.reshape(input_batch.shape[1] * 28, 28), cmap=plt.get_cmap('Greys'))
                ax.set_yticklabels(())
                ax.set_xticklabels(())
                ax.title.set_visible(False)
                plt.subplots_adjust(wspace=0.001)
                ax.axis('off')
            plt.show()
            plt.close()

        sys.exit()

    for epochs, lr in zip([250, 250, 250], [1e-3, 3e-4, 1e-4]):
        for epoch in xrange(epochs):
            if num_epochs < done_epochs:
                num_epochs += 1
                continue

            epoch_started = time.time()
            total_batches = 24345 / batch_size / episode_length
            for batch in xrange(total_batches):
                lb, pred_lb, i, _ = sess.run([avg_vlb, avg_pred_lb, global_step, train_op],
                                             feed_dict={learning_rate: lr})

                msg = '\repoch {0}, batch {1} '.format(epoch, i)
                for t in xrange(episode_length):
                    msg += ' %.2f' % pred_lb[t]
                sys.stdout.write(msg)
                if batch == total_batches-1:
                    print
                    log.info(msg)

            log.debug('time for epoch: %f', (time.time() - epoch_started))

            sess.run(increment_passed)
            if epoch % 30 == 0 and args.checkpoint is not None:
                saver.save(sess, args.checkpoint)

            if epoch % 20 == 0 and epoch > 0:
                test()

    coord.request_stop()
