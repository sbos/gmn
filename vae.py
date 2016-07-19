import scg
import tensorflow as tf
import numpy as np
import sys
import matplotlib.pyplot as plt


def recognition_model(input_data):
    obs = scg.Constant(input_data)(name='x')
    h1 = scg.Affine(28*28, 200, fun='softplus')(input=obs)
    h2 = scg.Affine(200, 200, fun='softplus')(input=h1)
    mu = scg.Affine(200, 50)(input=h2)
    sigma = scg.Affine(200, 50)(input=h2)
    z = scg.Normal(50)(mu=mu, pre_sigma=sigma, name='z')
    return z


def generative_model():
    z = scg.Normal(50)(name='z')
    h1 = scg.Affine(50, 200, fun='softplus')(input=z)
    h2 = scg.Affine(200, 200, fun='softplus')(input=h1)
    logit = scg.Affine(200, 28*28)(input=h2)
    obs = scg.Bernoulli()(logit=logit, name='x')

    return obs

s = tf.Session()

input_data = tf.placeholder(tf.float32, (None, 28*28))
binarized = tf.cast(tf.less_equal(tf.random_uniform(tf.shape(input_data)), input_data), tf.float32)

gen = generative_model()
rec = recognition_model(binarized)

cache = {}
rec_ll = scg.likelihood(rec, cache)
gen_ll = scg.likelihood(gen, cache)

gen_cache = gen.backtrace(batch=5)

lower_bound = tf.reduce_mean(gen_ll['x'] + gen_ll['z'] - rec_ll['z'])
train_op = tf.train.AdamOptimizer(beta1=0.99, epsilon=1e-4).minimize(-lower_bound)

mnist = np.load('data/mnist.npz')
mnist = mnist['X'] / 255.

s.run(tf.initialize_all_variables())

while True:
    random_batch = np.random.choice(50000, 200, False)
    ll, _ = s.run([lower_bound, train_op], feed_dict={input_data: mnist[random_batch]})
    sys.stdout.write('\rlower bound: {0}'.format(ll))

    if np.random.rand() < 0.001:
        sample = s.run(gen_cache['x'])
        plt.imshow(sample.reshape(28 * 5, 28))
        plt.savefig('samples.png')
