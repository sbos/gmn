import scg
import tensorflow as tf


s = tf.Session()
prior = scg.Normal(size=3)(batch=scg.Constant(2)(), name='z')
h1 = scg.Affine(3, 4)(input=prior, name='h1')
mu = scg.Slice(0, 2)(input=h1, name='mu')
sigma = scg.Slice(2, 2)(input=h1, name='sigma')
z = scg.Normal(2)(mu=mu, pre_sigma=sigma, name='z2')
cache = {}
z.backtrace(cache)

ll = scg.likelihood(z, cache)
print ll

s.run(tf.initialize_all_variables())
#print s.run([cache['mu'], cache['sigma']])
print s.run([cache['h1'], cache['z2'], ll['z2']])
