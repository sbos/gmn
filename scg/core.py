import numpy as np
import random
import tensorflow as tf
import string

name_random = random.Random()
name_random.seed(0)


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(name_random.choice(chars) for _ in range(size))


class Node:
    def __init__(self, name, prototype, input_nodes):
        self.prototype = prototype
        self.name = name
        self.input_nodes = input_nodes

        for node in self.input_nodes.itervalues():
            assert isinstance(node, Node), node

    def backtrace(self, cache=None, callback=lambda node, value, **inputs: None, **inputs):
        if cache is None:
            cache = {}

        input_values = {}
        for channel_name, node in self.input_nodes.iteritems():
            node.backtrace(cache, callback=callback, **inputs)
            assert cache[node.name] is not None
            input_values[channel_name] = cache[node.name]

        for input_name, value in inputs.iteritems():
            assert input_name not in input_values
            if input_name in self.prototype.flow.func_code.co_varnames:
                input_values[input_name] = value

        value = cache.get(self.name, None)
        if value is None:
            with tf.variable_scope(self.name):
                value = self.prototype.flow(**input_values)
            cache[self.name] = value

        callback(self, value, **input_values)

        return cache


class NodePrototype:
    def __init__(self):
        pass

    def flow(self, **inputs):
        pass

    def __call__(self, name=None, **input_nodes):
        if name is None:
            name = id_generator()

        return Node(name, self, input_nodes)


class StochasticPrototype(NodePrototype):
    def __init__(self):
        NodePrototype.__init__(self)

    def likelihood(self, value, **inputs):
        pass

    def noise(self, batch=1):
        pass

    def transform(self, eps, **inputs):
        pass

    def params(self, **inputs):
        pass

    def flow(self, batch=None, **inputs):
        if batch is None:
            if len(inputs) > 0:
                batch = tf.shape(inputs.itervalues().next())[0]
            else:
                raise Exception('can not infer batch size')
        eps = self.noise(batch)
        x = self.transform(eps, **inputs)
        return x


def likelihood(node, cache=None, ll=None):
    if ll is None:
        ll = {}

    def likelihood_callback(current_node, value, **inputs):
        if isinstance(current_node.prototype, StochasticPrototype):
            ll[current_node.name] = current_node.prototype.likelihood(value, **inputs)

    node.backtrace(cache, callback=likelihood_callback)

    return ll
