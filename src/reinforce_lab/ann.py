#!/usr/bin/env python

# Import modules
import tensorflow as tf
import numpy as np

# Import classes
from func_approx import *

class Network(FuncApprox):
    # define the placeholders
    input_ = None
    output_ = None

    optimizer_ = None
    var_init_ = None

    def __init__(self, layers_struct, output_type='deterministic' ):
        self.name_ = self.__class__.__name__

        input_dim = layers_struct[0]
        output_dim = layers_struct[-1]

        if output_type == 'stochastic':
            output_dim *= 2

        FuncApprox.__init__(self, input_dim, output_dim) 

        W = []
        b = []
        layer_output = []

        self.x_ = tf.placeholder(tf.float32, [None, self.input_dim_], name= 'input')
        self.y_ = tf.placeholder(tf.float32, [None, self.output_dim_], name= 'output')
        
        layer_output.append(self.x_)

        for i in range(1,layers_struct.size):
            W.append(tf.Variable(tf.random_normal([layers_struct[i-1], layers_struct[i]], stddev=0.03), name='W'+str(i)))
            b.append(tf.Variable(tf.random_normal([layers_struct[i]]), name='b'+str(i)))
            if i == layers_struct.size:
                layer_output.append(tf.nn.softmax(tf.add(tf.matmul(layer_output[i-1], W[i-1]), b[i-1])))
            else:
                layer_output.append(tf.nn.relu(tf.add(tf.matmul(layer_output[i-1], W[i-1]), b[i-1])))

        self.y_net_ = tf.clip_by_value(layer_output[-1], 1e-10, 0.9999999)

        self.grad_ = tf.gradients(self.y_net_, W)

        self._var_init_ = tf.global_variables_initializer()

    def predict(self, state, sess):
        sess.run(self._var_init_)
        return sess.run(self.y_net_, feed_dict={self.x_: state})

    def gradient_net(self, state, sess):
        sess.run(self._var_init_)
        return sess.run(self.grad_, feed_dict={self.x_: state})