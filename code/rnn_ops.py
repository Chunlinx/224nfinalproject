import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

class MatchLSTMCell(LSTMCell):

    def __init__(self, num_units, h_q, p_len, q_len, batch_size, scope=None):
        super(MatchLSTMCell, self).__init__(num_units)
        self._cell = LSTMCell(num_units)
        self._num_units = num_units
        self._state_size = LSTMStateTuple(num_units, num_units)
        self.p_len = p_len
        self.q_len = q_len
        self._output_size = num_units
        self.batch_size = batch_size
        self.Hq = tf.reshape(h_q, [-1, num_units])

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or self.__class__.__name__, 
                    initializer=tf.contrib.layers.xavier_initializer()):
            hp, h_r = inputs, state.h
            Wq = tf.get_variable('Wq', shape=(self._num_units, self._num_units))
        
            # None, 200, 45
            fixed_WH = tf.reshape(tf.matmul(self.Hq, Wq), [-1, 
                self._num_units, self.p_len])

            # Use same weights for fw/bw linear
            with vs.variable_scope('inner'):
                x = _linear([hp, h_r], self._num_units, True)
            x = tf.reshape(tf.tile(tf.expand_dims(x, 0), [self.batch_size,
                1, self.p_len]), [-1,  self._num_units, self.p_len])
            G = tf.reshape(tf.tanh(fixed_WH + x), [-1, self._num_units])

            # Use same weights for fw/bw linear
            with vs.variable_scope('outer'):
                attn = tf.nn.softmax(_linear(G, self._output_size, True)) # 1 x Q
            z = tf.concat([hp, tf.matmul(attn, self.Hq)], 0)

        return self._cell(z, state)

class AnsPtrLSTMCell(LSTMCell):

    def __init__(self, Hr, num_units, output_size, batch_size):
        super(MatchLSTMCell, self).__init__(num_units)
        self._cell = LSTMCell(state_size)   #  200
        self._state_size = LSTMStateTuple(num_units, num_units)
        self._output_size = num_units   # for linear: l = 200 
        self.batch_size = batch_size
        self.H = Hr

    def state_size(self):
        return self._state_size

    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or self.__class__.__name__, 
            initializer=tf.contrib.layers.xavier_initializer()):

            self.V = tf.get_variable('V', shape=(state_size, 2 * state_size))
            self.h_k = tf.get_variable('h_k', shape=(1, state_size))
            self.o_a = tf.get_variable('o_a', shape=(1, state_size))


def answer_pointer_lstm(cell, H_r, state_size, scope):

    with vs.variable_scope(scope, initializer=tf.contrib.layers.xavier_initializer()):

        p_len = H_r.get_shape().as_list()[2]
        batch_size = tf.shape(H_r)[0]
        beta = [0] * p_len

        o_a = tf.get_variable('o_a', shape=(1, state_size))

        V = tf.get_variable('V', shape=(2 * state_size, state_size))
        c = tf.get_variable('c', shape=(1, 1))

        h_k = tf.get_variable('h_k', shape=(1, state_size))
        h_k = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(h_k, 0),
                [batch_size, 1, 1]), perm=[0, 2, 1]), [-1, state_size])
        v_cp = tf.get_variable('v_cp', shape=(state_size, 1))

        H_r = tf.reshape(H_r, [-1, 2 * state_size])
        fixed_VH = tf.reshape(tf.matmul(H_r, V), [batch_size, state_size, p_len])

        for i in xrange(p_len):

            with vs.variable_scope('linear'):
                x = _linear(h_k, state_size, True)
                tf.get_variable_scope().reuse_variables()

            x = tf.tile(tf.expand_dims(x, 2), [batch_size, 1, p_len])
            F = tf.tanh(fixed_VH + x)   # None, 200, 750
            F = tf.reshape(F, [-1, state_size]) # None, 200

            b_k = tf.nn.softmax(tf.reshape(tf.matmul(F, v_cp), [-1, p_len]) +\
                tf.tile(c, [batch_size, p_len]))
            beta[i] = b_k   # None, 750
            m = tf.matmul(tf.reshape(H_r, [-1, p_len]), tf.reshape(b_k, [p_len, -1]))
            m = tf.reshape(m, [-1, 2 * state_size])  # None, 200
            o_a, h_k = cell(m, (o_a, h_k))
            h_k = h_k.h     # None, 200
            tf.get_variable_scope().reuse_variables()

        beta = tf.reshape(tf.stack(beta), [-1, p_len, p_len])
    return beta
