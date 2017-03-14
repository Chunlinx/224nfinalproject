import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

class MatchLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, h_p, h_q, output_size, state_size, batch_size, scope=None):
        super(MatchLSTMCell, self).__init__()
        self._cell = LSTMCell(num_units)
        self.h_p = h_p
        self.h_q = h_q
        self._state_size = state_size
        self._output_size = output_size
        self.batch_size = batch_size
        self.hidden_states = []

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or self.__class__.__name__, 
            initializer=tf.contrib.layers.xavier_initializer()):

            Wq = tf.get_variable('Wq', shape=(self._state_size, self._state_size))
            # Make all of these member variables
            p_len = self.h_p.get_shape().as_list()[1]
            q_len = self.h_q.get_shape().as_list()[1]
            Hq = tf.reshape(self.h_q, [-1, self._state_size])

            # Only use forward Hp, Hq here  # None, 200, 45
            fixed_WH = tf.reshape(tf.matmul(Hq, Wq), [self.batch_size, self._state_size, q_len])

            h_r = tf.get_variable('h_r', shape=(1, self._state_size))

            # Make h_r_fw/bw to fit batch size as well
            h_r = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(h_r, 0),
                [self.batch_size, self._state_size, 1]), perm=[0, 2, 1]), [-1, self._state_size])

            o = tf.get_variable('o', shape=(1, self._state_size))

            hp = tf.reshape(state, [-1, self._state_size])
            # tf.get_variable_scope().reuse_variables()
            # Use same weights for fw/bw linear
            with vs.variable_scope('inner') as scope:
                x = _linear([hp, h_r], self._state_size, True)
                scope.reuse_variables()

            x = tf.transpose(tf.tile(tf.expand_dims(x, 0),
                                    [self.batch_size, q_len, 1]), perm=[0, 2, 1])

            # l x Q     # try linear after debugging
            G = tf.reshape(tf.tanh(fixed_WH + x), [-1, self._state_size])

            # Use same weights for fw/bw linear
            with vs.variable_scope('outer') as scope:
                attn = tf.nn.softmax(_linear(G, self._output_size, True)) # 1 x Q
                scope.reuse_variables()

            z = tf.concat([hp, tf.matmul(attn, Hq)], 0)

            # Use different weights here for fw/bw cell
            o, h_r = self._cell(z, (o, h_r))

            self.hidden_states.append(h_r)

        return inputs, self._cell(z, state)

def bidirectional_match_lstm(Hp, Hq, fw_cell, bw_cell, output_size, T, num_units, scope):

    state_size = fw_cell.state_size.h
    batch_size = tf.shape(Hp)[0]
    # fw_hidden_states, bw_hidden_states = [0] * T, [0] * T

    # Define the MatchLSTMCell cell for the bidirectional_match_lstm
    fw_cell = MatchLSTMCell(num_units, Hp, Hq, output_size, state_size, batch_size)
    bw_cell = MatchLSTMCell(num_units, Hp, Hq, output_size, state_size, batch_size)

    # @TODO: Define what the inputs are
    inputs = [Hp, Hq]

    match_outputs, match_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                                                    Hq, sequence_length=None, dtype=tf.float32)

    # fw_hidden_states = tf.reshape(tf.stack(fw_hidden_states), [-1, state_size, p_len])
    # bw_hidden_states = tf.reshape(tf.stack(bw_hidden_states), [-1, state_size, p_len])

    # return fw_hidden_states, bw_hidden_states
    return match_outputs, match_states

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
