import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

def bidirectional_match_lstm(Hp, Hq, fw_cell, bw_cell, output_size, T, scope):

    state_size = fw_cell.state_size.h
    batch_size = tf.shape(Hp)[0]
    fw_hidden_states, bw_hidden_states = [0] * T, [0] * T

    with vs.variable_scope(scope, initializer=tf.contrib.layers.xavier_initializer()):

        p_len = Hp.get_shape().as_list()[1]
        q_len = Hq.get_shape().as_list()[1]
        Hq = tf.reshape(Hq, [-1, state_size])

        Wq = tf.get_variable('W_q', shape=(state_size, state_size))
        # Only use forward Hp, Hq here  # None, 200, 45
        fixed_WH = tf.reshape(tf.matmul(Hq, Wq), [batch_size, state_size, q_len])
        
        h_r_fw = tf.get_variable('h_r_fw', shape=(1, state_size))
        h_r_bw = tf.get_variable('h_r_bw', shape=(1, state_size))

        # Make h_r_fw/bw to fit batch size as well
        h_r_fw = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(h_r_fw, 0), 
            [batch_size, state_size, 1]), perm=[0, 2, 1]), [-1, state_size])
        h_r_bw = tf.reshape(tf.transpose(tf.tile(tf.expand_dims(h_r_bw, 0), 
            [batch_size, state_size, 1]), perm=[0, 2, 1]), [-1, state_size])
        
        o_fw = tf.get_variable('o_fw', shape=(1, state_size))
        o_bw = tf.get_variable('o_bw', shape=(1, state_size))
        
        for i in xrange(T):
            
            hp_fw = tf.reshape(Hp[:, i, :], [-1, state_size])
            hp_bw = tf.reshape(Hp[:, -i, :], [-1, state_size])

            with vs.variable_scope('inner'):
                # Use same weights for fw/bw linear
                x_fw = _linear([hp_fw, h_r_fw], state_size, True)
                tf.get_variable_scope().reuse_variables()
                x_bw = _linear([hp_bw, h_r_bw], state_size, True)

            x_fw = tf.transpose(tf.tile(tf.expand_dims(x_fw, 0), 
                [batch_size, q_len, 1]), perm=[0, 2, 1])
            x_bw = tf.transpose(tf.tile(tf.expand_dims(x_bw, 0), 
                [batch_size, q_len, 1]), perm=[0, 2, 1])

            # l x Q     # try linear after debugging
            G_fw = tf.reshape(tf.tanh(fixed_WH + x_fw) , [-1, state_size])
            G_bw = tf.reshape(tf.tanh(fixed_WH + x_bw), [-1, state_size])

            with vs.variable_scope('outer'):
                # Use same weights for fw/bw linear
                attn_fw = tf.nn.softmax(_linear(G_fw, output_size, True)) # 1 x Q
                tf.get_variable_scope().reuse_variables()
                attn_bw = tf.nn.softmax(_linear(G_bw, output_size, True)) # 1 x Q

            z_fw = tf.concat([hp_fw, tf.matmul(attn_fw, Hq)], 0)
            z_bw = tf.concat([hp_bw, tf.matmul(attn_bw, Hq)], 0)

            # Use different weights here for fw/bw cell
            with vs.variable_scope('forward'):
                o_fw, h_r_fw = fw_cell(z_fw, (o_fw, h_r_fw))
                
            with vs.variable_scope('backward'):
                o_bw, h_r_bw = bw_cell(z_bw, (o_bw, h_r_bw))

            h_r_fw = h_r_fw.h
            h_r_bw = h_r_bw.h
            fw_hidden_states[i] = h_r_fw
            bw_hidden_states[i] = h_r_bw
            tf.get_variable_scope().reuse_variables()

    fw_hidden_states = tf.reshape(tf.stack(fw_hidden_states), [-1, state_size, p_len])
    bw_hidden_states = tf.reshape(tf.stack(bw_hidden_states), [-1, state_size, p_len])

    return fw_hidden_states, bw_hidden_states

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
            #.get_shape().as_list()
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
        beta = tf.reshape(tf.stack(beta), [-1, state_size, p_len])
        print(beta.get_shape().as_list())
# class MatchLSTMCell(RNNCell):

# 	def __init__(self, state_size, output_size):
# 		self._cell = LSTMCell(state_size)
# 		self.state_size = state_size
# 		self.output_size = output_size

# 	def state_size(self):
# 		return self.state_size

# 	def output_size(self):
# 		return self.output_size
	
# 	def __call__(self, inputs, state, scope=None):

	 # with tf.variable_scope(scope or self.__class__.__name__):
  #           c_prev, h_prev = state
  #           x = tf.slice(inputs, [0, 0], [-1, self.state_size])
  #           q_mask = tf.slice(inputs, [0, self.state_size], [-1, self.output_size])  # [N, JQ]
  #           qs = tf.slice(inputs, [0, self.state_size + self.output_size], [-1, -1])
  #           qs = tf.reshape(qs, [-1, self.output_size, self.state_size])  # [N, JQ, d]
  #           x_tiled = tf.tile(tf.expand_dims(x, 1), [1, self.output_size, 1])  # [N, JQ, d]
  #           h_prev_tiled = tf.tile(tf.expand_dims(h_prev, 1), [1, self.output_size, 1])  # [N, JQ, d]
  #           f = tf.tanh(linear([qs, x_tiled, h_prev_tiled], self.state_size, True, scope='f'))  # [N, JQ, d]
  #           a = tf.nn.softmax(exp_mask(_linear(f, 1, True, squeeze=True, scope='a'), q_mask))  # [N, JQ]
  #           q = tf.reduce_sum(qs * tf.expand_dims(a, -1), 1)
  #           z = tf.concat(1, [x, q])  # [N, 2d]
  #           return self._cell(z, state)






