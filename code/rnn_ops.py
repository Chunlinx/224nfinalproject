import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

class MatchLSTMCell(LSTMCell):

    def __init__(self, num_units, h_q, p_len, q_len, scope=None):
        super(MatchLSTMCell, self).__init__(num_units)
        self._cell = LSTMCell(num_units)    # 400 for bidirection
        self.p_len = p_len
        self.q_len = q_len
        self.Hq = h_q  

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

            # Hq: # None, 45, 400
            Wq = tf.get_variable('Wq', shape=(self._num_units, self._num_units))
            Wp = tf.get_variable('Wp', shape=(self._num_units, self._num_units))
            Wr = tf.get_variable('Wr', shape=(self._num_units, self._num_units))
            w = tf.get_variable('w', shape=(1, self._num_units))
            bp = tf.get_variable('bp', shape=(self._num_units,))
            b = tf.get_variable('b', shape=())

            # WHq: None, 400, 45
            WHq = tf.reshape(tf.matmul(tf.reshape(self.Hq, 
                [-1, self._num_units]), Wq), [-1, self._num_units, self.q_len])
            # Whp: None, 400
            Whp = tf.reshape(tf.matmul(hp, Wp), [-1, self._num_units])
            # Whr: None, 400
            Whr = tf.matmul(h_r, Wr)
            # x: None, 400, 45
            x = Whp + Whr + bp
            x = tf.tile(tf.expand_dims(x, 2), [1, 1, self.q_len])
            # G: None, 400, 45
            G = tf.tanh(WHq + x)
            # wG: None, 45
            wG = tf.reshape(tf.matmul(w, tf.reshape(G, 
                [self._num_units, -1])), [-1, self.q_len])
            # attn: None, 45, 1
            attn = tf.expand_dims(tf.nn.softmax(wG + b), 2)

            # term: None, 400
            term = tf.matmul(tf.transpose(self.Hq, perm=[0, 2, 1]), attn)
            term = tf.reshape(term, [-1, self._num_units])

            # z: None, 800
            z = tf.concat([hp, term], 1)

        return self._cell(z, state)

class AnsPtrLSTMCell(LSTMCell):

    def __init__(self, Hr, num_units, p_len):
        super(AnsPtrLSTMCell, self).__init__(num_units)
        self._cell = LSTMCell(num_units)   # 400
        self._output_size = p_len
        self.H = Hr     # None, 750, 400

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, state, scope=None):

        with tf.variable_scope(scope or self.__class__.__name__,
            initializer=tf.contrib.layers.xavier_initializer()):

            V = tf.get_variable('V', shape=(2 * self._num_units, self._num_units))
            W = tf.get_variable('W', shape=(self._num_units, self._num_units))
            b = tf.get_variable('b', shape=(self._num_units,))

            # VHr: None, 750, 200
            VHr = tf.reshape(tf.matmul(tf.reshape(self.H, [-1, 2 * self._num_units]), V), 
                [-1, self._output_size, self._num_units])   

            # x: None, 750, 200
            x = tf.matmul(state.h, W) + b
            x = tf.tile(tf.expand_dims(x, 1), [1, self._output_size, 1])   

            # F: None, 750, 200
            F = tf.tanh(VHr + x)
            w = tf.get_variable('w', shape=(self._num_units, 1))
            c = tf.get_variable('c', shape=())

            # b_k: None, 750       
            b_k = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(F, 
                [-1, self._num_units]), w) + c), [-1, self._output_size])
            
            # m: None, 400
            m = tf.matmul(tf.transpose(self.H, perm=[0, 2, 1]), tf.expand_dims(b_k, 2))
            m = tf.reshape(m, [-1, 2 * self._num_units])
        
        return b_k, self._cell(m, state)[1]

def _linear_decode(H, num_units, p_len, scope='', span_search=False):

    with vs.variable_scope(scope + '_ans_s', 
            initializer=tf.contrib.layers.xavier_initializer()):      
        b_s = _decode_helper(H, num_units, p_len)
    with vs.variable_scope(scope + '_ans_e', 
            initializer=tf.contrib.layers.xavier_initializer()):
        b_e = _decode_helper(H, num_units, p_len)
    
    #  @TO-DO: p(a_s) x p(a_e)
    if span_search:
        x = 1
    ans_s, ans_e = b_s, b_e
    return ans_s, ans_e

def _decode_helper(H, num_units, p_len):
    """
    A helper to do equation 7 & 8 one pass without LSTM, to avoid repeating code
    """
    V = tf.get_variable('V', shape=(2 * num_units, num_units))
    VHr = tf.reshape(tf.matmul(tf.reshape(H, [-1, 2 * num_units]), V), 
        [-1, p_len, num_units])
    b = tf.get_variable('b', shape=(num_units,))
    F = tf.tanh(VHr + b)
    w = tf.get_variable('w', shape=(num_units, 1))
    c = tf.get_variable('c', shape=())

    # b_k: None, 750       
    b_k = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(F, 
        [-1, num_units]), w) + c), [-1, p_len])
    return b_k
