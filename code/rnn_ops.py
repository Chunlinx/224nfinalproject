import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMCell, LSTMStateTuple
from tensorflow.python.ops import variable_scope as vs

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import optimizer

class AdamaxOptimizer(optimizer.Optimizer):
    """Optimizer that implements the Adamax algorithm.
    See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
    ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
    @@__init__
    """

    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, use_locking=False, name="Adamax"):
        super(AdamaxOptimizer, self).__init__(use_locking, name)
        self._lr = learning_rate
        self._beta1 = beta1
        self._beta2 = beta2

        # Tensor versions of the constructor arguments, created in _prepare().
        self._lr_t = None
        self._beta1_t = None
        self._beta2_t = None

    def _prepare(self):
        self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
        self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1")
        self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2")

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        for v in var_list:
            self._zeros_slot(v, "m", self._name)
            self._zeros_slot(v, "v", self._name)

    def _apply_dense(self, grad, var):
        lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
        beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
        beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
        if var.dtype.base_dtype == tf.float16:
            eps = 1e-7  # Can't use 1e-8 due to underflow -- not sure if it makes a big difference.
        else:
            eps = 1e-8

        v = self.get_slot(var, "v")
        v_t = v.assign(beta1_t * v + (1. - beta1_t) * grad)
        m = self.get_slot(var, "m")
        m_t = m.assign(tf.maximum(beta2_t * m + eps, tf.abs(grad)))
        g_t = v_t / m_t

        var_update = state_ops.assign_sub(var, lr_t * g_t)
        return control_flow_ops.group(*[var_update, m_t, v_t])

    def _apply_sparse(self, grad, var):
        raise NotImplementedError("Sparse gradient updates are not supported.")

class MatchLSTMCell(LSTMCell):

    def __init__(self, num_units, h_q, p_len, q_len):
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

    def __init__(self, Hr, num_units, p_len, loss, model=''):
        super(AnsPtrLSTMCell, self).__init__(num_units)
        self._cell = LSTMCell(num_units)   # 400
        self._output_size = p_len
        if model == 'sequence':
            self.H = tf.concat([Hr, tf.zeros((tf.shape(Hr)[0], 1, 2 * num_units))], 1)
        else:   # Not concatenating for boundary model
            self.H = Hr
        self.loss = loss
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
            b_k = tf.reshape(tf.matmul(tf.reshape(F, 
                [-1, self._num_units]), w) + c, [-1, self._output_size])
            if self.loss == 'l2':
                b_k = tf.nn.softmax(b_k)
                reg_b_k = b_k
            else:
                if self.loss == 'softmax':
                    reg_b_k = tf.nn.softmax(b_k)
                elif self.loss == 'sigmoid':
                    reg_b_k = tf.nn.sigmoid(b_k)
                else:
                    raise NotImplementedError("Only allow following loss functions: l2, softmax CE, sigmoid CE") 
            # m: None, 400   
            m = tf.matmul(tf.transpose(self.H, perm=[0, 2, 1]), 
                tf.expand_dims(reg_b_k, 2))
            m = tf.reshape(m, [-1, 2 * self._num_units])
        
        return b_k, self._cell(m, state)[1]

def _linear_decode(H, num_units, p_len, loss='softmax'):

    with vs.variable_scope('ans_s', 
            initializer=tf.contrib.layers.xavier_initializer()):      
        b_s = _decode_helper(H, num_units, p_len, loss)
    with vs.variable_scope('ans_e', 
            initializer=tf.contrib.layers.xavier_initializer()):
        b_e = _decode_helper(H, num_units, p_len, loss)
    
    return b_s, b_e    

def _decode_helper(H, num_units, p_len, loss):
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
    if loss == 'l2':
        b_k = tf.reshape(tf.nn.softmax(tf.matmul(tf.reshape(F, 
            [-1, num_units]), w) + c), [-1, p_len])
    elif loss == 'softmax' or 'sigmoid':
        b_k = tf.reshape(tf.matmul(tf.reshape(F, 
            [-1, num_units]), w) + c, [-1, p_len])
    else:
        raise NotImplementedError("Only allow following loss functions: l2, softmax CE, sigmoid CE")
    return b_k


