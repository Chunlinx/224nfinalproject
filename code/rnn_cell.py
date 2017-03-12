import tensorflow as tf
from tensorflow.contrib.rnn import DropoutWrapper, RNNCell, LSTMStateTuple, LSTMCell
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin


class MatchLSTMCell(RNNCell):

	def __init__(self, state_size, output_size, time_step, query_embeddings):
		# State size is actually embedding dim
        self._state_size = state_size
        self._output_size = output_size
        self._T = time_step
        self._eQ = query_embeddings
        self._cell = LSTMCell(state_size)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def __call__(self, inputs, init_state, scope=None):

    	with vs.variable_scope(scope, True):
    		W_q = tf.get_variable('W_q', shape=[self._state_size, 
    			self._state_size], initializer=tf.contrib.layers.xavier_initializer())
    		for i in xrange(self._T):
    			if i == 0:
    				h_r = tf.get_variable('h_r', shape=(1, 
    					self._state_size)) if init_state is None else init_state 
    			x = _linear([:, i, :])


