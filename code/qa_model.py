from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

from evaluate import exact_match_score, f1_score

logging.basicConfig(level=logging.INFO)


def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    else:
        assert (False)
    return optfn


class Encoder(object):
    def __init__(self, size, state_size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

        self.n_classes = 2  # O or Answer
        self.state_size = state_size

    # def add_placeholders(self):

    def encode(self, inputs, seq_len_vec, encoder_state_input):
        """
        In a generalized encode function, you pass in your inputs,
        masks, and an initial hidden state input into this function.

        :param inputs: Symbolic representations of your input
        :param seq_len_vec: the actual length of the input (for masking)
        :param masks: this is to make sure tf.nn.dynamic_rnn doesn't iterate
                      through masked steps
        :param encoder_state_input: (Optional) pass this as initial hidden state
                                    to tf.nn.dynamic_rnn to build conditional representations
        :return: an encoded representation of your input.
                 It can be context-level representation, word-level representation,
                 or both.
        """
        # weights = {
        #     'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        #     'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
        # }
        # biases = {
        #     'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #     'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2]))
        # }

        # l1 = tf.nn.sigmoid(tf.add(tf.matmul(inputs, 
        #     weights['encoder_h1']), biases['encoder_b1']))
        # l2 = tf.nn.sigmoid(tf.add(tf.matmul(inputs, 
        #     weights['encoder_h2']), biases['encoder_b2']))

        # weight = tf.get_variable("W_encoder", shape=[2 * self.state_size, self.state_size])

        lstm_forward_cell = tf.contrib.rnn.LSTMCell(self.state_size)
        lstm_backward_cell = tf.contrib.rnn.LSTMCell(self.state_size)

        output, output_state_fw, output_state_bw = tf.contrib.rnn.bidirectional_dynamic_rnn(lstm_forward_cell, 
            lstm_backward_cell, inputs, dtype=tf.float32, sequence_length=seq_len_vec, 
            initial_state_fw=encoder_state_input, initial_state_bw=encoder_state_input)
        
        # encodings = tf.nn.sigmoid(output)
        return output

class Decoder(object):
    def __init__(self, output_size, state_size):
        self.output_size = output_size
        self.state_size = state_size

    def decode(self, knowledge_rep):
        """
        takes in a knowledge representation
        and output a probability estimation over
        all paragraph tokens on which token should be
        the start of the answer span, and which should be
        the end of the answer span.

        :param knowledge_rep: it is a representation of the paragraph and question,
                              decided by how you choose to implement the encoder
        :return:
        """
        # weights = {
        #     'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
        #     'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input]))
        # }

        # biases = {
        #     'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
        #     'decoder_b2': tf.Variable(tf.random_normal([n_input]))
        # }

        # Concatenation of both forward and backward
        lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.state_size * 2)
        lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.state_size * 2)

        output, output_state_fw, output_state_bw = tf.contrib.rnn.bidirectional_dynamic_rnn(lstm_forward_cell, 
            lstm_backward_cell, knowledge_rep, dtype=tf.float32, 
            initial_state_fw=encoder_state_input, initial_state_bw=encoder_state_input)

        return output

class QASystem(object):
    def __init__(self, encoder, decoder, batch_size, train_dir, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # Save your model parameters/checkpoints here
        self.train_dir = train_dir  
        self.encoder = encoder
        self.decoder = decoder

        self.batch_size = batch_size
        # ==== set up placeholder tokens ========


        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        pass


    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        inputs, seq_len = extract_input(self.train_dir)
        encoded = self.encoder(inputs, seq_len, tf.get_variable('hidden_init', shape=[self.batch_size, 
            self.encoder.state_size], initializer=tf.contrib.layers.xavier_initializer()))

        


    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        with vs.variable_scope("loss"):
            pass

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            pass

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        input_feed = {}

        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x

        output_feed = []

        outputs = session.run(output_feed, input_feed)

        return outputs

    def answer(self, session, test_x):

        yp, yp2 = self.decode(session, test_x)

        a_s = np.argmax(yp, axis=1)
        a_e = np.argmax(yp2, axis=1)

        return (a_s, a_e)

    def validate(self, sess, valid_dataset):
        """
        Iterate through the validation dataset and determine what
        the validation cost is.

        This method calls self.test() which explicitly calculates validation cost.

        How you implement this function is dependent on how you design
        your data iteration function

        :return:
        """
        valid_cost = 0

        for valid_x, valid_y in valid_dataset:
          valid_cost = self.test(sess, valid_x, valid_y)


        return valid_cost

    def evaluate_answer(self, session, dataset, sample=100, log=False):
        """
        Evaluate the model's performance using the harmonic mean of F1 and Exact Match (EM)
        with the set of true answer labels

        This step actually takes quite some time. So we can only sample 100 examples
        from either training or testing set.

        :param session: session should always be centrally managed in train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :param sample: how many examples in dataset we look at
        :param log: whether we print to std out stream
        :return:
        """

        f1 = 0.
        em = 0.

        if log:
            logging.info("F1: {}, EM: {}, for {} samples".format(f1, em, sample))

        return f1, em

    def train(self, session, dataset):
        """
        Implement main training loop

        TIPS:
        You should also implement learning rate annealing (look into tf.train.exponential_decay)
        Considering the long time to train, you should save your model per epoch.

        More ambitious appoarch can include implement early stopping, or reload
        previous models if they have higher performance than the current one

        As suggested in the document, you should evaluate your training progress by
        printing out information every fixed number of iterations.

        We recommend you evaluate your model performance on F1 and EM instead of just
        looking at the cost.

        :param session: it should be passed in from train.py
        :param dataset: a representation of our data, in some implementations, you can
                        pass in multiple components (arguments) of one dataset to this function
        :return:
        """

        # some free code to print out number of parameters in your model
        # it's always good to check!
        # you will also want to save your model parameters in self.train_dir
        # so that you can use your trained model to make predictions, or
        # even continue training

        tic = time.time()
        params = tf.trainable_variables()
        num_params = sum(map(lambda t: np.prod(tf.shape(t.value()).eval()), params))
        toc = time.time()
        logging.info("Number of params: %d (retreival took %f secs)" % (num_params, toc - tic))
