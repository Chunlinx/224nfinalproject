from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time, datetime
import logging
import random
import os

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear

from qa_data import PAD_ID
from qa_util import preprocess_dataset, get_minibatch, Progbar
from evaluate import exact_match_score, f1_score
import rnn_ops

logging.basicConfig(level=logging.INFO)
FLAGS = tf.app.flags.FLAGS

def get_optimizer(opt):
    if opt == "adam":
        optfn = tf.train.AdamOptimizer
    elif opt == "sgd":
        optfn = tf.train.GradientDescentOptimizer
    elif opt == "adagrad":
        optfn = tf.train.AdagradOptimizer
    elif opt == "adadelta":
        optfn = tf.train.AdadeltaOptimizer
    else:
        assert (False)
    return optfn

class Encoder(object):
    def __init__(self, size, vocab_dim):
        self.size = size
        self.vocab_dim = vocab_dim

        self.forward_cell = tf.contrib.rnn.LSTMCell(self.size)
        self.backward_cell = tf.contrib.rnn.LSTMCell(self.size)

    def encode(self, inputs, seq_len_vec, init_fw_encoder_state, init_bw_encoder_state, scope='',
        fw_dropout=1, bw_dropout=1):
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
        with vs.variable_scope(scope):
            (fw_out, bw_out), final_state_tuple = \
                tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.DropoutWrapper(self.forward_cell,
                    output_keep_prob=fw_dropout), tf.contrib.rnn.DropoutWrapper(self.backward_cell,
                    output_keep_prob=bw_dropout), inputs, dtype=tf.float32, sequence_length=seq_len_vec,
                        initial_state_fw=init_fw_encoder_state,
                        initial_state_bw=init_bw_encoder_state)
            tf.get_variable_scope().reuse_variables()
        #(N, T, d)   N: batch size   T: time steps  d: vocab_dim
        return fw_out, bw_out, final_state_tuple

    def encode_w_attn(self, h_p, h_q, p_len, q_len, seq_len_vec,
            scope='', bidirectional=False):
        state_size = 2 * self.size if bidirectional else self.size
        # Define the MatchLSTMCell cell for the bidirectional_match_lstm
        fw_cell = rnn_ops.MatchLSTMCell(state_size, h_q, p_len, q_len)
        bw_cell = rnn_ops.MatchLSTMCell(state_size, h_q, p_len, q_len)

        # @TODO: Define what the inputs are: h_p
        outputs, states = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell,
                h_p, sequence_length=seq_len_vec, dtype=tf.float32)

        return outputs, states

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.state_size = FLAGS.state_size
        self.answer_cell = tf.contrib.rnn.LSTMCell(self.state_size)

    def decode(self, h_q, h_c):
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
        # Linear mix: h_q * W1 + h_p * W2 + b
        with vs.variable_scope('a_s'):
            a_s = _linear([h_q, h_c], self.output_size, True)
        with vs.variable_scope('a_e'):
            a_e = _linear([h_q, h_c], self.output_size, True)
        return a_s, a_e

    def decode_w_attn(self, H, p_len, bidirectional=False ,scope=''):

        state_size = 2 * self.state_size if bidirectional else self.state_size
        cell = rnn_ops.AnsPtrLSTMCell(H, state_size, p_len)

        # inputs = tf.zeros((tf.shape(H)[0], p_len, p_len))
        beta, _ = tf.nn.dynamic_rnn(cell, H, dtype=tf.float32)

        flat_beta = tf.reshape(tf.contrib.layers.flatten(beta),
            [-1, p_len * p_len])

        with tf.variable_scope('answer_pointer_s'):
            ans_s = _linear(flat_beta, p_len, True)

        with tf.variable_scope('answer_pointer_e'):
            ans_e = _linear(flat_beta, p_len, True)

        #  @TO-DO: p(a_s) x p(a_e)

        return ans_s, ans_e

class QASystem(object):
    def __init__(self, encoder, decoder, *args):
        """
        Initializes your System

        :param encoder: an encoder that you constructed in train.py
        :param decoder: a decoder that you constructed in train.py
        :param args: pass in more arguments as needed
        """
        # Save your model parameters/checkpoints here
        self.encoder = encoder
        self.decoder = decoder

        self.embeddings = tf.constant(np.load(FLAGS.embed_path)['glove'], dtype=tf.float32)
        self.context_length = FLAGS.output_size
        self.question_length = FLAGS.question_size

        # ==== set up placeholder tokens ========
        self.context_placeholder = tf.placeholder(tf.int32, (None, self.context_length),
            name='context_input')
        self.question_placeholder = tf.placeholder(tf.int32, (None, self.question_length),
            name='question_input')
        self.context_mask_placeholder = tf.placeholder(tf.int32, (None,),
            name='context_mask_input')
        self.question_mask_placeholder = tf.placeholder(tf.int32, (None,),
            name='question_mask_input')
        self.answer_start_label_placeholder = tf.placeholder(tf.int32, (None,),
            name='a_s_label')
        self.answer_end_label_placeholder = tf.placeholder(tf.int32, (None,),
            name='a_e_label')
        self.fw_dropout_placeholder = tf.placeholder(tf.float32, (), name='fw_dropout')
        self.bw_dropout_placeholder = tf.placeholder(tf.float32, (), name='bw_dropout')
        # ==== assemble pieces ====
        with tf.variable_scope("qa", initializer=tf.uniform_unit_scaling_initializer(1.0)):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
        self.train_op = get_optimizer('adam')(FLAGS.learning_rate).minimize(self.loss)

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # put the network together (combine add loss, etc)

        q_fw_o, q_bw_o, q_h_tup = self.encoder.encode(self.question_embed,
            self.question_mask_placeholder, None, None, scope='question_encode',
            fw_dropout=self.fw_dropout_placeholder, bw_dropout=self.bw_dropout_placeholder)
        p_fw_o, p_bw_o, _ = self.encoder.encode(self.context_embed,
            self.context_mask_placeholder, q_h_tup[0], q_h_tup[1],
            scope='context_encode', fw_dropout=self.fw_dropout_placeholder,
            bw_dropout=self.bw_dropout_placeholder)

        # None, 750, 400
        H_p = tf.concat([p_fw_o, p_bw_o], 2)
        # None, 45, 400
        H_q = tf.concat([q_fw_o, q_bw_o], 2)

        # Bidirectional embeddings
        outputs, _ = self.encoder.encode_w_attn(H_p, H_q, self.context_length,
            self.question_length, self.context_mask_placeholder,
            scope='encode_attn', bidirectional=True)
        H_r = tf.concat([outputs[0], outputs[1]], 2)  # None, 750, 400

        # This is the predict op
        self.a_s, self.a_e = self.decoder.decode_w_attn(H_r, self.context_length,
            scope='decode_attn', bidirectional=True)

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        # Predict 2 numbers (in paper)
        with vs.variable_scope("loss"):

            mask = tf.sequence_mask(self.context_mask_placeholder, self.context_length)
            int_mask = tf.cast(mask, tf.float32)
            mask_s = self.a_s + tf.log(int_mask)
            mask_e = self.a_e + tf.log(int_mask)    # None, 750

            loss_vec_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.answer_start_label_placeholder,
                logits=mask_s)
            loss_vec_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.answer_end_label_placeholder,
                logits=mask_e)

            self.loss = tf.reduce_mean(loss_vec_1 + loss_vec_2)

    def setup_embeddings(self):
        """
        Loads distributed word representations based on placeholder tokens
        :return:
        """
        with vs.variable_scope("embeddings"):
            # Choosing to use constant
            self.context_embed = tf.reshape(tf.nn.embedding_lookup(self.embeddings,
                self.context_placeholder, name='context_embeddings'),
                    [-1, self.context_length, FLAGS.embedding_size])

            self.question_embed = tf.reshape(tf.nn.embedding_lookup(self.embeddings,
                self.question_placeholder, name='question_embeddings'),
                    [-1, self.question_length, FLAGS.embedding_size])

    def optimize(self, session, train_x, train_y):
        """
        Takes in actual data to optimize your model
        This method is equivalent to a step() function
        :return:
        """
        # fill in this feed_dictionary like:
        # input_feed['train_x'] = train_x
        input_feed = {}
        input_feed[self.context_placeholder] = train_x[0][0]
        input_feed[self.question_placeholder] = train_x[1][0]
        input_feed[self.context_mask_placeholder] = np.clip(train_x[0][1], 0, FLAGS.output_size)
        input_feed[self.question_mask_placeholder] = np.clip(train_x[1][1], 0, FLAGS.question_size)
        input_feed[self.answer_start_label_placeholder] = np.clip(train_y[0], 0, self.context_length)
        input_feed[self.answer_end_label_placeholder] = np.clip(train_y[1], 0, self.context_length)
        input_feed[self.fw_dropout_placeholder] = FLAGS.fw_dropout
        input_feed[self.bw_dropout_placeholder] = FLAGS.bw_dropout
        # Gradient norm
        output_feed = [self.loss, self.train_op]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_context, valid_question, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        # fill in this feed_dictionary like:
        # input_feed['valid_x'] = valid_x
        input_feed = {}
        input_feed[self.context_placeholder] = valid_context[0]
        input_feed[self.context_mask_placeholder] = valid_context[1]
        input_feed[self.question_placeholder] = valid_question[0]
        input_feed[self.question_mask_placeholder] = valid_question[1]
        input_feed[self.answer_start_label_placeholder] = valid_y[0]
        input_feed[self.answer_end_label_placeholder] = valid_y[1]
        input_feed[self.fw_dropout_placeholder] = 1.
        input_feed[self.bw_dropout_placeholder] = 1.

        output_feed = [self.loss]
        loss = session.run(output_feed, input_feed)
        return loss

    def decode(self, session, test_x):
        """
        Returns the probability distribution over different positions in the paragraph
        so that other methods like self.answer() will be able to work properly
        :return:
        """
        # fill in this feed_dictionary like:
        # input_feed['test_x'] = test_x
        input_feed = {}

        input_feed[self.context_placeholder] = test_x[0][0]
        input_feed[self.question_placeholder] = test_x[1][0]
        input_feed[self.context_mask_placeholder] = test_x[0][1]
        input_feed[self.question_mask_placeholder] = test_x[1][1]
        input_feed[self.fw_dropout_placeholder] = 1.
        input_feed[self.bw_dropout_placeholder] = 1.

        output_feed = [self.a_s, self.a_e]
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
        valid_c, valid_q, valid_a_s, valid_a_e = valid_dataset
        valid_cost = self.test(sess, valid_c, valid_q, (valid_a_s[0], valid_a_e[0]))

        return valid_cost

    def evaluate_answer(self, session, dataset_train, dataset_val, sample=100, log=False):
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
        # Sample each for half of total samples
        f1, em = 0., 0.
        context_padded_train_, question_padded_train_, \
            a_s_train_, a_e_train_ = dataset_train

        context_padded_val_, question_padded_val_, \
            a_s_val_, a_e_val_ = dataset_val
        # Context, query, ans labels are read correctly

        train_sample_idx = np.random.choice(np.arange(len(a_s_train_)), int(sample / 2), replace=False)
        val_sample_idx = np.random.choice(np.arange(len(a_e_val_)), sample - int(sample / 2), replace=False)

        context_train, context_train_len = context_padded_train_
        context_val, context_val_len = context_padded_val_

        query_train, query_train_len = question_padded_train_
        query_val, query_val_len = question_padded_val_

        merged_data = [(context_train[i], context_train_len[i],
            query_train[i], query_train_len[i], a_s_train_[i], a_e_train_[i]) for i in range(len(a_s_train_))]\
                + [(context_val[i], context_val_len[i], query_val[i],
                    query_val_len[i], a_s_val_[i], a_e_val_[i]) for i in range(len(a_e_val_))]

        selected_data = random.sample(merged_data, sample)
        feed_data = [((np.reshape(tp[0], (1, self.context_length)), np.reshape(tp[1], (1,))),
            (np.reshape(tp[2], (1, self.question_length)), np.reshape(tp[3], (1,))),
                tp[0][tp[4]: tp[5] + 1]) for tp in selected_data]
        ground_truth = [d[2].tolist() for d in feed_data]

        # Get the model back
        saver = tf.train.Saver()

        # Use the last checkpoint
        saver.restore(session, saver.last_checkpoints[-1])
        for i, d in enumerate(feed_data):
            a_s, a_e = self.answer(session, (d[0], d[1]))
            answer = d[0][0].flatten()[int(a_s): int(a_e) + 1].tolist()
            f1 += f1_score(answer, ground_truth[i]) / sample
            if exact_match_score(answer, ground_truth[i]):
                em += 1. / sample

        if log:
            logging.info("F1: {}, EM: {}%, for {} samples".format(f1, em * 100, sample))

        return f1, em

    def train(self, session, dataset, save_train_dir):
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
        train_data = preprocess_dataset(dataset['train'],
            self.context_length, self.question_length)
        val_data = preprocess_dataset(dataset['val'],
            self.context_length, self.question_length)

        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2)
        for epoch in range(FLAGS.epochs):
            prog = Progbar(target=1 + int(len(dataset['train']['context']) / FLAGS.batch_size))
            for i, batch in enumerate(get_minibatch(train_data, FLAGS.batch_size)):

                a_s_batch = batch[2]
                a_e_batch = batch[3]

                # Not annealing at this point yet
                # Return loss and gradient probably
                train_loss, _ = self.optimize(session, (batch[0], batch[1]),
                    (a_s_batch, a_e_batch))
                # print('Epoch {}, {}th batch: training loss {}'.format(epoch, i, train_loss))
                prog.update(i + 1, [("train loss", train_loss)])

            # Save model here for each epoch
            results_path = FLAGS.train_dir + "/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            model_path = results_path + "model.weights/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            saver.save(sess, model_path, global_step=epoch)
            val_loss = self.validate(session, val_data)
            print('Epoch {}, validation loss {}'.format(epoch, val_loss))

            # at the end of epoch
            self.evaluate_answer(session, train_data, val_data, FLAGS.evaluate)

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
