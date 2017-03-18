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
from qa_util import *
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
        self.preprocess_cell = tf.contrib.rnn.LSTMCell(self.size)

    def encode(self, inputs, seq_len_vec, init_fw_encoder_state, init_bw_encoder_state, scope='',
        reuse=None, fw_dropout=1, bw_dropout=1):
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
        with vs.variable_scope(scope, reuse=reuse):
            (fw_out, bw_out), final_state_tuple = \
                tf.nn.bidirectional_dynamic_rnn(tf.contrib.rnn.DropoutWrapper(
                self.preprocess_cell, output_keep_prob=fw_dropout), 
                tf.contrib.rnn.DropoutWrapper(self.preprocess_cell,
                output_keep_prob=bw_dropout), inputs, dtype=tf.float32, 
                sequence_length=seq_len_vec, initial_state_fw=init_fw_encoder_state, 
                initial_state_bw=init_bw_encoder_state, swap_memory=FLAGS.swap_memory)
        #(N, T, d)   N: batch size   T: time steps  d: vocab_dim
        return fw_out, bw_out, final_state_tuple

    def encode_w_attn(self, h_p, h_q, p_len, q_len, seq_len_vec,
            scope='', reuse=None):
        state_size = 2 * self.size if FLAGS.bidirectional_preprocess else self.size
        cell = rnn_ops.MatchLSTMCell(state_size, h_q, p_len, q_len)
        with vs.variable_scope(scope, reuse=reuse):  
            # Define the MatchLSTMCell cell for the bidirectional_match_lstm
            outputs, states = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                h_p, sequence_length=seq_len_vec, dtype=tf.float32, 
                swap_memory=FLAGS.swap_memory)
        return outputs, states

class Decoder(object):
    def __init__(self, output_size):
        self.output_size = output_size
        self.state_size = FLAGS.state_size
        self.answer_cell = tf.contrib.rnn.LSTMCell(self.state_size)

    def decode(self, h_q, h_p, scope=None, reuse=None):
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
            a_s = _linear([h_q, h_p], self.output_size, True)
        with vs.variable_scope('a_e'):
            a_e = _linear([h_q, h_p], self.output_size, True)
        return a_s, a_e

    def decode_w_attn(self, H, p_len, seq_len, init_state, scope='', reuse=None):

        state_size = 2 * self.state_size if FLAGS.bidirectional_preprocess else self.state_size
        # For boundary case, just random thing with p_len time steps and 1 output size
        inputs = H if FLAGS.model == 'sequence' else tf.zeros((tf.shape(H)[0], 1, 2 * state_size))
        seq_len = seq_len if FLAGS.model == 'sequence' else tf.ones((tf.shape(H)[0],), dtype=tf.int32)
        cell = rnn_ops.AnsPtrLSTMCell(H, state_size, p_len, FLAGS.loss, model=FLAGS.model)
        with vs.variable_scope(scope, reuse=reuse):
            # Two cases
            if FLAGS.bidirectional_answer_pointer:
                init_state_fw = init_state[0] if init_state else None
                init_state_bw = init_state[1] if init_state else None
                # TODO: Make sure this is reusing variables
                (beta_fw, beta_bw), states = tf.nn.bidirectional_dynamic_rnn(cell, cell,
                    inputs, dtype=tf.float32, sequence_length=seq_len,
                    initial_state_fw=init_state_fw, initial_state_bw=init_state_bw,
                    swap_memory=FLAGS.swap_memory)
                # simply adding the result from forward and backward
                beta = beta_fw + beta_bw    # None, p_len, p_len + 1
            else:
                beta, states = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, 
                    initial_state=init_state, sequence_length=seq_len, swap_memory=FLAGS.swap_memory)
        return beta, states

    def linear_decode(self, H, p_len, scope='', span_search=False):
        state_size = 2 * self.state_size if FLAGS.bidirectional_preprocess else self.state_size
        with vs.variable_scope(scope):
            return rnn_ops._linear_decode(H, state_size, p_len, FLAGS.loss,
                span_search)

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
        if FLAGS.train_embeddings:
            self.embeddings = tf.Variable(np.load(FLAGS.embed_path)['glove'], dtype=tf.float32)
        else:
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
        with tf.variable_scope("qa", initializer=tf.contrib.layers.xavier_initializer()):
            self.setup_embeddings()
            self.setup_system()
            self.setup_loss()

        # ==== set up training/updating procedure ====
            self.train_op = self.setup_train()

    def setup_system(self):
        """
        After your modularized implementation of encoder and decoder
        you should call various functions inside encoder, decoder here
        to assemble your reading comprehension system!
        :return:
        """
        # put the network together (combine add loss, etc)
        with vs.variable_scope('setup_system'):
            q_fw_o, q_bw_o, q_h_tup = self.encoder.encode(self.question_embed,
                self.question_mask_placeholder, None, None, scope='preprocess', 
                fw_dropout=self.fw_dropout_placeholder, bw_dropout=self.bw_dropout_placeholder)
            p_fw_o, p_bw_o, p_h_tup = self.encoder.encode(self.context_embed,
                self.context_mask_placeholder, q_h_tup[0], q_h_tup[1],
                fw_dropout=self.fw_dropout_placeholder, scope='preprocess', 
                bw_dropout=self.bw_dropout_placeholder, reuse=True)

            if FLAGS.model == 'baseline':
                q_h = tf.concat([q_h_tup[0].h, q_h_tup[1].h], 1)
                p_h = tf.concat([p_h_tup[0].h, p_h_tup[1].h], 1)
                self.a_s, self.a_e = self.decoder.decode(q_h, p_h)

            else:
                if FLAGS.bidirectional_preprocess:
                    H_p = tf.concat([p_fw_o, p_bw_o], 2) # None, 750, 400
                    H_q = tf.concat([q_fw_o, q_bw_o], 2) # None, 45, 400
                else:
                    H_p, H_q = p_fw_o, q_fw_o

                outputs, _ = self.encoder.encode_w_attn(H_p, H_q, self.context_length,
                    self.question_length, self.context_mask_placeholder,
                    scope='encode_attn')
                H_r = tf.concat([outputs[0], outputs[1]], 2)  # None, 750, 400

                # These are the predict ops
                if FLAGS.model == 'sequence':
                    # beta: None, 750, 751
                    beta, _ = self.decoder.decode_w_attn(H_r, self.context_length + 1,
                        self.context_mask_placeholder, None, scope='decode_attn_seq')
                    # TODO: verify this selection is correct
                    self.a_s = tf.reshape(beta[..., 0], [-1, self.context_length])
                    self.a_e = tf.reshape(beta[..., -1], [-1, self.context_length])

                elif FLAGS.model == 'boundary':
                    # beta: None, 1, 300
                    beta_s, h_s = self.decoder.decode_w_attn(H_r, self.context_length, 
                        self.context_mask_placeholder, None, scope='decode_attn_bnd_s')

                    # Now given a_s, find a_e
                    beta_e, _ = self.decoder.decode_w_attn(H_r, self.context_length, 
                        self.context_mask_placeholder, h_s, scope='decode_attn_bnd_e')

                    self.a_s = tf.reshape(beta_s, [-1, self.context_length])
                    self.a_e = tf.reshape(beta_e, [-1, self.context_length])
                elif FLAGS.model == 'linear':
                    self.a_s, self.a_e = self.decoder.linear_decode(H_r, self.context_length, 
                        scope='encode_attn_bnd', span_search=True)
                else:
                    raise NotImplementedError("Only allow following models: baseline, MatchLSTM/sequence, MatchLSTM/boundary, MatchLSTM/linear")

    def setup_loss(self):
        """
        Set up your loss computation here
        :return:
        """
        # Predict 2 numbers (in paper)
        with vs.variable_scope("loss"):
            # sequence of 1 and 0's
            mask = tf.cast(tf.sequence_mask(self.context_mask_placeholder,
                self.context_length), tf.float32)

            if FLAGS.loss == "softmax":
                # Exp mask for softmax
                mask_s = self.a_s + tf.log(mask)
                mask_e = self.a_e + tf.log(mask)    # None, 750
                loss_s = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.answer_start_label_placeholder,
                    logits=mask_s)
                loss_e = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.answer_end_label_placeholder,
                    logits=mask_e)
            else:
                # Elementwise, 0 for l2
                mask_s = tf.multiply(self.a_s, mask)
                mask_e = tf.multiply(self.a_e, mask)
                label_s = tf.one_hot(self.answer_start_label_placeholder,
                    self.context_length, dtype=tf.float32)
                label_e = tf.one_hot(self.answer_end_label_placeholder,
                    self.context_length, dtype=tf.float32)
                if FLAGS.loss == "l2":
                    loss_s = tf.nn.l2_loss(label_s - mask_s)
                    loss_e = tf.nn.l2_loss(label_e - mask_e)
                elif FLAGS.loss == "sigmoid":
                    loss_s = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=label_s, logits=mask_s)
                    loss_e = tf.nn.sigmoid_cross_entropy_with_logits(
                        labels=label_e, logits=mask_e)
                else:
                    raise NotImplementedError("Only allow following loss functions: l2, softmax CE, sigmoid CE")
        self.loss = tf.reduce_mean(loss_s + loss_e)

    def setup_train(self):
        optimizer = get_optimizer(FLAGS.optimizer)(
            FLAGS.learning_rate)
        grad, var = zip(*optimizer.compute_gradients(self.loss))
        grad, var = list(grad), list(var)
        grad, _ = tf.clip_by_global_norm(grad, FLAGS.max_gradient_norm)
        self.grad_norm = tf.global_norm(grad)
        train_op = optimizer.apply_gradients(zip(grad, var))
        return train_op

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
        input_feed = {}
        input_feed[self.context_placeholder] = train_x[0][0]
        input_feed[self.question_placeholder] = train_x[1][0]
        input_feed[self.context_mask_placeholder] = np.clip(train_x[0][1],
            0, self.context_length)
        input_feed[self.question_mask_placeholder] = np.clip(train_x[1][1],
            0, self.question_length)
        input_feed[self.answer_start_label_placeholder] = np.clip(train_y[0],
            0, self.context_length - 1)
        input_feed[self.answer_end_label_placeholder] = np.clip(train_y[1],
            0, self.context_length - 1)
        input_feed[self.fw_dropout_placeholder] = FLAGS.fw_dropout
        input_feed[self.bw_dropout_placeholder] = FLAGS.bw_dropout

        # Gradient norm
        output_feed = [self.loss, self.train_op, self.grad_norm]
        outputs = session.run(output_feed, input_feed)

        return outputs

    def test(self, session, valid_x, valid_y):
        """
        in here you should compute a cost for your validation set
        and tune your hyperparameters according to the validation set performance
        :return:
        """
        input_feed = {}
        input_feed[self.context_placeholder] = valid_x[0][0] # None, p_len
        input_feed[self.context_mask_placeholder] = np.clip(valid_x[0][1],
            0, self.context_length)
        input_feed[self.question_placeholder] = valid_x[1][0]   # None, q_len
        input_feed[self.question_mask_placeholder] = np.clip(valid_x[1][1],
            0, self.question_length)
        input_feed[self.answer_start_label_placeholder] = np.clip(valid_y[0],
            0, self.context_length - 1)  # None
        input_feed[self.answer_end_label_placeholder] = np.clip(valid_y[1],
            0, self.context_length - 1)
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
        valid_cost = self.test(sess, (valid_c, valid_q), (valid_a_s, valid_a_e))
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
        f1, em = 0., 0.
        # Sample each for half of total samples
        feed_data, ground_truth = get_sampled_data(dataset_train,
            dataset_val, self.context_length, self.question_length, sample=sample)

        for i, d in enumerate(feed_data):
            a_s, a_e = self.answer(session, (d[0], d[1]))
            answer = d[0][0].flatten()[int(a_s): int(a_e) + 1].tolist()
            truth = ' '.join([str(s) for s in ground_truth[i]])
            ans = ' '.join([str(s) for s in answer])
            f1 += f1_score(ans, truth) / sample
            if exact_match_score(ans, truth):
                em += 1. / sample
        if log:
            logging.info("F1: {}, EM: {}%, for {} samples".format(f1, em * 100, sample))
        return f1, em

    def train(self, session, train_data, val_data, save_train_dir):
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
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
        for epoch in range(FLAGS.epochs):
            prog = Progbar(target=1 + int(len(train_data[0][0]) / FLAGS.batch_size))
            for i, batch in enumerate(get_minibatch(train_data, FLAGS.batch_size)):
                a_s_batch = batch[2]
                a_e_batch = batch[3]
                train_loss, _, grad_norm = self.optimize(session, (batch[0], batch[1]),
                    (a_s_batch, a_e_batch))
                prog.update(i + 1, [("train loss", train_loss), ("global norm", grad_norm)])
            
            # Save model here for each epoch
            results_path = FLAGS.train_dir + "/{:%Y%m%d_%H%M%S}/".format(datetime.datetime.now())
            model_path = results_path + "model.weights/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            current_model = os.path.join(model_path, "model.%s" % epoch)
            saved_path = saver.save(session, current_model, global_step=epoch)
            print('Saved model at path {}'.format(saved_path))

            # Averaging validation cost
            val_loss = 0
            for j, batch in enumerate(get_minibatch(val_data, FLAGS.batch_size)):
                val_loss += self.validate(session, batch)[0]
            print('Epoch {}, validation loss {}'.format(epoch, val_loss / (j + 1)))   # epoch

            # at the end of epoch
            result = self.evaluate_answer(session, train_data, val_data,
                FLAGS.evaluate, log=True)

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


