from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import json

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from qa_model import preprocess_dataset
from qa_util import load_data
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)
tf.app.flags.DEFINE_string("data_dir", "../data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("train_dir", "../train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "../data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "../data/squad/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_integer("train_embeddings", 0, "1 for training embeddings, 0 for not.")
tf.app.flags.DEFINE_integer("ensemble", 0, "1 for using ensemble, 0 for not.")
tf.app.flags.DEFINE_boolean("swap_memory", True, "True for allowing swaping memory to CPU when GPU memory is exhausted, False for not.")
tf.app.flags.DEFINE_integer("config", 0, "Specify under which config to run the train")
FLAGS = tf.app.flags.FLAGS

def initialize_model(session, model, train_dir):
    ckpt = tf.train.get_checkpoint_state(train_dir)
    v2_path = ckpt.model_checkpoint_path + ".index" if ckpt else ""
    if ckpt and (tf.gfile.Exists(ckpt.model_checkpoint_path) or tf.gfile.Exists(v2_path)):
        logging.info("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        logging.info("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
        logging.info('Num params: %d' % sum(v.get_shape().num_elements() for v in tf.trainable_variables()))
    return model

def initialize_vocab(vocab_path):
    if tf.gfile.Exists(vocab_path):
        rev_vocab = []
        with tf.gfile.GFile(vocab_path, mode="rb") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip('\n') for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found.", vocab_path)

def get_normalized_train_dir(train_dir):
    """
    Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    file paths saved in the checkpoint. This allows the model to be reloaded even
    if the location of the checkpoint files has moved, allowing usage with CodaLab.
    This must be done on both train.py and qa_answer.py in order to work.
    """
    global_train_dir = '/tmp/cs224n-squad-train'
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    os.symlink(os.path.abspath(train_dir), global_train_dir)
    return global_train_dir

def load_config(current_config):
    """
    Load the current config specified by the user under which the train will
    run.
    """
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
        '../config', 'config_' + str(current_config) + '.json')
    if not config_path:
        raise Exception('Must specify a config for the QA system!')
    with open(config_path) as data_file:
        data = json.load(data_file)
        # Hyperparameters
        FLAGS.epochs = data['epochs']
        FLAGS.state_size = data['state_size']
        FLAGS.output_size = data['output_size']
        FLAGS.question_size = data['question_size']
        FLAGS.embedding_size = data['embedding_size']

        # Dropout layers
        FLAGS.context_fw_dropout = data['context_fw_dropout']
        FLAGS.context_bw_dropout = data['context_bw_dropout']
        FLAGS.query_fw_dropout = data['query_fw_dropout']
        FLAGS.query_bw_dropout = data['query_bw_dropout']
        FLAGS.match_fw_dropout = data['match_fw_dropout']
        FLAGS.match_bw_dropout = data['match_bw_dropout']
        FLAGS.as_fw_dropout = data['as_fw_dropout']
        FLAGS.as_bw_dropout = data['as_bw_dropout']
        FLAGS.ae_fw_dropout = data['ae_fw_dropout']
        FLAGS.ae_bw_dropout = data['ae_bw_dropout']

        # Learning options
        FLAGS.max_gradient_norm = data['max_gradient_norm']
        FLAGS.optimizer = data['optimizer']
        FLAGS.learning_rate = data['learning_rate']
        FLAGS.batch_size = data['batch_size']
        FLAGS.test_run = data['test_run']
        FLAGS.bidirectional_preprocess = data['bidirectional_preprocess']
        FLAGS.bidirectional_answer_pointer = data['bidirectional_answer_pointer']      
        FLAGS.model = data['model']
        FLAGS.loss = data['loss']
        FLAGS.evaluate = data['evaluate']

    print('Successfully loaded system config.')

def main(_):
    FLAGS.config = int(sys.argv[1])
    load_config(current_config=FLAGS.config)

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = load_data(FLAGS.data_dir) # ((question, context), answer)
    train_data = preprocess_dataset(dataset['train'],
        FLAGS.output_size, FLAGS.question_size)
    val_data = preprocess_dataset(dataset['val'],
        FLAGS.output_size, FLAGS.question_size)

    # print(dataset)
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)
    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, 
        "log" + '_config_' + str(FLAGS.config) + ".txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags" + '_config_' +\
        str(FLAGS.config) + ".json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)
        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, train_data, val_data, save_train_dir)
        qa.evaluate_answer(sess, train_data, val_data, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
