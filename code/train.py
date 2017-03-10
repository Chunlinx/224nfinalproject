from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json

import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np

from qa_model import Encoder, QASystem, Decoder
from os.path import join as pjoin

import logging

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.005, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 10.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 32, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("epochs", 0, "Number of epochs to train.")
tf.app.flags.DEFINE_integer("state_size", 200, "Size of each model layer.")
tf.app.flags.DEFINE_integer("output_size", 750, "The output size of your model.")   # 750
tf.app.flags.DEFINE_integer("question_size", 45, "The clip/padding length of question.")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained vocabulary.")
tf.app.flags.DEFINE_string("data_dir", "../data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("train_dir", "../train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_string("optimizer", "adam", "adam / sgd")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
tf.app.flags.DEFINE_string("vocab_path", "../data/squad/vocab.dat", "Path to vocab file (default: ./data/squad/vocab.dat)")
tf.app.flags.DEFINE_string("embed_path", "../data/squad/glove.trimmed.100.npz", "Path to the trimmed GLoVe embedding (default: ./data/squad/glove.trimmed.{vocab_dim}.npz)")
tf.app.flags.DEFINE_integer("evaluate", 100, "How many samples to evaluate EM and F1 score.")

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

def load_data(data_dir):

    dataset = {'train': {'context': [], 'question': [], 'answer_start': [], 'answer_end': []}, 
               'val': {'context': [], 'question': [], 'answer_start': [], 'answer_end': []}}

    train_context_id = data_dir + '/train.ids.context'
    val_context_id = data_dir + '/val.ids.context'
    train_question_id = data_dir + '/train.ids.question'
    val_question_id = data_dir + '/val.ids.question'
    # Labels for training/validation
    train_answer_id = data_dir + '/train.span'
    val_answer_id = data_dir + '/val.span'

    if tf.gfile.Exists(train_context_id):
        rev_context = []
        with tf.gfile.GFile(train_context_id, mode='r') as c:
            rev_context.extend(c.readlines())
        rev_context = [line.strip('\n').split() for line in rev_context]
        dataset['train']['context'] = [[int(s) for s in line] for line in rev_context]
    else:
        raise ValueError("Context file %s not found.", train_context_id)

    if tf.gfile.Exists(train_question_id):
        rev_question = []
        with tf.gfile.GFile(train_question_id, mode='r') as q:
            rev_question.extend(q.readlines())
        rev_question = [line.strip('\n').split() for line in rev_question]
        dataset['train']['question'] = [[int(s) for s in line] for line in rev_question]
    else:
        raise ValueError("Question file %s not found.", train_question_id)

    if tf.gfile.Exists(train_answer_id):
        rev_answer, start_label, end_label = [], [0] * len(dataset['train']['context']), [0] * len(dataset['train']['context'])
        with tf.gfile.GFile(train_answer_id, mode='r') as a:
            rev_answer.extend(a.readlines())
        rev_answer = [line.strip('\n').split() for line in rev_answer]
        rev_answer = [(int(s), int(e)) for (s, e) in rev_answer]
        assert len(rev_answer) == len(start_label) == len(end_label)
        for i in range(len(rev_answer)):
            start_label[i] = rev_answer[i][0]
            end_label[i] = rev_answer[i][1]
        # Using 1 to mark Answer and 0 for O
        dataset['train']['answer_start'] = start_label
        dataset['train']['answer_end'] = end_label
    else:
        raise ValueError("Answer span file %s not found.", train_answer_id)

    if tf.gfile.Exists(val_context_id):
        rev_context = []
        with tf.gfile.GFile(val_context_id, mode='r') as c:
            rev_context.extend(c.readlines())
        rev_context = [line.strip('\n').split() for line in rev_context]
        dataset['val']['context'] = [[int(s) for s in line] for line in rev_context]
    else:
        raise ValueError("Context file %s not found.", val_context_id)

    if tf.gfile.Exists(val_question_id):
        rev_question = []
        with tf.gfile.GFile(val_question_id, mode='r') as q:
            rev_question.extend(q.readlines())
        rev_question = [line.strip('\n').split() for line in rev_question]
        dataset['val']['question'] = [[int(s) for s in line] for line in rev_question]
    else:
        raise ValueError("Question file %s not found.", val_question_id)

    if tf.gfile.Exists(val_answer_id):
        rev_answer, start_label, end_label = [], [0] * len(dataset['val']['context']), [0] * len(dataset['val']['context'])
        with tf.gfile.GFile(val_answer_id, mode='r') as a:
            rev_answer.extend(a.readlines())
        rev_answer = [line.strip('\n').split() for line in rev_answer]
        rev_answer = [(int(s), int(e)) for (s, e) in rev_answer]
        assert len(rev_answer) == len(start_label) == len(end_label)
        for i in range(len(rev_answer)):
            start_label[i] = rev_answer[i][0]
            end_label[i] = rev_answer[i][1]
        # Using 1 to mark Answer and 0 for O
        dataset['val']['answer_start'] = start_label
        dataset['val']['answer_end'] = end_label
    else:
        raise ValueError("Answer span file %s not found.", train_answer_id)
    return dataset

def main(_):

    # Do what you need to load datasets from FLAGS.data_dir
    dataset = load_data(FLAGS.data_dir) # ((question, context), answer)

    # print(dataset)
    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.embedding_size))
    vocab_path = FLAGS.vocab_path or pjoin(FLAGS.data_dir, "vocab.dat")
    vocab, rev_vocab = initialize_vocab(vocab_path)

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    with tf.Session() as sess:
        load_train_dir = get_normalized_train_dir(FLAGS.load_train_dir or FLAGS.train_dir)
        initialize_model(sess, qa, load_train_dir)

        save_train_dir = get_normalized_train_dir(FLAGS.train_dir)
        qa.train(sess, dataset, save_train_dir)

        qa.evaluate_answer(sess, dataset, vocab, FLAGS.evaluate, log=True)

if __name__ == "__main__":
    tf.app.run()
