from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import os
import json
import sys
import random
from os.path import join as pjoin

from tqdm import tqdm
import numpy as np
from six.moves import xrange
import tensorflow as tf

from qa_model import Encoder, QASystem, Decoder
from preprocessing.squad_preprocess import data_from_json, maybe_download, squad_base_url, \
    invert_map, tokenize, token_idx_map
import qa_data, qa_util

import logging

logging.basicConfig(level=logging.INFO)
_PAD = b"<pad>"
_SOS = b"<sos>"
_UNK = b"<unk>"

tf.app.flags.DEFINE_string("dev_path", "../data/squad/dev-v1.1.json", "Path to the JSON dev set to evaluate against (default: ./data/squad/dev-v1.1.json)")
tf.app.flags.DEFINE_string("data_dir", "../data/squad", "SQuAD directory (default ./data/squad)")
tf.app.flags.DEFINE_string("load_train_dir", "", "Training directory to load model parameters from to resume training (default: {train_dir}).")
tf.app.flags.DEFINE_string("train_dir", "../train", "Training directory to save the model parameters (default: ./train).")
tf.app.flags.DEFINE_string("log_dir", "log", "Path to store log and flag files (default: ./log)")
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("keep", 0, "How many checkpoints to keep, 0 indicates keep all.")
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

def read_dataset(dataset, tier, vocab):
    """Reads the dataset, extracts context, question, answer,
    and answer pointer in their own file. Returns the number
    of questions and answers processed for the dataset"""

    context_data = []
    query_data = []
    question_uuid_data = []

    for articles_id in tqdm(range(len(dataset['data'])), desc="Preprocessing {}".format(tier)):
        article_paragraphs = dataset['data'][articles_id]['paragraphs']
        for pid in range(len(article_paragraphs)):
            context = article_paragraphs[pid]['context']
            # The following replacements are suggested in the paper
            # BidAF (Seo et al., 2016)
            context = context.replace("''", '" ')
            context = context.replace("``", '" ')

            context_tokens = tokenize(context)

            qas = article_paragraphs[pid]['qas']
            for qid in range(len(qas)):
                question = qas[qid]['question']
                question_tokens = tokenize(question)
                question_uuid = qas[qid]['id']

                context_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in context_tokens]
                qustion_ids = [str(vocab.get(w, qa_data.UNK_ID)) for w in question_tokens]

                context_data.append(' '.join(context_ids))
                query_data.append(' '.join(qustion_ids))
                question_uuid_data.append(question_uuid)

    return context_data, query_data, question_uuid_data


def prepare_dev(prefix, dev_filename, vocab):
    print("Downloading {}".format(dev_filename))
    # Don't check file size, since we could be using other datasets
    dev_dataset = maybe_download(squad_base_url, dev_filename, prefix)

    dev_data = data_from_json(os.path.join(prefix, dev_filename))
    context_data, question_data, question_uuid_data = read_dataset(dev_data, 'dev', vocab)

    return context_data, question_data, question_uuid_data


def generate_answers(sess, model, dataset, rev_vocab):
    """
    Loop over the dev or test dataset and generate answer.

    Note: output format must be answers[uuid] = "real answer"
    You must provide a string of words instead of just a list, or start and end index

    In main() function we are dumping onto a JSON file

    evaluate.py will take the output JSON along with the original JSON file
    and output a F1 and EM

    You must implement this function in order to submit to Leaderboard.

    :param sess: active TF session
    :param model: a built QASystem model
    :param rev_vocab: this is a list of vocabulary that maps index to actual words
    :return:
    """
    answers = {}
    size = len(dataset[0])
    batch_size = FLAGS.batch_size
    context_, query_, uuid = dataset

    context = [c.strip().split(' ') for c in context_]
    query = [q.strip().split(' ') for q in query_]
    context_padded, context_mask = qa_util.pad_sequence([[int(c) for c in m] for m in context],
        FLAGS.output_size)
    query_padded, query_mask = qa_util.pad_sequence([[int(q) for q in m] for m in query],
        FLAGS.question_size)
    prog = qa_util.Progbar(target=1 + int(len(context_) / batch_size))

    for i in xrange(0, size, batch_size):

        context_batch = context_padded[i: i + batch_size, :]

        context_mask_batch = context_mask[i: i + batch_size]
        query_batch = query_padded[i: i + batch_size, :]
        query_mask_batch = query_mask[i: i + batch_size]
        uuid_batch = uuid[i: i + batch_size]

        test_x = (
            (context_batch, context_mask_batch),
            (query_batch, query_mask_batch)
        )

        ans = np.dstack(model.answer(sess, test_x))[0]
        for j, (a_s, a_e) in enumerate(ans):
            if a_s <= a_e:
                answers[uuid_batch[j]] = ' '.join(
                    [rev_vocab[w] for w in context_batch[j][a_s: a_e + 1]])
            else: # if the start and end indexes are mixed up
                answers[uuid_batch[j]] = ' '.join(
                    [rev_vocab[v] for v in context_batch[j][a_e: a_s + 1]])  
        prog.update(i + 1, [("Answering batch", i)])      
    return answers


def main(_):

    FLAGS.config = int(sys.argv[1])
    load_config(current_config=FLAGS.config)

    vocab, rev_vocab = initialize_vocab(FLAGS.vocab_path)

    embed_path = FLAGS.embed_path or pjoin("data", "squad", "glove.trimmed.{}.npz".format(FLAGS.vocab_dim))

    global_train_dir = '/tmp/cs224n-squad-train'
    # Adds symlink to {train_dir} from /tmp/cs224n-squad-train to canonicalize the
    # file paths saved in the checkpoint. This allows the model to be reloaded even
    # if the location of the checkpoint files has moved, allowing usage with CodaLab.
    # This must be done on both train.py and qa_answer.py in order to work.
    if not os.path.exists(FLAGS.train_dir):
        os.makedirs(FLAGS.train_dir)
    if os.path.exists(global_train_dir):
        os.unlink(global_train_dir)
    os.symlink(os.path.abspath(FLAGS.train_dir), global_train_dir)
    train_dir = global_train_dir

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)
    file_handler = logging.FileHandler(pjoin(FLAGS.log_dir, "log.txt"))
    logging.getLogger().addHandler(file_handler)

    print(vars(FLAGS))
    with open(os.path.join(FLAGS.log_dir, "flags.json"), 'w') as fout:
        json.dump(FLAGS.__flags, fout)

    # ========= Load Dataset =========
    # You can change this code to load dataset in your own way

    dev_dirname = os.path.dirname(os.path.abspath(FLAGS.dev_path))
    dev_filename = os.path.basename(FLAGS.dev_path)
    context_data, question_data, question_uuid_data = prepare_dev(dev_dirname, dev_filename, vocab)
    dataset = (context_data, question_data, question_uuid_data)

    # ========= Model-specific =========
    # You must change the following code to adjust to your model

    encoder = Encoder(size=FLAGS.state_size, vocab_dim=FLAGS.embedding_size)
    decoder = Decoder(output_size=FLAGS.output_size)

    qa = QASystem(encoder, decoder, train_dir)

    with tf.Session() as sess:
        initialize_model(sess, qa, train_dir)
        answers = generate_answers(sess, qa, dataset, rev_vocab)

        # write to json file to root dir
        with io.open('dev-prediction.json', 'w', encoding='utf-8') as f:
            f.write(unicode(json.dumps(answers, ensure_ascii=False)))


if __name__ == "__main__":
  tf.app.run()
