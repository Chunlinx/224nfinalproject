from __future__ import division

import sys
import time
import logging
import StringIO
from collections import defaultdict, Counter, OrderedDict
import random
from numpy import array, zeros, allclose
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from qa_data import PAD_ID

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

    if tf.app.flags.FLAGS.test_run:
        train_context_id += '.small'
        val_context_id += '.small'
        train_question_id += '.small'
        val_question_id += '.small'
        # Labels for training/validation
        train_answer_id += '.small'
        val_answer_id += '.small'

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

def pad_sequence(sequences, max_length):
    """
    Given a list of sequences of word ids, pad each sequence to the 
    desired length or truncate each sequence to max length
    """
    padded, effective_len = [], []
    for seq in sequences:
        pad_len = max_length - len(seq)
        effective_len.append(len(seq))
        if pad_len <= 0:
            padded.append(seq[:max_length])
        else:
            padded.append(seq + [PAD_ID] * pad_len)
    return np.array(padded), np.array(effective_len)

def preprocess_dataset(dataset, context_len, question_len):

    context_, question_, a_s_, a_e_ = dataset['context'], \
        dataset['question'], dataset['answer_start'], dataset['answer_end']
    context_padded = pad_sequence(context_, context_len)
    question_padded = pad_sequence(question_, question_len) 
    assert len(context_padded[0]) == len(question_padded[1])
    return [context_padded, question_padded, a_s_, a_e_]

def get_minibatch(data, batch_size):
    """
    Given a complete dataset represented as dict, return the 
    batch sized data with shuffling as ((context, question), label)
    """
    def minibatch_helper(data_mini, batch_idx):
        # Return both data and seq_len_vec
        return [data_mini[0][batch_idx], data_mini[1][batch_idx]]

    data_size = len(data[0][0])
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for i in np.arange(0, data_size, batch_size):
        batch_indices = indices[i: i + batch_size]
        # Treat differently for data with mask and labels
        res = [minibatch_helper(d, batch_indices) for d in data[:2]] + \
            [np.array(d)[batch_indices] for d in data[2:]]
        yield res

def get_sampled_data(dataset_train, dataset_val, p_len, q_len, sample=100):
    """
    Randomly sample data in both train and validation data sets
    for evaluation
    """
    context_padded_train_, question_padded_train_, \
        a_s_train_, a_e_train_ = dataset_train

    context_padded_val_, question_padded_val_, \
        a_s_val_, a_e_val_ = dataset_val
    # Context, query, ans labels are read correctly

    train_sample_idx = np.random.choice(np.arange(len(a_s_train_)), 
        int(sample / 2), replace=False)
    val_sample_idx = np.random.choice(np.arange(len(a_e_val_)), 
        sample - int(sample / 2), replace=False)

    context_train, context_train_len = context_padded_train_
    context_val, context_val_len = context_padded_val_

    query_train, query_train_len = question_padded_train_
    query_val, query_val_len = question_padded_val_

    merged_data = [(context_train[i], context_train_len[i],
        query_train[i], query_train_len[i], a_s_train_[i], a_e_train_[i])\
         for i in range(len(a_s_train_))] + [(context_val[i], context_val_len[i], 
            query_val[i], query_val_len[i], a_s_val_[i], a_e_val_[i])\
                for i in range(len(a_e_val_))]

    selected_data = random.sample(merged_data, sample)
    feed_data = [((np.reshape(tp[0], (1, p_len)), np.reshape(tp[1], (1,))),
        (np.reshape(tp[2], (1, q_len)), np.reshape(tp[3], (1,))),
            tp[0][tp[4]: tp[5] + 1]) for tp in selected_data]
    ground_truth = [d[2].tolist() for d in feed_data]

    return feed_data, ground_truth

def map_id_to_words():
	pass


class Progbar(object):
    """
    Progbar class copied from keras (https://github.com/fchollet/keras/)
    Displays a progress bar.
    # Arguments
        target: Total number of steps expected.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose

    def update(self, current, values=None, exact=None):
        """
        Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            exact: List of tuples (name, value_for_last_step).
                The progress bar will display these values directly.
        """
        values = values or []
        exact = exact or []

        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far), current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        for k, v in exact:
            if k not in self.sum_values:
                self.unique_values.append(k)
            self.sum_values[k] = [v, 1]
        self.seen_so_far = current

        now = time.time()
        if self.verbose == 1:
            prev_total_width = self.total_width
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(self.target))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (current, self.target)
            prog = float(current)/self.target
            prog_width = int(self.width*prog)
            if prog_width > 0:
                bar += ('='*(prog_width-1))
                if current < self.target:
                    bar += '>'
                else:
                    bar += '='
            bar += ('.'*(self.width-prog_width))
            bar += ']'
            sys.stdout.write(bar)
            self.total_width = len(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            eta = time_per_unit*(self.target - current)
            info = ''
            if current < self.target:
                info += ' - ETA: %ds' % eta
            else:
                info += ' - %ds' % (now - self.start)
            for k in self.unique_values:
                if isinstance(self.sum_values[k], list):
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                else:
                    info += ' - %s: %s' % (k, self.sum_values[k])

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += ((prev_total_width-self.total_width) * " ")

            sys.stdout.write(info)
            sys.stdout.flush()

            if current >= self.target:
                sys.stdout.write("\n")

        if self.verbose == 2:
            if current >= self.target:
                info = '%ds' % (now - self.start)
                for k in self.unique_values:
                    info += ' - %s: %.4f' % (k, self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                sys.stdout.write(info + "\n")

    def add(self, n, values=None):
        self.update(self.seen_so_far+n, values)






