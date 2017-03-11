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
    def minibatch(data_mini, batch_idx):
        # Return both data and seq_len_vec
        return [data_mini[0][batch_idx], data_mini[1][batch_idx]]

    data_size = len(data[0][0])
    indices = np.arange(data_size)
    np.random.shuffle(indices)

    for i in np.arange(0, data_size, batch_size):
        batch_indices = indices[i: i + batch_size]
        # Treat differently for data with mask and labels
        res = [minibatch(d, batch_indices) for d in data[:2]] + \
            [np.array(d)[batch_indices] for d in data[2:]]
        yield res

def extract_answer_ids():

	pass

	
def map_id_to_words():
	pass






