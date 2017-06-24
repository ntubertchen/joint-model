import tensorflow as tf
from tensorflow.python.platform import flags
import numpy as np
from slu_preprocess import slu_data
from slu_model import slu_model
import argparse
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

FLAGS = flags.FLAGS

parser = argparse.ArgumentParser()
parser.add_argument('--glove', help='glove path')
parser.add_argument('--train_p', help='training data path')
parser.add_argument('--test_p', help='testing data path')
parser.add_argument('--test', action='store_true', help='test or train')
parser.add_argument('--slot_l', help='slot tag path')
parser.add_argument('--intent_l', help='intent path')
parser.add_argument('--intent', action='store_false', help='intent training')
parser.add_argument('--slot', action='store_false', help='slot training')
args = parser.parse_args()

def process_intent(batch_nl, batch_intent, max_seq_len, intent_pad_id, nl_pad_id, total_intent):
    train_tourist = list()
    train_guide = list()
    train_nl = list()
    train_target = list()
    target_idx = list()
    tourist_len = list()
    guide_len = list()
    nl_len = list()

    for i in batch_intent:
        temp_tourist_list = list()
        temp_guide_list = list()
        history = i[:-1]
        hist_len = len(history) / 2 # tourist, guide
        # tourist part
        for tup in history[:hist_len]:
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_tourist_list += d
        # guide part
        for tup in history[hist_len:]:
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_guide_list += d

        # pad the sequence
        train_tourist.append(temp_tourist_list+[intent_pad_id for _ in range(max_seq_len - len(temp_tourist_list))])
        tourist_len.append(len(temp_tourist_list))
        train_guide.append(temp_guide_list+[intent_pad_id for _ in range(max_seq_len - len(temp_guide_list))])
        guide_len.append(len(temp_guide_list))
        target_idx.append([i[-1][0]]+[attri for attri in i[-1][1]])

    for i in batch_nl:
        nl = i[-1]
        train_nl.append(nl+[nl_pad_id for _ in range(max_seq_len - len(nl))])
        nl_len.append(len(nl))

    # one-hot encode train_target
    for i in target_idx:
        target_l = [0.0 for _ in range(total_intent)]
        for idx in i:
            target_l[idx] = 1.0
        train_target.append(target_l)

    return train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len

if __name__ == '__main__':
    sess = tf.Session(config=config)
    data = slu_data()
    max_seq_len = 50
    total_intent = data.total_intent
    total_word = data.total_word
    model = slu_model(max_seq_len, total_intent)
    sess.run(model.init_model)
    # read in the glove embedding matrix
    sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix:data.embedding_matrix})

    epoch = 500
    use_intent = True
    # Train
    for cur_epoch in range(epoch):
        # get the data
        batch_nl, batch_intent = data.get_train_batch(256)
        train_intent = None
        train_nl = None
        if use_intent == True:
            train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len = process_intent(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        else:
            pass
        _, intent_output = sess.run([model.train_op, model.intent_output],
                feed_dict={
                    model.tourist_input:train_tourist,
                    model.guide_input:train_guide,
                    model.input_nl:train_nl,
                    model.tourist_len:tourist_len,
                    model.guide_len:guide_len,
                    model.nl_len:nl_len,
                    model.labels:train_target
                    })
   
        # calculate batch F1 score
        pred_vec = np.array(list())
        label_vec = np.array(list())
        for pred, label in zip(intent_output, train_target):
            label_bin = Binarizer(threshold=0.5)
            pred_bin = Binarizer(threshold=0.2)
            logit = pred_bin.fit_transform([pred])
            label = label_bin.fit_transform([label])
            pred_vec = np.append(pred_vec, logit)
            label_vec = np.append(label_vec, label)
        print "f1 score is:", f1_score(pred_vec, label_vec, average='binary')
        print "cur_epoch is:", cur_epoch

    # Test
    test_nl, test_intent = data.get_test_batch()
    if use_intent == True:
        test_tourist, test_guide, test_nl, test_target, tourist_len, guide_len, nl_len = process_intent(test_nl, test_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        test_output = sess.run(model.intent_output,
                feed_dict={
                    model.tourist_input:test_tourist,
                    model.guide_input:test_guide,
                    model.input_nl:test_nl,
                    model.tourist_len:tourist_len,
                    model.guide_len:guide_len,
                    model.nl_len:nl_len,
                    model.labels:test_target
                    })

    # calculate test F1 score
    pred_vec = np.array(list())
    label_vec = np.array(list())
    for pred, label in zip(test_output, train_target):
        label_bin = Binarizer(threshold=0.5)
        pred_bin = Binarizer(threshold=0.2)
        logit = pred_bin.fit_transform([pred])
        label = label_bin.fit_transform([label])
        pred_vec = np.append(pred_vec, logit)
        label_vec = np.append(label_vec, label)
    print "test f1 score is:", f1_score(pred_vec, label_vec, average='binary')
    sess.close()
