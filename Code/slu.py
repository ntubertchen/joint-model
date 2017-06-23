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
    train_intent = list()
    train_nl = list()
    train_target = list()
    target_idx = list()
    for i in batch_intent:
        temp_list = list()
        for tup in i[:-1]:
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_list += d
        # pad the intent sequence
        train_intent.append(temp_list+[intent_pad_id for _ in range(max_seq_len - len(temp_list))])
        target_idx.append([i[-1][0]]+[attri for attri in i[-1][1]])

    for i in batch_nl:
        nl = i[-1]
        train_nl.append(nl+[nl_pad_id for _ in range(max_seq_len - len(nl))])

    # one-hot encode train_target
    for i in target_idx:
        target_l = [0.0 for _ in range(total_intent)]
        for idx in i:
            target_l[idx] = 1.0
        train_target.append(target_l)

    return train_intent, train_nl, train_target

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

    epoch = 300
    use_intent = True
    for cur_epoch in range(epoch):
        for i in range(256):
            # get the data
            batch_nl, batch_intent = data.get_train_batch()
            train_intent = None
            train_nl = None
            if use_intent is True:
                train_intent, train_nl, train_target = process_intent(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            else:
                pass
            sess.run(model.train_op, 
                    feed_dict={
                        model.input_intent:train_intent,
                        model.input_nl:train_nl,
                        model.labels:train_target
                        })
        print (cur_epoch)

    sess.close()
    
