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

def process_nl(batch_nl, batch_intent, max_seq_len, intent_pad_id, nl_pad_id, total_intent):
    train_tourist = list()
    train_guide = list()
    train_nl = list()
    train_target = list()
    target_idx = list()
    tourist_len = list()
    guide_len = list()
    nl_len = list()

    for i in batch_nl:
        temp_tourist_list = list()
        temp_guide_list = list()
        history = i[:-1]
        hist_len = len(history) / 2 # tourist, guide
        # tourist part
        for nl in history[:hist_len]:
            temp_tourist_list += nl
        # guide part
        for nl in history[hist_len:]:
            temp_guide_list += nl
        # pad the sequence
        train_tourist.append(temp_tourist_list+[nl_pad_id for _ in range(max_seq_len - len(temp_tourist_list))])
        tourist_len.append(len(temp_tourist_list))
        train_guide.append(temp_guide_list+[nl_pad_id for _ in range(max_seq_len - len(temp_guide_list))])
        guide_len.append(len(temp_guide_list))

    for i in batch_intent:
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

def one_hot(idx, T):
    # intent dim is 26, 5 is act, 21 is attribute
    if T == 'act':
        ret = np.zeros(5)
        ret[idx] = 1.0
    elif T == 'attribute':
        ret = np.zeros(21)
        ret[idx] = 1.0
    return ret

if __name__ == '__main__':
    sess = tf.Session(config=config)
    data = slu_data()
    max_seq_len = 60
    total_intent = data.total_intent
    total_word = data.total_word
    model = slu_model(max_seq_len, total_intent)
    sess.run(model.init_model)
    # read in the glove embedding matrix
    sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix:data.embedding_matrix})

    epoch = 30
    use_intent = True # True: use intent tag as input, False: use nl as input

    # Train
    for cur_epoch in range(epoch):
        pred_outputs = list()
        train_targets = list()
        total_loss = 0.0
        for _ in range(50):
            # get the data
            batch_nl, batch_intent = data.get_train_batch(128)
            train_intent = None
            train_nl = None
            if use_intent == True:
                train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len = process_intent(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            else:
                train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len = process_nl(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            # add nl_indices to gather_nd of bi-rnn output
            nl_indices = list()
            for idx, indices in enumerate(nl_len):
                nl_indices.append([idx, indices])

            _, intent_output, loss = sess.run([model.train_op, model.intent_output, model.loss],
                    feed_dict={
                        model.tourist_input:train_tourist,
                        model.guide_input:train_guide,
                        model.input_nl:train_nl,
                        model.tourist_len:tourist_len,
                        model.guide_len:guide_len,
                        model.labels:train_target,
                        model.nl_len:nl_len,
                        })
            total_loss += loss
            for pred, label in zip(intent_output, train_target):
                pred_act = pred[:5] # first 5 is act
                pred_attribute = pred[5:] # remaining is attribute
                binary = Binarizer(threshold=0.5)
                act_logit = one_hot(np.argmax(pred_act), 'act')
                attribute_logit = binary.fit_transform([pred_attribute])
                if np.sum(attribute_logit) == 0:
                    attribute_logit = one_hot(np.argmax(pred_attribute), 'attribute')
                label = binary.fit_transform([label])
                pred_outputs = np.append(pred_outputs, np.append(act_logit, attribute_logit))
                train_targets = np.append(train_targets, label)
        # calculate batch F1 score
        print "cur_epoch is:", cur_epoch
        print "f1 score is:", f1_score(pred_outputs, train_targets, average='binary')
        print "loss is:", total_loss

        # Test
        test_nl, test_intent = data.get_test_batch()
        if use_intent == True:
            test_tourist, test_guide, test_nl, test_target, tourist_len, guide_len, nl_len = process_intent(test_nl, test_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        else:
            test_tourist, test_guide, test_nl, test_target, tourist_len, guide_len, nl_len = process_nl(test_nl, test_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        # add nl_indices to gather_nd of bi-rnn output
        nl_indices = list()
        for idx, indices in enumerate(nl_len):
            nl_indices.append([idx, indices])
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
        test_talker = open('Data/test/talker', 'r')
        pred_vec = np.array(list())
        label_vec = np.array(list())
        for pred, label, talker in zip(test_output, test_target, test_talker):
            if talker.strip('\n') == 'Guide':
                continue
            pred_act = pred[:5] # first 5 is act
            pred_attribute = pred[5:] # remaining is attribute
            binary = Binarizer(threshold=0.5)
            act_logit = one_hot(np.argmax(pred_act), 'act')
            attribute_logit = binary.fit_transform([pred_attribute])
            if np.sum(attribute_logit) == 0:
                attribute_logit = one_hot(np.argmax(pred_attribute), 'attribute')
            label = binary.fit_transform([label])
            pred_vec = np.append(pred_vec, np.append(act_logit, attribute_logit))
            label_vec = np.append(label_vec, label)
        print "test f1 score is:", f1_score(pred_vec, label_vec, average='binary')
        test_talker.close()
    sess.close()
