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
    nl_pad_id += 1
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
            # temp_tourist_list += nl
            temp_tourist_list.append(nl + [nl_pad_id for _ in range(max_seq_len - len(nl))])
        # guide part
        for nl in history[hist_len:]:
            # temp_guide_list += nl
            temp_guide_list.append(nl + [nl_pad_id for _ in range(max_seq_len - len(nl))])

        # pad the sequence
        # train_tourist.append(temp_tourist_list+[nl_pad_id for _ in range(max_seq_len - len(temp_tourist_list))])
        train_tourist.append(temp_tourist_list)

        # tourist_len.append(len(temp_tourist_list))
        tourist_len.append([len(tourist_nl) for tourist_nl in temp_tourist_list])

        # train_guide.append(temp_guide_list+[nl_pad_id for _ in range(max_seq_len - len(temp_guide_list))])
        train_guide.append(temp_guide_list)

        # guide_len.append(len(temp_guide_list))
        guide_len.append([len(temp_guide_list) for guide_nl in temp_guide_list])

    for i in batch_intent:
        # intent + attribute, ex: [2, 11, 12]
        target_idx.append([i[-1][0]]+[attri for attri in i[-1][1]])

    for i in batch_nl:
        nl = i[-1]
        # train_nl.append(nl+[nl_pad_id for _ in range(max_seq_len - len(nl))])
        train_nl.append(nl + [nl_pad_id for _ in range(max_seq_len - len(nl))])
        nl_len.append(len(nl))

    # one-hot encode train_target
    # i: target intent and attribute of each sentence
    # maybe a typo here: total_intent == len(act_dict) + len(attri_dict)
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

def calc_test_f1_score(test_output, test_target, test_role='All'):
    test_talker = open('Data/test/talker', 'r')
    pred_vec = np.array(list())
    label_vec = np.array(list())
    for pred, label, talker in zip(test_output, test_target, test_talker):
        if test_role == 'Tourist' and talker.strip('\n') == 'Guide':
            continue
        elif test_role == 'Guide' and talker.strip('\n') == 'Tourist':
            continue

        pred_act = pred[:5]  # first 5 is act
        pred_attribute = pred[5:]  # remaining is attribute
        binary = Binarizer(threshold=0.5)
        act_logit = one_hot(np.argmax(pred_act), 'act')
        attribute_logit = binary.fit_transform([pred_attribute])
        if np.sum(attribute_logit) == 0:
            attribute_logit = one_hot(np.argmax(pred_attribute), 'attribute')
        label = binary.fit_transform([label])
        pred_vec = np.append(pred_vec, np.append(act_logit, attribute_logit))
        label_vec = np.append(label_vec, label)
    f1sc = f1_score(pred_vec, label_vec, average='binary')
    test_talker.close()
    return f1sc

if __name__ == '__main__':
    sess = tf.Session(config=config)
    # max_seq_len = 70
    max_seq_len = 40
    epoch = 60
    batch_size = 128
    use_intent = False # True: use intent tag as input, False: use nl as input
    use_attention = True

    data = slu_data()
    total_intent = data.total_intent
    total_word = data.total_word
    model = slu_model(max_seq_len, total_intent, use_attention)
    sess.run(model.init_model)
    # read in the glove embedding matrix
    sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix: data.embedding_matrix})
    # test_f1_scores = list()
    test_f1_scores_all = list()
    test_f1_scores_guide = list()
    test_f1_scores_tourist = list()

    ####################################
    #             Training             #
    ####################################
    for cur_epoch in range(epoch):
        pred_outputs = list()
        train_targets = list()
        total_loss = 0.0
        for _ in range(50):
            # get the data
            batch_nl, batch_intent = data.get_train_batch(batch_size)
            train_intent = None
            train_nl = None

            if use_intent:
                train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len = process_intent(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            else:
                train_tourist, train_guide, train_nl, train_target, tourist_len, guide_len, nl_len = process_nl(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            # add nl_indices to gather_nd of bi-rnn output
            nl_indices = list()

            for idx, indices in enumerate(nl_len):
                nl_indices.append([idx, indices])

            _, intent_output, loss = sess.run(
                [model.train_op, model.intent_output, model.loss],
                feed_dict={
                    model.tourist_input: train_tourist,
                    model.guide_input: train_guide,
                    model.input_nl: train_nl,
                    model.tourist_len: tourist_len,
                    model.guide_len: guide_len,
                    model.labels: train_target,
                    model.nl_len: nl_len,
                    # model.dropout_keep_prob: 0.5
                })
            total_loss += loss
            for pred, label in zip(intent_output, train_target):
                pred_act = pred[:5]  # first 5 is act
                pred_attribute = pred[5:]  # remaining is attribute
                binary = Binarizer(threshold=0.5)
                act_logit = one_hot(np.argmax(pred_act), 'act')
                attribute_logit = binary.fit_transform([pred_attribute])
                if np.sum(attribute_logit) == 0:
                    attribute_logit = one_hot(np.argmax(pred_attribute), 'attribute')
                label = binary.fit_transform([label])
                pred_outputs = np.append(pred_outputs, np.append(act_logit, attribute_logit))
                train_targets = np.append(train_targets, label)

        # calculate batch F1 score
        print "========================="
        print "        Epoch:", cur_epoch
        print "========================="
        print "f1 score:", f1_score(pred_outputs, train_targets, average='binary')
        print "training loss:", total_loss

        ####################################
        #             Testing              #
        ####################################
        test_nl, test_intent = data.get_test_batch()
        if use_intent == True:
            test_tourist, test_guide, test_nl, test_target, tourist_len, guide_len, nl_len = process_intent(test_nl, test_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        else:
            test_tourist, test_guide, test_nl, test_target, tourist_len, guide_len, nl_len = process_nl(test_nl, test_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        test_output = sess.run(model.intent_output,
                feed_dict={
                    model.tourist_input:test_tourist[:5000],
                    model.guide_input:test_guide[:5000],
                    model.input_nl:test_nl[:5000],
                    model.tourist_len:tourist_len[:5000],
                    model.guide_len:guide_len[:5000],
                    model.nl_len:nl_len[:5000],
                    model.labels:test_target[:5000],
                    model.dropout_keep_prob:1.0
                    })

        test_output = np.concatenate((test_output, sess.run(model.intent_output,
                feed_dict={
                    model.tourist_input:test_tourist[5000:],
                    model.guide_input:test_guide[5000:],
                    model.input_nl:test_nl[5000:],
                    model.tourist_len:tourist_len[5000:],
                    model.guide_len:guide_len[5000:],
                    model.nl_len:nl_len[5000:],
                    model.labels:test_target[5000:],
                    model.dropout_keep_prob:1.0
                    })), axis=0)


        # calculate the f1 score, indicate the test role by arguments ('Guide', 'Tourist', 'All')
        f1sc = calc_test_f1_score(test_output, test_target, 'All')
        print "testing f1 score (All):", f1sc
        test_f1_scores_all.append(f1sc)

        f1sc = calc_test_f1_score(test_output, test_target, 'Guide')
        print "testing f1 score (Guide):", f1sc
        test_f1_scores_guide.append(f1sc)

        f1sc = calc_test_f1_score(test_output, test_target, 'Tourist')
        print "testing f1 score (Tourist):", f1sc
        test_f1_scores_tourist.append(f1sc)

        # print the max test f1 score of each role
        print ""
        print "max test f1 score (All):", max(test_f1_scores_all)
        print "max test f1 score (Guide):", max(test_f1_scores_guide)
        print "max test f1 score (Tourist):", max(test_f1_scores_tourist)
    sess.close()
