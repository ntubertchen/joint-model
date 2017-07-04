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

def one_hot(idx, T, weight=0):
    # intent dim is 26, 5 is act, 21 is attribute
    if T == 'act':
        ret = np.zeros(5)
        ret[idx] = 1.0
    elif T == 'attribute':
        ret = np.zeros(21)
        ret[idx] = 1.0
    elif T == 'mix':
        ret = np.zeros(26)
        for i in idx:
            ret[i] = 1.0 / float(weight)
    return ret

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
        temp_tourist_len = list()
        temp_guide_len = list()
        history = i[:-1]
        hist_len = len(history) / 2 # tourist, guide
        # tourist part
        for nl in history[:hist_len]:
            temp_tourist_list.append(nl+[nl_pad_id for _ in range(max_seq_len - len(nl))])
            temp_tourist_len.append(len(nl))
        # guide part
        for nl in history[hist_len:]:
            temp_guide_list.append(nl+[nl_pad_id for _ in range(max_seq_len - len(nl))])
            temp_guide_len.append(len(nl))
    
        train_tourist.append(temp_tourist_list)
        tourist_len.append(temp_tourist_len)
        train_guide.append(temp_guide_list)
        guide_len.append(temp_guide_len)

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

def process_intent(batch_nl, batch_intent, batch_distance, max_seq_len, intent_pad_id, nl_pad_id, total_intent):
    train_tourist = list()
    train_guide = list()
    train_nl = list()
    train_target = list()
    target_idx = list()
    tourist_len = list()
    guide_len = list()
    nl_len = list()

    for i, j in zip(batch_intent, batch_distance):
        temp_tourist_list = list()
        temp_guide_list = list()
        history = i[:-1]
        distance = j[:-1]
        hist_len = len(history) / 2 # tourist, guide
        # tourist part
        for tup, weight in zip(history[:hist_len], distance[:hist_len]):
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_tourist_list.append(one_hot(d, 'mix', weight))
        # guide part
        for tup, weight in zip(history[hist_len:], distance[hist_len:]):
            d = [tup[0]] + [attri for attri in tup[1]]
            temp_guide_list.append(one_hot(d, 'mix', weight))
        train_guide.append(temp_guide_list)
        train_tourist.append(temp_tourist_list)
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
    max_seq_len = 40
    epoch = 30
    batch_size = 256
    use_attention = "None"
    use_mid_loss = False

    data = slu_data()
    total_intent = data.total_intent
    total_word = data.total_word
    model = slu_model(max_seq_len, total_intent, use_attention, use_mid_loss)
    sess.run(model.init_model)
    # read in the glove embedding matrix
    sess.run(model.init_embedding, feed_dict={model.read_embedding_matrix:data.embedding_matrix})
    test_f1_scores = list()
    switch_epoch = 1
    # Train
    for cur_epoch in range(epoch):
        pred_outputs = list()
        train_targets = list()
        total_loss = 0.0
        for cnt in range(50):
            # get the data
            if cur_epoch < switch_epoch:
                batch_nl, batch_intent, batch_distance = data.get_original_train_batch(batch_size)
            else:
                batch_size = 16
                batch_nl, batch_intent, batch_distance = data.get_train_batch(batch_size)
            train_tourist_intent, train_guide_intent, train_nl, train_target_intent, tourist_len_intent, guide_len_intent, nl_len = process_intent(batch_nl, batch_intent, batch_distance, max_seq_len, total_intent-1, total_word-1, total_intent)
            train_tourist_nl, train_guide_nl, train_nl, train_target_nl, tourist_len_nl, guide_len_nl, nl_len = process_nl(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            assert train_target_intent == train_target_nl
            if cur_epoch < switch_epoch:
                loss_to_minimize = model.loss
                train_op = model.train_op
            else:
                train_op = model.sap_train_op
                loss_to_minimize = model.sap_loss

            _, intent_output, loss = sess.run([train_op, model.intent_output, loss_to_minimize],
                    feed_dict={
                        model.tourist_input_intent:train_tourist_intent,
                        model.guide_input_intent:train_guide_intent,
                        model.labels:train_target_intent,
                        model.tourist_input_nl:train_tourist_nl,
                        model.guide_input_nl:train_guide_nl,
                        model.predict_nl:train_nl,
                        model.tourist_len_nl:tourist_len_nl,
                        model.guide_len_nl:guide_len_nl,
                        model.predict_nl_len:nl_len,
                        model.dropout_keep_prob:0.75
                        })
                
            total_loss += loss
            for pred, label in zip(intent_output, train_target_intent):
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
        print "Epoch:", cur_epoch
        print "f1 score:", f1_score(pred_outputs, train_targets, average='binary')
        print "training loss:", total_loss
        if cur_epoch == 10:
            # print training result for Chen
            pred_outputs = list()
            train_targets = list()
            batch_nl, batch_intent, batch_distance = data.get_all_train()
            train_tourist_intent, train_guide_intent, train_nl, train_target_intent, tourist_len_intent, guide_len_intent, nl_len = process_intent(batch_nl, batch_intent, batch_distance, max_seq_len, total_intent-1, total_word-1, total_intent)
            train_tourist_nl, train_guide_nl, train_nl, train_target_nl, tourist_len_nl, guide_len_nl, nl_len = process_nl(batch_nl, batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
            assert train_target_intent == train_target_nl
            intent_output = sess.run(model.intent_output,
                    feed_dict={
                        model.tourist_input_intent:train_tourist_intent,
                        model.guide_input_intent:train_guide_intent,
                        model.labels:train_target_intent,
                        model.tourist_input_nl:train_tourist_nl,
                        model.guide_input_nl:train_guide_nl,
                        model.predict_nl:train_nl,
                        model.tourist_len_nl:tourist_len_nl,
                        model.guide_len_nl:guide_len_nl,
                        model.predict_nl_len:nl_len,
                        model.dropout_keep_prob:1.0
                        })
            for pred, label in zip(intent_output, train_target_intent):
                pred_act = pred[:5] # first 5 is act
                pred_attribute = pred[5:] # remaining is attribute
                binary = Binarizer(threshold=0.5)
                act_logit = one_hot(np.argmax(pred_act), 'act')
                attribute_logit = binary.fit_transform([pred_attribute])
                if np.sum(attribute_logit) == 0:
                    attribute_logit = one_hot(np.argmax(pred_attribute), 'attribute')
                label = binary.fit_transform([label])
                pred_outputs.append(np.append(act_logit, attribute_logit))
            ans = list()
            f_out = open('out.txt', 'w')
            for t in pred_outputs:
                s = ''
                for idx, tag in enumerate(t):
                    if tag == 1.0:
                        s = s + (data.whole_dict[idx]) + '-'
                f_out.write(s.strip('-')+'\n')
                ans.append(s)
            f_out.close()

        # Test
        if cur_epoch < switch_epoch:
            test_batch_nl, test_batch_intent, test_batch_distance = data.get_original_test_batch()
        else:
            test_batch_nl, test_batch_intent, test_batch_distance = data.get_test_batch()
        test_tourist_intent, test_guide_intent, test_nl, test_target_intent, tourist_len_intent, guide_len_intent, nl_len = process_intent(test_batch_nl, test_batch_intent, test_batch_distance, max_seq_len, total_intent-1, total_word-1, total_intent)
        test_tourist_nl, test_guide_nl, test_nl, test_target_nl, tourist_len_nl, guide_len_nl, nl_len = process_nl(test_batch_nl, test_batch_intent, max_seq_len, total_intent-1, total_word-1, total_intent)
        assert test_target_intent == test_target_nl
        test_output = sess.run(model.intent_output,
                feed_dict={
                    model.tourist_input_intent:test_tourist_intent[:5000],
                    model.guide_input_intent:test_guide_intent[:5000],
                    model.labels:test_target_intent[:5000],
                    model.tourist_input_nl:test_tourist_nl[:5000],
                    model.guide_input_nl:test_guide_nl[:5000],
                    model.predict_nl:test_nl[:5000],
                    model.tourist_len_nl:tourist_len_nl[:5000],
                    model.guide_len_nl:guide_len_nl[:5000],
                    model.predict_nl_len:nl_len[:5000],
                    model.dropout_keep_prob:1.0
                    })
        
        test_output = np.concatenate((test_output, sess.run(model.intent_output,
                feed_dict={
                    model.tourist_input_intent:test_tourist_intent[5000:],
                    model.guide_input_intent:test_guide_intent[5000:],
                    model.labels:test_target_intent[5000:],
                    model.tourist_input_nl:test_tourist_nl[5000:],
                    model.guide_input_nl:test_guide_nl[5000:],
                    model.predict_nl:test_nl[5000:],
                    model.tourist_len_nl:tourist_len_nl[5000:],
                    model.guide_len_nl:guide_len_nl[5000:],
                    model.predict_nl_len:nl_len[5000:],
                    model.dropout_keep_prob:1.0
                    })), axis=0)


        # calculate test F1 score
        test_talker = open('Data/test/talker', 'r')
        pred_vec = np.array(list())
        label_vec = np.array(list())
        output_test = list()
        if cur_epoch < switch_epoch:
            ground_truth = test_target_intent
        else:
            ground_truth = np.squeeze(np.split(np.array(test_target_intent).reshape(-1, 6, 26), 6, axis=1)[0])
        for pred, label, talker in zip(test_output, ground_truth, test_talker):
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
            output_test.append(np.append(act_logit, attribute_logit))
            label_vec = np.append(label_vec, label)
        f1sc = f1_score(pred_vec, label_vec, average='binary')
        print "testing f1 score:", f1sc
        test_f1_scores.append(f1sc)
        test_talker.close()
        if cur_epoch == 10:
            f_out_test = open('out_test.txt', 'w')
            for t in output_test:
                s = ''
                for idx, tag in enumerate(t):
                    if tag == 1.0:
                        s = s + (data.whole_dict[idx]) + '-'
                f_out_test.write(s.strip('-')+'\n')
            f_out_test.close()

    print "max test f1 score:", max(test_f1_scores)
    sess.close()
