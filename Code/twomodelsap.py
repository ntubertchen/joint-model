'''
A Bidirectional Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from sap_w2v import DataPrepare
import argparse
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer
'''
To classify images using a bidirectional recurrent neural network, we consider
every image row as a sequence of pixels. Because MNIST image shape is 28*28px,
we will then handle 28 sequences of 28 steps for every sample.
'''
# python s_v.py --glove "../glove/glove.6B.200d.txt" --train_p "../Data/" --slot_l "../Data/slot_l.txt" --intent_l "../Data/intent_l.txt"

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

parser = argparse.ArgumentParser()
parser.add_argument('--glove', help='glove path')
parser.add_argument('--train_p', help='training data path')
parser.add_argument('--test_p', help='testing data path')
parser.add_argument('--test',action='store_true',help='test or train')
parser.add_argument('--slot_l', help='slot tag path')
parser.add_argument('--intent_l', help='intent path')
parser.add_argument('--intent', action='store_false',help='intent training')
parser.add_argument('--slot', action='store_false',help='slot training')
args = parser.parse_args()


def get_glove(GloVe):
    d = {}
    dict_dim = 200
    with open(GloVe,'r') as f:
        for l in f:
            tmp = l.strip().split()
            d[tmp[0]] = [float(dim) for dim in tmp[1:]]
    n = np.zeros(dict_dim)
    n[dict_dim-1] = 999
    d['<unk>'] = n
    d['Empty'] = np.zeros(dict_dim)
    add_s = ['let','that','it','there','here','how','he','she','what']
    add_nt = ['do','did','were','have','does','would','was','has','should','is','are']
    add_re = ['we','they','you']
    add_ha = ['i','who','they','you','we']
    add_am = ['i']
    add_will = ['you','he','i','she','we','there','it','they']
    add_d = ['you','i','they','that','we']
    prefix = [add_s,add_nt,add_re,add_ha,add_am,add_will,add_d]
    syms = ['\'s','not','are','have','am','will','would']
    short = ['\'s','n\'t','\'re','\'ve','\'m','\'ll','\'d']
    for i in range(len(prefix)):
      for word in prefix[i]:
        d[word+short[i]] = np.add(d[word],d[syms[i]])
    d['won\'t'] = np.add(d['will'],d['not'])
    d['can\'t'] = np.add(d['can'],d['not'])
    return d

Glove = "../GloVe/glove.6B.200d.txt"
glove_dict = get_glove(Glove)
path = ["../GloVe/glove.6B.200d.txt" , "../All/Sap/train/seq.in" , "../All/Sap/train/seq.out" , "../All/Sap/train/intent","../All/Sap/train/info","../All/Sap/train/talker"]
t_path = ["../GloVe/glove.6B.200d.txt" , "../All/Sap/test/seq.in" , "../All/Sap/test/seq.out" , "../All/Sap/test/intent","../All/Sap/test/info","../All/Sap/test/talker"]

Data = DataPrepare(path,glove_dict)
t_data = DataPrepare(t_path,glove_dict)
# Parameters
learning_rate = 0.0005 
epoc = 10
batch_size = 1 ##################
display_step = 50

# Network Parameters
#n_input = 28 # MNIST data input (img shape: 28*28)
#n_steps = 28 # timesteps
#n_hidden = 128 # hidden layer num of features
#n_classes = 10 # MNIST total classes (0-9 digits)
n_words = Data.maxlength
n_slot = Data.slot_len
n_intent = Data.intent_len

"""
SAP Parameters
"""
S_learning_rate = 0.005#############
S_training_iters = 400000
S_batch_size = 1

# Network Parameters
S_vector_length = Data.intent_len #+ Data.slot_len # MNIST data input (img shape: 28*28) /////vector length 613    
S_n_sentences = 3 # timesteps /////number of sentences 
S_n_hidden = 128 # hidden layer num of features ###########
S_n_info = 3
S_n_labels = Data.intent_len # MNIST total classes (0-9 digits)

attention_weight = tf.Variable(tf.random_normal([3,1]),name='attention_weight')

# tf Graph input
sap_t = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
sap_g = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
sap_info = tf.placeholder("float",[None,S_n_sentences,S_n_info])
sap_y = tf.placeholder("float", [None, S_n_labels])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'SAP': tf.Variable(tf.random_normal([4*S_n_hidden, S_n_labels]),name="weight")
}
biases = {
    'SAP': tf.Variable(tf.random_normal([S_n_labels]),name="biase")
}

def SAP_BiRNN(x,scope):
    x = tf.unstack(x, S_n_sentences, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Get lstm cell output
    with tf.variable_scope(scope):
        _, fw, bw = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
    # Linear activation, using rnn inner loop last output

    return tf.concat([fw[0],bw[0]],-1)

tourist_in = tf.concat([sap_t,[sap_info[0]]],2)
guide_in = tf.concat([sap_g,[sap_info[1]]],2)
#add attention
            # attention_in = (tf.multiply(sap_in[0],attention_weight))
            # attention_sum = tf.reduce_mean(attention_in,0)
            # tmp = []
            # for i in range(3):
            #     tmp.append(tf.concat([sap_in[0][i],attention_sum],0))
#attention end
tourist_layer = SAP_BiRNN(tourist_in,'Tourist')
guide_layer = SAP_BiRNN(guide_in,'Guide')
concated_layer = tf.concat([tourist_layer,guide_layer],-1)
SAP_pred = tf.matmul(concated_layer,weights['SAP']) + biases['SAP']#8*hidden

_pred = tf.sigmoid(SAP_pred)

def toone(logit):
    for i in range(len(logit[0])):
        max_major = 0
        max_value = 0
        for j in range(len(Data.intentdict[0])):
            if logit[0][i][j] >= 0.5:
                max_major = j
                max_value = logit[0][i][j]
                logit[0][i][j] = 1
            elif logit[0][i][j] > max_value:
                max_major = j
                max_value = logit[0][i][j]
                logit[0][i][j] = 0
            else:
                logit[0][i][j] = 0
        logit[0][i][max_major] = 1
        max_minor = 0
        max_value = 0
        for j in range(len(Data.intentdict[0]),len(Data.intentdict[1])):
            if logit[0][i][j] >= 0.5:
                max_minor = j
                max_value = logit[0][i][j]
                logit[0][i][j] = 1
            elif logit[0][i][j] > max_value:
                max_minor = j
                max_value = logit[0][i][j]
                logit[0][i][j] = 0
            else:
                logit[0][i][j] = 0
        logit[0][i][max_minor] = 1
    return logit[0]

def intout(fp,pred_out):
    first = 1
    for i in range(len(pred_out)):
        if pred_out[i] > 0.5 and first == 1:
            fp.write(Data.rev_intentdict[i])
            first = 0
        elif pred_out[i] > 0.5:
            fp.write("-"+Data.rev_intentdict[i])
    fp.write('\n')

def acc_sap(logit,label,flp):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    logit = toone(logit)
    intout(flp,logit[-1])
    for i in range(len(logit)):
        for j in range(len(logit[i])):
            if logit[i][j] > 0.9 and label[i][j] > 0.9:
                tp += 1
            elif logit[i][j] < 0.5 and label[i][j] < 0.5:
                tn += 1
            elif logit[i][j] < 0.9 and label[i][j] > 0.9:
                fn += 1
            elif logit[i][j] > 0.5 and label[i][j] < 0.5:
                fp += 1
    if fp == 0 and fn == 0 and tp > 0:
        acc = 1
    else:
        acc = 0
    return tp,fp,fn,acc

def preprocess(logit,label):
    logit = toone(logit)
    if logit[-1][Data.intentdict[1]['none']] > 0.5 and label[-1][Data.intentdict[1]['none']] > 0.5:
        logit[-1][Data.intentdict[1]['none']] = int(0)
        label[-1][Data.intentdict[1]['none']] = int(0)
    bin = Binarizer(threshold=0.2)
    logit = bin.fit_transform([logit[-1]])
    label = bin.fit_transform([label[-1]])
    return logit[-1],label[-1]

sap_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sap_y,logits=SAP_pred))
_SAP_optimizer = tf.train.AdamOptimizer(learning_rate=S_learning_rate).minimize(sap_cost)

saver = tf.train.Saver()
# Initializing the variables
init = tf.global_variables_initializer()
seq_out,seq_int,seq_info,seq_talker = Data.get_all()
t_out,t_int,t_info,t_talker = t_data.get_all()
if args.test == False:
    # Launch the graph
    with tf.Session(config=config) as sess:
        sess.run(init)
        # Keep training until reach max iterations
        step = 0
        batch_seq = []
        #fout = open('./testout','w')
        for i in range(len(seq_out)):
            batch_seq.append(i)
        while step < epoc:
            np.random.shuffle(batch_seq)
            for i in range(len(seq_out)):
                train_batch = []
                train_y_batch = []
                train_info = []
                batch_slot, batch_intent,batch_info,batch_talker = seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_talker[batch_seq[i]]
                _s_nlu_out = batch_slot[:-1]
                summed_slot = np.sum(_s_nlu_out,axis=1)
                _sen_info = batch_info[:-1]
                _info_in = np.zeros((2,3,3))
                for k in range(3):
                    _info_in[0][k][0] = 1
                    _info_in[1][k][1] = 1
                    _info_in[0][k][2] = _sen_info[k]
                    _info_in[1][k][2] = _sen_info[k+3]
                t_i_nlu_out = batch_intent[0:3]
                g_i_nlu_out = batch_intent[3:6]
                #tags_con = np.concatenate((summed_slot,_i_nlu_out),axis=1)
                train_y_batch.append(batch_intent[-1])
                if batch_talker.strip() == "Guide":
                    _ = sess.run([_SAP_optimizer],feed_dict={sap_t:[t_i_nlu_out],sap_g:[g_i_nlu_out],sap_y:train_y_batch,sap_info:_info_in})
                if i * batch_size % 3000 == 0 and i != 0:
                    i1=i2=i3=sapacc=count=0
                    logit_list = []
                    label_list = []
                    sappred_out = open('../pred/saponly','w')
                    sap_ans = open('../pred/sapans','w')
                    for k in range(len(t_out)):
                        batch_slot=t_out[k]
                        batch_intent=t_int[k]
                        batch_info = t_info[k]
                        batch_talker = t_talker[k]
                        _sen_info = batch_info[:-1]
                        _info_in = np.zeros((2,3,3))
                        for k in range(3):
                            _info_in[0][k][0] = 1
                            _info_in[1][k][1] = 1
                            _info_in[0][k][2] = _sen_info[k]
                            _info_in[1][k][2] = _sen_info[k+3]
                        count += 1
                        _s_nlu_out = batch_slot[:-1]
                        summed_slot = np.sum(_s_nlu_out,axis=1)
                        t_i_nlu_out = batch_intent[0:3]
                        g_i_nlu_out = batch_intent[3:6]
                        #tags_con = np.concatenate((summed_slot,_i_nlu_out),axis=1)
                        if batch_talker.strip() == "Guide":
                            sapp = sess.run([_pred],feed_dict={sap_t:[t_i_nlu_out],sap_g:[g_i_nlu_out],sap_y:[batch_intent[-1]],sap_info:_info_in})
                        logit,label = preprocess(sapp,[batch_intent[-1]])
                        logit_list = np.concatenate((logit_list,logit),axis=0)
                        label_list = np.concatenate((label_list,label),axis=0)
                    print (f1_score(logit_list,label_list,average='binary'))
            step += 1
        save_path = saver.save(sess,"../model/sapmodel.ckpt")
        # print("Optimization Finished!")
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("../model")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)