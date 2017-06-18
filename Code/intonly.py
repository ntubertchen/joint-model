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
    nuk = ['midapril','fivestar','possi','anythings','nonair','don\'','travelator','guine','siobal','minutes\'','equarius','threestar','twostar','harbourville','backpackers\'','dimsum','cebupacific','welltaken','soclose','borderx','threestory','thirtymetre','exci','shoppi','specia','nonairconditioned','here;','nonalcohol','cocktailwise','themselve','twoday','camsur','ifly','sixtyminute','chapatti','briyani','how\'d','^um','aircondition','hours\'','\'kit','expent','dista','gues','northsouth','ezlink','althe','ninetyeight','mentio','lowrise','alri','that\'ll','thirtytwo','victo','foodcourt','koufu','movein','uptodate','onethreeone','thirtyfive','topup','livein','it\'','fiftyeight','threeinone','thirtythree','teeoff','nonweekend','fourstar','alloca','kne','thinki','highend','costy','vaction','its\'','underwaterworld','undewaterworld','reserves;','horbi','freeflight','pizzafari','furbished','mecan','couldn\'t','days\'','twent','panorail','offpeak','singap','don\'ts','bencoo','hereuh','longnosed','alacarte','westernlike','alarcarte','scific','spects','gogreen','ecogood','megazip','loated','tshirts','nonpeak','imbiah','sentos','floweries','airconditioner','inclu','curre','breakfa','deffinitely','coffeshops','transport;','firsttimer','twodays','twonight','fullfledged','selfservice','ghuat','straightly','onebyone','galatica','selegie','kwam']
    for i in range(len(nuk)):
      tmp = np.zeros(dict_dim)
      tmp[i] = 999
      d[nuk[i]] = tmp
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
slot_learning_rate = 0.001
int_learning_rate = 0.001
learning_rate = 0.0001
epoc = 10
batch_size = 3
display_step = 50

# Network Parameters
#n_input = 28 # MNIST data input (img shape: 28*28)
#n_steps = 28 # timesteps
n_hidden = 128 # hidden layer num of features
#n_classes = 10 # MNIST total classes (0-9 digits)
n_words = Data.maxlength
n_slot = Data.slot_len
n_intent = Data.intent_len
w2v_l = 200

"""
SAP Parameters
"""
S_learning_rate = 0.001
S_training_iters = 400000
S_batch_size = 1

# Network Parameters
S_vector_length = Data.slot_len + Data.intent_len # MNIST data input (img shape: 28*28) /////vector length 613
S_n_sentences = 3 # timesteps /////number of sentences 
S_n_hidden = 128 # hidden layer num of features
S_n_labels = Data.intent_len # MNIST total classes (0-9 digits)
S_n_info = 3

#sap_x = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
sap_y = tf.placeholder("float", [None, S_n_labels])

# tf Graph input
s1_all = tf.placeholder(tf.int32)
s1_len = tf.placeholder(tf.int32)
s2_all = tf.placeholder(tf.int32)
s2_len = tf.placeholder(tf.int32)
s3_all = tf.placeholder(tf.int32)
s3_len = tf.placeholder(tf.int32)
s4_len = tf.placeholder(tf.int32)
s5_len = tf.placeholder(tf.int32)
s6_len = tf.placeholder(tf.int32)
t_x = tf.placeholder("float", [None, n_words, w2v_l])
g_x = tf.placeholder("float", [None, n_words, w2v_l])

y_bio = tf.placeholder("float", [None, n_words, len(Data.slotdict[0])])
y_main = tf.placeholder("float", [None, n_words, len(Data.slotdict[1])])
y_from_to = tf.placeholder("float", [None, n_words, len(Data.slotdict[2])])
y_rel = tf.placeholder("float", [None, n_words, len(Data.slotdict[3])])
y_subcat = tf.placeholder("float", [None, n_words, len(Data.slotdict[4])])

y_intent = tf.placeholder("float", [None, 1 ,n_intent])
sap_info = tf.placeholder("float",[None,S_n_sentences,S_n_info])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'BIO': tf.Variable(tf.random_normal([2*n_hidden, len(Data.slotdict[0])])),
    'MAIN': tf.Variable(tf.random_normal([2*n_hidden, len(Data.slotdict[1])])),
    'FROM_TO': tf.Variable(tf.random_normal([2*n_hidden, len(Data.slotdict[2])])),
    'REL': tf.Variable(tf.random_normal([2*n_hidden, len(Data.slotdict[3])])),
    'SUBCAT': tf.Variable(tf.random_normal([2*n_hidden, len(Data.slotdict[4])])),
    't_out': tf.Variable(tf.random_normal([2*n_hidden, n_intent])),
    'g_out': tf.Variable(tf.random_normal([2*n_hidden, n_intent])),
    'SAP_out': tf.Variable(tf.random_normal([2*S_n_hidden, S_n_labels]))
}
biases = {
    'BIO': tf.Variable(tf.random_normal([len(Data.slotdict[0])])),
    'MAIN': tf.Variable(tf.random_normal([len(Data.slotdict[1])])),
    'FROM_TO': tf.Variable(tf.random_normal([len(Data.slotdict[2])])),
    'REL': tf.Variable(tf.random_normal([len(Data.slotdict[3])])),
    'SUBCAT': tf.Variable(tf.random_normal([len(Data.slotdict[4])])),
    't_out': tf.Variable(tf.random_normal([n_intent])),
    'g_out': tf.Variable(tf.random_normal([n_intent])),
    'SAP_out': tf.Variable(tf.random_normal([S_n_labels]))
}
with tf.variable_scope('BIO'):
    BIO_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('MAIN'):
    MAIN_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('FROM_TO'):
    FROM_TO_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('REL'):
    REL_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('SUBCAT'):
    SUBCAT_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('intent_cell'):
    t_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
    g_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
def slot_BiRNN(x, weights, biases,Lstmcell,scope):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope(scope+"rnn"):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'], Lstmcell['bw_lstm'], x,dtype=tf.float32)
    #senxbatchxlen
    outputs = tf.transpose(outputs,perm=[1,0,2])
    pred = []
    #pred.append(tf.matmul(tf.gather(outputs[0],s1_all),weights[scope]) + biases[scope])
    #pred.append(tf.matmul(tf.gather(outputs[1],s2_all),weights[scope]) + biases[scope])
    pred.append(tf.matmul(outputs[2],weights[scope]) + biases[scope])
    #batchxsenxlen
    return pred

def t_intent_BiRNN(x,weights,biases,Lstmcell,n_words):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('tourist'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'] , Lstmcell['bw_lstm'] , x,dtype=tf.float32)
    #senxbatchxlen
    outputs = tf.transpose(outputs,perm=[1,0,2])
    pred = []
    pred.append(tf.matmul(tf.gather(outputs[0],s1_len),weights['t_out'])+ biases['t_out'])
    pred.append(tf.matmul(tf.gather(outputs[1],s2_len),weights['t_out'])+ biases['t_out'])
    pred.append(tf.matmul(tf.gather(outputs[2],s3_len),weights['t_out'])+ biases['t_out'])
    #3x1xint    p
    # for i in range(batch_size):
    #     #3x1xintent_dim
    #     pred.append(tf.matmul([outputs[-1][i]],weights['intent_out']) + biases['intent_out'])
    return pred

def g_intent_BiRNN(x,weights,biases,Lstmcell,n_words):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('guide'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'] , Lstmcell['bw_lstm'] , x,dtype=tf.float32)
    #senxbatchxlen
    outputs = tf.transpose(outputs,perm=[1,0,2])
    pred = []
    pred.append(tf.matmul(tf.gather(outputs[0],s4_len),weights['g_out'])+ biases['g_out'])
    pred.append(tf.matmul(tf.gather(outputs[1],s5_len),weights['g_out'])+ biases['g_out'])
    pred.append(tf.matmul(tf.gather(outputs[2],s6_len),weights['g_out'])+ biases['g_out'])
    #3x1xint    p
    # for i in range(batch_size):
    #     #3x1xintent_dim
    #     pred.append(tf.matmul([outputs[-1][i]],weights['intent_out']) + biases['intent_out'])
    return pred

def slot_loss(logit,y):
    cost = []
    label = []
    #label.append(tf.gather(y[0],s1_all))
    #label.append(tf.gather(y[1],s2_all))
    #label.append(tf.gather(y[2],s3_all))
    label.append(y[2])
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logit[-1],labels=label[-1]))

def intent_loss(pred,y):
    # y = tf.transpose(y,perm=[1,0,2])
    # pred = tf.transpose(pred,perm=[1,0,2])
    #both 3x1xlen
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))

t_pred = t_intent_BiRNN(t_x,weights,biases,t_Lstmcell,n_words)
#g_pred = g_intent_BiRNN(g_x,weights,biases,g_Lstmcell,n_words)

int_pred = tf.sigmoid(t_pred)
#6x1xint

#i_pred = tf.sigmoid(intent_pred)
#3x1xint
_intent_loss = intent_loss(t_pred,y_intent)
tf.summary.scalar('intent loss', _intent_loss)

sap_x = []
int_out = tf.reduce_sum(t_pred,1)
#6xint
for i in range(3):
    sap_x.append(tf.concat([int_out[i],sap_info[0][i]],0))
#sap_x.append(tf.concat([tf.reduce_sum(i_pred,1)],1))

def SAP_BiRNN(x, weights, biases):
    x = tf.unstack(x, 3, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Get lstm cell output
    with tf.variable_scope('SAP_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['SAP_out']) + biases['SAP_out']

SAP_pred = SAP_BiRNN([sap_x], weights, biases)
_pred = tf.sigmoid(SAP_pred)

sap_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=sap_y,logits=SAP_pred))
#tf.summary.scalar('sap loss', sap_loss)
merged = tf.summary.merge_all()

_intent_optimizer_withsap = tf.train.AdamOptimizer(learning_rate=int_learning_rate).minimize(_intent_loss+sap_loss)
_intent_optimizer = tf.train.AdamOptimizer(learning_rate=int_learning_rate).minimize(_intent_loss)
_SAP_optimizer = tf.train.AdamOptimizer(learning_rate=S_learning_rate).minimize(sap_loss)

def toone(logit):
    for i in range(len(logit[-1])):
        max_major = 0
        max_value = 0
        for j in range(len(Data.intentdict[0])):
            if logit[-1][i][j] > max_value:
                max_major = j
                max_value = logit[-1][i][j]
                logit[-1][i][j] = 0
            else:
                logit[-1][i][j] = 0
        logit[-1][i][max_major] = 1
        max_minor = 0
        max_value = 0
        for j in range(len(Data.intentdict[0]),len(Data.intentdict[1])):
            if logit[-1][i][j] > max_value:
                max_minor = j
                max_value = logit[-1][i][j]
                logit[-1][i][j] = 0
            else:
                logit[-1][i][j] = 0
        logit[-1][i][max_minor] = 1
    return logit[-1]

def intpreprocess(logit,label):
    logit = toone(logit)
    #label = label[-1]
    #1xint
    if logit[-1][Data.intentdict[1]['none']] > 0.5 and label[-1][Data.intentdict[1]['none']] > 0.5:
        logit[-1][Data.intentdict[1]['none']] = int(0)
        label[-1][Data.intentdict[1]['none']] = int(0)
    bin = Binarizer(threshold=0.2)
    logit = bin.fit_transform(logit)
    label = bin.fit_transform(label)
    return logit[-1],label[-1]

def intout(logit,fout):
    logit = toone(logit)
    logit = logit[-1]
    first = 1
    for i in range(len(logit)):
        if logit[i] > 0.2:
            if first == 1:
                fout.write(Data.rev_intentdict[i])
                first = 0
            else:
                fout.write("-"+Data.rev_intentdict[i])
    fout.write('\n')

def sappreprocess(logit,label):
    logit = toone([logit])
    #1xint
    if logit[-1][Data.intentdict[1]['none']] > 0.5 and label[-1][Data.intentdict[1]['none']] > 0.5:
        logit[-1][Data.intentdict[1]['none']] = int(0)
        label[-1][Data.intentdict[1]['none']] = int(0)
    bin = Binarizer(threshold=0.2)
    logit = bin.fit_transform(logit)
    label = bin.fit_transform(label)
    return logit[-1],label[-1]

def toones(logit):
    for i in range(len(logit[-1])):
        if (logit[-1][i][Data.slotdict[0]['O']] > logit[-1][i][Data.slotdict[0]['B']] and logit[-1][i][Data.slotdict[0]['O']] > logit[-1][i][Data.slotdict[0]['I']]):
            for j in range(len(logit[-1][i])):
                logit[-1][i][j] = 0
            logit[-1][i][Data.slotdict[0]['O']] = 1
        else:
            logit[-1][i][Data.slotdict[0]['O']] = 0
            if logit[-1][i][Data.slotdict[0]['B']] >= logit[-1][i][Data.slotdict[0]['I']]:
                logit[-1][i][Data.slotdict[0]['B']] = 1
                logit[-1][i][Data.slotdict[0]['I']] = 0
            elif logit[-1][i][Data.slotdict[0]['B']] < logit[-1][i][Data.slotdict[0]['I']]:
                logit[-1][i][Data.slotdict[0]['B']] = 0
                logit[-1][i][Data.slotdict[0]['I']] = 1
            MAIN = ['AREA','DET','FEE','FOOD','LOC','TIME','TRSP','WEATHER']
            SUBCAT = ['COUNTRY','CITY','DISTRICT','NEIGHBORHOOD','ACCESS', 'BELIEF', 'BUILDING', 'EVENT', 'PRICE', 'NATURE', 'HISTORY', 'MEAL', 'MONUMENT','STROLL','VIEW','ATTRACTION', 'SERVICES', 'PRODUCTS','TEMPLE', 'RESTAURANT', 'SHOP', 'CULTURAL', 'GARDEN', 'ATTRACTION', 'HOTEL', 'WATERSIDE', 'EDUCATION', 'ROAD', 'AIRPORT','DATE', 'INTERVAL', 'START', 'END', 'OPEN', 'CLOSE','STATION', 'TYPE','MAIN']
            REL = ['NEAR', 'FAR', 'NEXT', 'OPPOSITE', 'NORTH', 'SOUTH', 'EAST','WEST','BEFORE', 'AFTER', 'AROUND','NONE']
            FROM_TO = ['FROM', 'TO','NONE']
            slot = [MAIN,FROM_TO,REL,SUBCAT]
            for j in range(len(slot)):
                max_value = 0
                max_index = 0
                for k in range(len(slot[j])):
                    if logit[-1][i][Data.slotdict[j+1][slot[j][k]]] >= max_value:
                        if max_index != 0:
                            logit[-1][i][max_index] = 0
                        max_value = logit[-1][i][Data.slotdict[j+1][slot[j][k]]]
                        max_index = Data.slotdict[j+1][slot[j][k]]
                        logit[-1][i][Data.slotdict[j+1][slot[j][k]]] = 0
                    else:
                        logit[-1][i][Data.slotdict[j+1][slot[j][k]]] = 0
                logit[-1][i][max_index] = 1
    return logit[-1]

def acc_sap(logit,label):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    logit = toone(logit)
    #1xlabel
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

def acc_slot(logits,labels,batch_l):
    label = labels[-1]
    logit = toones(logits)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    d_tp = 0
    d_fp = 0
    d_fn = 0
    c_tp = 0
    c_fp = 0
    c_fn = 0
    for i in range(batch_l):
        if logit[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(3):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 1:
                d_tp += 1
            else:
                d_fp += 1
        if label[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(3):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 0:
                d_fn += 1
    for i in range(batch_l):
        if logit[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(11):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 1:
                c_tp += 1
            else:
                c_fp += 1
        if label[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(11):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 0:
                c_fn += 1
    for i in range(batch_l):
        if logit[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(len(logit[i])):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 1:
                tp += 1
            else:
                fp += 1
        if label[i][Data.slotdict[0]['O']] < 0.2:
            success = 1
            for j in range(len(logit[i])):
                if (logit[i][j] > 0.2 and label[i][j] < 0.2) or (logit[i][j] < 0.2 and label[i][j] > 0.2):
                    success = 0
            if success == 0:
                fn += 1
    return tp,fp,fn,d_tp,d_fp,d_fn,c_tp,c_fp,c_fn
    
def acc_int(logits,labels):
    logit = toone(logits)
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(logit)):
        for j in range(len(logit[i])):
            if logit[i][j] > 0.9 and labels[-1][i][j] > 0.9:
                tp += 1
            elif logit[i][j] < 0.5 and labels[-1][i][j] < 0.5:
                tn += 1
            elif logit[i][j] < 0.9 and labels[-1][i][j] > 0.9:
                fn += 1
            elif logit[i][j] > 0.5 and labels[-1][i][j] < 0.5:
                fp += 1
    if fp == 0 and fn == 0 and tp > 0:
        acc = 1
    else:
        acc = 0
    return tp,fp,fn,acc

def slotout(logit,fout,nl):
    logit = toones(logit)
    for i in range(batch_l):
        first = 1
        for j in range(len(logit[i])):
            if logit[i][j] > 0.2:
                if first == 1:
                    fout.write(Data.rev_slotdict[j])
                    first = 0
                else:
                    fout.write("-"+Data.rev_slotdict[j])
        fout.write(' ')
    fout.write(nl)
    fout.write('\n')

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()
t_in,t_out,t_int,t_info,i_rev,i_talker = t_data.get_all()
if args.test == False:
    # Launch the graph
    with tf.Session(config=config) as sess:
        ac = 0
        train_writer = tf.summary.FileWriter('../tensorboard/joint',sess.graph)
        sess.run(init)
        # Keep training until reach max iterations
        seq_in,seq_out,seq_int,seq_info,seq_rev,seq_talker = Data.get_all()
        step = 0
        batch_seq = []
        for i in range(len(seq_in)):
            batch_seq.append(i)
        while step < epoc:
            for i in range(len(batch_seq)):
                batch_x, batch_slot, batch_intent, batch_info, batch_rev,batch_talker = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_rev[batch_seq[i]],seq_talker[batch_seq[i]]
                tour_in = batch_x[1:4]
                tour_nlu = [batch_intent[1:4]]
                int_in = [batch_intent[1:4]]
                tour_nlu = np.transpose(tour_nlu,(1,0,2))
                int_in = np.transpose(int_in,(1,0,2))
                _s1_len = len(batch_rev[0].strip().split())-1
                _s2_len = len(batch_rev[1].strip().split())-1
                _s3_len = len(batch_rev[2].strip().split())-1
                _s4_len = len(batch_rev[3].strip().split())-1
                _s5_len = len(batch_rev[4].strip().split())-1
                _s6_len = len(batch_rev[5].strip().split())-1
                if _s1_len <= 0:
                    _s1_len=0
                if _s2_len <= 0:
                    _s2_len=0
                if _s3_len <= 0:
                    _s3_len=0
                if _s4_len <= 0:
                    _s4_len=0
                if _s5_len <= 0:
                    _s5_len=0
                if _s6_len <= 0:
                    _s6_len=0
                _sen_info = batch_info[:-1]
                _info_in = np.zeros((2,3,3))
                for k in range(3):
                    _info_in[0][k][0] = 1
                    _info_in[1][k][1] = 1
                    _info_in[0][k][2] = _sen_info[k]
                    _info_in[1][k][2] = _sen_info[k+3]
                if batch_talker == "Guide":
                    _,summary = sess.run([_intent_optimizer,merged],feed_dict={t_x:tour_in,y_intent: int_in,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len],s4_len:[_s4_len],s5_len:[_s5_len],s6_len:[_s6_len]})
                    train_writer.add_summary(summary,i + batch_size*step*len(batch_seq))
                else:
                    _ = sess.run([_intent_optimizer],feed_dict={t_x:tour_in,y_intent: int_in,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len],s4_len:[_s4_len],s5_len:[_s5_len],s6_len:[_s6_len]})
                
                # else:
                #     if random.random() > 0.1:
                #         _,summary = sess.run([_SAP_optimizer,merged],feed_dict={x:nlu_in,y_slot: _s_nlu_out,y_intent: _i_nlu_out,sap_y:[batch_intent[-1]],s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len],sap_info:[_info_in],s1_all:_s1_all,s2_all:_s2_all,s3_all:_s3_all})
                #         train_writer.add_summary(summary,i)
                #     else:
                #         _,_ = sess.run([_slot_optimizer,_intent_optimizer],feed_dict={x:nlu_in,y_slot: _s_nlu_out,y_intent: _i_nlu_out,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len],s1_all:_s1_all,s2_all:_s2_all,s3_all:_s3_all})
                if i % 3000 == 0 and i != 0:
                    slotans = open('../pred/s_ans','w')
                    intpred = open('../pred/int','w')
                    int_logit_list = []
                    int_label_list = []
                    sap_logit_list = []
                    sap_label_list = []
                    for j in range(len(t_in)):
                        test_x,test_slot,test_intent,test_info,test_rev,test_talker = t_in[j] ,t_out[j],t_int[j],t_info[j],i_rev[j],i_talker[j]
                        tour_in = test_x[4:7]
                        tour_nlu = [test_intent[4:7]]
                        int_in = [test_intent[4:7]]
                        tour_nlu = np.transpose(tour_nlu,(1,0,2))
                        int_in = np.transpose(int_in,(1,0,2))
                        _s1_len = len(test_rev[0].strip().split())-1
                        _s2_len = len(test_rev[1].strip().split())-1
                        _s3_len = len(test_rev[2].strip().split())-1
                        _s4_len = len(test_rev[3].strip().split())-1
                        _s5_len = len(test_rev[4].strip().split())-1
                        _s6_len = len(test_rev[5].strip().split())-1
                        if _s1_len <= 0:
                            _s1_len=0
                        if _s2_len <= 0:
                            _s2_len=0
                        if _s3_len <= 0:
                            _s3_len=0
                        if _s4_len <= 0:
                            _s4_len=0
                        if _s5_len <= 0:
                            _s5_len=0
                        if _s6_len <= 0:
                            _s6_len=0
                        i_p = sess.run(int_pred,feed_dict={t_x:tour_in,y_intent: int_in,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len],s4_len:[_s4_len],s5_len:[_s5_len],s6_len:[_s6_len]})
                        logit,label = intpreprocess([i_p[2]],[test_intent[2]])
                        int_logit_list = np.concatenate((int_logit_list,logit),axis=0)
                        int_label_list = np.concatenate((int_label_list,label),axis=0)
                    print (i)
                    print (f1_score(int_logit_list,int_label_list,average='binary'))
                    #print ("intent:recall:",ir,",precision:",ip,",f score:",2 *ir*ip/ (ir+ip)," intent acc:",float(intacc)/countt)
                    #print ("sap:recall:",_r,",precision:",_p,",f score:",2*_r*_p/(_r+_p)," sap acc:",float(sapacc)/countg)
            step += 1
        save_path = saver.save(sess,"../jointmodel/model.ckpt")
        # print("Optimization Finished!")
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("../jointmodel")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        s1 = s2 = s3 = i1= i2 = i3 = 0
        for j in range(len(t_in)):
            test_x,test_slot,test_intent,test_info,test_rev = t_in[j] ,t_out[j],t_int[j],t_info[j],i_rev[j]
            if test_info[-1].strip() != "Tourist":
                continue
            test_in = test_x[4:]
            test_snlu = test_slot[4:]
            test_inlu = [test_intent[4:]]
            t_inlu = np.transpose(test_inlu,(1,0,2))
            i_p,s_p = sess.run([i_pred,s_pred],feed_dict={x:test_in,y_slot: test_snlu,y_intent: t_inlu})
            batch_l = test_rev[-1].strip().split()
            batch_l = len(batch_l)
            stp,sfp,sfn = acc_slot(s_p,test_snlu,batch_l)
            itp,ifp,ifn = acc_int(i_p,test_inlu)
            s1 += stp
            s2 += sfp
            s3 += sfn
            i1 += itp
            i2 += ifp
            i3 += ifn
        sr = float(s1)/(s1+s3)
        sp = float(s1)/(s1+s2)
        ir = float(i1)/(i1+i3)
        ip = float(i1)/(i1+i2)
        print ("slot:recall:",sr,",precision:",sp,",f score:",2 *sr*sp/(sr+sp))
        print ("intent:recall:",ir,",precision:",ip,",f score:",2 *ir*ip/ (ir+ip))