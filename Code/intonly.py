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
from w2v import DataPrepare
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
path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/train/seq.in" , "../All/Data/train/seq.out" , "../All/Data/train/intent","../All/Data/train/info"]
t_path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/test/seq.in" , "../All/Data/test/seq.out" , "../All/Data/test/intent","../All/Data/test/info"]

Data = DataPrepare(path,glove_dict)
t_data = DataPrepare(t_path,glove_dict)
# Parameters
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
s1_len = tf.placeholder(tf.int32)
s2_len = tf.placeholder(tf.int32)
s3_len = tf.placeholder(tf.int32)
t_x = tf.placeholder("float", [None, n_words, w2v_l])

y_intent = tf.placeholder("float", [None, 1 ,n_intent])

# Define weights
weights = {
    't_out': tf.Variable(tf.random_normal([2*n_hidden, n_intent]))
}
biases = {
    't_out': tf.Variable(tf.random_normal([n_intent]))
}

with tf.variable_scope('intent_cell'):
    t_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }


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
    return pred

def intent_loss(pred,y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))

t_pred = t_intent_BiRNN(t_x,weights,biases,t_Lstmcell,n_words)

int_pred = tf.sigmoid(t_pred)
#6x1xint

#3x1xint
_intent_loss = intent_loss(t_pred,y_intent)
tf.summary.scalar('intent loss', _intent_loss)

merged = tf.summary.merge_all()

_intent_optimizer = tf.train.AdamOptimizer(learning_rate=int_learning_rate).minimize(_intent_loss)

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

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()
t_in,t_out,t_int,t_info,i_rev = t_data.get_all()
if args.test == False:
    # Launch the graph
    with tf.Session(config=config) as sess:
        ac = 0
        train_writer = tf.summary.FileWriter('../tensorboard/joint',sess.graph)
        sess.run(init)
        # Keep training until reach max iterations
        seq_in,seq_out,seq_int,seq_info,seq_rev = Data.get_all()
        step = 0
        batch_seq = []
        for i in range(len(seq_in)):
            batch_seq.append(i)
        while step < epoc:
            for i in range(len(batch_seq)):
                batch_x, batch_slot, batch_intent, batch_info, batch_rev = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_rev[batch_seq[i]]
                tour_in = batch_x[3:6]
                tour_nlu = [batch_intent[3:6]]
                int_in = [batch_intent[3:6]]
                tour_nlu = np.transpose(tour_nlu,(1,0,2))
                int_in = np.transpose(int_in,(1,0,2))
                _s1_len = len(batch_rev[3].strip().split())
                _s2_len = len(batch_rev[4].strip().split())
                _s3_len = len(batch_rev[5].strip().split())
                _sen_info = batch_info[:-1]
                _,summary = sess.run([_intent_optimizer,merged],feed_dict={t_x:tour_in,y_intent: int_in,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len]})
                train_writer.add_summary(summary,i + batch_size*step*len(batch_seq))
                if i % 3000 == 0 and i != 0:
                    slotans = open('../pred/s_ans','w')
                    intpred = open('../pred/int','w')
                    int_logit_list = []
                    int_label_list = []
                    sap_logit_list = []
                    sap_label_list = []
                    for j in range(len(t_in)):
                        test_x,test_slot,test_intent,test_info,test_rev = t_in[j] ,t_out[j],t_int[j],t_info[j],i_rev[j]
                        if test_info[-1].strip() == "Tourist":
                            tour_in = test_x[4:7]
                            tour_nlu = [test_intent[4:7]]
                            int_in = [test_intent[4:7]]
                            tour_nlu = np.transpose(tour_nlu,(1,0,2))
                            int_in = np.transpose(int_in,(1,0,2))
                            _s1_len = len(test_rev[4].strip().split())
                            _s2_len = len(test_rev[5].strip().split())
                            _s3_len = len(test_rev[6].strip().split())
                            i_p = sess.run(int_pred,feed_dict={t_x:tour_in,y_intent: int_in,s1_len:[_s1_len],s2_len:[_s2_len],s3_len:[_s3_len]})
                            logit,label = intpreprocess([i_p[-1]],[test_intent[-1]])
                            int_logit_list = np.concatenate((int_logit_list,logit),axis=0)
                            int_label_list = np.concatenate((int_label_list,label),axis=0)
                    print (i)
                    print (f1_score(int_logit_list,int_label_list,average='binary'))
            step += 1
        save_path = saver.save(sess,"../jointmodel/model.ckpt")
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("../jointmodel")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
