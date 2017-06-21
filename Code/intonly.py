import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from w2v import DataPrepare
import argparse
import random
from sklearn.metrics import f1_score
from sklearn.preprocessing import Binarizer

config = tf.ConfigProto()
config.gpu_options.allow_growth = True


#load glove and give unknown word dimension.
def get_glove(GloVe):
    d = {}
    dict_dim = 200
    with open(GloVe,'r') as f:
        for l in f:
            tmp = l.strip().split()
            d[tmp[0]] = [float(dim) for dim in tmp[1:]]
    unknownword = ['midapril','fivestar','possi','anythings','nonair','don\'','travelator','guine','siobal','minutes\'','equarius','threestar','twostar','harbourville','backpackers\'','dimsum','cebupacific','welltaken','soclose','borderx','threestory','thirtymetre','exci','shoppi','specia','nonairconditioned','here;','nonalcohol','cocktailwise','themselve','twoday','camsur','ifly','sixtyminute','chapatti','briyani','how\'d','^um','aircondition','hours\'','\'kit','expent','dista','gues','northsouth','ezlink','althe','ninetyeight','mentio','lowrise','alri','that\'ll','thirtytwo','victo','foodcourt','koufu','movein','uptodate','onethreeone','thirtyfive','topup','livein','it\'','fiftyeight','threeinone','thirtythree','teeoff','nonweekend','fourstar','alloca','kne','thinki','highend','costy','vaction','its\'','underwaterworld','undewaterworld','reserves;','horbi','freeflight','pizzafari','furbished','mecan','couldn\'t','days\'','twent','panorail','offpeak','singap','don\'ts','bencoo','hereuh','longnosed','alacarte','westernlike','alarcarte','scific','spects','gogreen','ecogood','megazip','loated','tshirts','nonpeak','imbiah','sentos','floweries','airconditioner','inclu','curre','breakfa','deffinitely','coffeshops','transport;','firsttimer','twodays','twonight','fullfledged','selfservice','ghuat','straightly','onebyone','galatica','selegie','kwam']
    for i in range(len(unknownword)):
        tmp = np.zeros(dict_dim)
        tmp[i] = 999
        d[unknownword[i]] = tmp
    n = np.zeros(dict_dim)
    n[dict_dim-1] = 999
    d['<unk>'] = n
    d['Empty'] = np.zeros(dict_dim)
    #handle it's not in glove problem by adding it and 's vector
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

#glove path , natural language path, slot tag path, intent path, additional information(distance from current sentence), sentence talker (not yet added)
path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/train/seq.in" , "../All/Data/train/seq.out" , "../All/Data/train/intent","../All/Data/train/info","../All/Data/train/talker"]
t_path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/test/seq.in" , "../All/Data/test/seq.out" , "../All/Data/test/intent","../All/Data/test/info","../All/Data/test/talker"]

Data = DataPrepare(path,glove_dict)
t_data = DataPrepare(t_path,glove_dict)

int_learning_rate = 0.001
epoc = 10
batch_size = 1

# Network Parameters
n_hidden = 128 # hidden layer num of features
n_words = Data.maxlength
n_intent = Data.intent_len
w2v_l = 200

# tf Graph input
sentence_len = tf.placeholder(tf.int32)

nl_input = tf.placeholder("float", [None, n_words, w2v_l])

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
    pred.append(tf.matmul(tf.gather(outputs[0],sentence_len),weights['t_out'])+ biases['t_out'])
    return pred

def intent_loss(pred,y):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=pred,labels=y))

t_pred = t_intent_BiRNN(nl_input,weights,biases,t_Lstmcell,n_words)

intent_pred = tf.sigmoid(t_pred)
#6x1xint

#3x1xint
_intent_loss = intent_loss(t_pred,y_intent)
_intent_optimizer = tf.train.AdamOptimizer(learning_rate=int_learning_rate).minimize(_intent_loss)

tf.summary.scalar('intent loss', _intent_loss)
merged = tf.summary.merge_all()

def transform_to_onehot(logit):
    for i in range(len(logit[-1])):
        max_major = 0#intent act
        max_value = 0
        for j in range(len(Data.intentdict[0])):
            if logit[-1][i][j] > max_value:
                max_major = j
                max_value = logit[-1][i][j]
                logit[-1][i][j] = 0
            else:
                logit[-1][i][j] = 0
        logit[-1][i][max_major] = 1
        max_minor = 0#intent attritube
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

def intentpreprocess(logit,label):
    logit = transform_to_onehot(logit)
    #1xint
    #because sometimes attribute won't have any, so i establish a none option for model to determine weather to
    #choose a valid or not
    #but a none choose can't be counted as a label
    #this indicate that if both none, then good
    #but if label is none and logit is not, then will be negative positive
    if logit[-1][Data.intentdict[1]['none']] > 0.5 and label[-1][Data.intentdict[1]['none']] > 0.5:
        logit[-1][Data.intentdict[1]['none']] = int(0)
        label[-1][Data.intentdict[1]['none']] = int(0)
    else:
        label[-1][Data.intentdict[1]['none']] = 0
    bin = Binarizer(threshold=0.2)
    logit = bin.fit_transform(logit)
    label = bin.fit_transform(label)
    return logit[-1],label[-1]

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()
#in means natural input
#out means slot tag
#int means intent
#info means distance from current sentence
#rev is nl in string form
#talker means talker
t_in,t_out,t_int,t_info,i_rev,i_talker = t_data.get_all()
seq_in,seq_out,seq_int,seq_info,seq_rev,seq_talker = Data.get_all()

# Launch the graph
with tf.Session(config=config) as sess:
    train_writer = tf.summary.FileWriter('../tensorboard/joint',sess.graph)
    sess.run(init)
    # Keep training until reach max iterations
    step = 0
    batch_seq = []
    #random sequence
    for i in range(len(seq_in)):
        batch_seq.append(i)
    while step < epoc:
        for i in range(len(batch_seq)):
            batch_x, batch_slot, batch_intent, batch_info, batch_rev,batch_talker = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_rev[batch_seq[i]],seq_talker[batch_seq[i]]
            tour_in = [batch_x[-1]]
            tour_nlu = [[batch_intent[-1]]]
            int_in = [[batch_intent[-1]]]
            tour_nlu = np.transpose(tour_nlu,(1,0,2))
            #1x3xvector length
            int_in = np.transpose(int_in,(1,0,2))

            #length for each sentence
            _sentence_len = len(batch_rev[-1].strip().split())
            #information is how long this sentence from target sentence
            _,summary = sess.run([_intent_optimizer,merged],feed_dict={nl_input:tour_in,y_intent: int_in,sentence_len:[_sentence_len]})
            train_writer.add_summary(summary,i + batch_size*step*len(batch_seq))
            if i % 3000 == 0 and i != 0:
                int_logit_list = []
                int_label_list = []
                for j in range(len(t_in)):
                    test_x,test_slot,test_intent,test_info,test_rev,test_talker = t_in[j] ,t_out[j],t_int[j],t_info[j],i_rev[j],i_talker[j]
                    if test_talker.strip() == "Tourist":
                        tour_in = [test_x[-1]]
                        tour_nlu = [[test_intent[-1]]]
                        int_in = [[test_intent[-1]]]
                        tour_nlu = np.transpose(tour_nlu,(1,0,2))
                        int_in = np.transpose(int_in,(1,0,2))
                        _sentence_len = len(test_rev[-1].strip().split())
                        i_p = sess.run(intent_pred,feed_dict={nl_input:tour_in,y_intent: int_in,sentence_len:[_sentence_len]})
                        #take only final sentence as testing output
                        logit,label = intentpreprocess([i_p[-1]],[test_intent[-1]])
                        int_logit_list = np.concatenate((int_logit_list,logit),axis=0)
                        int_label_list = np.concatenate((int_label_list,label),axis=0)
                print (i)
                print (f1_score(int_logit_list,int_label_list,average='binary'))
        step += 1
    save_path = saver.save(sess,"../jointmodel/model.ckpt")