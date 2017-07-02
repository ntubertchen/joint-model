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

int_learning_rate = 0.005
int_learning_rate_t = 0.01
sap_learning_rate = 0.01
epoc = 20
batch_size = 1
history_size = 3
# Network Parameters
n_hidden = 128 # hidden layer num of features
n_words = Data.maxlength
n_intent = Data.intent_len
w2v_l = 200

# tf Graph input
# sentence_len_t1 = tf.placeholder(tf.int32)
# sentence_len_t2 = tf.placeholder(tf.int32)
# sentence_len_t3 = tf.placeholder(tf.int32)
# sentence_len_g1 = tf.placeholder(tf.int32)
# sentence_len_g2 = tf.placeholder(tf.int32)
# sentence_len_g3 = tf.placeholder(tf.int32)

nl_tourist = tf.placeholder("float", [None, n_words, w2v_l])
nl_guide = tf.placeholder("float",[None,n_words,w2v_l])

intent_tourist = tf.placeholder("float", [None,n_intent])
intent_guide = tf.placeholder("float", [None,n_intent])
sap_label = tf.placeholder('float',[1,n_intent])

# Define weights
weights = {
    'guide': tf.Variable(tf.random_normal([2*n_hidden, n_intent])),
    'tourist': tf.Variable(tf.random_normal([2*n_hidden, n_intent])),
    'sap': tf.Variable(tf.random_normal([4*n_hidden,n_intent]))
}
biases = {
    'guide': tf.Variable(tf.random_normal([n_intent])),
    'tourist': tf.Variable(tf.random_normal([n_intent])),
    'sap': tf.Variable(tf.random_normal([n_intent]))
}

with tf.variable_scope('intent_cell'):
    lstmcell = {
        'tourist_fw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'tourist_bw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'guide_fw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'guide_bw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('sap_cell'):
    sap_lstmcell = {
        'tourist_fw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'tourist_bw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'guide_fw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'guide_bw' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }

def intent_BiRNN(x,scope):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope(scope):
        if scope == "tourist":
            outputs, fw, bw = rnn.static_bidirectional_rnn(lstmcell['tourist_fw'] , lstmcell['tourist_bw'] , x,dtype=tf.float32)
        else:
            outputs, fw, bw = rnn.static_bidirectional_rnn(lstmcell['guide_fw'] , lstmcell['guide_bw'] , x,dtype=tf.float32)
    #senxbatchxlen
    #outputs = tf.transpose(outputs,perm=[1,0,2])
    pred = []
    if scope == "tourist":
        # pred.append(tf.matmul(tf.gather(outputs[0],sentence_len_t1),weights[scope])+ biases[scope])
        # pred.append(tf.matmul(tf.gather(outputs[1],sentence_len_t2),weights[scope])+ biases[scope])
        # pred.append(tf.matmul(tf.gather(outputs[2],sentence_len_t3),weights[scope])+ biases[scope])
        pred.append(tf.matmul(tf.concat([fw[-1],bw[-1]],-1),weights[scope])+biases[scope])
    else:
        # pred.append(tf.matmul(tf.gather(outputs[0],sentence_len_g1),weights[scope])+ biases[scope])
        # pred.append(tf.matmul(tf.gather(outputs[1],sentence_len_g2),weights[scope])+ biases[scope])
        # pred.append(tf.matmul(tf.gather(outputs[2],sentence_len_g3),weights[scope])+ biases[scope])
        pred.append(tf.matmul(tf.concat([fw[-1],bw[-1]],-1),weights[scope])+biases[scope])
    #3x1xlen
    #pred = tf.transpose(pred,perm=[1,0,2])
    #1x3xlen
    return pred

tourist_pred_tag = intent_BiRNN(nl_tourist,"tourist")
guide_pred_tag = intent_BiRNN(nl_guide,"guide")

tourist_pred_out = tf.sigmoid(tourist_pred_tag[0])
guide_pred_out = tf.sigmoid(guide_pred_tag[0])

def sap_BiRNN(x,scope):
    x = tf.unstack(x,history_size,1)
    with tf.variable_scope(scope):
        if scope == "tourist_sap":
            outputs, fw, bw = rnn.static_bidirectional_rnn(sap_lstmcell['tourist_fw'] , sap_lstmcell['tourist_bw'] , x,dtype=tf.float32)
        elif scope == "guide_sap":
            outputs, fw, bw = rnn.static_bidirectional_rnn(sap_lstmcell['guide_fw'] , sap_lstmcell['guide_bw'] , x,dtype=tf.float32)
    return tf.concat([fw[0],bw[0]],-1)

sap_tourist_hidden_layer = sap_BiRNN([tourist_pred_out],'tourist_sap')
sap_guide_hidden_layer = sap_BiRNN([guide_pred_out],'guide_sap')

concated_layer = tf.concat([sap_tourist_hidden_layer,sap_guide_hidden_layer],-1)

sap_tag = tf.matmul(concated_layer,weights['sap']) + biases['sap']
sap_pred_out = tf.sigmoid(sap_tag)

tourist_nl_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tourist_pred_tag[0],labels=intent_tourist))
guide_nl_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=guide_pred_tag[0],labels=intent_guide))
sap_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=sap_pred_out,labels=sap_label))


tourist_intent_optimizer = tf.train.AdamOptimizer(learning_rate=int_learning_rate_t).minimize(tourist_nl_cost)
guide_intent_optimizer = tf.train.AdamOptimizer(learning_rate=int_learning_rate).minimize(guide_nl_cost)
sap_optimizer = tf.train.AdamOptimizer(learning_rate=sap_learning_rate).minimize(sap_cost)

tf.summary.scalar('toursit nl cost', tourist_nl_cost)
tf.summary.scalar('guide nl cost', guide_nl_cost)
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

def intout(fp,pred_out):
    first = 1
    for i in range(len(pred_out)):
        if pred_out[i] > 0.5 and first == 1:
            fp.write(Data.rev_intentdict[i])
            first = 0
        elif pred_out[i] > 0.5:
            fp.write("-"+Data.rev_intentdict[i])
    fp.write('\n')

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

t_in,t_out,t_int,t_info,i_rev,i_talker = t_data.get_all()
seq_in,seq_out,seq_int,seq_info,seq_rev,seq_talker = Data.get_all()
tourist_fp = open('../pred/int','w')
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
        print ("step:",step)
        tourist_loss = 0
        guide_loss = 0
        sap_loss = 0
        counter2 = 0
        counter1 = 0
        counter3 = 0
        _nl_tourist = []
        _nl_guide = []
        _intent_tourist = []
        _intent_guide = []
        for i in range(len(batch_seq)):
            batch_x, batch_slot, batch_intent, batch_info, batch_rev,batch_talker = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_rev[batch_seq[i]],seq_talker[batch_seq[i]]
            if batch_talker.strip() == "Guide":
                _nl_guide.append(batch_x[-1])
                _intent_guide.append(batch_intent[-1])
                d = {}
                d[nl_guide] = _nl_guide
                d[intent_guide] = _intent_guide
                if len(_nl_guide) == 100:
                    g_cost,_ = sess.run([guide_nl_cost,guide_intent_optimizer],feed_dict=d)
                    _nl_guide = []
                    _intent_guide = []
                    counter1 += 1
                    guide_loss += g_cost
            elif batch_talker.strip() == "Tourist":
                _nl_tourist.append(batch_x[-1])
                _intent_tourist.append(batch_intent[-1])
                d = {}
                d[nl_tourist] = _nl_tourist
                d[intent_tourist] = _intent_tourist
                if len(_nl_tourist) == 100:
                    t_cost,_ = sess.run([tourist_nl_cost,tourist_intent_optimizer],feed_dict=d)
                    _nl_tourist = []
                    _intent_tourist = []
                    counter2 += 1
                    tourist_loss += t_cost
            if counter1 == 50:
                print ("guide loss:",float(guide_loss)/counter1)
                counter1 = 0
                guide_loss = 0
            if counter2 == 50:
                print ("tourist loss",float(tourist_loss)/counter2)
                counter2 = 0
                tourist_loss = 0
        
        if step > 3:
            for i in range(len(batch_seq)):
                batch_x, batch_slot, batch_intent, batch_info, batch_rev,batch_talker = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]],seq_info[batch_seq[i]],seq_rev[batch_seq[i]],seq_talker[batch_seq[i]]
                if batch_talker.strip() == "Guide":
                    _nl_tourist = batch_x[0:3]
                    _nl_guide = batch_x[3:6]
                    _intent_guide = batch_intent[0:3]
                    _intent_tourist = batch_intent[3:6]
                    _sap_label = batch_intent[6:]
                    d = {}
                    d[nl_guide] = _nl_guide
                    d[nl_tourist] = _nl_tourist
                    d[intent_guide] = _intent_guide
                    d[intent_tourist] = _intent_tourist
                    d[sap_label] = _sap_label
                    s_cost,_ = sess.run([sap_cost,sap_optimizer],feed_dict=d)
                    counter3 += 1
                    sap_loss += s_cost
                if counter3 == 400:
                    print ("sap loss:",float(sap_loss)/counter3)
                    counter3 = 0
                    sap_loss = 0
        # tourist_intent_logit_list = []
        # tourist_intent_label_list = []
        # guide_intent_logit_list = []
        # guide_intent_label_list = []
        # for j in range(len(t_in)):
        #     test_x,test_slot,test_intent,test_info,test_rev,test_talker = t_in[j] ,t_out[j],t_int[j],t_info[j],i_rev[j],i_talker[j]


        #     if test_talker.strip() == "Guide":
        #         _nl_guide = test_x[4:7]
        #         _intent_guide = test_intent[4:7]
        #         d = {} 
        #         d[nl_guide] = _nl_guide
        #         d[intent_guide] = _intent_guide
        #         guide_prediction_output = sess.run(guide_pred_out,feed_dict=d)
        #         logit,label = intentpreprocess([guide_prediction_output[-1:]],test_intent[-1:])
                
        #         guide_intent_logit_list = np.concatenate((guide_intent_logit_list,logit),axis=0)
        #         guide_intent_label_list = np.concatenate((guide_intent_label_list,label),axis=0)
        #     elif test_talker.strip() == "Tourist":
        #         _nl_tourist = test_x[4:7]
        #         _intent_tourist = test_intent[4:7]
        #         d = {}
        #         d[nl_tourist] = _nl_tourist
        #         d[intent_tourist] = _intent_tourist
        #         tourist_preditction_output = sess.run(tourist_pred_out,feed_dict=d)
        #         logit,label = intentpreprocess([tourist_preditction_output[-1:]],test_intent[-1:])
        #         intout(tourist_fp,logit)
        #         tourist_intent_logit_list = np.concatenate((tourist_intent_logit_list,logit),axis=0)
        #         tourist_intent_label_list = np.concatenate((tourist_intent_label_list,label),axis=0)
        # print (i)
        # print (f1_score(tourist_intent_logit_list,tourist_intent_label_list,average='binary'))
        # print (f1_score(guide_intent_logit_list,guide_intent_label_list,average='binary'))

        step += 1
    save_path = saver.save(sess,"../jointmodel/model.ckpt")