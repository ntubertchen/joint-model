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


path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/train/seq.in" , "../All/Data/train/seq.out" , "../All/Data/train/intent"]
t_path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/test/seq.in" , "../All/Data/test/seq.out" , "../All/Data/test/intent"]
slotpath = '../All/Data/slot_list'
intentpath = '../All/Data/intent_list'
Data = DataPrepare(path,slotpath,intentpath)
t_data = DataPrepare(t_path,slotpath,intentpath)
# Parameters
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
n_slot = Data.slot_dim
n_intent = Data.intent_dim
w2v_l = 200

"""
SAP Parameters
"""
S_learning_rate = 0.001
S_training_iters = 400000
S_batch_size = 1

# Network Parameters
S_vector_length = Data.slot_dim + Data.intent_dim # MNIST data input (img shape: 28*28) /////vector length 613    
S_n_sentences = 3 # timesteps /////number of sentences 
S_n_hidden = 128 # hidden layer num of features
S_n_labels = Data.intent_dim # MNIST total classes (0-9 digits)

#sap_x = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
#sap_y = tf.placeholder("float", [None, S_n_labels])

# tf Graph input
x = tf.placeholder("float", [None, n_words, w2v_l])
y_slot = tf.placeholder("float", [None, n_words, n_slot])
y_intent = tf.placeholder("float", [None, 1 ,n_intent])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'slot_out': tf.Variable(tf.random_normal([2*n_hidden, n_slot])),
    'intent_out': tf.Variable(tf.random_normal([2*n_hidden, n_intent]))
}
biases = {
    'slot_out': tf.Variable(tf.random_normal([n_slot])),
    'intent_out': tf.Variable(tf.random_normal([n_intent]))
}
with tf.variable_scope('slot_cell'):
    s_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
with tf.variable_scope('intent_cell'):
    i_Lstmcell = {
        'fw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0),
        'bw_lstm' : rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
    }
def slot_BiRNN(x, weights, biases,Lstmcell,n_words):
    # Prepare data shape to match `bidirectional_rnn` function requirements
    # Current data input shape: (batch_size, n_sentences, vector_length)
    # Required shape: 'n_sentences' tensors list of shape (batch_size, vector_length)
    # Permuting batch_size and n_steps
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('slot_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'], Lstmcell['bw_lstm'], x,dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    pred = []
    for i in range(len(outputs)):
        pred.append(tf.matmul(outputs[i],weights['slot_out']) + biases['slot_out'])
    return pred

def intent_BiRNN(x,weights,biases,Lstmcell,n_words):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('intent_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'] , Lstmcell['bw_lstm'] , x,dtype=tf.float32)
    pred = []
    for i in range(batch_size):
        pred.append(tf.matmul([outputs[-1][i]],weights['intent_out']) + biases['intent_out'])
    return pred

def slot_loss(pred,y,Data):
    cost = []
    y = tf.transpose(y,perm=[1,0,2])
    for i in range(Data.maxlength):
        logit = tf.slice(pred,[i,0,0],[1,-1,-1])
        label = tf.slice(y,[i,0,0],[1,-1,-1])
        cost.append((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit[0],labels=label[0]))))
    loss = tf.reduce_sum(cost)
    return loss

def intent_loss(pred,y):
    y = tf.transpose(y,perm=[1,0,2])
    pred = tf.transpose(pred,perm=[1,0,2])
    return tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=pred[0],labels=y[0]))

def tag_out(mertix,label_dict):
    #batch x sentence x vector
    batches = []
    for b in mertix:
        l = []
        for sen in b:
            l.append(label_dict[np.argmax(sen)])
        batches.append(l)
    return batches

def write_out(label_dict,fout,mertix):
    tags = tag_out(mertix,label_dict)
    for batch in tags:
        for word in batch:
            fout.write(word + " ")
        fout.write("***next*** ")
    fout.write("\n")

slot_pred = slot_BiRNN(x, weights, biases,s_Lstmcell,n_words)
SAP_slot = tf.reduce_sum(slot_pred,0)
#pred_out = slot_out(Data,pred)
_slot_loss = slot_loss(slot_pred,y_slot,Data)


intent_pred = intent_BiRNN(x,weights,biases,i_Lstmcell,n_words)
#SAP_input = tf.stack(SAP_slot)
_intent_loss = intent_loss(intent_pred,y_intent)



# else:
#     pred = intent_BiRNN(x,weights,biases)
#     pred_out = tf.argmax(pred,axis=1)
#     loss = intent_loss(pred,y_intent)
# Define loss and optimizer
slot_pred_acc = tf.equal(tf.argmax(tf.transpose(slot_pred,perm=[1,0,2]),axis=2),tf.argmax(y_slot,axis=2))
slot_acc = tf.reduce_mean(tf.cast(slot_pred_acc,tf.float32))

int_out = tf.argmax(intent_pred,axis=2)
int_pred_acc = tf.equal(tf.argmax(intent_pred,axis=2), tf.argmax(y_intent,axis=2))
int_acc = tf.reduce_mean(tf.cast(int_pred_acc, tf.float32))

_slot_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_slot_loss)
_intent_optimizer = tf.train.AdamOptimizer(learning_rate=S_learning_rate).minimize(_intent_loss)
# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) 
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

if args.test == False:
    # Launch the graph
    with tf.Session(config=config) as sess:
        ac = 0
        sess.run(init)
        # Keep training until reach max iterations
        seq_in,seq_out,seq_int = Data.get_all()
        t_in,t_out,t_int = t_data.get_all()
        step = 0
        batch_seq = []
        fout = open('./testout','w')
        for i in range(len(seq_in)):
            batch_seq.append(i)
        while step < epoc:
            np.random.shuffle(batch_seq)
            for i in range(len(seq_in)):
                batch_x, batch_slot, batch_intent = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]]
                nlu_in = batch_x[:len(batch_x)-2]
                _s_nlu_out = batch_slot[:len(batch_slot)-1]
                _i_nlu_out = batch_intent[:len(batch_intent)-1]
                intd = Data.rev_intentdict
                _,_ = sess.run([_slot_optimizer,_intent_optimizer],feed_dict={x:nlu_in,y_slot: _s_nlu_out,y_intent: _i_nlu_out})
                
                # if step % display_step == 0:
                #     # Calculate batch accuracy
                #     acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #     # Calculate batch loss
                if i % 100 == 0 and i != 0:
                    tmp = sess.run([int_out],feed_dict={x:nlu_in,y_slot: _s_nlu_out,y_intent: _i_nlu_out})
                    i_a,s_a = sess.run([int_acc,slot_acc],feed_dict={x:nlu_in,y_slot: _s_nlu_out,y_intent: _i_nlu_out})
                    print("Iter i:" +str(step) + " "+ str(i) +" {:.6f}".format(i_a)+" "+"{:.6f}".format(s_a))
                if i % 10000 == 0 and i != 0:
                    s_ac = 0
                    int_ac = 0
                    for k in range(len(t_in)):
                        test_x,test_s,test_i = t_in[k],t_out[k],t_int[k]
                        t_nlu_in = test_x[:len(test_x)-2]
                        t_snlu = test_s[:len(test_s)-1]
                        t_inlu = test_i[:len(test_i)-1]
                        i_a,s_a = sess.run([int_acc,slot_acc],feed_dict={x:t_nlu_in,y_slot: t_snlu,y_intent: t_inlu})
                        s_ac += s_a
                        int_ac += i_a
                        #s_p,int_p = sess.run([slot_pred,intent_pred],feed_dict={\
                        #    x:t_nlu_in,y_slot:t_snlu,y_intent:t_inlu})
                        #s_p = np.transpose(s_p,[1,0,2])
                        #int_p = np.transpose(int_p,[1,0,2])
                        #write_out(Data.rev_slotdict,fout,t_snlu)
                        #write_out(Data.rev_slotdict,fout,s_p)
                        #write_out(Data.rev_intentdict,fout,t_inlu)
                        #write_out(Data.rev_intentdict,fout,int_p)
                    print("Testing" +str(step) + " "+ str(i) +" {:.6f}".format(s_ac/len(t_in))+" "+"{:.6f}".format(int_ac/len(t_in)))
            step += 1
        save_path = saver.save(sess,"../model/model.ckpt")
        # print("Optimization Finished!")
else:
    print ("testing")
    with tf.Session(config=config) as sess:
        ckpt = tf.train.get_checkpoint_state("../model")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        t_in,t_out,t_int = t_data.get_all()
        s_ac = 0
        int_ac = 0
        for k in range(len(t_in)):
            test_x,test_s,test_i = t_in[k],t_out[k],t_int[k]
            t_nlu_in = test_x[:len(test_x)-2]
            t_snlu = test_s[:len(test_s)-1]
            t_inlu = test_i[:len(test_i)-1]
            i_a,s_a = sess.run([int_acc,slot_acc],feed_dict={x:t_nlu_in,y_slot: t_snlu,y_intent: t_inlu})
            s_ac += s_a
            int_ac += i_a
            #s_p,int_p = sess.run([slot_pred,intent_pred],feed_dict={\
            #    x:t_nlu_in,y_slot:t_snlu,y_intent:t_inlu})
            #s_p = np.transpose(s_p,[1,0,2])
            #int_p = np.transpose(int_p,[1,0,2])
            #write_out(Data.rev_slotdict,fout,t_snlu)
            #write_out(Data.rev_slotdict,fout,s_p)
            #write_out(Data.rev_intentdict,fout,t_inlu)
            #write_out(Data.rev_intentdict,fout,int_p)
        print("Testing" +" {:.6f}".format(s_ac/len(t_in))+" "+"{:.6f}".format(int_ac/len(t_in)))