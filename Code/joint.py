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


path = ["../GloVe/glove.6B.200d.txt" , "../Guide/Data/seq.in" , "../Guide/Data/seq.out" , "../Guide/Data/intent"]
slotpath = '../Guide/Data/slot_list'
intentpath = '../Guide/Data/intent_list'
Data = DataPrepare(path,slotpath,intentpath)

# Parameters
learning_rate = 0.0001
training_iters = 10000
batch_size = 1
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

print (n_words)
"""
SAP Parameters
"""
S_learning_rate = 0.0001
S_training_iters = 400000
S_batch_size = 1

# Network Parameters
S_vector_length = Data.slot_dim + Data.intent_dim # MNIST data input (img shape: 28*28) /////vector length 613    
S_n_sentences = 3 # timesteps /////number of sentences 
S_n_hidden = 128 # hidden layer num of features
S_n_labels = Data.intent_dim # MNIST total classes (0-9 digits)

sap_x = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
sap_y = tf.placeholder("float", [None, S_n_labels])

# tf Graph input
x = tf.placeholder("float", [None, n_words, w2v_l])
y_slot = tf.placeholder("float", [None, n_words, n_slot])
y_intent = tf.placeholder("float", [None, 1 ,n_intent])

# Define weights
weights = { 
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'slot_out': tf.Variable(tf.random_normal([2*n_hidden, n_slot])),
    'intent_out': tf.Variable(tf.random_normal([2*n_hidden, n_intent])),
    'SAP_out': tf.Variable(tf.random_normal([2*S_n_hidden, S_n_labels]))
}
biases = {
    'slot_out': tf.Variable(tf.random_normal([n_slot])),
    'intent_out': tf.Variable(tf.random_normal([n_intent])),
    'SAP_out': tf.Variable(tf.random_normal([S_n_labels]))
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
def slot_BiRNN(x, weights, biases,Lstmcell):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('slot_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'], Lstmcell['bw_lstm'], x,dtype=tf.float32)
    # Linear activation, using rnn inner loop last output
    pred = []
    for i in range(len(outputs)):
        pred.append(tf.matmul(outputs[i],weights['slot_out']) + biases['slot_out'])
    return pred

def intent_BiRNN(x,weights,biases,Lstmcell):
    x = tf.unstack(x, n_words, 1)
    with tf.variable_scope('intent_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(Lstmcell['fw_lstm'] , Lstmcell['bw_lstm'] , x,dtype=tf.float32)
    return tf.matmul(outputs[-1],weights['intent_out']) + biases['intent_out']

def slot_loss(pred,y):
    cost = []
    i = tf.constant(0)
    for i in range(Data.maxlength):
        logit = tf.slice(pred,[Data.maxlength-1-i,0,0],[1,1,-1])
        label = tf.slice(y,[0,i,0],[1,1,-1])
        cost.append((tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logit,labels=label))))
    loss = tf.reduce_sum(cost)
    return loss

def intent_loss(pred,y):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))

def slot_out(Data,pred):
    out = []
    for i in range(Data.maxlength):
        logit = tf.slice(pred,[i,0,0],[1,1,-1])
        out.append(tf.argmax(logit,axis=1))
    return out

def intent_out(Data,pred):
    return tf.argmax(pred)


slot_pred = slot_BiRNN(x, weights, biases,s_Lstmcell)
SAP_slot = tf.reduce_sum(slot_pred,0)
#pred_out = slot_out(Data,pred)
_slot_loss = slot_loss(slot_pred,y_slot)


intent_pred = intent_BiRNN(x,weights,biases,i_Lstmcell)
SAP_input = tf.reshape(tf.concat([SAP_slot,intent_pred],1),[-1])
#SAP_input = tf.stack(SAP_slot)
_intent_loss = intent_loss(intent_pred,y_intent)

def SAP_BiRNN(x, weights, biases):
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, S_vector_length])
    x = tf.split(x, S_n_sentences)
    lstm_fw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Backward direction cell
    lstm_bw_cell = rnn.BasicLSTMCell(S_n_hidden, forget_bias=1.0, input_size=None, state_is_tuple=True, activation=tf.tanh)
    # Get lstm cell output
    with tf.variable_scope('SAP_rnn'):
        outputs, _, _ = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['SAP_out']) + biases['SAP_out']

SAP_pred = SAP_BiRNN(sap_x, weights, biases)

# else:
#     pred = intent_BiRNN(x,weights,biases)
#     pred_out = tf.argmax(pred,axis=1)
#     loss = intent_loss(pred,y_intent)
# Define loss and optimizer

int_pred_acc = tf.equal(tf.argmax(intent_pred,axis=1), tf.argmax(y_intent,axis=1))
int_acc = tf.reduce_mean(tf.cast(int_pred_acc, tf.float32))



correct_pred = tf.equal(tf.argmax(SAP_pred,axis=1), tf.argmax(sap_y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sap_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sap_y,logits=SAP_pred))
_SAP_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(sap_cost)
_slot_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_slot_loss+sap_cost)
_intent_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(_intent_loss+sap_cost)
# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) 
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

if args.test == False:
    # Launch the graph
    with tf.Session() as sess:
        ac = 0
        sess.run(init)
        step = 1
        # Keep training until reach max iterations
        while step * batch_size < training_iters:
            batch_x, batch_slot, batch_intent = Data.get_trainbatch()
            sap_batch = []
            for i in range(len(batch_x)-1):
                sap_in = sess.run([SAP_input],feed_dict={x:[batch_x[i]]})
                sap_batch.append(sap_in[0])

            sess.run([_SAP_optimizer,_slot_optimizer,_intent_optimizer], feed_dict={x: [batch_x[3]],sap_x: [sap_batch],sap_y: batch_intent, y_slot: [batch_slot],y_intent:[batch_intent]})

            # if step % display_step == 0:
            #     # Calculate batch accuracy
            #     acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            #     # Calculate batch loss
            acc,sap_l,s_l,i_l = sess.run([accuracy,sap_cost,_slot_loss,_intent_loss], feed_dict={x: [batch_x[3]],sap_x: [sap_batch],sap_y: batch_intent, y_slot: [batch_slot],y_intent:[batch_intent]})
            ac += acc
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(s_l)  + " " +"{:.6f}".format(i_l)+ " " +"{:.6f}".format(sap_l)+ " " +"{:.6f}".format(float(ac)/step))
            step += 1
        save_path = saver.save(sess,"../jointmodel/model.ckpt")
        # print("Optimization Finished!")
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("../jointmodel")
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)
        # test_x,test_y,origin_x = Data.get_test()
        # slot_out = sess.run(pred,feed_dict={x: [test_x]})
        # origin_x = origin_x.strip().split()
        # out = []
        # for z in range(len(slot_out)):
        #     out.append(np.argmax(slot_out[z]))
        # for z in range(len(origin_x)):
        #     if Data.rev_slotdict[out[len(out)-1-z]] != 'O':
        #         print (Data.rev_slotdict[out[len(out)-1-z]],origin_x[z])