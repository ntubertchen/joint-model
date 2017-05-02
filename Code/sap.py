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


path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/train/seq.in" , "../All/Data/train/seq.out" , "../All/Data/train/intent"]
t_path = ["../GloVe/glove.6B.200d.txt" , "../All/Data/test/seq.in" , "../All/Data/test/seq.out" , "../All/Data/test/intent"]
slotpath = '../All/Data/slot_list'
intentpath = '../All/Data/intent_list'
Data = DataPrepare(path,slotpath,intentpath)
t_data = DataPrepare(t_path,slotpath,intentpath)
# Parameters
learning_rate = 0.0005
epoc = 5
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

# tf Graph input
sap_x = tf.placeholder("float", [None, S_n_sentences, S_vector_length])
sap_y = tf.placeholder("float", [None, S_n_labels])

# Define weights
weights = {
    # Hidden layer weights => 2*n_hidden because of forward + backward cells
    'SAP_out': tf.Variable(tf.random_normal([2*S_n_hidden, S_n_labels]),name="weight")
}
biases = {
    'SAP_out': tf.Variable(tf.random_normal([S_n_labels]),name="biase")
}

def SAP_BiRNN(x, weights, biases):
    x = tf.unstack(x, S_n_sentences, 1)
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
# Define loss and optimize

correct_pred = tf.equal(tf.argmax(SAP_pred,axis=1), tf.argmax(sap_y,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

sap_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=sap_y,logits=SAP_pred))
_SAP_optimizer = tf.train.AdamOptimizer(learning_rate=S_learning_rate).minimize(sap_cost)
# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1)) 
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

saver = tf.train.Saver()

# Initializing the variables
init = tf.global_variables_initializer()

if args.test == False:
    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        # Keep training until reach max iterations
        seq_in,seq_out,seq_int = Data.get_all()
        t_in,t_out,t_int = t_data.get_all()
        step = 0
        batch_seq = []
        #fout = open('./testout','w')
        for i in range(len(seq_in)):
            batch_seq.append(i)
        while step < epoc:
            np.random.shuffle(batch_seq)
            for i in range(len(seq_in)):
                batch_x, batch_slot, batch_intent = seq_in[batch_seq[i]],seq_out[batch_seq[i]],seq_int[batch_seq[i]]
                _s_nlu_out = batch_slot[:len(batch_slot)-1]
                summed_slot = np.sum(_s_nlu_out,axis=1)
                _i_nlu_out = batch_intent[:len(batch_intent)-1]
                summed_int = np.sum(_i_nlu_out,axis=1)
                tags_con = np.concatenate((summed_slot,summed_int),axis=1)
                if batch_x[len(batch_x)-1] == "Guide":
                    _ = sess.run([_SAP_optimizer],feed_dict={sap_x:[tags_con],sap_y:batch_intent[len(batch_intent)-1]})
                # if step % display_step == 0:
                #     # Calculate batch accuracy
                #     acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
                #     # Calculate batch loss
                # if i % 100 == 0 and i != 0:
                #     ac = sess.run([accuracy],feed_dict={sap_x:[tags_con],sap_y:batch_intent[len(batch_intent)-1]})
                #     print("Iter i:" +str(step) + " "+ str(i) +str(ac))
                if i % 1000 == 0:
                    sap_ac = 0
                    count = 0
                    for k in range(len(t_in)):
                        test_x,test_s,test_i = t_in[k],t_out[k],t_int[k]
                        if test_x[len(test_x)-1] == "Guide":
                            count+=1
                            t_snlu = test_s[:len(test_s)-1]
                            t_summed_slot = np.sum(t_snlu,axis=1)
                            t_inlu = test_i[:len(test_i)-1]
                            t_summed_int = np.sum(t_inlu,axis=1)
                            t_tags_con = np.concatenate((t_summed_slot,t_summed_int),axis=1)
                            sap_a = sess.run([accuracy],feed_dict={\
                                sap_x:[t_tags_con],sap_y:test_i[len(test_i)-1]})
                            sap_ac += sap_a[0]
                    print("Iter i:" +str(step) + " "+ str(i) +" "+str(sap_ac/count))
            step += 1
        save_path = saver.save(sess,"../model/sapmodel.ckpt")
        # print("Optimization Finished!")
else:
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state("../model")
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