import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 400002 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.add_variables()
        self.add_placeholders()
        self.add_variables()
        self.build_graph()
        self.add_loss()
        self.add_train_op()
        self.init_embedding()
        self.init_model = tf.global_variables_initializer()

    def init_embedding(self):
        self.init_embedding = self.embedding_matrix.assign(self.read_embedding_matrix)

    def add_variables(self):
        self.embedding_matrix = tf.Variable(tf.truncated_normal([self.total_word, self.embedding_dim]), dtype=tf.float32, name="glove_embedding")

    def add_placeholders(self):
        # natural language input sequence, which is also the utterance we are going to predict(intents)
        self.input_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])
        # pretrained word embedding matrix
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        # correct label that used to calculate sigmoid cross entropy loss, should be [batch_size, intent_dim]
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        # used to extract the right time step from output of bi-rnn
        self.nl_indices = tf.placeholder(tf.int32, [None, 2])

    def nl_biRNN(self):
	
        with tf.variable_scope("nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size, forget_bias=1.0)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
            fw_outputs = tf.gather_nd(outputs[0], self.nl_indices)
            bw_outputs = tf.gather_nd(outputs[1], self.nl_indices)
            outputs = tf.concat([fw_outputs, bw_outputs], axis=1) # concatenate forward and backward final states
            return outputs

    def build_graph(self):
        #outputs = self.nl_biRNN()
        outputs = self.nl_cnn()
        self.intent_output = tf.layers.dense(inputs=outputs, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.intent_output))
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(self.loss)
