import tensorflow as tf
import numpy as np

class slu_model(object):
    def __init__(self):
        self.hidden_size = 128
        self.intent_dim = 30 # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 10000 # total word embedding vectors
        self.add_variables()
        self.add_placeholders()
        self.add_variables()
        self.build_graph()
        self.add_loss()
        self.add_train_op()
        
    def init_embedding(self):
        self.init_embedding = self.embedding_matrix.assign(self.read_embedding_matrix)

    def add_variables(self):
        self.embedding_matrix = tf.get_variable(tf.float32, [self.total_word, self.embedding_dim], "embedding")

    def add_placeholders(self):
        # intent sequence, if we take previous n utterences as history, than its length is n*intent_dim
        self.input_x = tf.placeholder(tf.float32, [None, self.intent_dim])
        # natural language input sequence, which is also the utterance we are going to predict(intents)
        self.input_nl = tf.placeholder(tf.float32, [None])
        # pretrained word embedding matrix
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.embedding_dim, self.total_word])
        # correct label that used to calculate sigmoid cross entropy loss, should be [batch_size, intent_dim]
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])

    def hist_biRNN(self):
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        _, final_states = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, self.input_x)
        outputs = tf.concat(final_states) # concatenate forward and backward final states
        return outputs

    def nl_biRNN(self, history_summary):
        inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.input_nl)
        # we want to concat history_summary vector to every single input word vector
        # some trick here
        # first we need to replicate the history summary to [batch_size, seq_len]
        #TODO: batch_size = 1 or ?, need to pad the input batch to make sure correct concat
        tf.reduce_prod(tf.shape(inputs))
        tf.tile(history_summary, [shape[0], shape[1], shape[2]])
        concat_input = tf.concat(inputs, history_summary)
        lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
        lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
        _, final_states = bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input)
        outputs = tf.concat(final_states) # concatenate forward and backward final states
        return outputs

    def build_graph(self):
        tourist_output = self.hist_biRNN()
        guide_output = self.hist_biRNN()
        concat_output = tf.concat(tourist_output, guide_output)
        history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, activation=tf.nn.relu)
        self.final_output = self.nl_biRNN(history_summary)

    def add_loss(self):
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.final_output)
        
    def add_train_op(self):
        optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001, decay=0.999, momentum=0.1, epsilon=1e-8)
        gvs = optimizer.compute_gradients(self.loss)
        # clip the gradients
        capped_gvs = [(tf.clip_by_value(grad, 0., 1.), var) for grad, var in gvs]
        optimizer.apply_gradients(capped_gvs)
        self.train_op = optimizer.minimize(self.loss)
