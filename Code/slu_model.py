import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim, use_attention):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 400002 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.filter_sizes = [3,4,5]
        self.filter_depth = 64
        self.hist_len = 3
        self.use_attention = use_attention
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
        self.tourist_input_intent = tf.placeholder(tf.float32, [None, self.hist_len, self.intent_dim])
        self.guide_input_intent = tf.placeholder(tf.float32, [None, self.hist_len, self.intent_dim])
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.tourist_input_nl = tf.placeholder(tf.int32, [None, self.hist_len, self.max_seq_len])
        self.guide_input_nl = tf.placeholder(tf.int32, [None, self.hist_len, self.max_seq_len])
        self.tourist_len_nl = tf.placeholder(tf.int32, [None, self.hist_len])
        self.guide_len_nl = tf.placeholder(tf.int32, [None, self.hist_len])
        self.predict_nl_len = tf.placeholder(tf.int32, [None])
        self.predict_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.dropout_keep_prob = tf.placeholder(tf.float32)

    def hist_cnn(self, scope, idx):
        with tf.variable_scope(scope):
            if idx != 0:
                tf.get_variable_scope().reuse_variables()
            if scope == 'tourist':
                # tourist_input_nl should now have [batch_size, 3, max_seq_len]
                tourist_input_nl = tf.unstack(self.tourist_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, tourist_input_nl)
            elif scope == 'guide':
                guide_input_nl = tf.unstack(self.guide_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, guide_input_nl)
            pooled_outputs = list()
            for idx, filter_size in enumerate(self.filter_sizes):
                # convolution layer
                kernel_size = (filter_size)
                h = tf.layers.conv1d(inputs, self.filter_depth, kernel_size, activation=tf.nn.relu, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
                # max over time pooling
                pooled = tf.layers.max_pooling1d(h, (self.max_seq_len-filter_size+1), 1)
                pooled_outputs.append(pooled)
            num_filters_total = self.filter_depth * len(self.filter_sizes)
            h_pool_flat = tf.squeeze(tf.concat(pooled_outputs, axis=2), axis=1)
            h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
            dense_h = tf.layers.dense(inputs=h_drop, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
            return dense_h

    def hist_biRNN(self, scope):
        with tf.variable_scope(scope):
            if scope == 'tourist':
                inputs = tf.nn.embedding_lookup(self.intent_matrix, self.tourist_input)
                seq_len = self.tourist_len
            elif scope == 'guide':
                inputs = tf.nn.embedding_lookup(self.intent_matrix, self.guide_input)
                seq_len = self.guide_len

            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=seq_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs

    def nl_biRNN(self, history_summary):
        with tf.variable_scope("nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.predict_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            history_summary = tf.expand_dims(history_summary, axis=1)
            replicate_summary = tf.tile(history_summary, [1, self.max_seq_len, 1]) # [batch_size, self.max_seq_len, self.intent_dim]
            concat_input = tf.concat([inputs, replicate_summary], axis=2) # [batch_size, self.max_seq_len, self.intent_dim+self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input, sequence_length=self.predict_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs

    def role_attention(self):
        self.unstack_tourist_cnn_hist = list()
        self.unstack_guide_cnn_hist = list()
        for i in range(self.hist_len):
            self.unstack_tourist_cnn_hist.append(self.hist_cnn('tourist', i))
            self.unstack_guide_cnn_hist.append(self.hist_cnn('guide', i))
        # TODO: naively concat all the history
        tourist_cnn_hist = tf.sigmoid(tf.concat(self.unstack_tourist_cnn_hist, axis=1))
        guide_cnn_hist = tf.sigmoid(tf.concat(self.unstack_guide_cnn_hist, axis=1))

        with tf.variable_scope("role_attention"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.predict_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=self.predict_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            cur_rnn_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            # concat current output with cnn_hist
            outputs = tf.concat([tourist_cnn_hist, guide_cnn_hist, cur_rnn_outputs], axis=1)
            attention = tf.nn.softmax(tf.layers.dense(inputs=outputs, units=2, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer))
            #batch_size = tf.shape(attention)[0]
            #col_0 = tf.concat([tf.expand_dims(tf.range(0, batch_size), axis=1), tf.zeros([batch_size, 1], dtype=tf.int32)], axis=1)
            #col_1 = tf.concat([tf.expand_dims(tf.range(0, batch_size), axis=1), tf.ones([batch_size, 1], dtype=tf.int32)], axis=1)
            #role_attention = tf.concat([tf.multiply(tourist_summary, tf.expand_dims(tf.gather_nd(attention, col_0), axis=1)), tf.multiply(guide_summary, tf.expand_dims(tf.gather_nd(attention, col_1), axis=1))], axis=1)
            role_attention = tf.add(tf.multiply(tourist_cnn_hist, tf.expand_dims(tf.unstack(attention, axis=1)[0], axis=1)), tf.multiply(guide_cnn_hist, tf.expand_dims(tf.unstack(attention, axis=1)[1], axis=1)))
            return role_attention

    def build_graph(self):
        concat_output = self.role_attention()
        history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        final_output = self.nl_biRNN(history_summary)
        self.intent_output = tf.layers.dense(inputs=final_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

    def add_loss(self):
        loss_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.intent_output))
        stack_tourist_cnn_hist = tf.stack(self.unstack_tourist_cnn_hist, axis=1)
        stack_guide_cnn_hist = tf.stack(self.unstack_guide_cnn_hist, axis=1)
        loss_intent_tourist = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.tourist_input_intent, logits=stack_tourist_cnn_hist))
        loss_intent_guide = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.guide_input_intent, logits=stack_guide_cnn_hist))
        self.loss = loss_ce + loss_intent_tourist + loss_intent_guide
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
