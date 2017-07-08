import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

class slu_model(object):
    def __init__(self, max_seq_len, intent_dim, use_attention, use_mid_loss):
        self.hidden_size = 128
        self.intent_dim = intent_dim # one hot encoding
        self.embedding_dim = 200 # read from glove
        self.total_word = 400002 # total word embedding vectors
        self.max_seq_len = max_seq_len
        self.filter_sizes = [2,3,4]
        self.filter_depth = 128
        self.hist_len = 3
        self.use_attention = use_attention
        self.use_mid_loss = use_mid_loss
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
        self.read_embedding_matrix = tf.placeholder(tf.float32, [self.total_word, self.embedding_dim])
        self.labels = tf.placeholder(tf.float32, [None, self.intent_dim])
        self.history_nl = tf.placeholder(tf.int32, [None, self.hist_len*2, self.max_seq_len])
        self.current_nl_len = tf.placeholder(tf.int32, [None])
        self.current_nl = tf.placeholder(tf.int32, [None, self.max_seq_len])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.history_intent = tf.placeholder(tf.float32, [None, self.hist_len*2, self.intent_dim])

    def hist_cnn(self, scope, idx):
        with tf.variable_scope(scope):
            if idx != 0:
                tf.get_variable_scope().reuse_variables()
            if scope == "CNN_t":
                input_nl = tf.unstack(self.history_nl, axis=1)[idx]
            elif scope == "CNN_g":
                input_nl = tf.unstack(self.history_nl, axis=1)[idx+self.hist_len]
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, input_nl)
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

    def hist_biRNN(self, scope, idx):
        with tf.variable_scope(scope):
            reuse = False
            if idx != 0:
                tf.get_variable_scope().reuse_variables()
                reuse = True
            if scope == 'tourist':
                tourist_input_nl = tf.unstack(self.tourist_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, tourist_input_nl)
                seq_len = tf.unstack(self.tourist_len_nl, axis=1)[idx]
            elif scope == 'guide':
                guide_input_nl = tf.unstack(self.guide_input_nl, axis=1)[idx]
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, guide_input_nl)
                seq_len = tf.unstack(self.guide_len_nl, axis=1)[idx]

            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size, reuse=reuse)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size, reuse=reuse)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=seq_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            outputs = tf.layers.dense(inputs=outputs, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
            return outputs

    def nl_biRNN(self, history_summary):
        with tf.variable_scope("nl"):
            inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
            history_summary = tf.expand_dims(history_summary, axis=1)
            replicate_summary = tf.tile(history_summary, [1, self.max_seq_len, 1]) # [batch_size, self.max_seq_len, self.intent_dim]
            concat_input = tf.concat([inputs, replicate_summary], axis=2) # [batch_size, self.max_seq_len, self.intent_dim+self.embedding_dim]
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, concat_input, sequence_length=self.current_nl_len, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs

    def attention_biRNN(self, scope, inputs):
        with tf.variable_scope(scope):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
            return outputs

    def attention(self):
        self.unstack_hist_tourist = list()
        self.unstack_hist_guide = list()
        for i in range(self.hist_len):
            self.unstack_hist_tourist.append(self.hist_cnn('CNN_t', i))
            self.unstack_hist_guide.append(self.hist_cnn('CNN_g', i))

        serial_hist_tourist = tf.sigmoid(tf.stack(self.unstack_hist_tourist, axis=1))
        serial_hist_guide = tf.sigmoid(tf.stack(self.unstack_hist_guide, axis=1))

        with tf.variable_scope("serial_tourist"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, serial_hist_tourist, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            tourist_output = tf.concat([final_fw, final_bw], axis=1)

        with tf.variable_scope("serial_guide"):
            lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
            lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
            _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, serial_hist_guide, dtype=tf.float32)
            final_fw = tf.concat(final_states[0], axis=1)
            final_bw = tf.concat(final_states[1], axis=1)
            guide_output = tf.concat([final_fw, final_bw], axis=1)
        return tf.concat([tourist_output, guide_output], axis=1)
        
        # dummy workaround
        tourist_hist = tourist_output
        guide_hist = guide_output
        if self.use_attention == "None":
            return tf.concat([tourist_hist, guide_hist], axis=1)
        elif self.use_attention == "role":
            with tf.variable_scope("role_attention"):
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
                lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
                lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=self.current_nl_len, dtype=tf.float32)
                final_fw = tf.concat(final_states[0], axis=1)
                final_bw = tf.concat(final_states[1], axis=1)
                cur_rnn_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
                outputs = tf.concat([tourist_hist, guide_hist, cur_rnn_outputs], axis=1)
                self.attention = attention = tf.nn.softmax(tf.layers.dense(inputs=outputs, units=2, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer))
                role_attention = tf.add(tf.multiply(tourist_hist, tf.expand_dims(tf.unstack(attention, axis=1)[0], axis=1)), tf.multiply(guide_hist, tf.expand_dims(tf.unstack(attention, axis=1)[1], axis=1)))
                return role_attention       

        elif self.use_attention == "sentence":
            with tf.variable_scope("sentence_attention"):
                inputs = tf.nn.embedding_lookup(self.embedding_matrix, self.current_nl) # [batch_size, self.max_seq_len, self.embedding_dim]
                lstm_fw_cell = rnn.BasicLSTMCell(self.hidden_size)
                lstm_bw_cell = rnn.BasicLSTMCell(self.hidden_size)
                _, final_states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, inputs, sequence_length=self.current_nl_len, dtype=tf.float32)
                final_fw = tf.concat(final_states[0], axis=1)
                final_bw = tf.concat(final_states[1], axis=1)
                cur_rnn_outputs = tf.concat([final_fw, final_bw], axis=1) # concatenate forward and backward final states
                # concat current output with cnn_hist
                output_tourist = tf.concat([tourist_hist, cur_rnn_outputs], axis=1)
                weight_tourist = tf.unstack(tf.nn.softmax(tf.layers.dense(inputs=output_tourist, units=self.hist_len, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)), axis=1)
                output_guide = tf.concat([guide_hist, cur_rnn_outputs], axis=1)
                weight_guide = tf.unstack(tf.nn.softmax(tf.layers.dense(inputs=output_guide, units=self.hist_len, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)), axis=1)
                self.attention = weight_tourist
                tourist_attention = list()
                guide_attention = list()
                for i in range(3):
                    tourist_attention.append(tf.multiply(self.unstack_tourist_hist[i], tf.expand_dims(weight_tourist[i], axis=1)))
                    guide_attention.append(tf.multiply(self.unstack_guide_hist[i], tf.expand_dims(weight_guide[i], axis=1)))
                tourist_attention = tf.add_n(tourist_attention)
                guide_attention = tf.add_n(guide_attention)
                # change to rnn
                #rnn_tourist_attention = self.attention_biRNN("tourist_attention", tf.stack(tourist_attention, axis=1))
                #rnn_guide_attention = self.attention_biRNN("guide_attention", tf.stack(guide_attention, axis=1))
                sentence_attention = tf.concat([tourist_attention, guide_attention], axis=1)
                return sentence_attention

    def build_graph(self):
        concat_output = self.attention()
        history_summary = tf.layers.dense(inputs=concat_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)
        final_output = self.nl_biRNN(history_summary)
        self.intent_output = tf.layers.dense(inputs=final_output, units=self.intent_dim, kernel_initializer=tf.random_normal_initializer, bias_initializer=tf.random_normal_initializer)

    def add_loss(self):
        loss_ce = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.labels, logits=self.intent_output))
        loss_intent_tourist = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.split(self.history_intent, num_or_size_splits=2, axis=1)[0], logits=tf.stack(self.unstack_hist_tourist, axis=1)))
        loss_intent_guide = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.split(self.history_intent, num_or_size_splits=2, axis=1)[1], logits=tf.stack(self.unstack_hist_guide, axis=1)))
        if self.use_mid_loss == True:
            self.loss = loss_ce + loss_intent_tourist + loss_intent_guide
        else:
            self.loss = loss_ce
        
    def add_train_op(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
        self.train_op = optimizer.minimize(self.loss)
