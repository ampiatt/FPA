import tensorflow as tf
import math
import time
import os
import csv
import random
import datetime



class Model:
    stock_hist = 10
    vocabulary_size = 1000
    embedding_size = 10
    batch_size = 5
    num_hidden = 5
    num_classes = 2
    padded_length = 50
    filter_sizes = [3, 4, 5]
    num_filters = 128
    hidden_count = 384
    epoch_count = 1000

    def __init__(self):

        # text processing
        self.news_input = tf.placeholder(tf.int32, [None, self.padded_length])
        self.stock_input = tf.placeholder(tf.float32, [None, self.stock_hist, 7])
        self.price_output = tf.placeholder(tf.float32, [None, 1])
        self.dropout_keep_prob = tf.placeholder(tf.float32)
        self.learning_rate = tf.placeholder(tf.float32, None)

        self.W = tf.Variable(tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
        self.embedded_chars = tf.nn.embedding_lookup(self.W, self.news_input)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        self.pooled_outputs = []
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                self.filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                self.W = tf.Variable(tf.truncated_normal(self.filter_shape, stddev=0.1))
                self.b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                self.conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    self.W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                self.h = tf.nn.relu(tf.nn.bias_add(self.conv, self.b), name="relu")
                self.pooled = tf.nn.max_pool(
                    self.h,
                    ksize=[1, self.padded_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                self.pooled_outputs.append(self.pooled)
 
        # combine pooled features
        self.num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(self.pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])

        # historical stock processing
        self.cell = tf.nn.rnn_cell.LSTMCell(self.hidden_count, state_is_tuple=True)
        self.val, self.state = tf.nn.dynamic_rnn(self.cell, self.stock_input, dtype=tf.float32)
        self.val = tf.transpose(self.val, [1, 0, 2])
        self.last = tf.gather(self.val, int(self.val.get_shape()[0]) - 1)

        # convolutional combination
        self.concat = tf.concat([self.last, self.h_pool_flat], 1)
        self.weight = tf.Variable(tf.Variable(tf.truncated_normal([self.hidden_count * 2, 1])))
        self.bias = tf.Variable(tf.constant(0.1, shape=[1]))
        self.prediction = tf.matmul(self.concat, self.weight) + self.bias

        # loss and optimization
        self.loss = tf.reduce_mean(tf.square(self.prediction - self.price_output))
        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate)
        self.minimize = self.optimizer.minimize(self.loss)

    def get_date(self, date):
        return int((datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime(1970, 1, 1)).total_seconds())

    def clean(self, items):
        for item in items:
            if item == '&amp;' or item == '-':
                continue
            if item.startswith('b\'') or item.startswith('b"'):
                item = item[2:]
            if item.startswith('\'') or item.startswith('"'):
                item = item[1:]
            if item.endswith('\'') or item.endswith('"'):
                item = item[:-1]
            if item.endswith(',') or item.endswith('.'):
                item = item[:-1]
            yield item

    def process_sentences(self, data):
        # count words
        counts = {}
        for sentence in data.values():
            for word in sentence['news_input']:
                if not word in counts:
                    counts[word] = 0
                counts[word] += 1
        def get_highest():
            high_word = ''
            high_val = -1
            for word, val in counts.items():
                if val > high_val:
                    high_val = val
                    high_word = word
            del counts[high_word]
            return high_word
        # order words
        order = [get_highest() for i in range(self.vocabulary_size)]
        # return processed words
        for key, value in data.items():
            output = []
            for word in value['news_input']:
                if word in order:
                    val = order.index(word)
                else:
                    val = self.vocabulary_size - 1
                output.append(val)
            while len(output) < self.padded_length:
                output.append(self.vocabulary_size - 1)
            data[key]['news_input'] = output[:self.padded_length]

    def get_news_data(self):
        out = {}
        with open('Combined_News_DJIA.csv') as file:
            reader = csv.reader(file)
            i = 0
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                items = [item for item in row] 
                date = self.get_date(items[0])
                sentences = []
                for sentence in items[2:]:
                    sentences.extend([word for word in self.clean(sentence.split())])
                    out[date] = {'news_input': sentences}
        return out

    def get_stock_data(self, data):
        values = []
        indicies = {}
        with open('DJIA_table.csv') as file:
            reader = csv.reader(file)
            i = 0
            prev = []
            last_date = 0
            last_values = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            for row in reader:
                if i == 0:
                    i += 1
                    continue
                date = self.get_date(row[0])
                indicies[self.get_date(row[0])] = len(values)
                diff = last_date - date
                values.append([diff/86400, *[float(i) for i in row[1:]]])
                last_date = date
            to_remove = []
            for key in data:
                if key not in indicies:
                    to_remove.append(key)
                    continue
                idx = indicies[key]
                if idx <= self.stock_hist:
                    to_remove.append(key)
                    continue
                data[key]['output'] = [values[idx][-1]]
                data[key]['stock_input'] = values[idx-self.stock_hist:idx]
            for key in to_remove:
                del data[key]

    def get_data(self, data, keys):
        news = [data[key]['news_input'] for key in keys]
        stocks = [data[key]['stock_input'] for key in keys]
        out = [data[key]['output'] for key in keys]
        return news, stocks, out

    def get_learning_rate(self, epoch):
        return 0.99 ** float(epoch + 1)

    def train(self):
        # load the data
        inputs = self.get_news_data()
        self.process_sentences(inputs)
        self.get_stock_data(inputs)
        selection = random.sample(inputs.keys(), 1500)
        # open the session
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            # run the training
            news, stocks, out = self.get_data(inputs, selection)
            for i in range(self.epoch_count):
                _, error = session.run([self.minimize, self.loss], {self.news_input: news, self.stock_input: stocks, self.price_output: out, self.learning_rate: self.get_learning_rate(i)})
                print('Epoch: {}, error: {}'.format(i, error))
            # predict everything
            everything = [key for key in inputs.keys()]
            news, stocks, out = self.get_data(inputs, everything)
            prediction = session.run(self.prediction, {self.news_input: news, self.stock_input: stocks})
        return {everything[i]: prediction[i][0] for i in range(len(everything))}

    def predict(self, inputs):
        with tf.Session() as session:
            selection = random.sample(inputs.keys(), 10)
            news, stocks, out = self.get_data(inputs, selection)
            return session.run(prediction, {news_input: inputs['news_input'], stock_input: inputs['stock_input']})

