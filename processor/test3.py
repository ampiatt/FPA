import tensorflow as tf
import math
import time
import os
import csv
import random
import datetime

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
epoch_count = 500

def get_date(date):
    return int((datetime.datetime.strptime(date, '%Y-%m-%d') - datetime.datetime(1970, 1, 1)).total_seconds())

def clean(items):
    for item in items:
        if item == '&amp;':
            continue
        if item.startswith('b\'') or item.startswith('b"'):
            item = item[2:]
        if item.startswith('\'') or item.startswith('"'):
            item = item[1:]
        if item.endswith('\'') or item.endswith('"'):
            item = item[:-1]
        yield item

def process_sentences(data):
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
    order = [get_highest() for i in range(vocabulary_size)]
    # return processed words
    for key, value in data.items():
        output = []
        for word in value['news_input']:
            if word in order:
                val = order.index(word)
            else:
                val = vocabulary_size - 1
            output.append(val)
        while len(output) < padded_length:
            output.append(vocabulary_size - 1)
        data[key]['news_input'] = output[:padded_length]

def get_news_data():
    out = {}
    with open('Combined_News_DJIA.csv') as file:
        reader = csv.reader(file)
        i = 0
        for row in reader:
            if i == 0:
                i += 1
                continue
            items = [item for item in row] 
            date = get_date(items[0])
            sentences = []
            for sentence in items[2:]:
                sentences.extend([word for word in clean(sentence.split())])
            out[date] = {'news_input': sentences}
    return out

def get_stock_data(data):
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
            date = get_date(row[0])
            indicies[get_date(row[0])] = len(values)
            diff = last_date - date
            values.append([diff/86400, *[float(i) for i in row[1:]]])
            last_date = date
    to_remove = []
    for key in data:
        if key not in indicies:
            to_remove.append(key)
            continue
        idx = indicies[key]
        if idx <= stock_hist:
            to_remove.append(key)
            continue
        data[key]['output'] = [values[idx][-1]]
        data[key]['stock_input'] = values[idx-stock_hist:idx]
    for key in to_remove:
        del data[key]

inputs = get_news_data()
process_sentences(inputs)
get_stock_data(inputs)
selection = random.sample(inputs.keys(), 1000)

# text processing
news_input = tf.placeholder(tf.int32, [None, padded_length])
stock_input = tf.placeholder(tf.float32, [None, stock_hist, 7])
price_output = tf.placeholder(tf.float32, [None, 1])
dropout_keep_prob = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32, None)

with tf.device('/cpu:0'), tf.name_scope("embedding"):
    W = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0), name="W")
    embedded_chars = tf.nn.embedding_lookup(W, news_input)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    with tf.name_scope("conv-maxpool-%s" % filter_size):
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(
            embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
        pooled = tf.nn.max_pool(
            h,
            ksize=[1, padded_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
        pooled_outputs.append(pooled)
 
# combine pooled features
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

# historical stock processing
cell = tf.nn.rnn_cell.LSTMCell(hidden_count, state_is_tuple=True)
val, state = tf.nn.dynamic_rnn(cell, stock_input, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

# convolutional combination
concat = tf.concat([last, h_pool_flat], 1)
weight = tf.Variable(tf.Variable(tf.truncated_normal([hidden_count * 2, 1])))
bias = tf.Variable(tf.constant(0.1, shape=[1]))
prediction = tf.matmul(concat, weight) + bias

# loss and optimization
loss = tf.reduce_mean(tf.square(prediction - price_output))
optimizer = tf.train.RMSPropOptimizer(learning_rate)
minimize = optimizer.minimize(loss)

def get_data(data, keys):
    news = [data[key]['news_input'] for key in keys]
    stocks = [data[key]['stock_input'] for key in keys]
    out = [data[key]['output'] for key in keys]
    return news, stocks, out

def get_learning_rate(epoch):
    return 0.99 ** float(epoch + 1)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())

    news, stocks, out = get_data(inputs, selection)
    for i in range(2000):
        _, error = session.run([minimize, loss], {news_input: news, stock_input: stocks, price_output: out, learning_rate: get_learning_rate(i)})
        print('Epoch: {}, error: {}'.format(i, error))

    selection = random.sample(inputs.keys(), 10)
    news, stocks, out = get_data(inputs, selection)
    print(session.run(prediction, {news_input: news, stock_input: stocks}))
    print(out)

