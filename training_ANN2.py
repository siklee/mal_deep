import pickle
import tensorflow as tf
import numpy as np
import os, sys
import csv
import readfh

# for loading pickle data
BENIGN_NAME_LISTS_PIC = 'C:/Users/seongmin/Desktop/KISA/fh_benign_train_names.pickle'
MALWARE_NAME_LISTS_PIC = 'C:/Users/seongmin/Desktop/KISA/fh_malware_train_names.pickle'
BASE_PATH = os.path.normpath('C:/Users/seongmin/Desktop/KISA/train')

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

# for using tensorflow as hyper parameter
INPUT_SIZE = int(12288)
OUTPUT_SIZE = int(2)
LEARNING_RATE = 1e-4
BATCH_SIZE = int(256)
EPOCH = int(20000)
DROPOUT_PROB = 0.6
MODEL_STORAGE = 'C:\\data\\training_ANN'



def get_mini_batch(benign_path,mal_path, batch_size):
    total_benign_data = list()
    total_mal_data = list()
    mal_list = os.listdir(mal_path)
    benign_list = os.listdir(benign_path)
    for x in mal_list:
        total_mal_data.append(pickle.load(open(os.path.join(mal_path, x), 'rb')))
    for x in benign_list:
        total_benign_data.append(pickle.load(open(os.path.join(benign_path, x), 'rb')))

    while True:
        np.random.shuffle(total_benign_data)
        np.random.shuffle(total_mal_data)

        half_batch_size = int(batch_size/2)
        batch_data_lists, batch_label_lists = list(), list()

        batch_data_lists += total_benign_data[:half_batch_size]
        for i in range(half_batch_size):
            batch_label_lists.append([1, 0]) # benign

        batch_data_lists += total_mal_data[:half_batch_size]
        for i in range(half_batch_size):
            batch_label_lists.append([0, 1])

        yield (batch_data_lists, batch_label_lists)


#  KISnet
if __name__ == '__main__':

    with tf.device('/gpu:0'):
        # ANN network architecture
        prob = tf.placeholder(tf.float32)

        x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        dense_layer_1 = tf.layers.dense(inputs=x, units=2048, activation=tf.nn.relu)
        dense_drop_1 = tf.nn.dropout(dense_layer_1, prob)

        dense_layer_2 = tf.layers.dense(inputs=dense_drop_1, units=1024, activation=tf.nn.relu)
        dense_drop_2 = tf.nn.dropout(dense_layer_2, prob)

        dense_layer_3 = tf.layers.dense(inputs=dense_drop_2, units=512, activation=tf.nn.relu)
        dense_drop_3 = tf.nn.dropout(dense_layer_3, prob)

        dense_layer_4 = tf.layers.dense(inputs=dense_drop_3, units=256, activation=tf.nn.relu)
        dense_drop_4 = tf.nn.dropout(dense_layer_4, prob)

        dense_layer_5 = tf.layers.dense(inputs=dense_drop_4, units=128, activation=tf.nn.relu)
        dense_drop_5 = tf.nn.dropout(dense_layer_5, prob)

        y_ = tf.layers.dense(inputs=dense_drop_5, units=OUTPUT_SIZE)

        # loss function: softmax, cross-entropy
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))

        # optimizer: Adaptive momentum optimizer
        optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cost)

    # predict
    prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    # training session start
    init = tf.global_variables_initializer()
    train_iter = get_mini_batch(BENIGN_PATH, MAL_PATH, BATCH_SIZE)
    test_iter = get_mini_batch(BENIGN_PATH2, MAL_PATH2, BATCH_SIZE)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        print('learning start')
        for i in range(EPOCH):
            (training_data, training_label) = next(train_iter)
            sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB})
            if (i % 100 == 0):
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB}))
        print('------finish------')
        predicted_labels = list()
        for i in range(3000):
            (test_data, test_label) = next(test_iter)
            print("test accuracy %g" % sess.run(accuracy, feed_dict={x: test_data, y: test_label, prob: 1.0}))
    pass
