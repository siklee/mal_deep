import pickle
import tensorflow as tf
import numpy as np
import os, sys
import csv

# for loading pickle data
BENIGN_NAME_LISTS_PIC = 'C:/Users/seongmin/Desktop/KISA/fh_benign_train_names.pickle'
MALWARE_NAME_LISTS_PIC = 'C:/Users/seongmin/Desktop/KISA/fh_malware_train_names.pickle'
BASE_PATH = os.path.normpath('C:/Users/seongmin/Desktop/KISA/train')
# for using tensorflow as hyper parameter
INPUT_SIZE = int(779)
OUTPUT_SIZE = int(2)
LEARNING_RATE = 1e-4
BATCH_SIZE = int(256)
EPOCH = int(20000)
DROPOUT_PROB = 0.6
MODEL_STORAGE = 'C:\\data\\training_ANN'

DATA_PATHX = '‪E:\\kglo\\chungnam_sample.csv'
DATA_PATH = 'C:\\data\\180117_Wins_malware_feature(30000).csv'



def load_data():
    print('Data reading starts')
    with open(BENIGN_NAME_LISTS_PIC, 'rb') as f_name:
        ben_name_lists = pickle.load(f_name)
    with open(MALWARE_NAME_LISTS_PIC, 'rb') as f_name:
        mal_name_lists = pickle.load(f_name)

    print(len(ben_name_lists), len(mal_name_lists))

    ben_data_lists, mal_data_lists = list(), list()
    base_path = os.path.normpath(BASE_PATH)
    for ben_file in ben_name_lists:
        try:
            with open(os.path.join(base_path, ben_file+'.fh'), 'rb') as f :
                ben_data_lists.append(pickle.load(f))
        except:
            pass

    for mal_file in mal_name_lists:
        try:
            with open(os.path.join(base_path, mal_file+'.fh'), 'rb') as f :
                mal_data_lists.append(pickle.load(f))
        except:
            pass
    print('Data reading finished')
    return ben_data_lists, mal_data_lists


def make_list():
    list_data = []  # x value
    feature_name = []  # x name
    hash_name = []  # malware name
    label_value = []  # y value

    f = open(DATA_PATH, 'r')
    rdr = csv.reader(f,delimiter=',')
    for one in rdr:
        feature_name.append(one)
        break;
    for line in rdr:
        hash_name.append(line[0])
        label_value.append(int(line[-1]))
        tmp = line[1:]
        feature_value = []
        for tmp_item in tmp:
            feature_value.append(float(tmp_item))
        list_data.append(feature_value)
    f.close()
    return list_data, list_data


def init():
    # create model storage
    if not tf.gfile.Exists(MODEL_STORAGE):
        tf.gfile.MkDir(MODEL_STORAGE)
    # load data
    return make_list()


def get_mini_batch(ben_data_lists, mal_data_lists, batch_size):
    while True:
        np.random.shuffle(ben_data_lists)
        np.random.shuffle(mal_data_lists)

        half_batch_size = int(batch_size/2)
        batch_data_lists, batch_label_lists = list(), list()

        batch_data_lists += ben_data_lists[:half_batch_size]
        for i in range(half_batch_size):
            batch_label_lists.append([1, 0]) # benign

        batch_data_lists += mal_data_lists[:half_batch_size]
        for i in range(half_batch_size):
            batch_label_lists.append([1, 0])

        yield (batch_data_lists, batch_label_lists)


#  KISnet
if __name__ == '__main__':
    ben_data_lists, mal_data_lists = init()

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
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    train_iter = get_mini_batch(ben_data_lists, mal_data_lists, BATCH_SIZE)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)
        print('learning start')
        for i in range(EPOCH):
            (training_data, training_label) = next(train_iter)
            sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB})
            if (i % 100 == 0):
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB}))
                if (i % 1000 == 0):
                    model_saver.save(sess, os.path.normpath(MODEL_STORAGE + '\\model.ckpt'))
        print('------finish------')
        model_saver.save(sess, os.path.normpath(MODEL_STORAGE + '\\model.ckpt'))
    pass
