import tensorflow as tf
import numpy as np
import os, sys
import csv
import pickle

def trainANN(benign_data,mal_data,load_model_name,save_model_name):
    tf.gfile.MkDir(save_model_name)

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
    model_saver = tf.train.Saver()
    train_iter = get_mini_batch(benign_data, mal_data, BATCH_SIZE)

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
        sess.run(init)

        if(load_model_name != 'null'):
            print("check1")
            print(os.path.normpath(load_model_name + '\\model.ckpt'))
            model_saver.restore(sess, os.path.normpath(load_model_name + '\\model.ckpt'))

        print('learning start')
        for i in range(EPOCH):
            (training_data, training_label) = next(train_iter)
            sess.run(optimizer, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB})
            if (i % 100 == 0):
                print(i, sess.run(accuracy, feed_dict={x: training_data, y: training_label, prob: DROPOUT_PROB}))
        print('------finish------')
        print("check2")
        print(save_model_name)
        model_saver.save(sess, os.path.normpath(save_model_name + '\\model.ckpt'))
    pass