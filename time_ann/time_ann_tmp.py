import tensorflow as tf
import numpy as np
import os, sys
import csv
import pickle
import time

MAL_PATH = 'E:\\data\\tmp_test_mal'
BENIGN_PATH = 'E:\\data\\tmp_test_benign'
MODEL_PATH = 'E:\\data\\time\\window6\\20170824.20170830'
CSV_PATH='E:\\data\\time\\csv'
CSV_FILENAME = ''

# for using tensorflow as hyper parameter
INPUT_SIZE = int(12288)
OUTPUT_SIZE = int(2)
LEARNING_RATE = 1e-4


def collect_mal(mal_path):
    mal_list = os.listdir(mal_path)
    return mal_list


def collect_benign(benign_path):
    benign_list = os.listdir(benign_path)
    return benign_list


#  KISnet
def testANN(modelname,benign_path,mal_path):

    start_time = time.time()
    # read pickle files
    benign_list = collect_benign(benign_path)
    mal_list = collect_mal(mal_path)
    num_mal_data = mal_list.__len__()
    num_benign_data = benign_list.__len__()
    num_total_data=num_mal_data+num_benign_data
    tf.reset_default_graph()

    print("modelname:",modelname)
    print("mal_path:",mal_path)

    # CNN network architecture
    with tf.device('/gpu:0'):
        x = tf.placeholder(tf.float32, shape=[None, INPUT_SIZE])
        y = tf.placeholder(tf.float32, shape=[None, OUTPUT_SIZE])

        dense_layer_1 = tf.layers.dense(inputs=x, units=2048, activation=tf.nn.relu)

        dense_layer_2 = tf.layers.dense(inputs=dense_layer_1, units=1024, activation=tf.nn.relu)

        dense_layer_3 = tf.layers.dense(inputs=dense_layer_2, units=512, activation=tf.nn.relu)

        dense_layer_4 = tf.layers.dense(inputs=dense_layer_3, units=256, activation=tf.nn.relu)

        dense_layer_5 = tf.layers.dense(inputs=dense_layer_4, units=128, activation=tf.nn.relu)

        y_ = tf.layers.dense(inputs=dense_layer_5, units=OUTPUT_SIZE)
        y_test = tf.nn.softmax(y_)

    # testing session start
    model_saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    # 실행 과정에서 요구되는 만큼의 GPU memory만 할당
    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        model_saver.restore(sess, os.path.normpath(modelname + '\\model.ckpt'))

        mal_predicted_labels = list()
        benign_predicted_labels = list()
        err_cnt = 0
        print('data num :',num_total_data)
        print('--testing start--')

        for i in range(num_mal_data):
            try:
                haha = list()
                haha.append(pickle.load(open(os.path.join(mal_path, mal_list[i]), 'rb')))
                predicted_label = sess.run(y_test, feed_dict={x: haha})
                #print('pre ', np.array(predicted_label).reshape([-1]).argmax(-1), 'real ', '1')
                mal_predicted_labels.append([np.array(predicted_label).reshape([-1]).argmax(-1),1])

                
            except Exception as e:
                # print('error in {i}th: {err}'.format(i=i, err=e))
                err_cnt += 1
                print('err')
                mal_predicted_labels.append([1,1])  # if no files, just think of malware
                pass
                

        for i in range(num_benign_data):
            try:
                haha = list()
                haha.append(pickle.load(open(os.path.join(benign_path, benign_list[i]), 'rb')))
                predicted_label = sess.run(y_test, feed_dict={x: haha})
                #print('pre ', np.array(predicted_label).reshape([-1]).argmax(-1), 'real ', '0')
                benign_predicted_labels.append([np.array(predicted_label).reshape([-1]).argmax(-1), 0])

            except Exception as e:
                # print('error in {i}th: {err}'.format(i=i, err=e))
                err_cnt += 1
                print('err')
                benign_predicted_labels.append([0,0])  # if no files, just think of malware
                pass
        print('------finish------')
        print('error count: {}'.format(err_cnt))
    end_time = time.time()
    print('total time: {}s'.format(int(end_time - start_time)))


    # save result as csv file
    print('save the result as csv file')
    CSV_FILENAME = modelname
    print("CSV_FILENAME:",CSV_FILENAME)
    with open(CSV_FILENAME + '.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'predict','answer'])
        for i in range(num_mal_data):
            writer.writerow([mal_list[i], mal_predicted_labels[i][0],mal_predicted_labels[i][1]])
        for i in range(num_benign_data):
            writer.writerow([benign_list[i], benign_predicted_labels[i][0], benign_predicted_labels[i][1]])
        pass
    pass


def make_path():     # start_day end_day : int
    testANN(MODEL_PATH,BENIGN_PATH,MAL_PATH)




make_path()