import pickle, csv
import tensorflow as tf
import numpy as np
import os, sys, time
import csv
import csv2list

# import confmatrix
from sklearn.metrics import confusion_matrix

# for loading file
DATA_PATH = '‪E:\\kglo\\chungnam_sample.csv'
DATA_PATH2 = 'C:\\Users\\seek\\Desktop\\haha.csv'
DATA_PATH3 = 'C:\\data\\chungnam_sample.csv'
DATA_PATH4 = 'C:\\data\\chungnam_sample2.csv'

# for using tensorflow as hyper parameter
INPUT_SIZE = int(4096)
OUTPUT_SIZE = int(2)
MODEL_STORAGE = 'KISA_MODEL_ANN'


def load_data(can_save_data=False):
    print('Data reading starts')
    with open(FILE_NAME_LISTS_PIC, 'rb') as f_name:
        file_name_lists = pickle.load(f_name)

    if (can_save_data == True):
        file_data_lists = list()
        for file in file_name_lists:
            try:
                file_data_lists.append(pickle.load(open(os.path.join(BASE_PATH, file+'.fh'), 'rb')))
            except:
                pass
        print('Data reading finished')
        return file_data_lists
    else:
        print('Data reading finished')
        return file_name_lists
    pass


#  KISnet
if __name__ == '__main__':
    start_time = time.time()
    # read pickle files
    can_saved_data = False
    total_file_lists = load_data(can_saved_data)
    number_of_data = total_file_lists.__len__()

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

    tf_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    tf_config.gpu_options.allow_growth = True
    # 실행 과정에서 요구되는 만큼의 GPU memory만 할당
    with tf.Session(config=tf_config) as sess:
        sess.run(init)
        model_saver.restore(sess, os.path.normpath(MODEL_STORAGE + '\\model.ckpt'))

        predicted_labels = list()
        err_cnt = 0
        print('total: {}'.format(number_of_data))
        print('--testing start--')
        for i in range(number_of_data):
            try:
                if (can_saved_data == False):
                    predicted_label = sess.run(y_test,
                                               feed_dict={x: [pickle.load(
                                                   open(os.path.join(BASE_PATH, total_file_lists[i]+'.fh'), 'rb'))]})
                else:
                    predicted_label = sess.run(y_test, feed_dict={x: [total_file_lists[i]]})
                predicted_labels.append(np.array(predicted_label).reshape([-1]).argmax(-1))
                if (i % 100 == 0):
                    print(('{cnt}'.format(cnt=i)))
            except Exception as e:
                # print('error in {i}th: {err}'.format(i=i, err=e))
                err_cnt += 1
                predicted_labels.append(1)  # if no files, just think of malware
                pass
        print('------finish------')
        print('error count: {}'.format(err_cnt))
    end_time = time.time()
    print('total time: {}s'.format(int(end_time - start_time)))

    # save result as csv file
    print('save the result as csv file')
    with open('result_malware_jeongseongmin3.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'class'])
        for i in range(number_of_data):
            writer.writerow([os.path.splitext(total_file_lists[i])[0], predicted_labels[i]])
        pass
    pass