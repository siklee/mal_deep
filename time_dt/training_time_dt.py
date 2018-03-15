import tensorflow as tf
import numpy as np
import os, sys
import csv
import pickle
import time
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import readfh
import graphviz

MAL_PATH = 'C:\\data\\time\\mal'  # 피쳐해쉬 폴더목록
BENIGN_PATH = 'C:\\data\\time\\benign_train'
BASE_PATH = 'C:\\data\\time\\model_d\\'
START_DAY = 20170727
END_DAY = 20170830
MONITER_SIZE = 2


def collect_mal(mal_path_list,mal_path):
    mal_list = os.listdir(mal_path)
    for x in mal_list:
        mal_path_list.append(os.path.join(mal_path, x))
    print(mal_path_list[0])
    return mal_path_list


def collect_benign(benign_path_list,benign_path):
    benign_list = os.listdir(benign_path)
    for x in benign_list:
        benign_path_list.append(os.path.join(benign_path, x))
    return benign_path_list


def make_path(start_day,end_day):     # start_day end_day : int
    date_list = os.listdir(MAL_PATH)
    date_list.sort()
    benign_list = list()
    benign_list = collect_benign(benign_list, BENIGN_PATH)
    benign_data = list()
    for x in benign_list:
        benign_data.append(pickle.load(open(x, 'rb')))
    for d_day in date_list:
        d_day_int = int(d_day)
        mal_list = list()
        xtrain = list()
        ytrain = list()
        if d_day_int >= start_day and d_day_int <= end_day:
            if date_list.index(d_day)+MONITER_SIZE >= date_list.__len__():
                print("THE END")
                break;
            for dd_day in date_list[date_list.index(d_day):date_list.index(d_day)+MONITER_SIZE]:
                mal_path = os.path.join(MAL_PATH, dd_day)  # mal_path C:\\data\\time\\mal\\20170829
                mal_list = collect_mal(mal_list, mal_path)  # mal_list C:\\data\\time\\mal\\20170829\\ff4a3b697310d126e46ebae462d712a1.fh
                print(dd_day)
            for x in mal_list:
                xtrain.append(pickle.load(open(x, 'rb')))
                ytrain.append(1)
            for x in benign_data:
                xtrain.append(x)
                ytrain.append(0)
            tree = DecisionTreeClassifier(random_state=0, max_depth=100, max_leaf_nodes=500)
            tree.fit(xtrain, ytrain)
            save_model_name = BASE_PATH + str(date_list[date_list.index(d_day)]) + "." + str(date_list[date_list.index(d_day) + MONITER_SIZE])
            pickle.dump(tree,open(save_model_name,'wb'))

make_path(START_DAY,END_DAY)