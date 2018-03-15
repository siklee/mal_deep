import numpy as np
import os, sys
import csv

MAL_PATH = 'E:\\data\\time\\mal'
CSV_PATH='E:\\data\\time\\csv'

def collect_mal(mal_path_list,mal_path):
    mal_list = os.listdir(mal_path)
    print(mal_list)
    for x in mal_list:
        mal_path_list.append(os.path.join(mal_path, x))
    return mal_path_list

def check_overlap():
    date_list = os.listdir(MAL_PATH)
    date_list.sort()
    mal_list = list()
    for d_day in date_list:
        mal_path = os.path.join(MAL_PATH, d_day)  # mal_path C:\\data\\time\\mal\\20170829
        mal_list += os.listdir(mal_path)
        print(mal_list)
    print(mal_list.__len__())
    print(set(mal_list).__len__())

check_overlap()