import os, sys
import pickle

#MAL_PATH = "C:\\data\\fh_mal_train"
#BENIGN_PATH = "C:\\data\\fh_benign_train"

#ben_data.append(pickle.load(open('C:\\data\\fh_benign\\000abdf5d3e31514801b44b954e1cf91.fh', 'rb')))
#print(ben_data)

def load_data(benign_path,mal_path):
    benign_data = list()
    mal_data = list()
    mal_list = os.listdir(mal_path)
    benign_list = os.listdir(benign_path)
    for x in mal_list:
        mal_data.append(pickle.load(open(os.path.join(mal_path, x), 'rb')))
    for x in benign_list:
        benign_data.append(pickle.load(open(os.path.join(benign_path, x), 'rb')))

    return benign_data,mal_data

def load_data2(benign_path,mal_path):
    total_xdata = list()
    total_ydata =list()
    mal_list = os.listdir(mal_path)
    benign_list = os.listdir(benign_path)
    for x in mal_list:
        total_xdata.append(pickle.load(open(os.path.join(mal_path, x), 'rb')))
        total_ydata.append(1)
    for x in benign_list:
        total_xdata.append(pickle.load(open(os.path.join(benign_path, x), 'rb')))
        total_ydata.append(0)

    return total_xdata, total_ydata
