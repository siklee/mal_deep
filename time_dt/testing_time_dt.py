from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import os
import pickle
import csv

MODEL_PATH = 'C:\\data\\time\\model_d\\'
MAL_PATH = 'C:\\data\\time\\mal'  # 피쳐해쉬 폴더목록
BENIGN_PATH = 'C:\\data\\time\\benign_train'

def collect_mal(mal_path):
    mal_list = os.listdir(mal_path)
    return mal_list

def collect_benign(benign_path):
    benign_list = os.listdir(benign_path)
    return benign_list


def test_dt(modelname,mal_path):
    file = open(MODEL_PATH+modelname, 'rb')
    model_data = pickle.load(file)
    file.close()

    mal_list =list()
    mal_list = collect_mal(mal_path)
    num_mal_data = mal_list.__len__()

    benign_list = collect_benign(BENIGN_PATH)
    num_benign_data = benign_list.__len__()

    x_testbenign = list()
    y_testbenign = list()
    x_testmal = list()
    y_testmal = list()
    scorebenign = list()
    scoremal = list()

    for i in range(num_mal_data):
        x_testmal.append(pickle.load(open(os.path.join(mal_path, mal_list[i]), 'rb')))
        y_testmal.append(1)

    for i in range(num_benign_data):
        x_testbenign.append(pickle.load(open(os.path.join(BENIGN_PATH, benign_list[i]), 'rb')))
        y_testbenign.append(0)

    scorebenign = model_data.predict(x_testbenign)
    scoremal = model_data.predict(x_testmal)
    """
    precision = average_precision_score(y_test,score)
    recall = recall_score(y_test, score, average='binary')
    print("pricision :",precision)
    print("recall :", recall)
    """

    print('save the result as csv file')
    CSV_FILENAME = modelname
    print("CSV_FILENAME:", CSV_FILENAME)
    with open(CSV_FILENAME + '.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'predict', 'answer'])
        for i in range(num_mal_data):
            writer.writerow([mal_list[i], y_testmal[i], scoremal[i]])
        for i in range(num_benign_data):
            writer.writerow([benign_list[i], y_testbenign[i], scorebenign[i]])
        pass
    pass

def make_path():     # start_day end_day : int
    model_list = os.listdir(MODEL_PATH)
    mal_path = list()
    for x in model_list:
        tmp = x.split('.')[1]
        mal_path.append(os.path.join(MAL_PATH,tmp))
    for x in range(model_list.__len__()-1):
        print('model :',model_list[x])
        print('mal :',mal_path[x+1])
        test_dt(model_list[x],mal_path[x+1])

make_path()
