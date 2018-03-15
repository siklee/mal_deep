import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import csv

FILE_PATH = "C:\\data\\fh_mal_train"
GRAPH_PATH = "C:\\code\\count_mal.pickle"

def get_size(start_path = '.'):
    total_size = 0
    total_count =0
    for dirpath, dirnames, filenames in os.walk(start_path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
            total_count +=1
    return total_size,total_count


def haha(start_path):
    cal = list()
    for x in  os.listdir(start_path):
        start_path2 = os.path.join(start_path, x)
        if os.path.isdir(start_path2):
            year = x
            for x2 in os.listdir(start_path2):
                start_path3 = os.path.join(start_path2, x2)
                if os.path.isdir(start_path3):
                    month = x2
                    for x3 in os.listdir(start_path3):
                        start_path4 = os.path.join(start_path3, x3)
                        if os.path.isdir(start_path4):
                            day = x3
                            start_path6 = os.path.join(start_path4, 'total')
                            if os.path.isdir(start_path6):
                                haha= list(year,month,day,get_size(start_path6))
                                cal.append(haha)
    return cal


def draw_graph(gpath):
    f = open('count_mal.pickle','rb')
    dirdata = pickle.load(f)
    x=list()
    y=list()
    for t in dirdata:
        x.append(t[2])
        y.append(t[3][0])
    print(dirdata)
    plt.xlabel("day")
    plt.ylabel("data size")
    plt.plot(y)
    plt.show()

def writecsv(gpath):
    f = open('haha.csv','w',encoding='utf-8',newline='')
    wr = csv.writer(f)
    wr.writerow(['year','month','date','size','count'])
    f2 = open('count_mal.pickle', 'rb')
    dirdata = pickle.load(f2)
    dirdata = sorted(dirdata)
    for t in dirdata:
        t[2]=t[2][4:]
        wr.writerow([t[0],t[1],t[2],t[3][0],t[3][1]])

#print(haha(GRAPH_PATH))
#draw_graph(GRAPH_PATH)
writecsv(GRAPH_PATH)

