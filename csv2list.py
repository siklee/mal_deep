import csv


DATA_PATH = '‪E:\\kglo\\chungnam_sample.csv'
DATA_PATH2 = 'C:\\Users\\seek\\Desktop\\haha.csv'
DATA_PATH3 = 'C:\\data\\chungnam_sample.csv'
DATA_PATH4 = 'C:\\data\\chungnam_sample2.csv'


list_data = []      # x value
feature_name = []   # x name
hash_name=[]        #  malware name
label_value = []   # y value

def make_list():
    f = open(DATA_PATH4, 'r')
    rdr = csv.reader(f,delimiter=',')
    for one in rdr:
        feature_name.append(one)
        break;
    for line in rdr:
        hash_name.append(line[0])
        label_value.append(int(line[-1]))
        tmp = line[1:-2]
        feature_value = []
        for x in tmp:
            feature_value.append(float(x))
        list_data.append(feature_value)
    f.close()

make_list()
print(hash_name)
print(feature_name)
print(list_data)
print(label_value)