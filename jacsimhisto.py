import pandas as pd
import os
import sys
from collections import Counter

CSVPATH = 'C:\\data\\time\\result2\\'


def find_similarity():
    csv_list = os.listdir(CSVPATH)
    csv_list.sort()
    jac_sim = list()
    for x in csv_list:
        csv1 = pd.read_csv(os.path.join(CSVPATH, x))
        csv1 = csv1[(csv1['predict'] == 1) & (csv1['answer'] == 0)]
        csv1list = csv1['ID'].values.tolist()
        jac_sim = jac_sim + csv1list
    df = pd.DataFrame.from_dict(Counter(jac_sim),orient='index')
    df.to_csv('jachist.csv')

find_similarity()
