import pandas as pd
import os
import sys

CSVPATH = 'C:\\data\\time\\result2\\'


def find_similarity():
    csv_list = os.listdir(CSVPATH)
    csv_list.sort()
    jac_sim = list()
    for x in csv_list:
        csv1id = list()
        csv1 = pd.read_csv(os.path.join(CSVPATH, x))
        csv1 = csv1[(csv1['predict'] == 0) & (csv1['answer'] == 1)]
        csv1list = csv1['ID'].values.tolist()

        for y in csv_list:
            csv2id = list()
            csv2 = pd.read_csv(os.path.join(CSVPATH, y))
            csv2 = csv2[(csv2['predict'] == 0) & (csv2['answer'] == 1)]
            csv2list = csv2['ID'].values.tolist()

            union = list()
            union = csv1list+csv2list
            jacnum = ((len(union) - set(union).__len__()) / set(union).__len__())*100
            jac_sim.append([x,y,len(csv1list),len(csv2list),len(union),set(union).__len__(), jacnum])
    df = pd.DataFrame(jac_sim, columns=['name_csv1', 'name_csv2', 'num_csv1', 'num_csv2','num_sum', 'num_union','jac_similarity'])
    df.to_csv('jac.csv', index=False)



find_similarity()