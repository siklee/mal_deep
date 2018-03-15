import pandas as pd
import os

CSVPATH = 'E:\\data\\time\\csv\\'

csv_list = os.listdir(CSVPATH)
csv_list.sort()

result_csv = list()

for x in csv_list:
    ha = pd.read_csv(os.path.join(CSVPATH, x))
    name = x
    tp=ha[(ha['predict'] == 1) & (ha['answer'] == 1)].__len__()
    fp=ha[(ha['predict'] == 1) & (ha['answer'] == 0)].__len__()
    fn=ha[(ha['predict'] == 0) & (ha['answer'] == 1)].__len__()
    tn=ha[(ha['predict'] == 0) & (ha['answer'] == 0)].__len__()

    accuracy = (tp+tn)/(tp+tn+fp+fn)
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    total_num= tp+tn+fp+fn

    result_csv.append([name,accuracy,precision,recall])
    #result_csv.append([name,fp,fn,total_num])

df = pd.DataFrame(result_csv,columns=['name','accuracy','precision','recall'])
#df = pd.DataFrame(result_csv,columns=['name','fp','fn','total_num'])
df.to_csv('seek_result.csv',index=False)