import sklearn
import readfh
from sklearn.neighbors import KNeighborsClassifier

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

X_train,y_train = readfh.load_data2(BENIGN_PATH,MAL_PATH)
X_test,y_test = readfh.load_data2(BENIGN_PATH2,MAL_PATH2)

clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train,y_train)

pre_data = clf.predict(X_test)
count_true = 0
count_false = 0
for x in range(len(pre_data)):
    if pre_data[x] == y_test[x]:
        count_true+=1
    else:
        count_false+=1

print("결과 :","참: ",count_true,"거짓: ",count_false,"\n")
print("정확도: ")
print("%.2f" %(count_true/(count_true+count_false)))