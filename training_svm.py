from sklearn.svm import SVC
import readfh

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

X_train,y_train = readfh.load_data2(BENIGN_PATH,MAL_PATH)
X_test,y_test = readfh.load_data2(BENIGN_PATH2,MAL_PATH2)

svc = SVC()
svc.fit(X_train,y_train)

print("훈련 세트 정확도: {:.3f}".format(svc.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(svc.score(X_test, y_test)))