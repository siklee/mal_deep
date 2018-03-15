import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import readfh

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

X_train,y_train = readfh.load_data2(BENIGN_PATH,MAL_PATH)
X_test,y_test = readfh.load_data2(BENIGN_PATH2,MAL_PATH2)

model = TSNE(learning_rate=100,n_components=2)
pre_data = model.fit_transform(X_train)

malx=list()
maly=list()
benignx=list()
benigny=list()

print(pre_data[0])
for x in range(len(y_train)):
    if y_train[x] == 1:
        malx.append(pre_data[x][0])
        maly.append(pre_data[x][1])
    else:
        benignx.append(pre_data[x][0])
        benigny.append(pre_data[x][1])

plt.scatter(benignx,benigny,c='r')
plt.scatter(malx,maly,c='b')

plt.show()
