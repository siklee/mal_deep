import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz
import readfh
import graphviz

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

X_train,y_train = readfh.load_data2(BENIGN_PATH,MAL_PATH)
X_test,y_test = readfh.load_data2(BENIGN_PATH2,MAL_PATH2)

tree = RandomForestClassifier(n_estimators=300,max_depth= 500)
tree.fit(X_train,y_train)

dot_graph = export_graphviz(tree,out_file="randomforest.dot",class_names=["malware","benign"],impurity=False,filled=True)

dot = graphviz.Source(dot_graph)
dot.format = 'png'
dot.render(filename='randomforest.png')

haha = tree.predict(X_test)
for x in range(len(haha)):
    print("pre :",haha[x],"ans :",y_test[x])

print("훈련 세트 정확도: {:.3f}".format(tree.score(X_train, y_train)))
print("테스트 세트 정확도: {:.3f}".format(tree.score(X_test, y_test)))