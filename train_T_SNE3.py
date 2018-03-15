'''
==============
3D scatterplot
==============

Demonstration of a basic scatterplot in 3D.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import readfh

MAL_PATH = "C:\\data\\fh_mal_train"
BENIGN_PATH = "C:\\data\\fh_benign_train"
MAL_PATH2 = "C:\\data\\fh_mal_test"
BENIGN_PATH2 = "C:\\data\\fh_benign_test"

X_train,y_train = readfh.load_data2(BENIGN_PATH,MAL_PATH)
X_test,y_test = readfh.load_data2(BENIGN_PATH2,MAL_PATH2)

model = TSNE(learning_rate=100,n_components=3)
pre_data = model.fit_transform(X_train)

def randrange(n, vmin, vmax):
    '''
    Helper function to make an array of random numbers having shape (n, )
    with each number distributed Uniform(vmin, vmax).
    '''
    return (vmax - vmin)*n + vmin

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



# For each set of style and range settings, plot n random points in the box
# defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
for x in range(len(y_train)):
    if y_train[x] == 1:
        xs = randrange(pre_data[x][0], -1500000, 1500000)
        ys = randrange(pre_data[x][1], -1500000, 1500000)
        zs = randrange(pre_data[x][2], -1500000, 1500000)
        ax.scatter(xs, ys, zs, c='r', marker='o')
    else:
        xs = randrange(pre_data[x][0], -1500000, 1500000)
        ys = randrange(pre_data[x][1], -1500000, 1500000)
        zs = randrange(pre_data[x][2], -1500000, 1500000)
        ax.scatter(xs, ys, zs, c='b', marker='^')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
