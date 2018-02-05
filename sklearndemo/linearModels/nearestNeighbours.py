from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
import numpy as np
def nearestN():
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(x)
    distances, indices = nbrs.kneighbors(x)
    print distances, indices

def nearestCen():
    x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    y = np.array([1, 1, 1, 2, 2, 2])
    clf = NearestCentroid()
    clf.fit(x, y)
    print(clf.predict([[-0.8, -1]]))


if __name__ == '__main__':
    #nearestN()
    nearestCen()
