import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.datasets.samples_generator import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def draw():
    data = np.random.randint(0, 255, size=[40, 40, 40])
    xx, yy = make_classification(n_samples=1000, n_features=3, \
                               n_redundant=0, n_classes=3, \
                               n_informative=2, n_clusters_per_class=1, \
                               class_sep=0.5, random_state=10)
    x, y, z = data[0], data[1], data[2]
    print xx, yy
    print xx[:, 2]
    fig = plt.figure()
    ax = plt.subplot(221, projection='3d')
    ax2 = plt.subplot(223)
    ax3 = plt.subplot(224)
    #ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=30, azim=20)
    #ax.scatter(x[:10][0], y[:10][0], z[:10][0], c='y')

    #for pca
    pca = PCA(n_components=2)
    pca.fit(xx)
    xx_new2 = pca.transform(xx)
    print xx_new2

    #for lda
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(xx, yy)
    xx_new3 = lda.transform(xx)


    for i in range(1000):
        arr = xx[i]
        if yy[i] == 0:
            ax.scatter(arr[0], arr[1], arr[2], c='y')
            ax2.scatter(xx_new2[i][0], xx_new2[i][1], c='y')
            ax3.scatter(xx_new3[i][0], xx_new3[i][1], c='y')
        if yy[i] == 1:
            ax.scatter(arr[0], arr[1], arr[2], c='b')
            ax2.scatter(xx_new2[i][0], xx_new2[i][1], c='b')
            ax3.scatter(xx_new3[i][0], xx_new3[i][1], c='b')
        if yy[i] == 2:
            ax.scatter(arr[0], arr[1], arr[2], c='r')
            ax2.scatter(xx_new2[i][0], xx_new2[i][1], c='r')
            ax3.scatter(xx_new3[i][0], xx_new3[i][1], c='r')
    # ax.scatter(x[10:20], y[10:20], z[10:20], c='r')
    # ax.scatter(x[30:40], y[30:40], z[30:40], c='g')
    ax.set_zlabel('Z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

if __name__ == '__main__':
    #pca()
    draw()

