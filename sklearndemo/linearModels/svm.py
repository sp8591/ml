from sklearn import svm

def svmcdemo():
    x = [[0, 0], [1, 1], [2, 3]]
    y = [0, 1, 2]
    clf = svm.SVC()
    clf.fit(x, y)
    print clf.predict([[2, 2], [2, 2]])
    print clf.support_vectors_
    print clf.n_support_


def svmrdemo():
    x = [[0, 0], [1, 1], [2, 3]]
    y = [0, 1, 2]
    clf = svm.SVR()
    clf.fit(x, y)
    print clf.predict([[2, 2], [2, 2]])
    print clf.support_vectors_
    print clf.n_support_


if __name__ == '__main__':
    #svmcdemo()
    svmrdemo()