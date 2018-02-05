import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
def split():
    iris = datasets.load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size = 0.4, random_state = 0)
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
    print clf.score(X_test, y_test)

def cross():
    iris = datasets.load_iris()
    clf = svm.SVC(kernel='linear', C=1)
    scores = cross_val_score(clf, iris.data, iris.target, cv=5)
    print scores



split()
cross()
