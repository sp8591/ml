from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
def variance():
    x = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    print sel.fit_transform(x)

def x2():
    iris = load_iris()
    X, y = iris.data, iris.target
    X_new = SelectKBest(chi2, k=2).fit_transform(X, y)
    print X_new.shape

def l1():
    iris = load_iris()
    X, y = iris.data, iris.target
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(X)
    print X_new.shape

def trees():
    iris = load_iris()
    X, y = iris.data, iris.target
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    print X_new.shape



#variance()
x2()
l1()
trees()

