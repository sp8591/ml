from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn import datasets
def baggingex():
    iris = datasets.load_iris()
    clf = BaggingClassifier(KNeighborsClassifier(),max_samples = 0.5, max_features = 0.5)
    clf = RandomForestClassifier(n_estimators=10)
    clf = AdaBoostClassifier(n_estimators=100)
    clf.fit(iris.data, iris.target)
    print clf.predict(iris.data[1:, :])
    scores = cross_val_score(clf, iris.data, iris.target)
    print scores.mean()


baggingex()