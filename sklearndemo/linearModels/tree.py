from sklearn import tree
import pydotplus
from sklearn.datasets import load_iris
def treed():
    iris = load_iris()
    clf = tree.DecisionTreeClassifier()
    clf.fit(iris.data, iris.target)
    print clf.predict(iris.data[:1, :])
    print clf.predict_proba(iris.data[:1, :])
    dot_data = tree.export_graphviz(clf, out_file=None)
    dot_data = tree.export_graphviz(clf, out_file=None,
                                    feature_names=iris.feature_names,
                                    class_names=iris.target_names,
                                    filled=True, rounded=True,
                                    special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("iris.pdf")

treed()