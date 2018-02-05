from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
def bayes():
    iris = datasets.load_iris()
    gnb = GaussianNB()
    y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
    print y_pred

if __name__ == '__main__':
    bayes()