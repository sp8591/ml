import numpy
import sklearndemo
from sklearn import linear_model



def linear_model_demo():
    reg = linear_model.LinearRegression()
    x = [[0, 0], [1, 1], [2, 2.1], [3, 3.2], [4, 3.9]]
    x1 = [[5, 5.2], [6, 6], [7, 6.5]]
    y = [2, 3, 4, 5, 6]
    reg.fit(x, y)
    y1 = reg.predict(x1)
    print reg.coef_
    print reg.intercept_
    print y1

def linear_model_ridge():
    reg = linear_model.Ridge(alpha=.5)
    x = [[0, 0], [1, 1], [2, 1.5], [3, 3.2], [4, 3.9]]
    x1 = [[5, 5.2], [6, 6], [7, 6.5]]
    y = [2, 3, 3.5, 5, 6]
    y1 = [7, 8, 8.5]
    reg.fit(x, y)
    y1 = reg.predict(x1)
    print reg.coef_
    print reg.intercept_
    print y1

def linear_model_lasso():
    reg = linear_model.Lasso(alpha=0.1)
    x = [[0, 0], [1, 1], [2, 1.5], [3, 3.2], [4, 3.9]]
    x1 = [[5, 5.2], [6, 6], [7, 6.5]]
    y = [2, 3, 3.5, 5, 6]
    y1 = [7, 8, 8.5]
    reg.fit(x, y)
    y1 = reg.predict(x1)
    print reg.coef_
    print reg.intercept_
    print y1

def linear_model_lars_lasso():
    reg = linear_model.LassoLars(alpha=0.1)
    x = [[0, 0], [1, 1], [2, 1.5], [3, 3.2], [4, 3.9]]
    x1 = [[5, 5.2], [6, 6], [7, 6.5]]
    y = [2, 3, 3.5, 5, 6]
    y1 = [7, 8, 8.5]
    reg.fit(x, y)
    y1 = reg.predict(x1)
    print reg.coef_
    print reg.intercept_
    print y1

def linear_model_bayesian_ridge():
    reg = linear_model.BayesianRidge()
    x = [[0, 0], [1, 1], [2, 1.5], [3, 3.2], [4, 3.9]]
    x1 = [[5, 5.2], [6, 6], [7, 6.5]]
    y = [2, 3, 3.5, 5, 6]
    y1 = [7, 8, 8.5]
    reg.fit(x, y)
    y1 = reg.predict(x1)
    print reg.coef_
    print reg.intercept_
    print y1


if __name__ == '__main__':
    #linear_model_demo()
    #linear_model_ridge()
    #linear_model_lasso()
    #linear_model_lars_lasso()
    linear_model_bayesian_ridge()