
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
import numpy as np
from sklearn import svm

def linear_regression_model(data, periods=1):
    y = data.values
    y = y[periods:]
    X_data = np.zeros((periods, len(y)))
    for i in range(periods):
        X = data.shift(periods=i+1)
        X.dropna(inplace=True)
        X = X.values
        if -periods+i+1 < 0:
            X = X[periods-i-1:]
        X = X.reshape(1, -1)
        X_data[i] = X[0]
    X_data = X_data.transpose()
    lr = LinearRegression()
    # lr = Ridge(alpha=0.01)
    lr.fit(X_data, y)
    print(lr.coef_)
    print('-------------------------')
    print(lr.intercept_)
    return lr



def SVR(data, periods=1):
    y = data.values
    y = y[periods:]
    X_data = np.zeros((periods, len(y)))
    for i in range(periods):
        X = data.shift(periods=i+1)
        X.dropna(inplace=True)
        X = X.values
        if -periods+i+1 < 0:
            X = X[periods-i-1:]
        X = X.reshape(1, -1)
        X_data[i] = X[0]
    X_data = X_data.transpose()
    model = svm.SVR(kernel='poly')
    model.fit(X_data, y)
    return model

