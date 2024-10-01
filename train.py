import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from tools.dataloader import data_loader
from models.regression_model import linear_regression_model, SVR
from sklearn.model_selection import train_test_split
from tools.model_test import test_data, test_model
# parameters
periods = 20


# dataloader
path = './data/data.mat'
df = data_loader(path)
df.dropna()
df_row = df[0]
df_row = df_row[:20000]
# X = df[0].values
# X = X[0:100]
# result = adfuller(X)
# print('ADF Statistic: %f' % result[0])
# print('p-value: %f' % result[1])
# print('Critical Values:')
# for key, value in result[4].items():
#     print('\t%s: %.3f' % (key, value))

# if result[0] < result[4]["5%"]:
#     print ("Reject Ho - Time Series is Stationary")
# else:
#     print ("Failed to Reject Ho - Time Series is Non-Stationary")
     
# plot_pacf(X, lags=10)
# plt.show()

train_df, test_df = train_test_split(df_row, test_size=0.2, shuffle=False) # to be continue

# Model
lr = linear_regression_model(train_df, periods=periods)
# lr = SVR(train_df, periods=periods)

# test
test_df_shifted, target = test_data(data=test_df, periods=periods)
prediction = test_model(lr, test_df[:200], periods=periods)
# print(test_df_shifted[0])
# print(prediction)
# print(target)
plt.plot(target[-200:], label="Actual Values")
plt.plot(prediction[-200:], label="Predicted Values")
plt.legend()
plt.show()


# MSE
from sklearn.metrics import mean_squared_error # 均方误差
from sklearn.metrics import mean_absolute_error # 平方绝对误差
from sklearn.metrics import r2_score # R square

print(mean_squared_error(target,prediction))
print(mean_absolute_error(target,prediction))
print(r2_score(target,prediction))

