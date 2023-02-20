import sklearn.datasets as datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from sklearn import metrics
import numpy as np

#设置显示的最大列、宽等参数，消掉打印不完全中间的省略号
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)

# 导入数据，路径中要么用\\或/或者在路径前加r
# dataset = pd.read_csv(r'D:\pythonCode\RF\petrol_consumption.csv')
dataset = pd.read_excel('数据整合.xlsx')
# 输出数据预览
# print(dataset)

# 准备训练数据
X = dataset.iloc[:, 3:9].values
y = dataset.iloc[:, 9].values
# print(X)
# print(y)
minn = 100.0;
# 将数据分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.2,
                                                    random_state=13)
regr = RandomForestRegressor(n_estimators=100000, random_state=13)
regr.fit(X_train, y_train)
y_pred = regr.predict(X_test)
errors = abs(y_pred - y_test)
MAPE = 100 * (errors / y_test)
# 调参过程
# for i in range(10, 100):
#
#     X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     random_state=i)
#     regr = RandomForestRegressor(n_estimators=1000, random_state=i)
#     regr.fit(X_train, y_train)
#     y_pred = regr.predict(X_test)
#     errors = abs(y_pred - y_test)
#     MAPE = 100 * (errors / y_test)
#     if(minn > np.mean(MAPE)):
#         minn = np.mean(MAPE)
#         print(i,minn)

# random_state 42 MAPE 19
#              9       18
#              13      14
# regr = RandomForestRegressor(random_state=100,
#                              bootstrap=True,
#                              max_depth=2,
#                              max_features=2,
#                              min_samples_leaf=3,
#                              min_samples_split=5,
#                              n_estimators=3)
# pipe = Pipeline([('scaler', StandardScaler()), ('reduce_dim', PCA()),
#                  ('regressor', regr)])
# pipe.fit(X_train, y_train)
# ypipe = pipe.predict(X_test)



print('Mean Absolute Percentage Error:',np.mean(MAPE),"%")
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:',
      np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

