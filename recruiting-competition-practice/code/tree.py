#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 12:49:04 2017

@author: zexi
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:49:49 2017

@author: zexi
"""

import numpy as np
import scipy as sp 
from sknn.mlp import Regressor, Layer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy import stats
import pandas as pd
#from learning import X_train, X_test, y_train, y_test

df8 = pd.read_csv('severeWeather.csv',sep=',')
mask_test = (df8['item_nbr'] == 5)
df_test = df8.loc[mask_test]


#============================================================
#df7 = pd.read_csv('ordered.csv',sep=',')
df8 = pd.read_csv('severeWeather.csv',sep=',')
mask_test = (df8['item_nbr'] == 5)
df_test = df8.loc[mask_test]

#df_test = df_test[['date','units','tavg','depart','snowfall','preciptotal','stnpressure','sealevel','resultdir','avgspeed','station_nbr']]
df_test = df_test[['date','units','tavg','station_nbr']]
# weekdays search

df_test['weekdays'] = pd.to_datetime(df_test['date']).dt.dayofweek
df_test['month'] = pd.to_datetime(df_test['date']).dt.month

def qualTrans(df, b_var):
    """
    Reads a pandas dataframe and returns a dataframe with additional 
    features that numeric the quality and text features. Note that
    this function does not modify the input dataframe
    Parameters
    ----------
    df = the input dataframe
    b_var = the index list of quality features
    """
    df2 = df.copy()
    features = []

    for index, row in df.iterrows():
        for names in b_var:
            if names+'_'+str(row[names]) not in features:
                features.append(names+'_'+str(row[names]))

    for item in features:
        df2[item] = 0

    for index, row in df2.iterrows():
        for names in b_var:
            df2.set_value(index, names+'_'+str(row[names]), 1)
    return df2

b_var = ['month','weekdays','station_nbr']

df_test2 = qualTrans(df_test, b_var)
df_test = df_test2
print df_test
drop_cl = ['date','month','weekdays','station_nbr']

for item in drop_cl:
    df_test = df_test.drop(item, 1)

#============================================================
df_test = df_test.interpolate()
df_test = df_test.convert_objects(convert_numeric=True).dropna()
#df_test = df_test.apply(pd.to_numeric, errors='raise').dropna()
df_test = df_test._get_numeric_data()

X = df_test.drop('units', 1)
y = df_test[['units']]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
#scaler2 = StandardScaler()
X = scaler.fit_transform(X)
#y = scaler2.fit_transform(y)

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
######3
#####################33
#############################
###############################3
#X_trainn = X_train.as_matrix()
#X_testn = X_test.as_matrix()
#y_trainn = y_train.as_matrix()
#y_testn = y_test.as_matrix()
X_trainn = X_train
X_testn = X_test
y_trainn = y_train
y_testn = y_test


from sklearn import tree

from sklearn.tree import DecisionTreeRegressor
regression_tree = tree.DecisionTreeRegressor(
        min_samples_split=30, min_samples_leaf=10,random_state=0)

regression_tree.fit(X_trainn,y_trainn)
#score = np.mean(cross_validation.cross_val_score(regression_tree, X_trainn, y_trainn,
#                                scoring='neg_mean_squared_error', cv=10,
#                                n_jobs=1))
score = np.mean(cross_validation.cross_val_score(regression_tree, X, y,
                                scoring='neg_mean_squared_error', cv=10,
                                n_jobs=1))
print 'Mean squared error: %.3f' % abs(score)


 
y_hat=regression_tree.predict(X_testn)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y_test_p = y_test
y_hat_p = y_hat

ax.scatter(y_test_p, y_hat_p)
ax.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

from sklearn import metrics

print("RMSLE:",np.sqrt(metrics.mean_squared_error(np.log(y_test_p+1.0), np.log(y_hat_p+1.0)))/110)