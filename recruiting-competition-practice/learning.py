#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 12:00:49 2017

@author: zexi
"""
import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math


#df7 = pd.read_csv('ordered.csv',sep=',')
df8 = pd.read_csv('severeWeather.csv',sep=',')
mask_test = (df8['item_nbr'] == 5)
df_test = df8.loc[mask_test]

#df_test = df_test[['date','units','tavg','depart','snowfall','preciptotal','stnpressure','sealevel','resultdir','avgspeed','station_nbr']]
df_test = df_test[['date','units','tavg','preciptotal','stnpressure','sealevel','resultdir','avgspeed','station_nbr']]
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


from sklearn.linear_model import LinearRegression
from sklearn import svm
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV

#########################3 for tunning##################333
#linreg2 = svm.SVR()
#params = {'C': np.logspace(-5,3,9), 'gamma': np.logspace(-5,3,9), 'degree': [1],
#        'kernel':["rbf",'linear'],}
#max_lin = RandomizedSearchCV(linreg2,param_distributions=params,n_iter=10,cv=3,random_state=101,scoring='neg_mean_squared_error',verbose=10,\
#                             pre_dispatch=None)
#model_tuning=max_lin.fit(X_train,y_train)
##model_tuning=max_lin.fit(X_train,y_train)
#print ("best score %s" % model_tuning.best_score_)
#print ("best parameters %s" % model_tuning.best_params_)
############################################3

#linreg = LinearRegression()
#linreg = svm.SVR(kernel='rbf', C=10000.0, gamma=100000.0)
linreg = svm.SVR(kernel='rbf', C=100.0, gamma=10.0)
#linreg = svm.SVR(kernel='rbf', C=10.0, gamma=0.01)
#linreg = svm.SVR(kernel='linear', C=10.0, degree=1, gamma=1e-05)
linreg.fit(X_train, y_train)
#linreg.fit(X_train,y_train)

#print(linreg.intercept_)
#print(linreg.coef_)

#y_pred = np.exp(linreg.predict(X_test))-1.0
#y_pred = linreg.predict(X_test)
#from sklearn import metrics
## MSE
#print ("MSE:",metrics.mean_squared_error(y_test, y_pred))
## RMSE
#print ("RMSE:",np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# vectorized error calc
#def rmsle(y, y0):
#    assert len(y) == len(y0)
#    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
#
##looping error calc
#def rmsle_loop(y, y_pred):
#    assert len(y) == len(y_pred)
#    terms_to_sum = [(math.log(y_pred[i] + 1) - math.log(y[i] + 1)) ** 2.0 for i,pred in enumerate(y_pred)]
#    return (sum(terms_to_sum) * (1.0/len(y))) ** 0.5

#print ("RMSLE:",rmsle(y_test, y_pred))

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(linreg, X, y, cv=10)

from sklearn import metrics
# MSE
print("MSE:",metrics.mean_squared_error(y, predicted))
# RMSE
print("RMSE:",np.sqrt(metrics.mean_squared_error(y, predicted)))

print("RMSLE:",np.sqrt(metrics.mean_squared_error(np.log(y+1.0), np.log(predicted+1.0))/110))
# RMLSE
#print ("RMSLE:",rmsle(y, predicted))

#==================================================
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y, predicted)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

#import matplotlib.pyplot as plt
#
#fig, ax = plt.subplots()
#ax.scatter(y_test, y_pred)
#ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()

#============================================



