#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 20:47:24 2017

@author: zexi
"""

import pandas as pd
import numpy as np
import sys
import re
import os


df_ordered = pd.read_csv("../code/ordered.csv")
df_ordered['train'] = 1
df_orderedtest = pd.read_csv("../code/ordered_test.csv")
df_orderedtest['train']=0
df_orderedtest['units']=np.nan #last version0

print df_ordered.shape
print df_orderedtest.shape

print df_orderedtest.head()

df_main = pd.concat([df_ordered,df_orderedtest])
#df_main = df_main.drop(['Unnamed: 0','station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed'],axis=1)
df_main = df_main.drop(['Unnamed: 0','station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed','avgspeed','preciptotal','resultdir','tavg'],axis=1)
df_ordered = []
df_orderedtest = []

df_main = df_main.sort_values(['store_nbr','item_nbr','date']).reset_index(drop=True)
df_main['units'] = df_main['units'].interpolate() #added

df_main['Before_Event'] = (pd.rolling_mean(df_main['Condition'], window=3).shift(-3) > 0)
df_main['After_Event'] = (pd.rolling_mean(df_main['Condition'], window=3).shift(1) > 0)

df_main['Before_Sales'] = pd.rolling_mean(df_main['units'], window=6).shift(-6)
df_main['After_Sales'] = pd.rolling_mean(df_main['units'], window=6).shift(1)

item_nbr_list = list(set(df_main['item_nbr']))
store_nbr_list = list(set(df_main['store_nbr']))

df_ordered = df_main.loc[(df_main['train'] == 1)].reset_index(drop=True)
df_orderedtest = df_main.loc[(df_main['train'] == 0)].reset_index(drop=True)

df_ordered['year'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.year
df_ordered['month'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.month
df_ordered['weekday'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.weekday

df_orderedtest['year'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.year
df_orderedtest['month'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.month
df_orderedtest['weekday'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.weekday


## drop unused columns

#df_ordered = df_ordered.drop(['station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed'],axis=1)
#df_orderedtest = df_orderedtest.drop(['station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed'],axis=1)

## get_dummies

df_ordered = pd.get_dummies(df_ordered, columns=['year','month','weekday'])
df_ordered = df_ordered.drop(['date'],axis=1)

df_orderedtest = pd.get_dummies(df_orderedtest, columns=['year','month','weekday'])

df_orderedtest['id'] = df_orderedtest['store_nbr'].astype(str) + '_' + df_orderedtest['item_nbr'].astype(str) + '_' + df_orderedtest['date'].astype(str)
df_orderedtest = df_orderedtest.set_index('id')

df_orderedtest = df_orderedtest.drop(['date'],axis=1)


# why not
result = df_orderedtest['units']
result = result.reset_index()
mask = (result['units'] < 0)
result.loc[mask,'units'] = 0
result.to_csv("../code/result_0_new.csv",sep=",",index=False)
## train test split

item_nbr_list = list(set(df_ordered['item_nbr']))
store_nbr_list = list(set(df_ordered['store_nbr']))

#df_orderedtest.insert(12, 'year_2012', 0) # this one is right
df_orderedtest.insert(12, 'year_2012', 0)
df_ordered['units'] = df_ordered['units'].apply(lambda x: np.log(x+1)) #  new ver4
df_orderedtest['units'] = df_orderedtest['units'].apply(lambda x: np.log(x+1)) # new ver4

print df_ordered.columns
print df_orderedtest.columns
## Set 0 
#result = df_orderedtest['units']
#result = result.reset_index()
#mask = (result['units'] < 0)
#result.loc[mask,'units'] = 0
#result.to_csv("../code/result_0.csv",sep=",",index=False)

## SVM =================================================================================================
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from sklearn.svm import SVR

# for result analysis
paras = []
paras5 = []
paras9 = []
paras45 = []
paras47 = []
paras68 = []
store1 = []
store2 = []
store3 = []
#===========

for store in store_nbr_list:
    for item in item_nbr_list:
        mask = ((df_ordered['item_nbr'] == item) & (df_ordered['store_nbr'] == store))
        y_train = df_ordered.loc[mask,'units']
        X_train = df_ordered.loc[mask,df_ordered.columns != 'units']
        
        mask = ((df_orderedtest['item_nbr'] == item) & (df_orderedtest['store_nbr'] == store))
        X_test = df_orderedtest.loc[mask,df_orderedtest.columns != 'units']
        
        if X_test.empty == 0:
            if y_train.max() != 0:
       
                scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
                X_train = scaling.transform(X_train)
                X_test = scaling.transform(X_test)
        
                svr = SVR(kernel='linear')
                #svr = SVR()
                #svr = SVR(kernel='rbf', C=100.0, gamma=10.0)
                svr.fit(X_train, y_train)
                df_orderedtest.loc[mask,'units'] = svr.predict(X_test)
                
                paras.append(svr.coef_)
                if item == 5:
                    paras5.append(svr.coef_)
                if item == 9:
                    paras9.append(svr.coef_)
                if item == 45:
                    paras45.append(svr.coef_)
                if item == 48:
                    paras47.append(svr.coef_)
                if item == 68:
                    paras68.append(svr.coef_)
                if item == 16:
                    store1.append(svr.coef_)
                if item == 44:
                    store2.append(svr.coef_)
                if item == 51:
                    store3.append(svr.coef_)
                
    print store

df_ordered['units'] = df_ordered['units'].apply(lambda x: np.exp(x)-1)
df_orderedtest['units'] = df_orderedtest['units'].apply(lambda x: np.exp(x)-1)
result = df_orderedtest['units']
result = result.reset_index()
mask = (result['units'] < 0)
result.loc[mask,'units'] = 0
result.to_csv("../code/result_fastsvm.csv_v5",sep=",",index=False)

##+==================================================================================================
### Decision Tree
from sklearn.tree import DecisionTreeRegressor
#
#df_orderedtest['units'] = 0
#
for store in store_nbr_list:
    for item in item_nbr_list:
        mask = ((df_ordered['item_nbr'] == item) & (df_ordered['store_nbr'] == store))
        y_train = df_ordered.loc[mask,'units']
        X_train = df_ordered.loc[mask,df_ordered.columns != 'units']
#        
        mask = ((df_orderedtest['item_nbr'] == item) & (df_orderedtest['store_nbr'] == store))
        X_test = df_orderedtest.loc[mask,df_orderedtest.columns != 'units']
        
        if X_test.empty == 0:
            if y_train.max() != 0:
               scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
               X_train = scaling.transform(X_train)
               X_test = scaling.transform(X_test)
        
               #regression_tree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10,random_state=42)
               regression_tree = DecisionTreeRegressor(random_state=42)
               regression_tree.fit(X_train, y_train)
               df_orderedtest.loc[mask,'units'] = regression_tree.predict(X_test)
        
    print store
        
result = df_orderedtest['units']
result = result.reset_index()
mask = (result['units'] < 0)
result.loc[mask,'units'] = 0
result.to_csv("../code/result_tree.csv",sep=",",index=False)

### NN ===============================================================================================================

import gc
import tensorflow as tf
import tensorflow.contrib.learn as skflow
import itertools 

tf.logging.set_verbosity(tf.logging.ERROR)
df_orderedtest['units'] = 0
for store in store_nbr_list:
    for item in item_nbr_list:
        mask = ((df_ordered['item_nbr'] == item) & (df_ordered['store_nbr'] == store))
        y_train = df_ordered.loc[mask,'units']
        X_train = df_ordered.loc[mask,df_ordered.columns != 'units']
        
        mask = ((df_orderedtest['item_nbr'] == item) & (df_orderedtest['store_nbr'] == store))
        X_test = df_orderedtest.loc[mask,df_orderedtest.columns != 'units']
        
        if X_test.empty == 0:
            if y_train.max() != 0:
                scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
                X_train = scaling.transform(X_train)
                X_test = scaling.transform(X_test)
        
                reg3 = skflow.DNNRegressor(
                        feature_columns = tf.contrib.learn.infer_real_valued_columns_from_input(X_train),
                        hidden_units=[35,35,35],
                        optimizer='Adam')

                reg3.fit(X_train,y_train, steps = 600)
                y_predict = reg3.predict(X_test)
                predictions = list(itertools.islice(y_predict, len(df_orderedtest.loc[mask,'units'])))
                df_orderedtest.loc[mask,'units'] = predictions
        
        print store,item
        
result = df_orderedtest['units']
result = result.reset_index()
mask = (result['units'] < 0)
result.loc[mask,'units'] = 0
result.to_csv("../code/result_nn.csv",sep=",",index=False)