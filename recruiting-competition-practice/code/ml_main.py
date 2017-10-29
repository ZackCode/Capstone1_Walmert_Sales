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

df_ordered = pd.read_csv("../code/ordered.csv")
df_orderedtest = pd.read_csv("../code/ordered_test.csv")

mask = (df_ordered['WEvent'] == 1)

df_ordered['Before_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(-3) > 0)
df_ordered['After_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(1) > 0)

mask = (df_orderedtest['WEvent'] == 1)

df_orderedtest['Before_Event'] = (pd.rolling_mean(df_orderedtest['Condition'], window=3).shift(-3) > 0)
df_orderedtest['After_Event'] = (pd.rolling_mean(df_orderedtest['Condition'], window=3).shift(1) > 0)

df_ordered['year'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.year
df_ordered['month'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.month
df_ordered['weekday'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.weekday

df_orderedtest['year'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.year
df_orderedtest['month'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.month
df_orderedtest['weekday'] = pd.to_datetime(df_orderedtest['date'], infer_datetime_format=True).dt.weekday


## drop unused columns

df_ordered = df_ordered.drop(['station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed'],axis=1)
df_orderedtest = df_orderedtest.drop(['station_nbr','tmax','tmin','depart','dewpoint','wetbulb','heat','cool','sunrise','sunset','codesum','snowfall','stnpressure','sealevel','resultspeed'],axis=1)

## get_dummies

df_ordered = pd.get_dummies(df_ordered, columns=['year','month','weekday'])
df_ordered = df_ordered.drop(['Unnamed: 0','date'],axis=1)

df_orderedtest = pd.get_dummies(df_orderedtest, columns=['year','month','weekday'])

df_orderedtest['id'] = df_orderedtest['store_nbr'].astype(str) + '_' + df_orderedtest['item_nbr'].astype(str) + '_' + df_orderedtest['date'].astype(str)
df_orderedtest = df_orderedtest.set_index('id')

df_orderedtest = df_orderedtest.drop(['Unnamed: 0','date'],axis=1)
## train test split

item_nbr_list = list(set(df_ordered['item_nbr']))
store_nbr_list = list(set(df_ordered['store_nbr']))

df_orderedtest.insert(12, 'year_2012', 0)
df_orderedtest['units'] = 0
## Set 0 
#result = df_orderedtest['units']
#result = result.reset_index()
#mask = (result['units'] < 0)
#result.loc[mask,'units'] = 0
#result.to_csv("../code/result_0.csv",sep=",",index=False)
## SVM
from sklearn.preprocessing import MinMaxScaler
#from collections import defaultdict
#from sklearn.svm import SVR
#
#for store in store_nbr_list:
#    for item in item_nbr_list:
#        mask = ((df_ordered['item_nbr'] == item) & (df_ordered['store_nbr'] == store))
#        y_train = df_ordered.loc[mask,'units']
#        X_train = df_ordered.loc[mask,df_ordered.columns != 'units']
#        
#        mask = ((df_orderedtest['item_nbr'] == item) & (df_orderedtest['store_nbr'] == store))
#        X_test = df_orderedtest.loc[mask,df_orderedtest.columns != 'units']
#        
#        if X_test.empty == 0:
#       
#            scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
#            X_train = scaling.transform(X_train)
#            X_test = scaling.transform(X_test)
#        
#            svr = SVR()
#            svr.fit(X_train, y_train)
#            df_orderedtest.loc[mask,'units'] = svr.predict(X_test)
#        
#    print store
#        
#result = df_orderedtest['units']
#result = result.reset_index()
#mask = (result['units'] < 0)
#result.loc[mask,'units'] = 0
#result.to_csv("../code/result.csv",sep=",",index=False)

### Decision Tree
#from sklearn.tree import DecisionTreeRegressor
#
#df_orderedtest['units'] = 0
#
#for store in store_nbr_list:
#    for item in item_nbr_list:
#        mask = ((df_ordered['item_nbr'] == item) & (df_ordered['store_nbr'] == store))
#        y_train = df_ordered.loc[mask,'units']
#        X_train = df_ordered.loc[mask,df_ordered.columns != 'units']
#        
#        mask = ((df_orderedtest['item_nbr'] == item) & (df_orderedtest['store_nbr'] == store))
#        X_test = df_orderedtest.loc[mask,df_orderedtest.columns != 'units']
#        
#        if X_test.empty == 0:
#       
#            scaling = MinMaxScaler(feature_range=(0,1)).fit(X_train)
#            X_train = scaling.transform(X_train)
#            X_test = scaling.transform(X_test)
#        
#            #regression_tree = DecisionTreeRegressor(min_samples_split=30, min_samples_leaf=10,random_state=42)
#            regression_tree = DecisionTreeRegressor(random_state=42)
#            regression_tree.fit(X_train, y_train)
#            df_orderedtest.loc[mask,'units'] = regression_tree.predict(X_test)
#        
#    print store
#        
#result = df_orderedtest['units']
#result = result.reset_index()
#mask = (result['units'] < 0)
#result.loc[mask,'units'] = 0
#result.to_csv("../code/result_tree.csv",sep=",",index=False)

### NN

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