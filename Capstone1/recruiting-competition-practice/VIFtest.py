#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:47:33 2017

@author: zexi
"""
import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math


#df7 = pd.read_csv('ordered.csv',sep=',')
df8 = pd.read_csv('severeWeather.csv',sep=',')
#=====================================
mask_test = (df8['item_nbr'] == 5)
df_test = df8.loc[mask_test]


df11 = df8

mask = (df11['item_nbr'] == 11)
df11 = df11.loc[mask]

df12 = df11[['units','tavg','depart']]
#df12 = df12.convert_objects(convert_numeric=True).dropna()
df12 = df12.apply(pd.to_numeric, errors='coerce').dropna()
df12 = df12._get_numeric_data().reset_index(drop=True)

#===============================================================
df13 = df12[['tavg','depart']]

features = "+".join(df13.columns)
y, X = dmatrices('units ~' + features, df12, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)

#=============================================================
df11 = df8

mask = (df11['item_nbr'] == 11)
df11 = df11.loc[mask]

df12 = df11[['units','snowfall','preciptotal']]
df12 = df12.apply(pd.to_numeric, errors='coerce').dropna()
df12 = df12._get_numeric_data()
df12.reset_index(drop=True)

df13 = df12[['snowfall','preciptotal']]

features = "+".join(df13.columns)
y, X = dmatrices('units ~' + features, df12, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns

vif.round(1)
#==============================================================
df11 = df8

mask = (df11['item_nbr'] == 11)
df11 = df11.loc[mask]

df12 = df11[['units','stnpressure','sealevel','resultdir','avgspeed']]
df12 = df12.apply(pd.to_numeric, errors='coerce').dropna()
df12 = df12._get_numeric_data()
df12.reset_index(drop=True)

df13 = df12[['stnpressure','sealevel','resultdir','avgspeed']]

features = "+".join(df13.columns)
y, X = dmatrices('units ~' + features, df12, return_type='dataframe')

vif = pd.DataFrame()
vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif.round(1)
