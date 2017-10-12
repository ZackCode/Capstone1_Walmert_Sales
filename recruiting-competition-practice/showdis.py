#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:50:07 2017

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


#============================================================
df_test = df_test.dropna()
df_test = df_test.convert_objects(convert_numeric=True).dropna()
#df_test = df_test.apply(pd.to_numeric, errors='raise').dropna()
df_test = df_test._get_numeric_data()

X = df_test[['tavg','depart','snowfall','preciptotal','stnpressure','sealevel','resultdir','avgspeed']]
y = df_test[['units']]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(y, X['avgspeed'])
ax.set_xlabel('Measured')
ax.set_ylabel('X')
plt.show()