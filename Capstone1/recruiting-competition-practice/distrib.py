#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 14:12:54 2017

@author: zexi
"""

import pandas as pd
import numpy as np
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor
import math

df8 = pd.read_csv('severeWeather.csv',sep=',')

max_i = []
mean_u = 0

for num in range(1,df8['item_nbr'].max()):
    mask_test = (df8['item_nbr'] == num)
    df_test = df8.loc[mask_test]
    if df_test['units'].mean() > mean_u:
        max_i.insert(0,num)
        mean_u = df_test['units'].mean()
    else:
        max_i.append(num)

print max_i


mask_test = (df8['item_nbr'] == max_i[2])
df_test = df8.loc[mask_test]
print df_test['units'].describe()
