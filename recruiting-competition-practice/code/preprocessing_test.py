#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 18:02:11 2017

@author: zexi
"""
import pandas as pd
import numpy as np

df_key = pd.read_csv("../input/key.csv")
df_test = pd.read_csv("../input/test.csv")
df_weather = pd.read_csv("../input/weather.csv")

df_test['date'] = pd.to_datetime(df_test['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

temp = pd.merge(df_test, df_key,how='left', on=['store_nbr'])
df_main_test = pd.merge(temp, df_weather, how='left', on=['station_nbr','date'])

df_ordered = df_main_test.sort_values(['store_nbr','item_nbr','date']).reset_index(drop=True)
#df7 = df7.apply(pd.to_numeric, errors='coerce')
df_ordered = df_ordered.convert_objects(convert_numeric=True)
df_ordered['preciptotal'] = df_ordered['preciptotal'].fillna(0)
df_ordered['snowfall'] = df_ordered['snowfall'].fillna(0)
df_ordered = df_ordered.interpolate()


patternRA = 'RA'
patternSN = 'SN'
df_ordered['RA'], df_ordered['SN'] = df_ordered['codesum'].str.contains(patternRA), df_ordered['codesum'].str.contains(patternSN)
df_ordered['Condition'] = (df_ordered['RA'] & (df_ordered['preciptotal']>1.0)) | (df_ordered['SN'] & (df_ordered['preciptotal']>2.0))
df_ordered['WEvent'] = (pd.rolling_mean(df_ordered['Condition'],window=7,center=True) > 0)

df_ordered.to_csv('ordered_test.csv', sep=',')