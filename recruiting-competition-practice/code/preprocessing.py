#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 11:02:09 2017

@author: zexi

This document preprocess the training data and store it as ordered.csv
"""
# load the data
import pandas as pd
import numpy as np

df_key = pd.read_csv("../input/key.csv")
df_train = pd.read_csv("../input/train.csv")
df_test = pd.read_csv("../input/test.csv")
df_weather = pd.read_csv("../input/weather.csv")

# merge the data
df_train['date'] = pd.to_datetime(df_train['date'])
df_weather['date'] = pd.to_datetime(df_weather['date'])

temp = pd.merge(df_train, df_key,how='left', on=['store_nbr'])
df_main_train = pd.merge(temp, df_weather, how='left', on=['station_nbr','date'])
# evaluate
#print(df_train.shape)
#print(temp.shape)
#print(df_main_train.shape)
#print(list(df_main_train))

# sort the data and interpolate
df_ordered = df_main_train.sort_values(['store_nbr','item_nbr','date']).reset_index(drop=True)
#df7 = df7.apply(pd.to_numeric, errors='coerce')
df_ordered = df_ordered.convert_objects(convert_numeric=True)
df_ordered['preciptotal'] = df_ordered['preciptotal'].fillna(0) #replace all letter records in preciptotal with 0
df_ordered['snowfall'] = df_ordered['snowfall'].fillna(0)
df_ordered.loc[((df_ordered['units'] == 3369) | (df_ordered['units'] == 5568)),'units'] = np.nan
df_ordered = df_ordered.interpolate()

# add features for weather event
patternRA = 'RA'
patternSN = 'SN'
df_ordered['RA'], df_ordered['SN'] = df_ordered['codesum'].str.contains(patternRA), df_ordered['codesum'].str.contains(patternSN)
df_ordered['Condition'] = (df_ordered['RA'] & (df_ordered['preciptotal']>1.0)) | (df_ordered['SN'] & (df_ordered['preciptotal']>2.0))
df_ordered['WEvent'] = (pd.rolling_mean(df_ordered['Condition'],window=7,center=True) > 0)

mask = (df_ordered['WEvent'] == True)
df_severeWeather = df_ordered.loc[mask]

# save result
df_severeWeather.to_csv('severeWeather.csv', sep=',')
df_ordered.to_csv('ordered.csv', sep=',')