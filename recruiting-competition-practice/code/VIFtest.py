#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 10:47:33 2017

@author: zexi
"""
import pandas as pd
import numpy as np
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# load the data
df_ordered = pd.read_csv('ordered.csv',sep=',')
#===================================== define function VIF

def VIF(whole_list,feature_list):
    '''
    This function perform VIF on given list, whole_list includes target varaible and all features, and feature_list includes only features
    '''
    df = df_ordered

    mask = (df['item_nbr'] == 5)
    df = df.loc[mask]

    df_temperature_related = df[whole_list]
    df_temperature_related = df_temperature_related.convert_objects(convert_numeric=True).dropna()
    df_temperature_related = df_temperature_related._get_numeric_data()
    df_temperature_related = df_temperature_related.reset_index(drop=True)

    df_temperature_related_features = df_temperature_related[feature_list]

    features = "+".join(df_temperature_related_features.columns)
    y, X = dmatrices('units ~' + features, df_temperature_related, return_type='dataframe')

    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    vif["features"] = X.columns

    print vif.round(1)
    
#==========================performing on temperature features
whole_list = ['units','tmax','tmin','tavg','depart','dewpoint','wetbulb','heat','cool']
feature_list = ['tmax','tmin','tavg','depart','dewpoint','wetbulb','heat','cool']

VIF(whole_list,feature_list)

whole_list = ['units','tmin','tavg','depart','dewpoint','wetbulb','heat','cool']
feature_list = ['tmin','tavg','depart','dewpoint','wetbulb','heat','cool']

VIF(whole_list,feature_list)

whole_list = ['units','tavg','depart','dewpoint','wetbulb','heat','cool']
feature_list = ['tavg','depart','dewpoint','wetbulb','heat','cool']

VIF(whole_list,feature_list)

whole_list = ['units','tavg','depart','wetbulb','heat','cool']
feature_list = ['tavg','depart','wetbulb','heat','cool']

VIF(whole_list,feature_list)

whole_list = ['units','tavg','depart','wetbulb','cool']
feature_list = ['tavg','depart','wetbulb','cool']

VIF(whole_list,feature_list)

whole_list = ['units','tavg','depart','cool']
feature_list = ['tavg','depart','cool']

VIF(whole_list,feature_list)

#=======================performing on wind related features
whole_list = ['units','stnpressure','sealevel','resultspeed','resultdir','avgspeed']
feature_list = ['stnpressure','sealevel','resultspeed','resultdir','avgspeed']
VIF(whole_list,feature_list)

whole_list = ['units','stnpressure','sealevel','resultdir','avgspeed']
feature_list = ['stnpressure','sealevel','resultdir','avgspeed']
VIF(whole_list,feature_list)

#====================performing on rainfall features
whole_list = ['units','snowfall','preciptotal']
feature_list = ['snowfall','preciptotal']
VIF(whole_list,feature_list)