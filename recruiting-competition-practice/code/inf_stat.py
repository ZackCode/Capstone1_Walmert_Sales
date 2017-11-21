#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:10:43 2017

@author: zexi
"""
import pandas as pd
import numpy as np
import sys
import re
df_ordered = pd.read_csv("../code/ordered.csv")
df_severeWeather = pd.read_csv("../code/severeWeather.csv")

# Define some useful functions first

def perm_diff(data1,data2,targetfun):
    conc_data = np.concatenate((data1,data2))
    value = targetfun(data1) - targetfun(data2)
    value_diff = np.empty(10000)

    for i in range(10000):
        perm_data = np.random.permutation(conc_data)
        perm_data1 = perm_data[:len(data1)]
        perm_data2 = perm_data[len(data1):]
        
        perm_value1 = targetfun(perm_data1)
        perm_value2 = targetfun(perm_data2)
        value_diff[i] = perm_value1 - perm_value2
        
        if value>0:
            p = np.sum(value_diff > value)/float(len(value_diff))
        else:
            p = np.sum(value_diff < value)/float(len(value_diff))
    print "p value:",  p
    print "value", value
    print "99% null hypothesis interval:",  np.percentile(value_diff, [0.5, 99.5])
    
def bsfromfunc(observes,targetfunc):
    value = targetfunc(observes)
    bs_target = np.empty(10000)
    for i in range(10000):
        bs_sample = np.random.choice(observes,size=len(observes))
        bs_target[i] = targetfunc(bs_sample)
        
    print 'value, ', value
    print '99% interval, ', np.percentile(bs_target, [0.5, 99.5])

# null hypoth test: item 45
def item_sales_inferential(item):

    mask_event = ((df_ordered['WEvent'] == 1) & (df_ordered['item_nbr'] == item))
    mask_no_event = ((df_ordered['WEvent'] == 0) & (df_ordered['item_nbr'] == item))

    data1 = list(df_ordered.loc[mask_event, 'units'])
    data2 = list(df_ordered.loc[mask_no_event, 'units'])

    perm_diff(data1,data2,np.mean)

item_sales_inferential(45)

# null hypoth test: item 9
item_sales_inferential(9)

# null hypoth test: item 44
item_sales_inferential(44)

# a lot of similar tests...
print "item 5"
item_sales_inferential(5)
print "\n"

print "item 68"
item_sales_inferential(68)
print "\n"

print "item 16"
item_sales_inferential(16)
print "\n"

print "item 25"
item_sales_inferential(25)
print "\n"

print "item 48"
item_sales_inferential(48)
print "\n"

print "item 36"
item_sales_inferential(36)
print "\n"

# test: In extreme weather days, people buy item 5 more and they often do so before extreme weather comes
mask = (df_ordered['WEvent'] == 1)

df_ordered['Before_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(-3) > 0)
df_ordered['After_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(1) > 0)

mask_event = ((df_ordered['After_Event'] == 0) & (df_ordered['WEvent'] == 1) & (df_ordered['item_nbr'] == 5))
mask_no_event = ((df_ordered['After_Event'] == 1) & (df_ordered['item_nbr'] == 5))

data1 = list(df_ordered.loc[mask_event, 'units'])
data2 = list(df_ordered.loc[mask_no_event, 'units'])

perm_diff(data1,data2,np.mean)

# test: People buy item 93 more on the day of extreme weather. They also buy this item before an extreme weather.
mask_event = ((df_ordered['Condition'] == 1) & (df_ordered['item_nbr'] == 93))
mask_no_event = ((df_ordered['Condition'] == 0) & (df_ordered['item_nbr'] == 93))

data1 = list(df_ordered.loc[mask_event, 'units'])
data2 = list(df_ordered.loc[mask_no_event, 'units'])

perm_diff(data1,data2,np.mean)

mask_event = ((df_ordered['Condition'] == 0) & (df_ordered['item_nbr'] == 93) & (df_ordered['Before_Event'] == 1))
mask_no_event = ((df_ordered['Condition'] == 0) & (df_ordered['item_nbr'] == 93) & (df_ordered['Before_Event'] == 0))

data1 = list(df_ordered.loc[mask_event, 'units'])
data2 = list(df_ordered.loc[mask_no_event, 'units'])

perm_diff(data1,data2,np.mean)

# test: When facing a long time extreme weather event, people do less shopping on item 5, 45, 44
def examinie_long_term(item):

    mask_event = ((df_ordered['Before_Event'] ^ df_ordered['After_Event'] == 1) & (df_ordered['item_nbr'] == item) & (df_ordered['WEvent'] == 1))
    mask_no_event = ((df_ordered['Before_Event'] ^ df_ordered['After_Event'] == 0) & (df_ordered['item_nbr'] == item) & (df_ordered['WEvent'] == 1))

    data1 = list(df_ordered.loc[mask_event, 'units'])
    data2 = list(df_ordered.loc[mask_no_event, 'units'])

    perm_diff(data1,data2,np.mean)
    
print "item 5: "
examinie_long_term(5)
print "item 45: "
examinie_long_term(45)
print "item 44: "
examinie_long_term(44)

#test: Even when it is a sunny day, the sales record close to bad weather still differ from normal case, with item 5 being the best seller and item 45 at the third place.
mask_event = ((df_ordered['preciptotal'] == 0) & (df_ordered['item_nbr'] == 5) & (df_ordered['WEvent'] == 1))
mask_no_event = ((df_ordered['preciptotal'] == 0) & (df_ordered['item_nbr'] == 5) & (df_ordered['WEvent'] == 0))

data1 = list(df_ordered.loc[mask_event, 'units'])
data2 = list(df_ordered.loc[mask_no_event, 'units'])

perm_diff(data1,data2,np.mean)

mask_event = ((df_ordered['preciptotal'] == 0) & (df_ordered['item_nbr'] == 45) & (df_ordered['WEvent'] == 1))
mask_no_event = ((df_ordered['preciptotal'] == 0) & (df_ordered['item_nbr'] == 45) & (df_ordered['WEvent'] == 0))

data1 = list(df_ordered.loc[mask_event, 'units'])
data2 = list(df_ordered.loc[mask_no_event, 'units'])

perm_diff(data1,data2,np.mean)

# test: item 5 year: Sales record steady goes down year by year.
df_ordered['year'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.year

mask2012 = ((df_ordered['year'] == 2012) & (df_ordered['item_nbr'] == 5))
mask2013 = ((df_ordered['year'] == 2013) & (df_ordered['item_nbr'] == 5))
mask2014 = ((df_ordered['year'] == 2014) & (df_ordered['item_nbr'] == 5))

data2012 = list(df_ordered.loc[mask2012,'units'])
data2013 = list(df_ordered.loc[mask2013,'units'])
data2014 = list(df_ordered.loc[mask2014,'units'])

print "2012 item 5 sales per day: "
print bsfromfunc(data2012,np.mean)

print "2013 item 5 sales per day: "
print bsfromfunc(data2013,np.mean)

print "2014 item 5 sales per day: "
print bsfromfunc(data2014,np.mean)

# test: weekdays: monday saturday sunday sales better than other days:
df_ordered['weekday'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.weekday

maskMon = ((df_ordered['weekday'] == 0) & (df_ordered['item_nbr'] == 5))
maskTue = ((df_ordered['weekday'] == 1) & (df_ordered['item_nbr'] == 5))
maskWed = ((df_ordered['weekday'] == 2) & (df_ordered['item_nbr'] == 5))
maskThu = ((df_ordered['weekday'] == 3) & (df_ordered['item_nbr'] == 5))
maskFri = ((df_ordered['weekday'] == 4) & (df_ordered['item_nbr'] == 5))
maskSat = ((df_ordered['weekday'] == 5) & (df_ordered['item_nbr'] == 5))
maskSun = ((df_ordered['weekday'] == 6) & (df_ordered['item_nbr'] == 5))

list_para = [maskMon,maskTue,maskWed,maskThu,maskFri,maskSat,maskSun]
weekdays = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']

for ind in range(len(list_para)):
    data = list(df_ordered.loc[list_para[ind],'units'])
    print 'The average sales for ' + weekdays[ind] + ':'
    print bsfromfunc(data,np.mean)
    
# test: rainfall: For item 5: Rainfall/Snowfall: People tend to buy item 5 on a sunny day. But when facing major weather events people will go and buy them as well.
mask_normal_sunny = ((df_ordered['preciptotal'] == 0) & (df_ordered['WEvent'] == 0))
mask_noraml_not_summy = ((df_ordered['preciptotal'] > 0) & (df_ordered['WEvent'] == 0))
mask_weather = (df_ordered['WEvent'] == 1)

data_noraml_sunny = list(df_ordered.loc[mask_normal_sunny,'units'])
data_noraml_not_summy = list(df_ordered.loc[mask_noraml_not_summy,'units'])
data_weather = list(df_ordered.loc[mask_weather,'units'])

print "sunny vs non-sunny days in normal days"
perm_diff(data_noraml_sunny,data_noraml_not_summy,np.mean)

print "non-sunny normal days vs weather event days"
perm_diff(data_weather,data_noraml_not_summy,np.mean)