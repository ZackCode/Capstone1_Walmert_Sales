#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 09:57:36 2017

@author: zexi
"""
import pandas as pd
import numpy as np
import sys
import re
df_ordered = pd.read_csv("../code/ordered.csv")
df_severeWeather = pd.read_csv("../code/severeWeather.csv")

# top10 sales per day
data = df_ordered[['units','item_nbr']]
count = data.groupby(['item_nbr']).aggregate(np.mean).unstack()

import matplotlib.pyplot as plt
plt.style.use('ggplot')

count.sort_values().tail(10).plot(kind='barh')
plt.title('Top 10 best selling product in all stores per day')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# comparsion: normal vs extreme
data = df_ordered[['units','item_nbr','WEvent']]
count = data.groupby(['item_nbr','WEvent']).aggregate(np.mean).unstack()
count.columns = ['normal','extrme_weather']
count.sort_values(by='extrme_weather').tail(10).plot(kind='barh')
plt.title('Top 10 best selling product in all stores per day during extreme weather periods compared to normal days')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# comparsion: before extreme vs others
mask = (df_ordered['WEvent'] == 1)

df_ordered['Before_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(-3) > 0)
df_ordered['After_Event'] = (pd.rolling_mean(df_ordered['Condition'], window=3).shift(1) > 0)

mask = (df_ordered['WEvent'] == 1)
data = df_ordered.loc[mask,['units','item_nbr','Before_Event']]
count = data.groupby(['item_nbr','Before_Event']).aggregate(np.mean).unstack()
count.columns = ['Not_Before_Event','Before_Event']
count.sort_values(by='Before_Event').tail(10).plot(kind='barh')
plt.title('Top 10 best selling product in all stores per day before extreme weather days compared by not before extreme weather days')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# comparsion: after extreme days vs others
mask = (df_ordered['WEvent'] == 1)

data = df_ordered.loc[mask,['units','item_nbr','After_Event']]
count = data.groupby(['item_nbr','After_Event']).aggregate(np.mean).unstack()
count.columns = ['Not_After_Event','After_Event']
count.sort_values(by='After_Event').tail(10).plot(kind='barh')
plt.title('Top 10 best selling product in all stores per day after extreme weather days compared not after extreme weather days')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

#comparsion: on extreme days vs others
mask = (df_ordered['WEvent'] == 1)

data = df_ordered.loc[mask,['units','item_nbr','Condition']]
count = data.groupby(['item_nbr','Condition']).aggregate(np.mean).unstack()
count.columns = ['Not_on_The_Day','The_Day']
count.sort_values(by='The_Day').tail(10).plot(kind='barh')
plt.title('Top 10 best selling product in all stores per day on extreme weather days compared by not on extreme weather days')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# comparsion: normal less rainfall days VS less rainfall days close to extreme weather
mask = (df_ordered['Condition'] == 0)

data = df_ordered.loc[mask,['units','item_nbr','WEvent']]
count = data.groupby(['item_nbr','WEvent']).aggregate(np.mean).unstack()
count.columns = ['Normal_Days','Days_in_Extreme_Event']
count.sort_values(by='Days_in_Extreme_Event').tail(10).plot(kind='barh')
plt.title('normal less rainfall days VS less rainfall days close to extreme weather')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# compasion: conclusion
mask = (df_ordered['WEvent'] == 1)

data = df_ordered.loc[mask,['units','item_nbr','Condition','Before_Event','After_Event']]
count = data.groupby(['item_nbr','Condition','Before_Event','After_Event']).aggregate(np.mean).unstack().unstack().unstack()
count.columns = ['None','on the day only','before event only','both on the day and before event','after event only','both after event and on the day','both after event and before event','both on the day, before an event and after an event']
count = count.drop('None',1)
count.sort_values('on the day only').tail(4).plot.barh(figsize=(20,10))
plt.title('Average item sold per day based on conditions')
plt.xlabel('average items per day')
plt.ylabel('item number')
plt.show()

# item 5: year
plt.style.use('ggplot')

df_ordered['year'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.year

mask = (df_ordered['item_nbr'] == 5)
data = df_ordered.loc[mask,['units','year']]

count = data.groupby(['year']).aggregate(np.mean).unstack()
count.index = count.index.droplevel()


mask = ((df_ordered['item_nbr'] == 5) & (df_ordered['WEvent'] == 1))
data = df_ordered.loc[mask,['units','year']]

count_severe = data.groupby(['year']).aggregate(np.mean).unstack()
count_severe.index = count_severe.index.droplevel()

count = pd.concat([count, count_severe], axis=1)
count.columns = ['normal','extreme weather period']

count.plot(kind='bar')

plt.title('Daily sold item by year')
plt.xlabel('year')
plt.ylabel('average items per day')


plt.show()

# item 51: month
df_ordered['month'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.month

mask = (df_ordered['item_nbr'] == 51)
data = df_ordered.loc[mask,['units','month']]

count = data.groupby(['month']).aggregate(np.mean).unstack()
count.index = count.index.droplevel()


mask = ((df_ordered['item_nbr'] == 51) & (df_ordered['WEvent'] == 1))
data = df_ordered.loc[mask,['units','month']]

count_severe = data.groupby(['month']).aggregate(np.mean).unstack()
count_severe.index = count_severe.index.droplevel()

count = pd.concat([count, count_severe], axis=1)
count.columns = ['normal','extreme weather period']

count.plot(kind='bar')

plt.title('Daily sold item by month')
plt.xlabel('month')
plt.ylabel('average items per day')


plt.show()

# item 5: weekday
df_ordered['weekday'] = pd.to_datetime(df_ordered['date'], infer_datetime_format=True).dt.weekday

mask = (df_ordered['item_nbr'] == 5)
data = df_ordered.loc[mask,['units','weekday']]

count = data.groupby(['weekday']).aggregate(np.mean).unstack()
count.index = count.index.droplevel()


mask = ((df_ordered['item_nbr'] == 5) & (df_ordered['WEvent'] == 1))
data = df_ordered.loc[mask,['units','weekday']]

count_severe = data.groupby(['weekday']).aggregate(np.mean).unstack()
count_severe.index = count_severe.index.droplevel()

count = pd.concat([count, count_severe], axis=1)
count.columns = ['normal','extreme weather period']
import calendar
count.index = list(calendar.day_name)

count.plot(kind='bar')

plt.title('Daily sold item by weekday')
plt.xlabel('weekday')
plt.ylabel('average items per day')


plt.show()

# scatter plot rainfall vs sales
mask = (df_ordered['item_nbr'] == 5)
data = df_ordered.loc[mask,['units','preciptotal']]

plt.scatter(data['preciptotal'],data['units'])
plt.title('rainfall/snowfall')
plt.xlabel('item 5 sold per day')
plt.ylabel('item 5 sold per day against daily rainfall/snowfall')

plt.show()

# subgroup boxplot: rainfall vs sales
import seaborn as sns

def bsfromfunc(observes,targetfunc):
    value = targetfunc(observes)
    bs_target = np.empty(10000)
    for i in range(10000):
        bs_sample = np.random.choice(observes,size=len(observes))
        bs_target[i] = targetfunc(bs_sample)
        
    print 'value, ', value
    print '99% interval, ', np.percentile(bs_target, [0.5, 99.5])
    
    return value,np.percentile(bs_target, [0.5, 99.5])

def cont_boxplot(data,xlabel,ylabel,seg=10):
    data_range = np.percentile(data[xlabel], [100*float(x)/seg for x in range(seg+1)])
    data_range = sorted(list(set(data_range)))
    
    print data_range
    
    seg = len(data_range) - 1
    data['x_label']=str(data_range[seg-1])+' to '+str(data_range[seg])
    value = []
    value_interv_h = []
    value_interv_l = []
    number = []
    order = []
    
    data.loc[(data[xlabel]==data_range[0]),'x_label'] = str(data_range[0])
    obsers = data.loc[data['x_label']==str(data_range[0]),ylabel]
    print '\n'+'for '+xlabel+' range in '+str(data_range[0])
    print 'number of observations: '+str(len(obsers))
    t_value,t_interv = bsfromfunc(obsers,np.mean)
    value.append(t_value)
    value_interv_l.append(t_interv[0])
    value_interv_h.append(t_interv[1])
    number.append(len(obsers))
    order.append(str(data_range[0]))
    for i in range(seg):
        data.loc[(data[xlabel]>data_range[i])&(data[xlabel]<=data_range[i+1]),'x_label']=str(data_range[i])+' to '+str(data_range[i+1])
        obsers = data.loc[data['x_label']==str(data_range[i])+' to '+str(data_range[i+1]),ylabel]
        print '\n'+'for '+xlabel+' range in '+str(data_range[i])+' to '+str(data_range[i+1])
        print 'number of observations: '+str(len(obsers))
        t_value,t_interv = bsfromfunc(obsers,np.mean)
        value.append(t_value)
        value_interv_l.append(t_interv[0])
        value_interv_h.append(t_interv[1])
        number.append(len(obsers))
        order.append(str(data_range[i])+' to '+str(data_range[i+1]))
    f, (ax1, ax2, ax3) = plt.subplots(3, figsize=(16,8))
    sns.boxplot(x='x_label',y=ylabel,data=data,ax=ax1,order=order)
    ax1.set_title(xlabel+' VS '+ylabel)
    ax1.set_xlabel(xlabel + ' group range')
    ax1.set_ylabel(ylabel)
    #ax1.set_xticklabels(data['x_label'],rotation='vertical')
    
    order.insert(0,'Somehow the set_xticklabel start from 1')
    ax2.plot(value,c='red', label='mean')
    ax2.plot(value_interv_l,c='green', label='low 99% interval board')
    ax2.plot(value_interv_h,c='green', label='high 99% interval board')
    ax2.plot()
    ax2.set_title('mean and its confidence intervals')
    ax2.set_xlabel(xlabel + ' group range')
    ax2.set_ylabel('mean and 99% confidence interval')
    ax2.set_xticklabels(order)
    ax2.legend()
    
    #ax2.plot(number)
    ax3.bar(np.array(np.arange(len(number))), np.array(number), width=0.8)
    ax3.set_xticklabels(order)
    ax3.set_title('number of observations')
    ax3.set_xlabel(xlabel + ' group range')
    ax3.set_ylabel('number of observations')  
    ax3.set_xticklabels(order)
    plt.tight_layout()
    plt.show()
    #return value, value_interv_l, value_interv_h
    
mask = (df_ordered['item_nbr'] == 5)
data = df_ordered.loc[mask,['units','preciptotal']]
cont_boxplot(data,'preciptotal','units',seg=20)

# change intervals
mask = (df_severeWeather['item_nbr'] == 5)
data = df_severeWeather.loc[mask,['units','preciptotal']]
cont_boxplot(data,'preciptotal','units',seg=10)

# the week around extreme days
mask = ((df_severeWeather['item_nbr'] == 5) & (df_severeWeather['Condition']))
data = df_severeWeather.loc[mask,['units','preciptotal']]
cont_boxplot(data,'preciptotal','units',seg=6)

# temperature vs sales
mask = (df_ordered['item_nbr'] == 5)
data = df_ordered.loc[mask,['units','tavg']]
cont_boxplot(data,'tavg','units',seg=8)

mask = (df_severeWeather['item_nbr'] == 5)
data = df_severeWeather.loc[mask,['units','tavg']]
cont_boxplot(data,'tavg','units',seg=10)

mask = ((df_severeWeather['item_nbr'] == 5) & (df_severeWeather['Condition']))
data = df_severeWeather.loc[mask,['units','tavg']]
cont_boxplot(data,'tavg','units',seg=6)