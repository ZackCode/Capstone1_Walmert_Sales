#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 13:49:49 2017

@author: zexi
"""

import numpy as np
import scipy as sp 
from sknn.mlp import Regressor, Layer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn import datasets
from sklearn.grid_search import RandomizedSearchCV
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy import stats
from learning import X_train, X_test, y_train, y_test



X_trainn = X_train.as_matrix()
X_testn = X_test.as_matrix()
y_trainn = y_train.as_matrix()
y_testn = y_test.as_matrix()
#X_trainn = X_train
#X_testn = X_test
#y_trainn = y_train
#y_testn = y_test


#max_net = Regressor(layers= [Layer("Rectifier",units=1),
#                                       Layer("Rectifier",units=1),
#                                       Layer("Rectifier",units=1),
#                                       Layer("Rectifier",units=1),
#                                       Layer("Linear")])
#params={'learning_rate': [.002],
#        'hidden0__units': sp.stats.randint(2, 8),
#        'hidden0__type': ["Rectifier"],
#        'hidden1__units': sp.stats.randint(2, 8),
#        'hidden1__type': ["Rectifier"],
#        'hidden2__units': sp.stats.randint(2, 8),
#        'hidden2__type': ["Rectifier"],
#        'learning_rule':["adam","rmsprop","sgd"]}
#max_net2 = RandomizedSearchCV(max_net,param_distributions=params,n_iter=10,cv=3,random_state=101,scoring='neg_mean_squared_error',verbose=10,\
#                             pre_dispatch=None)
#
#model_tuning=max_net2.fit(X_trainn,y_trainn)
#
#print ("best score %s" % model_tuning.best_score_)
#print ("best parameters %s" % model_tuning.best_params_)

reg2 = Regressor(
    layers=[
        Layer("Rectifier", units=5),   
        Layer("Rectifier", units=4), 
        Layer("Rectifier", units=4), 
        Layer("Linear")],    learning_rate=0.002, learning_rule='rmsprop',random_state=201,n_iter=200)

model1=reg2.fit(X_trainn, y_trainn)
y_hat=reg2.predict(X_testn)


import matplotlib.pyplot as plt

fig, ax = plt.subplots()

y_test_p = y_test
y_hat_p = y_hat

ax.scatter(y_test_p, y_hat_p)
ax.plot([y_test_p.min(), y_test_p.max()], [y_test_p.min(), y_test_p.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

from sklearn import metrics

print("RMSE:",np.sqrt(metrics.mean_squared_error(y_test_p, y_hat_p)))