#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import math
import sklearn.metrics as sms
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns
import plotly.express as px
import matplotlib.dates as mdates
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.ensemble import RandomForestRegressor
import itertools
import statsmodels.api as sm
import timeit
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from keras.layers import Bidirectional


#Random helper functions
def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

def train_test(data, split_amount, n):
    #returns training data and testing data
    return data[0:split_amount], data[split_amount:split_amount+n]


#Data Preperation function
def prep_data (data_type, model = None):
    bmc_monthly_train = pd.read_excel('bmc_hospital_monthly_train.xlsx')
    bmc_monthly_test = pd.read_excel('bmc_hospital_monthly_test.xlsx')

    bmc_daily_train = pd.read_excel('bmc_hospital_daily_train.xlsx')
    bmc_daily_test = pd.read_excel('bmc_hospital_daily_test.xlsx')

    bmc_monthly_train.date = pd.DatetimeIndex(bmc_monthly_train.date)
    bmc_monthly_test.date = pd.DatetimeIndex(bmc_monthly_test.date)
    bmc_daily_train.date = pd.DatetimeIndex(bmc_daily_train.date)
    bmc_daily_test.date = pd.DatetimeIndex(bmc_daily_test.date)
    
    bmc_daily_train = bmc_daily_train.drop(['week'],axis=1)
    bmc_daily_test = bmc_daily_test.drop(['week'],axis=1)
    if model == None:
        if (data_type == 'modeling_monthly'):
            modeling_monthly = bmc_monthly_train.reset_index().set_index('date')
            X = modeling_monthly['y'].resample('M').sum()

        elif (data_type == 'modeling_daily'):
            modeling_daily = bmc_daily_train.reset_index().set_index('date')
            X = modeling_daily['y'].resample('D').sum()

        elif (data_type == 'testing_monthly'):
            frames = [bmc_monthly_train, bmc_monthly_test]
            train_test_monthly = pd.concat(frames)
            train_test_monthly = train_test_monthly.reset_index().set_index('date')
            X = train_test_monthly['y'].resample('M').sum()

        elif (data_type == 'testing_daily'):
            frames = [bmc_daily_train, bmc_daily_test]
            train_test_daily = pd.concat(frames)
            train_test_daily = train_test_daily.reset_index().set_index('date')
            X = train_test_daily['y'].resample('D').sum()
    
    elif model == 'random_forest':
        if (data_type == 'modeling_monthly'):
            modeling_monthly = bmc_monthly_train.reset_index().set_index('date')
            data = modeling_monthly['y'].resample('M').sum()

        elif (data_type == 'modeling_daily'):
            modeling_daily = bmc_daily_train.reset_index().set_index('date')
            data = modeling_daily['y'].resample('D').sum()

        elif (data_type == 'testing_monthly'):
            frames = [bmc_monthly_train, bmc_monthly_test]
            train_test_monthly = pd.concat(frames)
            train_test_monthly = train_test_monthly.reset_index().set_index('date')
            data = train_test_monthly['y'].resample('M').sum()

        elif (data_type == 'testing_daily'):
            frames = [bmc_daily_train, bmc_daily_test]
            train_test_daily = pd.concat(frames)
            train_test_daily = train_test_daily.reset_index().set_index('date')
            data = train_test_daily['y'].resample('D').sum()
        
        X = data.reset_index()
        X.columns=['ds','y']
        
        dt = pd.DatetimeIndex(X.ds)
        
        X['x1'] = pd.DatetimeIndex(X['ds']).year
        X['x2'] = pd.DatetimeIndex(X['ds']).month
        X['x3'] = pd.DatetimeIndex(X['ds']).day
        
        # One-hot encoding 
        X = pd.concat([X,pd.get_dummies(X['x1'], prefix='year')],axis=1)
        X = pd.concat([X,pd.get_dummies(X['x2'], prefix='month')],axis=1)
        X = pd.concat([X,pd.get_dummies(X['x3'], prefix='date')],axis=1)
        X = X.drop(['ds'], axis=1)
        return X, dt
    
    return X 

#Performance functions
def rmse(y,y_):
    rmse = math.sqrt(sms.mean_squared_error(y, y_))
    return rmse

def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape_measure(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def model_performance(y, y_):
    NRMSE= round(math.sqrt(sms.mean_squared_error(y, y_))/(np.asarray(y).max()-np.asarray(y).min()),2)
    RMSE = round(math.sqrt(sms.mean_squared_error(y, y_)), 2)
    MAPE = round(mape_measure(y,y_), 2)
    ND = round((sum(abs(np.asarray(y)-np.asarray(y_)))/sum(np.asarray(y)))[0],2)
    MAE = round(mean_absolute_error(np.asarray(y), np.asarray(y_)),2)
    return MAPE, RMSE, MAE


#SARIMA functions
def sarima(p,d,q,P,D,Q,s,y):
    model = sm.tsa.statespace.SARIMAX(y,order=(p, d, q),seasonal_order=(P, D, Q, s),enforce_stationarity=False,enforce_invertibility=False)
    results = model.fit()
    return results

#LSTM functions
def lstm_function(steps,nodes,activation,optimize):
    model = Sequential()
    model.add(Bidirectional(LSTM(nodes, activation=activation, input_shape=(steps, 1))))
    model.add(Dense(nodes, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimize, loss='mse')
    return model 

def split_series(t_series, datetime, previous_steps):
    xx, yy, dt = list(), list(), list()
    for i in range(len(t_series)):
        # end of the pattern
        end_ix = i + previous_steps
        # Don't exceed the end of the series!
        if end_ix > len(t_series)-1:
            break
        # input , output in an  autoregressive manner
        seq_x, seq_y, seq_dt = t_series[i:end_ix], t_series[end_ix], datetime[end_ix]
        xx.append(seq_x)
        yy.append(seq_y)
        dt.append(seq_dt)
    return np.array(xx), np.array(yy), np.array(dt)

def train_test_lstm(x_train, y_label, dt, train_amount, n):
    return x_train[0:train_amount], y_label[train_amount:train_amount+n], dt[train_amount:train_amount+n]