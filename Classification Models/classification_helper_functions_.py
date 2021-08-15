#!/usr/bin/env python
# coding: utf-8
# helper functions for classification models

# import libraries
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
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import timeit
from datetime import datetime, timedelta
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import SGD
from keras.layers import Bidirectional
from scipy.spatial import distance

#### random helper functions ####
def convert(seconds): 
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)

# train-test set generation functions
def train_test(data, split_amount, n):
    #returns training data and testing data
    return data[0:split_amount], data[split_amount:split_amount+n]

# builds and returns the training and validation sets 
def train_validation(data, split_amount): 
    idx = int(len(data)*split_amount)
    return data[0:idx], data[idx:] # training set, validation set 


#### performance functions ####

# main function to calculate all performance measures for classifiers 
def model_performance(y_test, y_pred):
    ACCURACY = round(sms.accuracy_score(y_test, y_pred),2)
    F1SCORE = round(sms.f1_score(y_test, y_pred, average = "weighted"),2)
    PRECISION = round(sms.precision_score(y_test, y_pred, average = "weighted"),2)
    RECALL = round(sms.recall_score(y_test, y_pred, average = "weighted"),2)
    return ACCURACY, F1SCORE, PRECISION, RECALL


#### graphing functions ####
def sns_plot(df_test,df_results):
    import seaborn as sns
    sns.set()
    fig = df_test.plot(figsize=(20,10), linewidth=5, fontsize=20)

# plots actual train set, actual test set and predicted test set
def plot(train, test, prediction):
    sns.set()
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(30,10))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()
    

# plots actual test set and predicted test set
def plot_acutal_predict(y,y_,title):
    sns.set()
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(40,15)).autofmt_xdate()
    plt.plot(y[-300:], label='Y Actual', color='grey', linewidth=4)
    plt.plot(y_[-300:], label='Y Prediction', color='blue', linewidth=4)
    plt.legend(fontsize="x-large")
    plt.tick_params(labelsize=35)
    plt.ylabel('Number of Patients', fontsize=35)
    plt.xticks(rotation=30)
    date_format = mdates.DateFormatter('%d-%m-%Y')
    plt.gca().xaxis.set_major_formatter(date_format)
    for idx, label in enumerate(plt.gca().xaxis.get_ticklabels()):
        if idx % 1 != 0:
            label.set_visible(False)
    plt.savefig(str(title)+'.png')

#### data preperation functions ####
def train_test_split_classification(data, split_amount, n):
    train = data[0:split_amount]
    test = data[split_amount:split_amount+n]

    x_train = train.drop(['y_class'], axis=1)
    y_train = train.y_class

    x_test = test.drop(['y_class'], axis=1)
    y_test = test.y_class
    
    return x_train, x_test, y_train, y_test

# function to load and prepare the data
def load_data(): 
    df = pd.read_excel('small_db_prepared_mod.xlsx')
    df.datehour = pd.DatetimeIndex(df.datehour)
    df["category"].replace({np.nan: "No Category", "": "No Category",'Follow up':'Follow Up', "Other":"No Category" }, inplace=True)
    
    categoryVolume = df.groupby(['datehour','country','category']).agg({'patientID':'nunique'}).reset_index()
    categoryVolume.columns = ['datetime','country','category','y']
    categoryVolume.set_index(['datetime','country'], inplace=True)
    categoryVolume = categoryVolume.pivot_table(
        values='y', 
        index=['datetime', 'country'], 
        columns='category', 
        aggfunc=np.sum)
    categoryVolume.columns = ['covid_19', 'cold_or_flu', 'emergency_tool', 'follow_up','gastrointestinal', 
                  'no_category', 'pregnancy', 'sexual_health','skin']
    categoryVolume=categoryVolume.fillna(0)
    categoryVolume['total'] = categoryVolume.sum(axis=1)
    categoryVolume=categoryVolume.reset_index().set_index('datetime')
    return categoryVolume

# filter the dataset based on country of interest 
def filter_country(country, category, df):
    if country == 'NORTH_AMERICA':
        North_America_Filter = ['CA','US']
        df = df[df.country.isin(North_America_Filter)]
    elif country == 'ENTIRE':
        df = df[[category]]
        df.columns=['y']
        return df
    else:
        df = df[df['country']==country]
    
    df = df[[category]]
    df.columns=['y']
    return df

# function to prepare the dataset for training
def ts_data(country = 'ENTIRE', category = 'total', frequency = '1H', model = None):
    df = load_data()
    df = df.sort_values(by=['datetime'])
    
    df = filter_country(country, category, df)

    df.index = pd.DatetimeIndex(df.index)
    X = df['y'].resample(frequency).sum()
    
    X = X.reset_index()
    X.columns=['ds','y']

    bin_labels = [0, 1, 2, 3] # 4 unique classes for classification
    X['y_bins'] = pd.cut(X['y'], bins=4)
    X['y_class'] = pd.cut(X['y'], bins=4, labels=bin_labels)
    labels = X.groupby(['y_bins','y_class']).agg({'y_class':'count'}) # map the regression dataset to a multiclass classification one
    
    if model == 'naive_baseline':
        X = X.drop(['y','y_bins'], axis=1)
        X = X.reset_index().set_index('ds')
        X = X.drop(['index'], axis=1)
        return X

    else: # add additional training features to the classification dataset
        dt = pd.DatetimeIndex(X.ds)
        X['x1'] = pd.DatetimeIndex(X['ds']).year
        X['x2'] = pd.DatetimeIndex(X['ds']).month
        X['x3'] = pd.DatetimeIndex(X['ds']).day
            
        # One-hot encoding 
        X = pd.concat([X,pd.get_dummies(X['x1'], prefix='year')],axis=1)
        X = pd.concat([X,pd.get_dummies(X['x2'], prefix='month')],axis=1)
        X = pd.concat([X,pd.get_dummies(X['x3'], prefix='date')],axis=1)
            
            
        X = X.drop(['ds','y','y_bins'], axis=1)
        return X, dt, labels



# prepare the table storing the performance of each model
def performance_table(result_dict):
    df_ENTIRE = pd.DataFrame()
    df_NA = pd.DataFrame()

    df_NA['Frequency'] = result_dict['NORTH_AMERICA']['Frequency']
    df_NA['MODEL'] = result_dict['NORTH_AMERICA']['MODEL']
    df_NA['RUN_TIME'] = result_dict['NORTH_AMERICA']['RUN_TIME']
    df_NA['Prediction Window'] = result_dict['NORTH_AMERICA']['Prediction_Window']
    df_NA['ACCURACY'] = result_dict['NORTH_AMERICA']['ACCURACY']
    df_NA['F1SCORE'] = result_dict['NORTH_AMERICA']['F1SCORE']
    df_NA['PRECISION'] = result_dict['NORTH_AMERICA']['PRECISION']
    df_NA['RECALL'] = result_dict['NORTH_AMERICA']['RECALL']
    df_NA['Country'] = 'NORTH AMERICA'

    df_ENTIRE['Frequency'] = result_dict['ENTIRE']['Frequency']
    df_ENTIRE['MODEL'] = result_dict['ENTIRE']['MODEL']
    df_ENTIRE['RUN_TIME'] = result_dict['ENTIRE']['RUN_TIME']
    df_ENTIRE['Prediction Window'] = result_dict['ENTIRE']['Prediction_Window']
    df_ENTIRE['ACCURACY'] = result_dict['ENTIRE']['ACCURACY']
    df_ENTIRE['F1SCORE'] = result_dict['ENTIRE']['F1SCORE']
    df_ENTIRE['PRECISION'] = result_dict['ENTIRE']['PRECISION']
    df_ENTIRE['RECALL'] = result_dict['ENTIRE']['RECALL']
    df_ENTIRE['Country'] = 'ENTIRE'

    FRAMES = [df_ENTIRE, df_NA]
    performance_table = pd.concat(FRAMES)
    
    return performance_table


#Model Helper Functions

#Naive Baseline
def naive_baseline(input_train, input_test_index):
    dt_ = input_test_index - timedelta(days=7)
    result = input_train.loc[dt_]
    return result

# helper functions for LSTM: lstm(), train_test_lstm(), split_series() and performance_table_lstm()
def lstm(steps,nodes,activation,optimize):
    model = Sequential()
    model.add(Bidirectional(LSTM(nodes, activation=activation, input_shape=(steps, 1))))
    model.add(Dense(nodes, activation=activation))
    model.add(Dense(1))
    model.compile(optimizer=optimize, loss='mse')
    return model 

def train_test_lstm(x_train, y_label, dt, train_amount, n):
	return x_train[0:train_amount], y_label[train_amount:train_amount+n], dt[train_amount:train_amount+n]

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

def performance_table_lstm(result_dict):
    df = pd.DataFrame()
    df['Frequency'] = result_dict['NORTH_AMERICA']['Frequency']
    df['MODEL'] = result_dict['NORTH_AMERICA']['MODEL']
    df['RUN_TIME'] = result_dict['NORTH_AMERICA']['RUN_TIME']
    df['Prediction Window'] = result_dict['NORTH_AMERICA']['Prediction_Window']
    df['ACCURACY'] = result_dict['NORTH_AMERICA']['ACCURACY']
    df['F1SCORE'] = result_dict['NORTH_AMERICA']['F1SCORE']
    df['PRECISION'] = result_dict['NORTH_AMERICA']['PRECISION']
    df['RECALL'] = result_dict['NORTH_AMERICA']['RECALL']
    df['Country'] = 'NORTH AMERICA'
    
    return performance_table

# DTW helper function used for KNN-DTW
def DTW(a, b):   
    an = a.size
    bn = b.size
    pointwise_distance = distance.cdist(a.reshape(-1,1),b.reshape(-1,1))
    cumdist = np.matrix(np.ones((an+1,bn+1)) * np.inf)
    cumdist[0,0] = 0
    
    for ai in range(an):
        for bi in range(bn):
            minimum_cost = np.min([cumdist[ai, bi+1],
                                   cumdist[ai+1, bi],
                                   cumdist[ai, bi]])
            cumdist[ai+1, bi+1] = pointwise_distance[ai,bi] + minimum_cost

    return cumdist[an, bn]