#!/usr/bin/env python
# coding: utf-8


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
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from sklearn.model_selection import train_test_split
from sklearn import datasets, ensemble
from sklearn.ensemble import RandomForestRegressor
import itertools
import statsmodels.api as sm
import timeit
from datetime import datetime, timedelta

#### RANDOM HELPER FUNCTIONS ####

def convert(seconds):
# returns the time in the format "%d:%02d:%02d" % (hour, minutes, seconds)
    seconds = seconds % (24 * 3600) 
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


def train_test(data, split_amount, n):
# returns the train and test sets split based on the split_amount 
    return data[0:split_amount], data[split_amount:split_amount+n]


def train_validation(data, split_amount):
# returns the training and validation sets 
    idx = int(len(data)*split_amount)
    return data[0:idx], data[idx:] # training set, validation set 


#### HELPER FUNCTIONS TO CALCULATE PERFORMANCE ####

def rmse(y,y_):
# returns RMSE 
    rmse = math.sqrt(sms.mean_squared_error(y, y_))
    return rmse

def percentage_error(actual, predicted):
# calculate precentage error for MAPE measure 
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape(y_true, y_pred): 
# returns MAPE 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100


def model_performance(y, y_):
# main helper function to calculate 
    NRMSE= round(rmse(y, y_)/(np.asarray(y).max()-np.asarray(y).min()),2)
    RMSE = round(rmse(y, y_), 2)
    MAPE = round(mape(y,y_), 2)
    ND = round((sum(abs(np.asarray(y)-np.asarray(y_)))/sum(np.asarray(y)))[0],2)
    return MAPE, RMSE, NRMSE, ND


#### HELPER FUNCTIONS FOR PLOTTING ####

def sns_plot(df_test,df_results):
    import seaborn as sns
    sns.set()
    fig = df_test.plot(figsize=(20,10), linewidth=5, fontsize=20)


def plot(train, test, prediction):
# plots actual train set, actual test set and predicted test set
    sns.set()
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=(30,10))
    plt.plot(train, label='Train')
    plt.plot(test, label='Test')
    plt.plot(prediction, label='Prediction')
    plt.legend()
    plt.show()
    


def plot_acutal_predict(y,y_,title):
# plots actual test set and predicted test set
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


#### HELPER FUNCTIONS FOR DATA PREPERATION ####

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


def filter_country(country, category, df):
# filter the dataset based on country of interest 
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

# function to prepare the dataset for train set for different models 
def ts_data(country = 'ENTIRE', category = 'total', frequency = '1H', model = None):
    df = load_data()
    df = df.sort_values(by=['datetime'])
    
    df = filter_country(country, category, df)

    #prepare the data for the model
    if model == None:
        X = df['y'].resample(frequency).sum()
        return X
    
    elif model == 'gbr_rf':
        X = df['y'].resample(frequency).sum()
        X = X.reset_index()
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

def performance_table_single(result_dict):
    country = list(result_dict.keys())[0]
    df= pd.DataFrame()
    
    df['Frequency'] = result_dict[country]['Frequency']
    df['MODEL'] = result_dict[country]['MODEL']
    df['RUN_TIME'] = result_dict[country]['RUN_TIME']
    df['Prediction Window'] = result_dict[country]['Prediction_Window']
    df['MAPE'] = result_dict[country]['MAPE']
    df['RMSE'] = result_dict[country]['RMSE']
    df['NRMSE'] = result_dict[country]['NRMSE']
    df['ND'] = result_dict[country]['ND']
    df['Country'] = country
    
    return df

#prepare the table storing the performance of each model
def performance_table(result_dict):
    df_US = pd.DataFrame()
    df_CA = pd.DataFrame()
    df_GB = pd.DataFrame()
    df_ENTIRE = pd.DataFrame()
    df_NA = pd.DataFrame()

    df_NA['Frequency'] = result_dict['NORTH_AMERICA']['Frequency']
    df_NA['MODEL'] = result_dict['NORTH_AMERICA']['MODEL']
    df_NA['RUN_TIME'] = result_dict['NORTH_AMERICA']['RUN_TIME']
    df_NA['Prediction Window'] = result_dict['NORTH_AMERICA']['Prediction_Window']
    df_NA['MAPE'] = result_dict['NORTH_AMERICA']['MAPE']
    df_NA['RMSE'] = result_dict['NORTH_AMERICA']['RMSE']
    df_NA['NRMSE'] = result_dict['NORTH_AMERICA']['NRMSE']
    df_NA['ND'] = result_dict['NORTH_AMERICA']['ND']
    df_NA['Country'] = 'NORTH AMERICA'

    df_US['Frequency'] = result_dict['US']['Frequency']
    df_US['MODEL'] = result_dict['US']['MODEL']
    df_US['RUN_TIME'] = result_dict['US']['RUN_TIME']
    df_US['Prediction Window'] = result_dict['US']['Prediction_Window']
    df_US['MAPE'] = result_dict['US']['MAPE']
    df_US['RMSE'] = result_dict['US']['RMSE']
    df_US['NRMSE'] = result_dict['US']['NRMSE']
    df_US['ND'] = result_dict['US']['ND']
    df_US['Country'] = 'US'

    df_CA['Frequency'] = result_dict['CA']['Frequency']
    df_CA['MODEL'] = result_dict['CA']['MODEL']
    df_CA['RUN_TIME'] = result_dict['CA']['RUN_TIME']
    df_CA['Prediction Window'] = result_dict['CA']['Prediction_Window']
    df_CA['MAPE'] = result_dict['CA']['MAPE']
    df_CA['RMSE'] = result_dict['CA']['RMSE']
    df_CA['NRMSE'] = result_dict['CA']['NRMSE']
    df_CA['ND'] = result_dict['CA']['ND']
    df_CA['Country'] = 'CA'
    
    df_GB['Frequency'] = result_dict['GB']['Frequency']
    df_GB['MODEL'] = result_dict['GB']['MODEL']
    df_GB['RUN_TIME'] = result_dict['GB']['RUN_TIME']
    df_GB['Prediction Window'] = result_dict['GB']['Prediction_Window']
    df_GB['MAPE'] = result_dict['GB']['MAPE']
    df_GB['RMSE'] = result_dict['GB']['RMSE']
    df_GB['NRMSE'] = result_dict['GB']['NRMSE']
    df_GB['ND'] = result_dict['GB']['ND']
    df_GB['Country'] = 'GB'
    
    df_ENTIRE['Frequency'] = result_dict['ENTIRE']['Frequency']
    df_ENTIRE['MODEL'] = result_dict['ENTIRE']['MODEL']
    df_ENTIRE['RUN_TIME'] = result_dict['ENTIRE']['RUN_TIME']
    df_ENTIRE['Prediction Window'] = result_dict['ENTIRE']['Prediction_Window']
    df_ENTIRE['MAPE'] = result_dict['ENTIRE']['MAPE']
    df_ENTIRE['RMSE'] = result_dict['ENTIRE']['RMSE']
    df_ENTIRE['NRMSE'] = result_dict['ENTIRE']['NRMSE']
    df_ENTIRE['ND'] = result_dict['ENTIRE']['ND']
    df_ENTIRE['Country'] = 'ENTIRE'
    
    FRAMES = [df_US, df_CA, df_GB, df_ENTIRE, df_NA]
    performance_table = pd.concat(FRAMES)
    
    return performance_table

#### HELPER FUNCTIONS FOR REGRESSION MODELS - GRID SEARCH, etc. ####

#SES
def grid_search_ses(input_ses):
    alpha = np.round(np.linspace(0,1,num=10),2)
    x, y = train_validation(input_ses,0.8)
    minError, optParam = float('inf'), None
    for a in alpha:
        y_hat = SimpleExpSmoothing(x).fit(smoothing_level=a,optimized=False).forecast(len(y))
        error = rmse(y,y_hat)
        if minError > error:
            minError, optParam = error, a
    return optParam

#GBR
def grid_search_gbr(input_gbr):
    # split train set into training subset and validation subset; 80% and 20% split respectively 
    train_input = input_gbr.drop(['y'], axis=1)
    t_train, t_test, v_train, v_test = train_test_split(train_input, input_gbr.y, test_size=0.20, shuffle=False)
    # p, q and d values to iter through
    n_estimators = [2,4,6,8,10,12,14,16,18,20,30,40,50,60,70,80,90,100]
    learning_rate = [0.1,0.05,0.02,0.01]
    max_depth = [2,3,4,5,6]
    minError, optParam = float('inf'), None
    for i in n_estimators:
        for j in learning_rate :
            for k in max_depth:
                params = {'n_estimators': i,
                  'max_depth': k,
                  'learning_rate': j,
                  'loss': 'ls',
                  'criterion': 'mse'}
                y_hat = gbr(t_train,v_train,t_test,params)
                error = rmse(y_hat, v_test)
                if error < minError:
                    minError, optParam = error, params
    return optParam

def gbr(x_train,y_train,x_test,params):
    model = ensemble.GradientBoostingRegressor(**params)
    model.fit(x_train, y_train)
    result = model.predict(x_test)
    return result

#RF
def grid_search_rf(input_rf,gbr1=False):
    # split train set into training subset and validation subset; 80% and 20% split respectively 
    train_input = input_rf.drop(['y'], axis=1)
    x_train, x_test, y_train, y_test = train_test_split(train_input, input_rf.y, test_size=0.20, shuffle=False)
    
    n_estimators = [2,4,6,8,10,12,14,16,18,20,30,40,50,100,150,200]
    max_features = [1,2,3,4,5,6,8,10,15,20,25,30,40]
    max_depth = [1,2,3,4,5,6,12,24,32]
    
    minError, optParam = float('inf'), None
    for i in n_estimators:
        for j in max_features :
            for k in max_depth:
                params = (i, j, k)
                model = RandomForestRegressor(n_estimators = i, max_features = j, max_depth = k, random_state = 42).fit(x_train, y_train)
                y_hat = model.predict(x_test)
                error = rmse(y_hat, y_test)
                if error < minError:
                    minError, optParam = error, params
    return optParam

#SARIMA
def sarima(p,d,q,P,D,Q,y):
    model = sm.tsa.statespace.SARIMAX(y,order=(p, d, q),seasonal_order=(P, D, Q, 12),enforce_stationarity=False,enforce_invertibility=False)
    results = model.fit()
    return results

def grid_search_sarima(input_sarima):
    AICscores = list()
    p = d = q = range(0, 2)    
    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
    
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                model = sm.tsa.statespace.SARIMAX(input_sarima,
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
                results = model.fit()
                AICscores.append([param, param_seasonal, results.aic])
            except:
                 continue
    AICscores = sorted(AICscores,key=lambda x:x[2])
    return AICscores[0]

#ARIMA
def arima_predict(train, parameters, end_length):
	p = ARIMA(train, parameters).fit().predict(start=len(train),end=end_length-1)
	return p

def grid_search_arima(input_arima):
    # split train set into training subset and validation subset; 80% and 20% split respectively 
    t, v = train_validation(input_arima, 0.8)
    # p, q and d values to iter through
    p = [0,1,2,3]
    q = [0,1,2,3]
    d = [0,1,2]
    minError, optParam = float('inf'), (0,0,0)
    for i in p:
	    for j in d:
	        for k in q:
	            parameters = (i,j,k)
	            try:
	                y_hat = arima_predict(t, parameters, len(input_arima))
	                error = rmse(y_hat, v)
	                if error < minError:
	                    minError, optParam = error, parameters
	            except:
	                continue
    return optParam


# Naive Baseline
def naive_baseline(input_train, input_test):
    dt_ = input_test.index - timedelta(days=7)
    result = input_train[dt_]
    return result