import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize']=20,10
%matplotlib inline

import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM,Dropout,Dense
from keras import optimizers
import datetime as dt
import os
import warnings
warnings.filterwarnings('ignore')

df=yf.download('MSFT', '2000-01-01', '2021-01-01')
df.head()
df.shape

plt.figure(figsize=(16,8))
plt.plot(df["Close"])

plt.title('Close Price history')
plt.show()

stock_prices=df.head(100)

                            
plt.figure()

up = stock_prices[stock_prices.Close >= stock_prices.Open]
down = stock_prices[stock_prices.Close < stock_prices.Open]
  
col1 = 'red'
col2 = 'green'
  
width = .5
width2 = .05
  
plt.bar(up.index, up.Close-up.Open, width, bottom=up.Open, color=col1)
plt.bar(up.index, up.High-up.Close, width2, bottom=up.Close, color=col1)
plt.bar(up.index, up.Low-up.Open, width2, bottom=up.Open, color=col1)
  
plt.bar(down.index, down.Close-down.Open, width, bottom=down.Open, color=col2)
plt.bar(down.index, down.High-down.Open, width2, bottom=down.Open, color=col2)
plt.bar(down.index, down.Low-down.Close, width2, bottom=down.Close, color=col2)
plt.title(" Candlestick pattern for the stock",size=20)

plt.xticks(rotation=30,ha="right")

plt.show()

df.corr()['Close']

def fit_model(train,val,timesteps,hl,lr,batch,epochs):
    X_train = []
    Y_train = []
    X_val = []
    Y_val = []
      
    for i in range(timesteps,train.shape[0]):
        X_train.append(train[i-timesteps:i])
        Y_train.append(train[i][0])
    X_train,Y_train = np.array(X_train),np.array(Y_train)
      
    for i in range(timesteps,val.shape[0]):
        X_val.append(val[i-timesteps:i])
        Y_val.append(val[i][0])
    X_val,Y_val = np.array(X_val),np.array(Y_val)
        
    model = Sequential()
    model.add(LSTM(X_train.shape[2],input_shape = (X_train.shape[1],X_train.shape[2]),return_sequences = True,
                   activation = 'relu'))
    for i in range(len(hl)-1):        
        model.add(LSTM(hl[i],activation = 'relu',return_sequences = True))
    model.add(LSTM(hl[-1],activation = 'relu'))
    model.add(Dense(1))
    model.compile(optimizer = optimizers.Adam(lr = lr), loss = 'mean_squared_error')
    
  
    
    history = model.fit(X_train,Y_train,epochs = epochs,batch_size = batch,validation_data = (X_val, Y_val),verbose = 0,
                        shuffle = False)
    model.reset_states()
    return model, history.history['loss'], history.history['val_loss']

def evaluate_model(model,test,timesteps):
    X_test = []
    Y_test = []   
    for i in range(timesteps,test.shape[0]):
        X_test.append(test[i-timesteps:i])
        Y_test.append(test[i][0])
    X_test,Y_test = np.array(X_test),np.array(Y_test)
    
    Y_hat = model.predict(X_test)
    mse = mean_squared_error(Y_test,Y_hat)
    rmse = sqrt(mse)
    r = r2_score(Y_test,Y_hat)
    return mse, rmse, r, Y_test, Y_hat

features = df[['Close','High','Volume']] # Picking the series with high correlation
features.shape

features.head()

train_start = dt.date(2000,1,1)
train_end = dt.date(2016,12,31)
train_data = features.loc[train_start:train_end]

val_start = dt.date(2017,1,1)
val_end = dt.date(2019,12,31)
val_data = features.loc[val_start:val_end]

test_start = dt.date(2020,1,1)
test_end = dt.date(2021,1,1)
test_data = features.loc[test_start:test_end]

sc = MinMaxScaler()
train = sc.fit_transform(train_data)
val = sc.transform(val_data)
test = sc.transform(test_data)


timesteps = 50 #using 50 days to predict 51st day ka price
hl = [8,5]
lr = 1e-3
batch_size = 32
epochs = 50

model,train_loss,val_loss = fit_model(train,val,timesteps,hl,lr,batch_size,epochs)

plt.plot(train_loss,c = 'r')
    plt.plot(val_loss,c = 'b')
    plt.ylabel('Loss')
    plt.legend(['train','val'],loc = 'upper right')
    plt.show()

    
mse, rmse, r2_value,true,predicted = evaluate_model(model,test,timesteps)
plt.plot(predicted,c = 'r')
    plt.plot(true,c = 'y')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.title('Stock Prediction Graph using Multivariate-LSTM model')
    plt.legend(['Actual','Predicted'],loc = 'lower right')
    plt.show()

print('MSE = {}'.format(mse))
print('RMSE = {}'.format(rmse))
print('R-Squared Score = {}'.format(r2_value))
    





