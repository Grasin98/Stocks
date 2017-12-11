# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 11:29:00 2017

@author: NISARG
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense,Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
import time


#from numpy._distributor_init import NUMPY_MKL 

train=pd.read_csv("C:/Users/NISARG/Desktop/WinPython/DEEP LEARNING/Recurrent_Neural_Networks/Google_Stock_Price_Train.csv")
test=pd.read_csv("C:/Users/NISARG/Desktop/WinPython/DEEP LEARNING/Recurrent_Neural_Networks/Google_Stock_Price_Test.csv")

train=train.iloc[:,1:2].values
'''
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
train=sc.inverse_transform(train)

'''
X_train=train[0:1257]
X_test=test[0:1257]
y_train=train[1:1258]

X_train=np.reshape(X_train,(1257,1,1))

model=Sequential()

model.add(LSTM(input_dim=1,output_dim=50,
          return_sequences=True))
model.add(Dropout(0.3))

model.add(LSTM(100,return_sequences=False))

model.add(Dropout(0.2))

model.add(Dense(output_dim=1))
model.add(Activation("relu"))

start=time.time()
model.compile(loss="mse",optimizer="rmsprop")


model.fit(X_train,y_train,batch_size=512,nb_epoch=100,validation_split=0.05)
