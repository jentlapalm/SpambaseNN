# -*- coding: utf-8 -*-
"""
Created on Wed Sep 26 17:03:29 2018

@author: GEASTON
"""

#%% Get Spiral Data

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import optimizers

NEpochs = 10000
BatchSize=250
#Optimizer=optimizers.SGD(lr=0.01)
Optimizer=optimizers.RMSprop(learning_rate=0.01)

# Read in the data

TrainData = pd.read_csv('SpiralTrain.csv',sep=',',header=0,quotechar='"')
list(TrainData)
ValData = pd.read_csv('SpiralVal.csv',sep=',',header=0,quotechar='"')
list(ValData)

# Rescale the training data

TrColorCode = np.array(TrainData['Color'].astype('category').cat.codes)
TrainData[['Color']].astype('category')
TrColorCode
# Blue is 0, red is 1

TrX = np.array(TrainData.loc[:,['x','y']])

TrXrsc = (TrX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(TrXrsc.shape)
print(TrXrsc.min(axis=0))
print(TrXrsc.max(axis=0))

# No need to rescale the Y (which is Color Code) because it is categorical

# Rescale the validation data

ValColorCode = np.array(ValData['Color'].astype('category').cat.codes)
ValData[['Color']].astype('category')
ValColorCode
# Blue is 0, red is 1

ValX = np.array(ValData.loc[:,['x','y']])

# Be sure to use the parameters from the training datq

ValXrsc = (ValX - TrX.min(axis=0))/TrX.ptp(axis=0)
print(ValXrsc.shape)
print(ValXrsc.min(axis=0))
print(ValXrsc.max(axis=0))

#%% Set up Neural Net Model

SpiralNN = Sequential()

SpiralNN.add(Dense(units=10,input_shape=(TrXrsc.shape[1],),activation="relu",use_bias=True))
SpiralNN.add(Dense(units=10,activation="relu",use_bias=True))
SpiralNN.add(Dense(units=10,activation="relu",use_bias=True))
SpiralNN.add(Dense(units=10,activation="relu",use_bias=True))
SpiralNN.add(Dense(units=10,activation="relu",use_bias=True))
SpiralNN.add(Dense(units=10,activation="relu",use_bias=True))
SpiralNN.add(Dense(units=1,activation="sigmoid",use_bias=True))

SpiralNN.compile(loss='binary_crossentropy', optimizer=Optimizer,metrics=['binary_crossentropy','accuracy'])
print(SpiralNN.summary())
#%% Fit NN Model

from tensorflow.keras.callbacks import EarlyStopping

StopRule = EarlyStopping(monitor='val_loss',mode='min',verbose=0,patience=50,min_delta=0.0)
FitHist = SpiralNN.fit(TrXrsc,TrColorCode,validation_data=(ValXrsc,ValColorCode), \
                    epochs=NEpochs,batch_size=BatchSize,verbose=0, \
                    callbacks=[StopRule])
    
#FitHist = SpiralNN.fit(TrXrsc,TrColorCode,epochs=NEpochs,batch_size=BatchSize,verbose=0)

print("Number of Epochs = "+str(len(FitHist.history['accuracy'])))
print("Final training accuracy: "+str(FitHist.history['accuracy'][-1]))
print("Recent history for training accuracy: "+str(FitHist.history['accuracy'][-10:-1]))
print("Final validation accuracy: "+str(FitHist.history['val_accuracy'][-1]))
print("Recent history for validation accuracy: "+str(FitHist.history['val_accuracy'][-10:-1]))

#%% Make Predictions

TrPRed = SpiralNN.predict(TrXrsc,batch_size=TrXrsc.shape[0])
ValPRed = SpiralNN.predict(ValXrsc,batch_size=TrXrsc.shape[0])


GridN = 100
x1g = np.arange(0,GridN)/(GridN-1)*(max(TrX[:,0])-min(TrX[:,0]))+min(TrX[:,0])
x1g = np.tile(x1g,GridN)

x2g = np.arange(0,GridN)/(GridN-1)*(max(TrX[:,1])-min(TrX[:,1]))+min(TrX[:,1])
x2g = np.repeat(x2g,GridN)

Xgrid = np.concatenate((np.reshape(x1g,(len(x1g),1)),np.reshape(x2g,(len(x2g),1))),axis=1)


Ygrid = SpiralNN.predict((Xgrid-Xgrid.min(axis=0))/TrX.ptp(axis=0),batch_size=Xgrid.shape[0])


#%% Write out prediction

TrainData['TrPRed'] = TrPRed
list(TrainData)
ValData['ValPRed'] = ValPRed

TrainData.to_csv('SpiralTrainOut.csv',sep=',',na_rep="NA",header=True,index=False)
ValData.to_csv('SpiralValOut.csv',sep=',',na_rep="NA",header=True,index=False)

Out = pd.DataFrame(np.concatenate((Xgrid,Ygrid),axis=1),columns=["x1g","x2g","Ygrid"],copy=True)
Out.to_csv('SpiralMachineOut.csv',sep=',',na_rep="NA",header=True,index=False)


